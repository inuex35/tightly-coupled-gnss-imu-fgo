// Structured runtime state for the TC pipeline (state/runtime_state.py,
// state/epoch_data.py).
#pragma once

#include "gnss_fgo/common.h"
#include "gnss_fgo/nav_state.h"
#include "gnss_fgo/rinex.h"

#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gnss_fgo {

/// All bookkeeping for a single (sat, freq) pair.
struct SatState {
  int outc = 0;                              // epochs since last seen
  std::optional<gtsam::Key> amb_key;         // GTSAM symbol for N
  int amb_gen = 0;                           // ++ on slip / reset
  std::optional<double> held_value;          // conditioned-out held integer [cyc]
  std::optional<double> last_held_value;     // last held integer for re-seed [cyc]
  bool release_seed_pending = false;         // one-shot prior on first float epoch
  int fix_streak = 0;                        // consecutive Fix epochs

  /// Move this ambiguity out of the graph and pin its integer externally.
  void activate_hold(double value) {
    held_value = value;
    last_held_value = value;
    release_seed_pending = false;
  }
  /// Drop the active hold, optionally queueing a one-shot re-seed prior.
  void release_hold(bool seed = true) {
    if (held_value) last_held_value = *held_value;
    held_value.reset();
    release_seed_pending = seed && last_held_value.has_value();
  }
  /// Forget all held-integer state (slip / reset / outage path).
  void clear_hold() {
    held_value.reset();
    last_held_value.reset();
    release_seed_pending = false;
  }
};

/// Per-(sat, freq) SatState map (insertion ordered) + per-sat GF memory.
class SatStateMap {
 public:
  int maxout = 5;
  std::map<int, double> gf;   // sat -> geometry-free combination [m]

  /// Get-or-create.
  SatState& get(int sat, int freq) {
    auto& p = track_[SfKey{sat, freq}];
    if (!p) p = std::make_unique<SatState>();
    return *p;
  }
  /// Lookup without creating (an empty state when absent).
  const SatState& at(int sat, int freq) const {
    static const SatState empty;
    const auto* p = track_.find(SfKey{sat, freq});
    return p ? **p : empty;
  }
  SatState* find(const SfKey& k) {
    auto* p = track_.find(k);
    return p ? p->get() : nullptr;
  }
  bool contains(const SfKey& k) const { return track_.contains(k); }

  /// Iterate (key, state) in insertion order.
  template <typename F>
  void for_each(F&& f) {
    for (auto& [k, p] : track_) f(k, *p);
  }
  template <typename F>
  void for_each(F&& f) const {
    for (const auto& [k, p] : track_) f(k, static_cast<const SatState&>(*p));
  }
  std::vector<SfKey> keys() const {
    std::vector<SfKey> out;
    for (const auto& [k, _] : track_) out.push_back(k);
    return out;
  }

  /// Snapshot {(sat, f): amb_key}, sorted by (sat, f).
  std::map<SfKey, gtsam::Key> amb_keys_dict() const {
    std::map<SfKey, gtsam::Key> out;
    for (const auto& [k, p] : track_)
      if (p->amb_key) out.emplace(k, *p->amb_key);
    return out;
  }
  /// (key, amb_key) in insertion order.
  std::vector<std::pair<SfKey, gtsam::Key>> amb_items() const {
    std::vector<std::pair<SfKey, gtsam::Key>> out;
    for (const auto& [k, p] : track_)
      if (p->amb_key) out.emplace_back(k, *p->amb_key);
    return out;
  }
  int amb_count() const {
    int n = 0;
    for (const auto& [k, p] : track_) n += p->amb_key ? 1 : 0;
    return n;
  }
  /// Active holds, insertion order.
  std::vector<std::pair<SfKey, double>> held_items() const {
    std::vector<std::pair<SfKey, double>> out;
    for (const auto& [k, p] : track_)
      if (p->held_value) out.emplace_back(k, *p->held_value);
    return out;
  }
  std::vector<gtsam::Key> amb_key_values() const {
    std::vector<gtsam::Key> out;
    for (const auto& [k, p] : track_)
      if (p->amb_key) out.push_back(*p->amb_key);
    return out;
  }

 private:
  OrderedMap<SfKey, std::unique_ptr<SatState>> track_;
};

/// The first AR attempt's joint-marginal extraction, sliced by retries.
struct ArJointCache {
  gtsam::Key key_pose = 0;
  std::map<SfKey, int> pos;
  Eigen::MatrixXd cov, cross;
};

/// State that lives exactly one epoch; replaced wholesale each epoch.
struct CurrentEpochState {
  std::optional<ArJointCache> ar_joint_cache;
  std::map<int, int> ref_sats;          // sys -> reference sat
  std::set<SfKey> seeded_amb_keys;      // (re)seeded this epoch -> no AR
};

/// Per-AR-call diagnostic counters.
struct ArDiagnostics {
  std::string outcome = "not_called";
  std::optional<std::string> exception;
  std::optional<int> orphan_cp_count, amb_dict_size, held_size, cp_visible_size,
      amb_estimate_missing, amb_vsat1, amb_vsat0_young, amb_not_in_obs,
      held_not_in_obs, sat_in_obs_size, resamb_raw_nb, exact_fit_accept,
      amb_el_min_deg, amb_el_median_deg, amb_el_above15, amb_el_above25,
      partial_dropped;
};

/// Cross-cutting recovery / degradation flags.
struct RecoveryState {
  int recov_cp_hold = 0;
  int cp_hold_retrigger_streak = 0;
  bool pim_discontinuity = false;
  int ddpr_bad_count = 0;
  std::optional<gtsam::Pose3> zupt_anchor_pose;
  std::optional<int> zupt_anchor_start_ep;

  /// Engage the global CP-hold for ``hold_n`` epochs (idempotent max).
  bool start_cp_hold(int hold_n, const std::string& reason, Info& info,
                     std::optional<double> value, bool skip_if_active) {
    if (hold_n <= 0) return false;
    if (recov_cp_hold > 0) {
      ++cp_hold_retrigger_streak;
      info.set("cp_hold_retrigger_streak", cp_hold_retrigger_streak);
    } else {
      cp_hold_retrigger_streak = 0;
    }
    if (skip_if_active && recov_cp_hold > 0) return false;
    recov_cp_hold = std::max(recov_cp_hold, hold_n);
    if (value) info.set("cp_hold_" + reason, *value);
    else info.set("cp_hold_" + reason, true);
    return true;
  }
  /// One CP-hold countdown step (only while the hold is active).
  void tick_cp_hold(Info& info) {
    recov_cp_hold -= 1;
    info.set("recov_cp_hold", recov_cp_hold + 1);
  }
};

/// Snapshot of the previous-epoch main DDPR residuals.
struct MresSignalsState {
  double last_res = 0.0;
  OrderedMap<int, double> per_sat;
};

/// The GNSS inputs of one epoch (what ``process`` receives).
struct EpochInput {
  ObsPtr obs, obsb;
  Eigen::MatrixXd rs, vs, rsb;
  std::vector<int> sat;
  std::vector<double> el;
  std::vector<int> iu;
  std::map<int, int> ir_map;   // base sat -> row of obsb
  /// The rover signal plan (``obs_sd.sig``).
  const SignalPlan& sig() const { return *obs->sig; }
};

/// Per-epoch mutable context shared by the Phase-2 stages.
struct EpochData {
  EpochInput in;
  Info info;
  int ns = 0;
  Vec3 init_ecef = Vec3::Zero();
  Mat3 R_enu2ecef = Mat3::Identity();

  // imu_prediction
  int key_idx = -1;
  double tow = 0.0;
  int imu_idx_prev = 0;
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> pim;
  int n_imu = 0;
  std::optional<Vec3> gyro_mean;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::Values estimate;
  gtsam::Pose3 pose_p;
  Vec3 vel_prev = Vec3::Zero();
  gtsam::imuBias::ConstantBias bias_prev;
  gtsam::NavState pred_nav;

  // quality_gate
  bool skip_cp_now = false;
  Vec3 pred_ecef = Vec3::Zero();
  std::map<SfKey, std::pair<gtsam::Key, double>> prev_amb_values;

  // solve
  int nv = 0;
  std::size_t nf_before = 0;
  gtsam::Pose3 pose_tc;
  Vec3 ecef_tc = Vec3::Zero();
  int nb = 0;
  std::optional<StateVec> xa;   // LAMBDA-fixed state

  // validate_fix / report
  Vec3 sol = Vec3::Zero();
  std::string tag;
};

}  // namespace gnss_fgo
