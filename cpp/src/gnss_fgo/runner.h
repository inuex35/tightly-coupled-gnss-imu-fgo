// ImuGnssTc — two-phase IMU/GNSS tight coupling runner.
//
// The estimator is the factor graph. The GNSS front-end (RTKLIB-backed
// RINEX input, satellite states, observation editing) is a library it
// calls; ``nav`` is the small state the two share (see nav_state.h).
#pragma once

#include "gnss_fgo/config.h"
#include "gnss_fgo/frontend.h"
#include "gnss_fgo/nav_state.h"
#include "gnss_fgo/state.h"
#include "gnss_fgo/utils.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace gnss_fgo {

/// Result of one ``process`` call.
struct EpochResult {
  Vec3 sol = Vec3::Zero();
  std::string tag;   // "FIX" / "FLT"
  int nb = 0;
  Info info;
};

/// One Fix epoch collected during Phase 1 for the Phase-2 seed.
struct CollectedFix {
  Vec3 ecef = Vec3::Zero();
  Vec3 vel = Vec3::Zero();
  std::vector<ImuSample> imu;
  EpochInput gnss;
};

/// AR context rejection details.
struct ArContextReject {
  int nb = 0;
  double main_ddpr_res = 0.0, worst_sat_res = 0.0;
};

/// Subset-AR search diagnostics.
struct ArSubsetDebug {
  int candidates = 0;
  int max_drop = 0;
  bool used = false;
  std::string skip_reason;
  int dirty_n = 0;
  std::vector<int> drop_sats;
  int nb = 0;
  double ratio = 0.0;
};

/// Origin of an ambiguity table: the Phase-1 plain dict, or the per-sat
/// ``amb_key`` fields (Phase 2's ``amb_keys_tc`` view).
class AmbDictRef {
 public:
  explicit AmbDictRef(std::map<SfKey, gtsam::Key>* plain) : plain_(plain) {}
  explicit AmbDictRef(SatStateMap* states) : states_(states) {}
  bool contains(const SfKey& k) const {
    if (plain_) return plain_->count(k) > 0;
    const SatState& st = static_cast<const SatStateMap*>(states_)->at(k.first, k.second);
    return st.amb_key.has_value();
  }
  void set(const SfKey& k, gtsam::Key key) {
    if (plain_) (*plain_)[k] = key;
    else states_->get(k.first, k.second).amb_key = key;
  }

 private:
  std::map<SfKey, gtsam::Key>* plain_ = nullptr;
  SatStateMap* states_ = nullptr;
};

using Smoother = gtsam::IncrementalFixedLagSmoother;

class ImuGnssTc {
 public:
  // Symbol helpers
  static gtsam::Key Xp(int i) { return gtsam::Symbol('x', i); }     // Phase-1 pose
  static gtsam::Key Xpose(int i) { return gtsam::Symbol('P', i); }  // Phase-2 pose
  static gtsam::Key Vel(int i) { return gtsam::Symbol('v', i); }
  static gtsam::Key Bias(int i) { return gtsam::Symbol('b', i); }
  static gtsam::Key N(int s, int f, int64_t gen = 0) {
    return gtsam::Symbol('n', uint64_t(gen * 1000000 + int64_t(s) * 10 + f));
  }
  static gtsam::SharedNoiseModel noise1(double sigma) {
    return gtsam::noiseModel::Isotropic::Sigma(1, sigma);
  }

  ImuGnssTc(const NavData& navdata, int nf, const Vec3& pos0, const Vec3& base_ecef,
            std::vector<ImuSample> imu_data, const Vec3& lever_arm,
            const TcConfig& cfg);

  /// Double-difference preparation through the front-end.
  std::optional<DDMeasurements> prepare_double_difference_measurements(
      const ObsEpoch& obs, const ObsEpoch* obsb) {
    return frontend.prepare(obs, obsb, nav);
  }

  /// Process one epoch.
  EpochResult process(const EpochInput& in);
  /// Advance the Phase-2 graph by IMU only (no GNSS).
  EpochResult process_imu_only(const ObsEpoch& obs);

  // ── helpers used across the stage modules ──────────────────────
  void update_epoch_dt(const ObsEpoch& obs);
  void reset_current_epoch();
  Vec3 antenna_ecef(const gtsam::Pose3& pose, const Vec3& ecef_body) const;
  std::shared_ptr<gtsam::PreintegrationCombinedParams> make_imu_params_() const;
  DdprLsResult ddpr_only_position(const EpochInput& in, const gtsam::Pose3& pose_init);
  double compute_gdop_(const gtsam::NavState& pred, int ns, const EpochInput& in) const;
  std::shared_ptr<Smoother> make_isam2(double lag) const;
  gtsam::imuBias::ConstantBias bias_or_init() const {
    return tc_bias ? *tc_bias : (tc_bias_init ? *tc_bias_init : gtsam::imuBias::ConstantBias());
  }

  // ── configuration / frames ─────────────────────────────────────
  TcConfig cfg;
  NavState nav;
  DdFrontend frontend;
  Vec3 base_ecef;
  std::vector<ImuSample> imu_data;
  Vec3 lever_arm;
  Mat3 R_enu2ecef;
  gtsam::Pose3 ecef_T_nav;
  double body_rot_std, accel_noise, gyro_noise, accel_bias_sigma, gyro_bias_sigma;

  // ── run state ──────────────────────────────────────────────────
  RecoveryState recovery;
  MresSignalsState mres_signals;
  int phase = 1;
  int epoch = 0;
  int imu_idx = 0;
  int tc_epoch = 0;
  double phase1_t = 0.0;
  std::shared_ptr<Smoother> isam;                   // Phase-1 smoother
  std::map<SfKey, gtsam::Key> amb_keys;             // Phase-1 ambiguities
  std::set<gtsam::Key> isam_p1_inserted;
  bool collecting = false;
  std::vector<CollectedFix> collected_fixes;
  SatStateMap sat_states;

  std::shared_ptr<Smoother> isam2;                  // Phase-2 smoother
  std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params;
  std::optional<gtsam::imuBias::ConstantBias> tc_bias, tc_bias_init;
  Vec3 lever_arm_tc = Vec3::Zero();
  double tc_time = 0.0;
  double fls_lag = 0.0;

  std::optional<gnss::GTime> last_obs_t;
  double epoch_dt = 0.2;
  CurrentEpochState current_epoch;
  ArDiagnostics ar_diag;

  std::optional<ArContextReject> ar_context_reject;
  std::optional<ArSubsetDebug> ar_subset_debug;
  int ar_starve_streak = 0;
  std::optional<Vec3> nhc_lever_cache;
  std::string nhc_lever_src;
  std::optional<gtsam::Key> ar_key_pose;
  std::vector<std::pair<double, Vec3>> sol_hist;

  // Per-epoch DDCP / DDPR factor bookkeeping (reset in build_dd_factors)
  std::set<SfKey> ar_cp_visible_sf;
  std::set<std::size_t> last_custom_ddcp_local;
  std::map<std::pair<gtsam::Key, gtsam::Key>, std::tuple<int, int, int>>
      last_custom_ddcp_global;
  std::vector<std::tuple<std::size_t, int, int, int>> last_ddpr_sat_tags;
  OrderedMap<int, double> last_per_sat_res;
  std::optional<double> cached_ddpr_res_pre;
  std::optional<Vec3> last_sol_ecef;

  // AR ratio stash
  double last_s0 = 0.0, last_s1 = 0.0, last_ar_thres = 0.0;
  std::optional<std::set<SfKey>> ar_partial_holdable;
};

/// Pack an epoch result and advance the epoch counter / nav time.
EpochResult advance_epoch_and_pack(ImuGnssTc& tc, const Vec3& sol, const std::string& tag,
                                   int nb, Info info, const ObsEpoch& obs);

/// Default diagnostics for a new epoch.
Info make_epoch_diagnostics(const ImuGnssTc& tc);

}  // namespace gnss_fgo
