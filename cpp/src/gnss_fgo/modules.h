// Free functions of the pipeline modules, grouped as in the Python
// packages: factors/, ar/, integrity/, pipeline/, initialization.
#pragma once

#include "gnss_fgo/runner.h"

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace gnss_fgo {

using PrevAmbValues = std::map<SfKey, std::pair<gtsam::Key, double>>;
using AmbDict = std::map<SfKey, gtsam::Key>;

// ── factors ─────────────────────────────────────────────────────────

/// DD carrier-phase factor with 0/1 float-ambiguity keys: the held
/// integers are folded into ``offset_m``. Error =
///   (rho_rov_ref - rho_rov_j) + offset - (rho_base_ref - rho_base_j) - dd_obs
///   + coeff * N_float
class HeldDdcpFactor : public gtsam::NoiseModelFactor {
 public:
  HeldDdcpFactor(gtsam::Key key_pose, std::optional<gtsam::Key> key_float,
                 const gtsam::SharedNoiseModel& noise, const Vec3& sat_ref_rov,
                 const Vec3& sat_j_rov, const Vec3& sat_ref_base, const Vec3& sat_j_base,
                 const Vec3& base_ecef, double dd_obs_cp, const Vec3& lever,
                 const gtsam::Pose3& ecef_T_nav, double offset_m, double coeff_m);

  gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                gtsam::OptionalMatrixVecType H = nullptr) const override;
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<HeldDdcpFactor>(*this);
  }
  bool has_float() const { return has_float_; }

 private:
  bool has_float_;
  Vec3 sat_ref_rov_, sat_j_rov_, lever_;
  gtsam::Pose3 ecef_T_nav_;
  double err_const_, coeff_m_;
};

/// RTKLIB-demo5 varerr DD sigma [m] (code=1 PR, 0 CP).
double varerr_dd_sigma(const ImuGnssTc& tc, int code, double el_rad, double dt_s);
/// DD reference satellite (index into ``sat``, sat number) for one system.
std::pair<int, int> pick_ref_sat_idx(const ImuGnssTc& tc, int sys_id,
                                     const std::vector<int>& idx_sys,
                                     const std::vector<int>& sat,
                                     const std::vector<double>& el);
/// DD pseudorange + carrier-phase factors; returns the factor count.
int build_dd_factors(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph,
                     gtsam::Values& values, const EpochInput& in, gtsam::Key key_pose,
                     const Vec3& lever, AmbDictRef amb_dict, int dd_epoch = 0,
                     const PrevAmbValues* prev_amb_values = nullptr, bool skip_cp = false);
/// Between-satellite single-differenced Doppler.
void add_sd_doppler_factors(ImuGnssTc& tc, EpochData& epoch);
/// Non-holonomic constraint; true when added.
bool add_nhc_factor(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, int key_idx,
                    double speed, const std::optional<Vec3>& gyro_mean);
/// ZUPT / ZARU / stationary anchor; true when any factor fired.
bool add_zupt_factors(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, int key_idx,
                      int imu_idx_prev, int n_imu, Info& info,
                      const std::optional<gtsam::Pose3>& pose_prev, bool gnss_available,
                      const std::optional<Vec3>& vel_prev);
void add_zupt_factors_for_stage(ImuGnssTc& tc, EpochData& epoch);
/// Preintegrate up to ``target_tow`` from tc.imu_idx (advances it).
PimResult build_pim_tc(ImuGnssTc& tc, const gtsam::imuBias::ConstantBias& bias,
                       std::optional<double> target_tow);
/// CombinedImuFactor + bias regularisation, or the chain break after a reset.
void add_imu_chain(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                   int key_idx, const gtsam::PreintegratedCombinedMeasurements& pim,
                   const gtsam::Pose3& pose_p, const Vec3& vel_prev, Info& info);

// ── ar ──────────────────────────────────────────────────────────────

/// LAMBDA AR with validation; (nb, xa) with nb == 0 for no fix.
std::pair<int, std::optional<StateVec>> run_ar(ImuGnssTc& tc, const std::vector<int>& sat,
                                               const std::vector<double>& el,
                                               const AmbDict& amb_dict);
/// Publish float/held ambiguities, vsat and the AR pose key.
void publish_marginals(ImuGnssTc& tc, const gtsam::Values& estimate, gtsam::Key key_pose,
                       const AmbDict& amb_dict);
/// Fix-and-hold (armode 3).
void apply_fix_and_hold(ImuGnssTc& tc, const AmbDict& amb_dict, const StateVec& xa,
                        const std::optional<std::set<SfKey>>& allow = std::nullopt);

// ── integrity ───────────────────────────────────────────────────────

/// LLI + GF slip detection and outage expiry; returns the reset count.
int detect_slips_and_reset_ambiguities(ImuGnssTc& tc, const EpochInput& in);
/// DDPR sanity ladder; a result when the recovery fired.
std::optional<EpochResult> run_ddpr_sanity(ImuGnssTc& tc, EpochData& epoch);
void warm_reset_phase2(ImuGnssTc& tc, const Vec3& ecef_seed, const gtsam::Rot3& rot_seed,
                       bool break_pim = true);
int reset_ambiguities_with_cp_hold(ImuGnssTc& tc);
/// DDPR-only solve + warm reset; the antenna position when it took.
std::optional<Vec3> try_ddpr_reset(ImuGnssTc& tc, const EpochInput& in,
                                   const gtsam::Pose3& pose_init, const gtsam::Rot3& rot_seed,
                                   Info& info, const std::string& reason);
EpochResult handle_solve_exception(ImuGnssTc& tc, const std::exception& ex, EpochData& epoch);
bool trigger_cp_hold(ImuGnssTc& tc, const std::string& reason, Info& info,
                     std::optional<double> value, bool skip_if_active = false);
int effective_cp_hold_epochs(const ImuGnssTc& tc);

// ── pipeline ────────────────────────────────────────────────────────

EpochResult run_tc_epoch(ImuGnssTc& tc, const EpochInput& in, Info info, int ns,
                         const Vec3& init_ecef, const Mat3& R);
void fls_update(ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
                const gtsam::Values& values, int key_idx,
                const std::vector<gtsam::Key>& keep_keys = {}, bool advance_time = true,
                bool include_prev = true);

struct MainDdprResult {
  double rms = 0.0;
  OrderedMap<int, double> per_sat;
  std::vector<PairRow> pairs;
};
MainDdprResult main_ddpr_residuals(const ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
                                   const gtsam::Values& estimate);
std::optional<double> ddpr_res_at_fixed_pose(const ImuGnssTc& tc,
                                             const gtsam::NonlinearFactorGraph& graph,
                                             const gtsam::Values& estimate,
                                             gtsam::Key key_pose, const StateVec& xa);
/// Post-fit FDE on the current epoch's factor slots.
gtsam::Values apply_fde(ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph, int key_idx,
                        std::optional<int> nv, gtsam::Values estimate, Info& info,
                        std::optional<std::size_t> fi_start, Smoother* smoother = nullptr,
                        std::optional<gtsam::Key> pose_key = std::nullopt);
std::map<std::string, std::pair<double, int>> all_factor_residuals(
    const ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& estimate);

// ── initialization ──────────────────────────────────────────────────

EpochResult run_init_epoch(ImuGnssTc& tc, const EpochInput& in, Info info,
                           const Vec3& init_ecef, const Mat3& R);

}  // namespace gnss_fgo
