// Shared helpers: frames/rotations, IMU loading and preintegration, the
// stand-alone DDPR least-squares solver (utils/*.py).
#pragma once

#include "gnss_fgo/common.h"
#include "gnss_fgo/state.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gnss_fgo {

struct ImuSample {
  double tow = 0.0;
  Vec3 acc = Vec3::Zero();    // [m/s^2], body FLU
  Vec3 gyro = Vec3::Zero();   // [rad/s], body FLU
};

// ── geometry ────────────────────────────────────────────────────────
extern const Mat3 R_ENU2NED, R_NED2ENU, R_FRD2FLU, R_FLU2FRD;

/// Reference NED/FRD Euler angles -> internal FLU body-to-NED / -ENU.
Mat3 euler_to_R_body2ned(double roll, double pitch, double heading);
Mat3 euler_to_R_body2enu(double roll, double pitch, double heading);
/// "x,y,z" -> vector.
Vec3 parse_lever(const std::string& s);
/// Raw IMU CSV (dataset sensor frame, gyro in deg/s).
std::vector<ImuSample> load_imu_csv(const std::string& path);
/// Heading [rad] from horizontal ENU velocity, else displacement, else fallback.
double heading_from_vel(const Vec3& vel, double fallback,
                        const std::optional<Vec3>& disp_enu = std::nullopt,
                        double vel_speed_min = 0.5, double disp_min = 0.01);
/// Body forward-axis heading [deg] of a body->ENU pose.
double heading_from_pose(const gtsam::Pose3& pose);
/// GDOP at a predicted pose (999 when degenerate).
double compute_gdop(const Vec3& pred_trans_enu, int ns, const Eigen::MatrixXd& rs,
                    const std::vector<int>& iu, const Mat3& R_enu2ecef,
                    const Vec3& base_ecef);

// ── IMU ─────────────────────────────────────────────────────────────
std::shared_ptr<gtsam::PreintegrationCombinedParams> make_imu_params(
    double accel_noise, double gyro_noise, double accel_bias_sigma,
    double gyro_bias_sigma, double integ_cov);

struct PimResult {
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> pim;
  int n = 0;
  Vec3 gyro_mean = Vec3::Zero();
  int new_idx = 0;
};
/// Integrate samples up to ``target_tow`` (or ``max_samples``).
PimResult build_pim(const std::shared_ptr<gtsam::PreintegrationCombinedParams>& params,
                    const gtsam::imuBias::ConstantBias& bias,
                    const std::vector<ImuSample>& imu, int imu_idx,
                    std::optional<double> target_tow, int max_samples = 100,
                    double dt = 0.01);
/// Integrate a provided sample list.
PimResult build_pim_from_samples(
    const std::shared_ptr<gtsam::PreintegrationCombinedParams>& params,
    const gtsam::imuBias::ConstantBias& bias, const std::vector<ImuSample>& samples,
    double dt = 0.01);
/// Coarse stationary bias over the first n_max samples.
std::pair<Vec3, Vec3> estimate_stationary_bias(const std::vector<ImuSample>& imu,
                                               int n_max, int n_min = 100,
                                               double g_expected = 9.81);
struct ZuptStats {
  int n = 0;
  double acc_std = 0, gyro_std = 0, gyro_median = 0, acc_norm_mean = 0;
};
std::optional<ZuptStats> compute_zupt_stats(const ImuSample* begin, const ImuSample* end,
                                            const std::optional<Vec3>& bias_acc_ref,
                                            const std::optional<Vec3>& bias_gyro_ref);
/// Raw samples up to ``target_tow`` (or n_samples); advances the cursor.
std::vector<ImuSample> collect_imu_samples(const std::vector<ImuSample>& imu, int* imu_idx,
                                           int n_samples, std::optional<double> target_tow);

// ── DDPR-only least squares ─────────────────────────────────────────
struct DDPRContext {
  Mat3 R_enu2ecef;
  Vec3 base_ecef;
  gtsam::Pose3 ecef_T_nav;
  Vec3 lever;
  int nav_nf = 3;
  double sigma_pr = 0.3;
  double fde_pr = 4.0;
  /// (sys, idx_sys, sat, el) -> (ref_idx, ref_sat)
  std::function<std::pair<int, int>(int, const std::vector<int>&, const std::vector<int>&,
                                    const std::vector<double>&)>
      pick_ref_sat_idx;
};

struct DdprLsResult {
  std::optional<Vec3> ecef;     // body point
  int n_active = 0;
  double res_rms = std::numeric_limits<double>::infinity();
};
DdprLsResult ddpr_only_position(const EpochInput& in, const gtsam::Pose3& pose_init,
                                const DDPRContext& ctx);

/// Deterministic (sat, freq) order helper for map-shaped ambiguity dicts.
template <typename V>
std::vector<std::pair<SfKey, V>> sorted_items(const std::map<SfKey, V>& m) {
  return {m.begin(), m.end()};
}

}  // namespace gnss_fgo
