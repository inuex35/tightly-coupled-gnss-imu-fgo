// TcConfig — tunables for ImuGnssTc. Every field is an env var
// (UPPER_CASE of the field name, or the override listed in config.cpp);
// the defaults and their rationale are documented in src/gnss_fgo/config.py.
#pragma once

#include <cmath>
#include <limits>
#include <string>

namespace gnss_fgo {

// X(type, name, default)
#define GNSS_FGO_CONFIG_FIELDS(X)                       \
  X(double, sigma_pr, 0.3)                              \
  X(double, sigma_cp, 0.003)                            \
  X(double, sigma_amb0, 30.0)                           \
  X(double, sigma_cont, 1.0)                            \
  X(double, sigma_n_between, 0.01)                      \
  X(double, sigma_n_between_flt, 0.1)                   \
  X(int, betweenn_enable, 1)                            \
  X(double, el_mask_deg, 10.0)                          \
  X(double, elmin_deg, 15.0)                            \
  X(double, cnr_min_dbhz, 25.0)                         \
  X(double, snr_ref_dbhz, 42.0)                         \
  X(int, varerr_enable, 1)                              \
  X(double, err_a, 0.001)                               \
  X(double, err_b, 0.001)                               \
  X(double, err_eratio_pr, 100.0)                       \
  X(double, err_sclkstab, 5e-12)                        \
  X(double, fde_pr, 4.0)                                \
  X(double, fde_cp, 0.5)                                \
  X(double, fde_max_frac, 0.5)                          \
  X(int, strict_sat_gate, 1)                            \
  X(int, fde_enable, 1)                                 \
  X(int, fde_max_iter, 1)                               \
  X(double, ar_thresar_min, 1.5)                        \
  X(double, ar_thresar_max, 3.0)                        \
  X(int, ddpr_sanity_enable, 1)                         \
  X(int, sanity_break_pim, 1)                           \
  X(double, sanity_pose_replace_thresh, 5.0)            \
  X(double, varholdamb, 0.001)                          \
  X(double, pim_break_trans_sigma, 1.0)                 \
  X(double, thres_slip, 0.15)                           \
  X(int, doppler_adaptive_sigma, 1)                     \
  X(double, doppler_fde_k, 4.0)                         \
  X(int, doppler_snr_weight, 1)                         \
  X(double, doppler_gdop_max, 0.0)                      \
  X(int, doppler_require_dd, 1)                         \
  X(double, doppler_sd_sigma, 0.5)                      \
  X(double, doppler_huber, 1.0)                         \
  X(int, min_dd_for_solve, 4)                           \
  X(double, propagate_pose_sigma, 1.0)                  \
  X(double, propagate_vel_sigma, 1.0)                   \
  X(double, propagate_bias_sigma, 0.1)                  \
  X(double, propagate_amb_sigma, 0.1)                   \
  X(int, recov_cp_hold, 5)                              \
  X(double, sanity_max_median_ratio, 5.0)               \
  X(int, sanity_max_median_min_sats, 6)                 \
  X(double, main_ddpr_res_thresh, 3.0)                  \
  X(int, ddpr_sanity_persist, 3)                        \
  X(double, ddpr_max_res, 2.0)                          \
  X(double, main_ddpr_res_catastrophic, 15.0)           \
  X(double, ar_ddpr_xvalidate_thresh, 10.0)             \
  X(double, ddpr_fast_worst_sat_min, 30.0)              \
  X(double, per_sat_res_thresh, 3.0)                    \
  X(std::string, imu_grade, "tactical")                 \
  X(double, imu_integ_cov, 1e-3)                        \
  X(double, accel_noise, 2.84e-3)                       \
  X(double, gyro_noise, 4.01e-3)                        \
  X(double, accel_bias_sigma, 3.14e-4)                  \
  X(double, gyro_bias_sigma, 9.70e-6)                   \
  X(double, bias_between_acc_sigma, 3e-4)               \
  X(double, bias_between_gyro_sigma, 3e-5)              \
  X(double, bias_prior_acc_sigma, 3e-3)                 \
  X(double, bias_prior_gyro_sigma, 3e-4)                \
  X(int, bias_prior_mode, 2)                            \
  X(double, init_pitch_deg, std::numeric_limits<double>::quiet_NaN()) \
  X(int, ar_mode, 3)                                    \
  X(int, ar_starve_reset, 50)                           \
  X(double, ar_starve_max_res, 2.0)                     \
  X(double, ar_gdop_max, 8.0)                           \
  X(double, ar_thresar, 3.0)                            \
  X(int, ar_minfixsats, 4)                              \
  X(int, subset_ar_enable, 1)                           \
  X(int, subset_ar_max_candidates, 5)                   \
  X(int, subset_ar_min_nb, 4)                           \
  X(int, subset_ar_max_drop, 2)                         \
  X(int, subset_ar_max_dirty_sats, 2)                   \
  X(double, subset_ar_dirty_sat_res_m, 1.0)             \
  X(int, exclude_bds_geo, 1)                            \
  X(double, ar_context_main_ddpr_max, 1.2)              \
  X(double, ar_context_worst_sat_max, 4.0)              \
  X(int, ar_context_nb_max, 6)                          \
  X(int, low_nb_fix_reject_nb_max, 6)                   \
  X(int, low_nb_fix_only_after_flt, 1)                  \
  X(int, low_nb_fix_reject_max_prev_fix_streak, 2)      \
  X(double, lambda_corr_hard_max, 1.0)                  \
  X(int, diag_factor_residuals, 0)                      \
  X(double, vel_thresh, 1.0)                            \
  X(int, n_collect, 5)                                  \
  X(double, body_rot_std, 1.0)                          \
  X(int, nhc_enable, 1)                                 \
  X(double, nhc_min_speed, 0.0)                         \
  X(double, nhc_sigma_lat, 0.3)                         \
  X(double, nhc_sigma_vert, 0.2)                        \
  X(std::string, nhc_lever, "0,0,0")                    \
  X(int, zupt_enable, 1)                                \
  X(double, zupt_max_acc_std, 0.55)                     \
  X(double, zupt_max_gyro_std, 0.030)                   \
  X(double, zupt_max_gyro_median, 0.020)                \
  X(double, zupt_g_dev_thr, 0.0)                        \
  X(double, zupt_max_speed, 0.0)                        \
  X(int, zupt_min_samples, 5)                           \
  X(double, zupt_sigma_zero_velocity, 0.5)              \
  X(double, zupt_sigma_zero_rotation, 0.010)            \
  X(double, zupt_anchor_sigma_translation, 1.0)         \
  X(double, zupt_anchor_sigma_rotation, 0.1)            \
  X(double, fls_lag, 1.0)                               \
  X(double, phase1_fls_lag, 1.0)                        \
  X(int, isam2_relinearize_skip, 10)                    \
  X(double, isam2_relinearize_threshold, 0.05)

struct TcConfig {
#define GNSS_FGO_DECLARE(type, name, dflt) type name = dflt;
  GNSS_FGO_CONFIG_FIELDS(GNSS_FGO_DECLARE)
#undef GNSS_FGO_DECLARE

  /// Defaults, then every field from its env var, then the IMU preset.
  static TcConfig from_env();
  /// Apply the imu_grade preset when no individual IMU sigma was set.
  void apply_imu_preset();
};

}  // namespace gnss_fgo
