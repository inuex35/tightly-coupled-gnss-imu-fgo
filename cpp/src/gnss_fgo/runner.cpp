#include "gnss_fgo/runner.h"

#include "gnss_fgo/modules.h"

#include <cmath>

namespace gnss_fgo {

ImuGnssTc::ImuGnssTc(const NavData& navdata, int nf, const Vec3& pos0, const Vec3& base_ecef_,
                     std::vector<ImuSample> imu, const Vec3& lever, const TcConfig& cfg_)
    : cfg(cfg_),
      nav(nf),
      frontend(navdata, nf),
      base_ecef(base_ecef_),
      imu_data(std::move(imu)),
      lever_arm(lever) {
  nav.x.pos = pos0;
  nav.rb = base_ecef;

  // ENU frame anchored at the base station.
  const Vec3 llh = gnss::ecef2pos(base_ecef);
  const double sl = std::sin(llh[0]), cl = std::cos(llh[0]);
  const double sn = std::sin(llh[1]), cn = std::cos(llh[1]);
  R_enu2ecef << -sn, -sl * cn, cl * cn,
                cn, -sl * sn, cl * sn,
                0, cl, sl;
  ecef_T_nav = gtsam::Pose3(gtsam::Rot3(R_enu2ecef), base_ecef);

  // TcConfig onto the shared nav state.
  nav.armode = cfg.ar_mode;
  nav.elmin = cfg.elmin_deg * gnss::PI / 180.0;
  nav.cnr_min = cfg.cnr_min_dbhz;
  nav.strict_sat_gate = bool(cfg.strict_sat_gate);
  nav.thresar = cfg.ar_thresar;
  nav.minfixsats = cfg.ar_minfixsats;
  body_rot_std = cfg.body_rot_std;
  accel_noise = cfg.accel_noise;
  gyro_noise = cfg.gyro_noise;
  accel_bias_sigma = cfg.accel_bias_sigma;
  gyro_bias_sigma = cfg.gyro_bias_sigma;

  isam = make_isam2(cfg.phase1_fls_lag);
}

void ImuGnssTc::update_epoch_dt(const ObsEpoch& obs) {
  if (last_obs_t) {
    const double dt = gnss::timediff(obs.t, *last_obs_t);
    if (dt > 0) epoch_dt = dt;
  }
  last_obs_t = obs.t;
}

void ImuGnssTc::reset_current_epoch() {
  // Replaced, not wiped: epoch-scoped state cannot outlive the epoch.
  current_epoch = CurrentEpochState();
  ar_diag = ArDiagnostics();
  // amb_gen is part of the N-key value: it resets with the epoch.
  sat_states.for_each([](const SfKey&, SatState& st) { st.amb_gen = 0; });
}

Vec3 ImuGnssTc::antenna_ecef(const gtsam::Pose3& pose, const Vec3& ecef_body) const {
  if (lever_arm_tc.norm() == 0) return ecef_body;
  const Mat3 R_body = ecef_T_nav.compose(pose).rotation().matrix();
  return ecef_body + R_body * lever_arm_tc;
}

std::shared_ptr<gtsam::PreintegrationCombinedParams> ImuGnssTc::make_imu_params_() const {
  return make_imu_params(accel_noise, gyro_noise, accel_bias_sigma, gyro_bias_sigma,
                         cfg.imu_integ_cov);
}

DdprLsResult ImuGnssTc::ddpr_only_position(const EpochInput& in, const gtsam::Pose3& pose_init) {
  DDPRContext ctx;
  ctx.R_enu2ecef = R_enu2ecef;
  ctx.base_ecef = base_ecef;
  ctx.ecef_T_nav = ecef_T_nav;
  ctx.lever = lever_arm_tc;
  ctx.nav_nf = nav.nf;
  ctx.sigma_pr = cfg.sigma_pr;
  ctx.fde_pr = cfg.fde_pr;
  ctx.pick_ref_sat_idx = [this](int sys, const std::vector<int>& idx_sys,
                                const std::vector<int>& sat, const std::vector<double>& el) {
    return pick_ref_sat_idx(*this, sys, idx_sys, sat, el);
  };
  return gnss_fgo::ddpr_only_position(in, pose_init, ctx);
}

double ImuGnssTc::compute_gdop_(const gtsam::NavState& pred, int ns, const EpochInput& in) const {
  return compute_gdop(pred.pose().translation(), ns, in.rs, in.iu, R_enu2ecef, base_ecef);
}

Info make_epoch_diagnostics(const ImuGnssTc& tc) {
  Info info;
  info.set("phase", tc.phase);
  info.set("tc_epoch", 0);
  info.set("collecting", tc.collecting);
  info.set("n_collected", tc.collected_fixes.size());
  return info;
}

EpochResult ImuGnssTc::process(const EpochInput& in) {
  update_epoch_dt(*in.obs);
  reset_current_epoch();
  const int ns = int(in.sat.size());
  const Vec3 init_ecef = nav.x.pos;
  Info info = make_epoch_diagnostics(*this);
  // Phase 1 with ns < 4 cannot form DDs: just advance time.
  if (phase == 1 && ns < 4) return advance_epoch_and_pack(*this, nav.x.pos, "FLT", 0, info, *in.obs);
  if (phase == 1) return run_init_epoch(*this, in, std::move(info), init_ecef, R_enu2ecef);
  return run_tc_epoch(*this, in, std::move(info), ns, init_ecef, R_enu2ecef);
}

}  // namespace gnss_fgo
