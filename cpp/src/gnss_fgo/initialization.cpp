// Phase 1 — stationary GNSS-only Pose3 RTK, and the Phase-1 -> Phase-2
// handoff that seeds the tightly coupled graph from the collected fixes.
#include "gnss_fgo/modules.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>

namespace gnss_fgo {

namespace {

gtsam::SharedNoiseModel diag6(double a, double b, double c, double d, double e, double f) {
  return gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector6() << a, b, c, d, e, f).finished());
}

/// Rebuild the Phase-1 smoother from scratch (numerical-failure recovery).
void p1_fresh_restart(ImuGnssTc& tc) {
  tc.isam = tc.make_isam2(tc.cfg.phase1_fls_lag);
  tc.amb_keys.clear();
  tc.isam_p1_inserted.clear();
  tc.sat_states.for_each([](const SfKey&, SatState& st) {
    st.amb_gen += 1;
    st.amb_key.reset();
  });
}

/// Phase 1A — per-epoch DD graph, FLS update + three relinearisation
/// passes, Phase-1 FDE. nullopt on a double smoother failure.
std::optional<gtsam::Values> p1_build_and_solve(ImuGnssTc& tc, const EpochInput& in,
                                                Vec3 init_ecef, const Mat3& R, Info& info) {
  gtsam::NonlinearFactorGraph g;
  gtsam::Values v;
  const int ep = tc.epoch;
  Vec3 enu_pp = R.transpose() * (init_ecef - tc.base_ecef);
  gtsam::Pose3 pose0(gtsam::Rot3(), enu_pp);
  const bool prev_missing = ep > 0 && !tc.isam_p1_inserted.count(ImuGnssTc::Xp(ep - 1));
  const bool fresh_start = ep == 0 || prev_missing;
  if (fresh_start) {
    // The RINEX header position can be km-stale: seed the anchor from a
    // code-only DD LS, insensitive to that initial error, and re-anchor
    // the ambiguity seeding geometry with it.
    try {
      const DdprLsResult ls = tc.ddpr_only_position(in, pose0);
      if (ls.ecef && ls.n_active >= 4 && std::isfinite(ls.res_rms)) {
        init_ecef = *ls.ecef;
        tc.nav.x.pos = init_ecef;
        enu_pp = R.transpose() * (init_ecef - tc.base_ecef);
        pose0 = gtsam::Pose3(gtsam::Rot3(), enu_pp);
      }
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Value);
    }
    g.addPrior(ImuGnssTc::Xp(ep), pose0, diag6(0.01, 0.01, 0.01, 3, 3, 3));
  } else {
    g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        ImuGnssTc::Xp(ep - 1), ImuGnssTc::Xp(ep), gtsam::Pose3(),
        diag6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1));
  }
  v.insert(ImuGnssTc::Xp(ep), pose0);

  const Vec3 lever = Vec3::Zero();
  build_dd_factors(tc, g, v, in, ImuGnssTc::Xp(ep), lever, AmbDictRef(&tc.amb_keys), 0);

  for (gtsam::Key k : v.keys())
    if (tc.isam_p1_inserted.count(k)) v.erase(k);
  tc.phase1_t += tc.epoch_dt;
  gtsam::FixedLagSmootherKeyTimestampMap ts;
  ts[ImuGnssTc::Xp(ep)] = tc.phase1_t;
  if (!fresh_start) ts[ImuGnssTc::Xp(ep - 1)] = tc.phase1_t;
  for (gtsam::Key k : v.keys()) ts[k] = tc.phase1_t;
  for (const auto& [sf, k] : tc.amb_keys) ts[k] = tc.phase1_t;
  std::size_t nf_before = tc.isam->getFactors().size();
  try {
    tc.isam->update(g, v, ts);
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
    // A GNSS-only bootstrap holds no irreplaceable state: restart Phase 1.
    p1_fresh_restart(tc);
    g = gtsam::NonlinearFactorGraph();
    v = gtsam::Values();
    g.addPrior(ImuGnssTc::Xp(ep), pose0, diag6(0.01, 0.01, 0.01, 3, 3, 3));
    v.insert(ImuGnssTc::Xp(ep), pose0);
    build_dd_factors(tc, g, v, in, ImuGnssTc::Xp(ep), lever, AmbDictRef(&tc.amb_keys), 0);
    // phase1_t was already advanced above.
    ts = gtsam::FixedLagSmootherKeyTimestampMap();
    for (gtsam::Key k : v.keys()) ts[k] = tc.phase1_t;
    try {
      nf_before = tc.isam->getFactors().size();
      tc.isam->update(g, v, ts);
    } catch (const std::exception& e2) {
      GNSS_FGO_CATCH_ONLY(e2, PyErr::Runtime, PyErr::Index);
      return std::nullopt;
    }
  }
  tc.isam_p1_inserted.insert(ImuGnssTc::Xp(ep));
  for (gtsam::Key k : v.keys()) tc.isam_p1_inserted.insert(k);
  gtsam::Values est;
  try {
    for (int i = 0; i < 3; ++i)
      tc.isam->update(gtsam::NonlinearFactorGraph(), gtsam::Values(),
                      gtsam::FixedLagSmootherKeyTimestampMap());
    est = tc.isam->calculateEstimate();
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index);
    p1_fresh_restart(tc);
    return std::nullopt;
  }
  // Marginalised variables leave the inserted-set, or a satellite that
  // reappears under the same key would lose its fresh Values insert.
  for (auto it = tc.isam_p1_inserted.begin(); it != tc.isam_p1_inserted.end();)
    it = est.exists(*it) ? std::next(it) : tc.isam_p1_inserted.erase(it);

  // Phase-1 FDE: the same postfit screen Phase 2 runs.
  return apply_fde(tc, g, -1, std::nullopt, est, info, nf_before, tc.isam.get(),
                   ImuGnssTc::Xp(ep));
}

struct P1Solution {
  Vec3 sol;
  std::string tag;
  int nb;
};

/// Phase 1B — nav.x from the estimate, marginals, LAMBDA after warm-up.
P1Solution p1_emit_and_run_ar(ImuGnssTc& tc, const gtsam::Values& est, const EpochInput& in,
                              const Mat3& R) {
  const int ep = tc.epoch;
  const Vec3 enu = est.at<gtsam::Pose3>(ImuGnssTc::Xp(ep)).translation();
  tc.nav.x.pos = R * enu + tc.base_ecef;
  publish_marginals(tc, est, ImuGnssTc::Xp(ep), tc.amb_keys);
  tc.nav.smode = 5;
  int nb = 0;
  std::optional<StateVec> xa;
  if (ep >= 5) {
    std::tie(nb, xa) = run_ar(tc, in.sat, in.el, tc.amb_keys);
    if (nb > 0 && tc.nav.armode == 3) apply_fix_and_hold(tc, tc.amb_keys, *xa);
  }
  const bool fixed = nb > 0 && tc.nav.smode == 4;
  return {fixed ? xa->pos : tc.nav.x.pos, tc.nav.smode == 4 ? "FIX" : "FLT", nb};
}

// ── Phase-1 -> Phase-2 handoff ──────────────────────────────────────

/// Smooth per-fix heading seed from the collected fix path (ENU).
std::pair<double, std::vector<double>> init_tc_heading_series(
    const ImuGnssTc& tc, const std::vector<CollectedFix>& fixes) {
  const Mat3& R = tc.R_enu2ecef;
  const int n = int(fixes.size());
  std::vector<Vec3> enu;
  Vec3 vel_mean = Vec3::Zero();
  for (const auto& f : fixes) {
    enu.push_back(R.transpose() * (f.ecef - tc.base_ecef));
    vel_mean += f.vel;
  }
  vel_mean /= n;
  const Vec3 total = enu.back() - enu.front();
  const double seed = total.head<2>().norm() > 0.01
                          ? std::atan2(total[0], total[1])
                          : heading_from_vel(vel_mean, 0.0, total);
  std::vector<double> series;
  double prev_h = seed;
  for (int i = 0; i < n; ++i) {
    Vec3 disp;
    if (n == 1) disp = total;
    else if (i == 0) disp = enu[1] - enu[0];
    else if (i == n - 1) disp = enu[n - 1] - enu[n - 2];
    else disp = enu[i + 1] - enu[i - 1];
    const double h = disp.head<2>().norm() > 0.01 ? std::atan2(disp[0], disp[1]) : prev_h;
    series.push_back(h);
    prev_h = h;
  }
  return {seed, series};
}

void add_tc_seed_epoch(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& g, gtsam::Values& v, int i,
                       const CollectedFix& fix, const gtsam::imuBias::ConstantBias& bias0,
                       double roll, double pitch, double heading_i, double body_rot_std_rad,
                       const Vec3& lever) {
  const Vec3 fix_enu = tc.R_enu2ecef.transpose() * (fix.ecef - tc.base_ecef);
  const double speed = fix.vel.head<2>().norm();
  const Mat3 R_b2e = euler_to_R_body2enu(roll, pitch, heading_i);
  const gtsam::Pose3 pose_i(gtsam::Rot3(R_b2e), fix_enu - R_b2e * lever);
  v.insert(ImuGnssTc::Xpose(i), pose_i);
  v.insert(ImuGnssTc::Vel(i), gtsam::Vector3(fix.vel));
  v.insert(ImuGnssTc::Bias(i), bias0);
  const double brs2 = body_rot_std_rad * body_rot_std_rad;
  if (i == 0) {
    // Roll/pitch tight, yaw sigma from the velocity.
    const double std_yaw = speed > 0.1 ? std::sqrt(brs2 + std::pow(0.1 / speed, 2)) : 0.5;
    g.addPrior(ImuGnssTc::Xpose(0), pose_i, diag6(0.02, 0.02, std_yaw, 0.05, 0.05, 0.1));
  } else {
    const double std_yaw = std::sqrt(brs2 + std::pow(0.2 / std::max(speed, 0.2), 2));
    g.addPrior(ImuGnssTc::Xpose(i), pose_i, diag6(0.01, 0.01, std_yaw, 0.10, 0.10, 0.10));
  }
  g.addPrior<gtsam::Vector3>(ImuGnssTc::Vel(i), fix.vel,
                             gtsam::noiseModel::Isotropic::Sigma(3, 0.5));
  g.addPrior(ImuGnssTc::Bias(i), bias0, gtsam::noiseModel::Isotropic::Sigma(6, 0.01));
  if (i > 0) {
    const PimResult pr = build_pim_from_samples(tc.imu_params, bias0, fix.imu);
    g.emplace_shared<gtsam::CombinedImuFactor>(
        ImuGnssTc::Xpose(i - 1), ImuGnssTc::Vel(i - 1), ImuGnssTc::Xpose(i), ImuGnssTc::Vel(i),
        ImuGnssTc::Bias(i - 1), ImuGnssTc::Bias(i), *pr.pim);
  }
  g.emplace_shared<gtsam::GPSFactorArm>(ImuGnssTc::Xpose(i), gtsam::Point3(fix_enu),
                                        gtsam::Point3(lever),
                                        gtsam::noiseModel::Isotropic::Sigma(3, 0.02));
}

struct TransitionInfo {
  double pitch, roll, heading;
  Vec3 bias_acc, bias_gyro;
};

/// Initialise Phase 2 with the accumulated fixes + IMU.
TransitionInfo transition_to_tc(ImuGnssTc& tc) {
  const std::vector<CollectedFix> fixes = tc.collected_fixes;
  const int n = int(fixes.size());
  const auto [bias_acc, bias_gyro] =
      estimate_stationary_bias(tc.imu_data, std::min(3000, tc.imu_idx));
  const gtsam::imuBias::ConstantBias bias0(bias_acc, bias_gyro);
  tc.tc_bias = bias0;
  tc.tc_bias_init = bias0;  // anchor for the weak prior

  const int n_static = std::min<int>(500, int(tc.imu_data.size()));
  Vec3 acc_avg = Vec3::Zero();
  for (int i = 0; i < n_static; ++i) acc_avg += tc.imu_data[i].acc;
  acc_avg /= n_static;
  const Vec3 tilt = acc_avg - bias_acc;
  double pitch = std::atan2(tilt[0], std::sqrt(tilt[1] * tilt[1] + tilt[2] * tilt[2]));
  const double roll = std::atan2(tilt[1], tilt[2]);
  if (std::isfinite(tc.cfg.init_pitch_deg)) pitch = tc.cfg.init_pitch_deg * gnss::PI / 180.0;

  const double brs = tc.body_rot_std * gnss::PI / 180.0;
  const auto [heading, series] = init_tc_heading_series(tc, fixes);
  tc.imu_params = tc.make_imu_params_();
  tc.lever_arm_tc = tc.lever_arm;
  tc.fls_lag = tc.cfg.fls_lag;
  tc.isam2 = tc.make_isam2(tc.fls_lag);
  tc.tc_time = 0.0;

  gtsam::NonlinearFactorGraph g;
  gtsam::Values v;
  for (int i = 0; i < n; ++i)
    add_tc_seed_epoch(tc, g, v, i, fixes[i], bias0, roll, pitch, series[i], brs, tc.lever_arm);
  tc.sat_states.for_each([](const SfKey&, SatState& st) { st.amb_key.reset(); });
  for (int i = 0; i < n; ++i)
    build_dd_factors(tc, g, v, fixes[i].gnss, ImuGnssTc::Xpose(i), tc.lever_arm_tc,
                     AmbDictRef(&tc.sat_states), 0);  // shared keys, no prev

  // Timestamps spread within the lag window (real seconds, dt apart).
  const double dt = tc.epoch_dt;
  gtsam::FixedLagSmootherKeyTimestampMap ts;
  for (int i = 0; i < n; ++i) {
    ts[ImuGnssTc::Xpose(i)] = i * dt;
    ts[ImuGnssTc::Vel(i)] = i * dt;
    ts[ImuGnssTc::Bias(i)] = i * dt;
  }
  for (gtsam::Key k : v.keys()) ts.emplace(k, (n - 1) * dt);  // ambiguities at the last epoch
  tc.tc_time = (n - 1) * dt;
  if (tc.fls_lag < n * dt) {
    tc.fls_lag = n * dt + dt;
    tc.isam2 = tc.make_isam2(tc.fls_lag);
  }
  tc.isam2->update(g, v, ts);

  // AR bookkeeping on the initial graph (last epoch's GNSS).
  const gtsam::Values est = tc.isam2->calculateEstimate();
  publish_marginals(tc, est, ImuGnssTc::Xpose(n - 1), tc.sat_states.amb_keys_dict());
  const gtsam::Pose3 pose_last = est.at<gtsam::Pose3>(ImuGnssTc::Xpose(n - 1));
  const Vec3 ecef_last = tc.R_enu2ecef * pose_last.translation() + tc.base_ecef;
  tc.nav.x.pos = tc.antenna_ecef(pose_last, ecef_last);
  tc.nav.smode = 5;

  tc.tc_epoch = n - 1;
  tc.phase = 2;
  return {pitch, roll, heading, bias_acc, bias_gyro};
}

/// Phase 1C — IMU collection, velocity from a 1.5 s fix window, fix
/// accumulation once moving, and the Phase-2 transition.
void p1_collect_and_maybe_transition(ImuGnssTc& tc, const EpochInput& in, const Vec3& sol,
                                     Info& info, const Mat3& R) {
  const double tow = gnss::time2gpst(in.obs->t);
  std::vector<ImuSample> imu_samples = collect_imu_samples(tc.imu_data, &tc.imu_idx, 100, tow);
  if (tc.nav.smode == 4) {
    tc.sol_hist.emplace_back(tow, sol);
    while (!tc.sol_hist.empty() && tow - tc.sol_hist.front().first > 1.5)
      tc.sol_hist.erase(tc.sol_hist.begin());
  }
  Vec3 est_vel = Vec3::Zero();
  if (tc.sol_hist.size() >= 2) {
    const auto& [t0, s0] = tc.sol_hist.front();
    const auto& [t1, s1] = tc.sol_hist.back();
    const double dt = t1 - t0;
    if (dt >= 0.8) est_vel = R.transpose() * (s1 - s0) / dt;
  }
  if (!tc.collecting && est_vel.head<2>().norm() > tc.cfg.vel_thresh && tc.nav.smode == 4)
    tc.collecting = true;

  if (tc.collecting && tc.nav.smode == 4) {
    tc.collected_fixes.push_back({sol, est_vel, std::move(imu_samples), in});
    info.set("n_collected", tc.collected_fixes.size());
    // The heading seed is the displacement direction: wait for real motion.
    if (int(tc.collected_fixes.size()) > tc.cfg.n_collect)
      tc.collected_fixes.erase(tc.collected_fixes.begin());
    const double disp = tc.collected_fixes.size() >= 2
                            ? (tc.collected_fixes.back().ecef - tc.collected_fixes.front().ecef).norm()
                            : 0.0;
    if (int(tc.collected_fixes.size()) >= tc.cfg.n_collect && disp > 2.0) {
      const TransitionInfo t = transition_to_tc(tc);
      info.set("transition", true);
      info.set("pitch", t.pitch * 180.0 / gnss::PI);
      info.set("roll", t.roll * 180.0 / gnss::PI);
      info.set("heading", t.heading * 180.0 / gnss::PI);
      info.set("bias_acc_init", t.bias_acc);
      info.set("bias_gyro_init", t.bias_gyro);
      info.set("n_amb_tc", tc.sat_states.amb_count());
    }
  } else if (tc.collecting && tc.nav.smode != 4) {
    // Lost Fix during collection — reset.
    tc.collecting = false;
    tc.collected_fixes.clear();
  }
}

}  // namespace

EpochResult run_init_epoch(ImuGnssTc& tc, const EpochInput& in, Info info,
                           const Vec3& init_ecef, const Mat3& R) {
  const auto est = p1_build_and_solve(tc, in, init_ecef, R, info);
  if (!est) return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, *in.obs);
  const P1Solution s = p1_emit_and_run_ar(tc, *est, in, R);
  p1_collect_and_maybe_transition(tc, in, s.sol, info, R);
  tc.last_sol_ecef = s.sol;
  return advance_epoch_and_pack(tc, s.sol, s.tag, s.nb, info, *in.obs);
}

}  // namespace gnss_fgo
