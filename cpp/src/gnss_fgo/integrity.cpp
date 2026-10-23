// integrity/: slip detection, the DDPR sanity ladder and the recovery
// actions (IMU-only epochs, warm reset, arc purge, global CP-hold).
#include "gnss_fgo/modules.h"

#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>

namespace gnss_fgo {

namespace {

gtsam::SharedNoiseModel diag6(double a, double b, double c, double d, double e, double f) {
  return gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector6() << a, b, c, d, e, f).finished());
}

}  // namespace

// ── slip detection ──────────────────────────────────────────────────

namespace {

/// Kill the ambiguity of every slipped (sat, f).
int reset_slipped_ambiguities(ImuGnssTc& tc, const std::set<SfKey>& reset_keys) {
  int n = 0;
  for (const SfKey& key : reset_keys) {
    SatState& st = tc.sat_states.get(key.first, key.second);
    if (st.amb_key) {
      st.amb_key.reset();
      ++n;
    }
    st.clear_hold();
    st.amb_gen += 1;
  }
  return n;
}

/// Geometry-free L1-L2 jump -> slip on every band of the satellite.
void detslp_gf(ImuGnssTc& tc, const ObsEpoch& obs, int row, int s,
               const std::vector<double>& lams, int nf, std::set<SfKey>& reset_keys) {
  const double L1 = obs.L(row, 0), L2 = obs.L(row, 1);
  if (L1 == 0 || L2 == 0 || lams[0] <= 0 || lams[1] <= 0) return;
  const double gf = L1 * lams[0] - L2 * lams[1];
  auto& prev = tc.sat_states.gf;
  auto it = prev.find(s);
  if (it != prev.end() && it->second != 0 && std::abs(gf - it->second) > tc.cfg.thres_slip)
    for (int f = 0; f < nf; ++f) reset_keys.insert({s, f});
  prev[s] = gf;
}

}  // namespace

int detect_slips_and_reset_ambiguities(ImuGnssTc& tc, const EpochInput& in) {
  const int nf = tc.nav.nf;
  const ObsEpoch& obs = *in.obs;
  std::set<SfKey> reset_keys, current;
  for (std::size_t i = 0; i < in.sat.size(); ++i) {
    const int s = in.sat[i];
    const std::vector<double> lams = in.sig().wavelengths(s);
    for (int f = 0; f < nf; ++f) {
      current.insert({s, f});
      tc.sat_states.get(s, f).outc = 0;
      if (obs.lli(in.iu[i], f) == 1) reset_keys.insert({s, f});
    }
    if (nf >= 2 && lams.size() >= 2) detslp_gf(tc, obs, in.iu[i], s, lams, nf, reset_keys);
  }
  // Outage counter for satellites not seen this epoch.
  for (const SfKey& key : tc.sat_states.keys()) {
    if (current.count(key)) continue;
    SatState& st = tc.sat_states.get(key.first, key.second);
    st.outc += 1;
    if (st.outc > tc.sat_states.maxout) reset_keys.insert(key);
  }
  return reset_slipped_ambiguities(tc, reset_keys);
}

// ── recovery ────────────────────────────────────────────────────────

EpochResult advance_epoch_and_pack(ImuGnssTc& tc, const Vec3& sol, const std::string& tag,
                                   int nb, Info info, const ObsEpoch& obs) {
  tc.epoch += 1;
  tc.nav.t = obs.t;
  return EpochResult{sol, tag, nb, std::move(info)};
}

int effective_cp_hold_epochs(const ImuGnssTc& tc) { return tc.cfg.recov_cp_hold; }

bool trigger_cp_hold(ImuGnssTc& tc, const std::string& reason, Info& info,
                     std::optional<double> value, bool skip_if_active) {
  return tc.recovery.start_cp_hold(effective_cp_hold_epochs(tc), reason, info, value,
                                   skip_if_active);
}

void warm_reset_phase2(ImuGnssTc& tc, const Vec3& ecef_seed, const gtsam::Rot3& rot_seed,
                       bool break_pim) {
  const Vec3 enu_seed = tc.R_enu2ecef.transpose() * (ecef_seed - tc.base_ecef);
  const gtsam::Pose3 pose_seed(rot_seed, enu_seed);
  const Vec3 vel_seed = Vec3::Zero();   // a carried velocity belongs to the broken chain

  tc.isam2 = tc.make_isam2(tc.fls_lag);
  tc.tc_time = 0.0;
  tc.tc_epoch = 0;
  tc.sat_states.for_each([](const SfKey&, SatState& st) {
    if (st.amb_key) st.amb_gen += 1;
    st.clear_hold();
    st.amb_key.reset();
  });
  const gtsam::imuBias::ConstantBias bias0 = tc.tc_bias ? *tc.tc_bias : gtsam::imuBias::ConstantBias();

  gtsam::NonlinearFactorGraph g;
  gtsam::Values v;
  v.insert(ImuGnssTc::Xpose(0), pose_seed);
  v.insert(ImuGnssTc::Vel(0), gtsam::Vector3(vel_seed));
  v.insert(ImuGnssTc::Bias(0), bias0);
  const double rs = tc.body_rot_std * gnss::PI / 180.0;
  g.addPrior(ImuGnssTc::Xpose(0), pose_seed, diag6(rs, rs, 0.3, 2.0, 2.0, 3.0));
  g.addPrior<gtsam::Vector3>(ImuGnssTc::Vel(0), vel_seed,
                             gtsam::noiseModel::Isotropic::Sigma(3, 3.0));
  g.addPrior(ImuGnssTc::Bias(0), bias0, gtsam::noiseModel::Isotropic::Sigma(6, 0.01));
  tc.tc_bias = bias0;

  gtsam::FixedLagSmootherKeyTimestampMap ts;
  tc.tc_time += tc.epoch_dt;
  ts[ImuGnssTc::Xpose(0)] = tc.tc_time;
  ts[ImuGnssTc::Vel(0)] = tc.tc_time;
  ts[ImuGnssTc::Bias(0)] = tc.tc_time;
  tc.isam2->update(g, v, ts);

  tc.recovery.recov_cp_hold = effective_cp_hold_epochs(tc);
  tc.nav.x.pos = tc.antenna_ecef(pose_seed, ecef_seed);
  tc.recovery.pim_discontinuity = break_pim;
}

int reset_ambiguities_with_cp_hold(ImuGnssTc& tc) {
  const std::vector<gtsam::Key> keys = tc.sat_states.amb_key_values();
  const std::set<gtsam::Key> n_keys(keys.begin(), keys.end());
  gtsam::FactorIndices remove;
  if (!n_keys.empty()) {
    try {
      const gtsam::NonlinearFactorGraph& facs = tc.isam2->getFactors();
      for (std::size_t i = 0; i < facs.size(); ++i) {
        if (!facs[i]) continue;
        for (gtsam::Key k : facs[i]->keys())
          if (n_keys.count(k)) {
            remove.push_back(i);
            break;
          }
      }
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
      remove.clear();
    }
  }
  // Bump amb_gen and clear bookkeeping so the next epoch fresh-inits.
  tc.sat_states.for_each([](const SfKey&, SatState& st) {
    if (st.amb_key) st.amb_gen += 1;
    st.clear_hold();
    st.amb_key.reset();
  });
  tc.recovery.recov_cp_hold = effective_cp_hold_epochs(tc);
  if (!remove.empty()) {
    try {
      tc.isam2->update(gtsam::NonlinearFactorGraph(), gtsam::Values(),
                       gtsam::FixedLagSmootherKeyTimestampMap(), remove);
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index, PyErr::Value);
      return 0;   // nothing actually left the graph
    }
  }
  return int(remove.size());
}

std::optional<Vec3> try_ddpr_reset(ImuGnssTc& tc, const EpochInput& in,
                                   const gtsam::Pose3& pose_init, const gtsam::Rot3& rot_seed,
                                   Info& info, const std::string& reason) {
  const DdprLsResult r = tc.ddpr_only_position(in, pose_init);
  if (!r.ecef) return std::nullopt;
  if (r.res_rms > tc.cfg.ddpr_max_res) {
    info.set(reason + "_untrusted", r.res_rms);
    return std::nullopt;
  }
  info.set(reason, true);
  warm_reset_phase2(tc, *r.ecef, rot_seed);
  tc.recovery.ddpr_bad_count = 0;
  // The DDPR solve carries the lever: report the antenna.
  const Vec3 enu_body = tc.R_enu2ecef.transpose() * (*r.ecef - tc.base_ecef);
  return tc.antenna_ecef(gtsam::Pose3(rot_seed, enu_body), *r.ecef);
}

namespace {

void outage_tick_sat_outc(ImuGnssTc& tc, Info& info) {
  int n = 0;
  const int maxout = tc.sat_states.maxout;
  tc.sat_states.for_each([&](const SfKey&, SatState& st) {
    st.outc += 1;
    if (st.outc > maxout) {
      st.clear_hold();
      if (st.amb_key) {
        st.amb_key.reset();
        st.amb_gen += 1;
        ++n;
      }
    }
  });
  if (n) info.set("skip_amb_reset", n);
}

}  // namespace

EpochResult ImuGnssTc::process_imu_only(const ObsEpoch& obs) {
  ImuGnssTc& tc = *this;
  tc.update_epoch_dt(obs);
  tc.reset_current_epoch();
  Info info = make_epoch_diagnostics(tc);
  info.set("gnss_skip", true);
  info.set("imu_only", true);
  const double tow_obs = gnss::time2gpst(obs.t);

  if (tc.phase == 1) {
    // Drain IMU to keep the sample cursor aligned with real time.
    while (tc.imu_idx < int(tc.imu_data.size()) && tc.imu_data[tc.imu_idx].tow <= tow_obs + 1e-6)
      ++tc.imu_idx;
    return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, obs);
  }
  info.set("gnss_skip", true);
  info.set("outage_source", "imu_only");

  tc.tc_epoch += 1;
  const int key_idx = tc.tc_epoch;
  info.set("tc_epoch", key_idx);
  outage_tick_sat_outc(tc, info);

  const int imu_idx_prev = tc.imu_idx;
  const PimResult pr = build_pim_tc(tc, tc.bias_or_init(), tow_obs);
  if (pr.n == 0) return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, obs);
  info.set("n_imu", pr.n);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  const gtsam::Values estimate = tc.isam2->calculateEstimate();
  if (!estimate.exists(Xpose(key_idx - 1)))  // previous pose marginalised
    return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, obs);

  const gtsam::Pose3 pose_p = estimate.at<gtsam::Pose3>(Xpose(key_idx - 1));
  const Vec3 vel_prev = estimate.at<gtsam::Vector3>(Vel(key_idx - 1));
  const auto bias_prev = estimate.at<gtsam::imuBias::ConstantBias>(Bias(key_idx - 1));
  const gtsam::NavState pred = pr.pim->predict(gtsam::NavState(pose_p, vel_prev), bias_prev);
  values.insert(Xpose(key_idx), pred.pose());
  values.insert(Vel(key_idx), gtsam::Vector3(pred.velocity()));
  values.insert(Bias(key_idx), bias_prev);
  add_imu_chain(tc, graph, values, key_idx, *pr.pim, pose_p, vel_prev, info);

  // SKIP-only tight bias anchor (measured sigmas).
  const double sa = 1e-4, sg = 3e-6;
  graph.addPrior(Bias(key_idx), tc.bias_or_init(), diag6(sa, sa, sa, sg, sg, sg));
  // Outage pseudo-measurements: NHC + ZUPT/ZARU/anchor.
  if (add_nhc_factor(tc, graph, key_idx, vel_prev.head<2>().norm(), pr.gyro_mean))
    info.set("nhc", true);
  const int n_imu = int(info.num("n_imu", tc.imu_idx - imu_idx_prev));
  add_zupt_factors(tc, graph, key_idx, imu_idx_prev, n_imu, info, pose_p, false, vel_prev);

  Vec3 sol = tc.nav.x.pos;
  try {
    fls_update(tc, graph, values, key_idx, tc.sat_states.amb_key_values());
    const gtsam::Values est = tc.isam2->calculateEstimate();
    const gtsam::Pose3 pose_tc = est.at<gtsam::Pose3>(Xpose(key_idx));
    tc.tc_bias = est.at<gtsam::imuBias::ConstantBias>(Bias(key_idx));
    const Vec3 ecef_tc = tc.R_enu2ecef * pose_tc.translation() + tc.base_ecef;
    sol = tc.antenna_ecef(pose_tc, ecef_tc);
    tc.nav.x.pos = sol;
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index, PyErr::Value);
    info.set("error", std::string(e.what()));
    sol = tc.nav.x.pos;
  }
  tc.nav.smode = 5;
  return advance_epoch_and_pack(tc, sol, "FLT", 0, info, obs);
}

EpochResult handle_solve_exception(ImuGnssTc& tc, const std::exception& ex, EpochData& ep) {
  Info& info = ep.info;
  info.set("error", std::string(ex.what()));
  const auto ddpr = try_ddpr_reset(tc, ep.in, ep.pred_nav.pose(), ep.pred_nav.pose().rotation(),
                                   info, "ddpr_exception_recover");
  if (ddpr) return advance_epoch_and_pack(tc, *ddpr, "FLT", 0, info, *ep.in.obs);
  try {
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    const int k = ep.key_idx;
    v.insert(ImuGnssTc::Xpose(k), ep.pred_nav.pose());
    v.insert(ImuGnssTc::Vel(k), gtsam::Vector3(ep.pred_nav.velocity()));
    v.insert(ImuGnssTc::Bias(k), ep.bias_prev);
    g.addPrior(ImuGnssTc::Xpose(k), ep.pred_nav.pose(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
    g.addPrior<gtsam::Vector3>(ImuGnssTc::Vel(k), ep.pred_nav.velocity(),
                               gtsam::noiseModel::Isotropic::Sigma(3, 1.0));
    g.addPrior(ImuGnssTc::Bias(k), ep.bias_prev, gtsam::noiseModel::Isotropic::Sigma(6, 0.1));
    fls_update(tc, g, v, k);
    tc.tc_bias = ep.bias_prev;
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index, PyErr::Value);
    info.set("fallback_error", exc_label(e));
  }
  return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, *ep.in.obs);
}

// ── DDPR sanity ladder ──────────────────────────────────────────────

namespace {

/// One satellite dwarfing the median: punish the satellite, not the pose.
bool ddpr_multipath_dominated(const ImuGnssTc& tc, Info& info) {
  const double ratio_thr = tc.cfg.sanity_max_median_ratio;
  if (ratio_thr <= 0 || !info.main_ddpr_per_sat) return false;
  const auto& per_sat = *info.main_ddpr_per_sat;
  const int n = int(per_sat.size());
  if (n < tc.cfg.sanity_max_median_min_sats) return false;
  std::vector<double> vals;
  for (const auto& [s, v] : per_sat) vals.push_back(v);
  std::sort(vals.begin(), vals.end());
  const double median = vals[n / 2], max_v = vals.back();
  if (median <= 1e-3) return false;
  const double ratio = max_v / median;
  if (ratio > ratio_thr) {
    info.set("sanity_skipped_multipath_ratio", ratio);
    return true;
  }
  return false;
}

double compute_res_at_pred(const ImuGnssTc& tc, const EpochData& ep, Info& info) {
  try {
    gtsam::Values v;
    v.insert(ImuGnssTc::Xpose(ep.key_idx), ep.pred_nav.pose());
    const double res = main_ddpr_residuals(tc, ep.graph, v).rms;
    info.set("ddpr_res_at_pred", res);
    return res;
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
    return std::numeric_limits<double>::infinity();
  }
}

/// Pose to report when sanity recovery fires (TC or IMU-predicted).
gtsam::Pose3 sanity_report_pose(const ImuGnssTc& tc, const EpochData& ep, double pred_res,
                                Info& info) {
  const double thr = tc.cfg.sanity_pose_replace_thresh;
  if (thr <= 0) return ep.pose_tc;
  if (pred_res > thr) {
    info.set("sanity_pose_replace_pred_dirty", pred_res);
    return ep.pose_tc;
  }
  const gtsam::Pose3 pred_pose = ep.pred_nav.pose();
  const double gap = (ep.pose_tc.translation() - pred_pose.translation()).norm();
  info.set("sanity_pose_gap", gap);
  if (gap > thr) {
    info.set("sanity_pose_replaced", 1);
    return pred_pose;
  }
  return ep.pose_tc;
}

EpochResult apply_sanity_reset(ImuGnssTc& tc, EpochData& ep, double pred_res) {
  Info& info = ep.info;
  info.set("sanity_dd_removed", reset_ambiguities_with_cp_hold(tc));
  tc.recovery.ddpr_bad_count = 0;
  if (tc.cfg.sanity_break_pim) tc.recovery.pim_discontinuity = true;
  const gtsam::Pose3 pose = sanity_report_pose(tc, ep, pred_res, info);
  const Vec3 ecef_body = tc.R_enu2ecef * pose.translation() + tc.base_ecef;
  return advance_epoch_and_pack(tc, tc.antenna_ecef(pose, ecef_body), "FLT", 0, info,
                                *ep.in.obs);
}

}  // namespace

std::optional<EpochResult> run_ddpr_sanity(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  const double main_res = info.num("main_ddpr_res", 0.0);
  // Escalation step 1: a clean residual signal resets the bad count.
  if (!(main_res > tc.cfg.main_ddpr_res_thresh)) {
    tc.recovery.ddpr_bad_count = 0;
    return std::nullopt;
  }
  if (ddpr_multipath_dominated(tc, info)) return std::nullopt;
  const double pred_res = compute_res_at_pred(tc, ep, info);

  // Fast path for catastrophic residual spikes.
  if (main_res > tc.cfg.main_ddpr_res_catastrophic && ep.nb <= 0) {
    const double worst = info.main_ddpr_sat_worst ? info.main_ddpr_sat_worst->second : 0.0;
    if (worst >= tc.cfg.ddpr_fast_worst_sat_min) {
      info.set("ddpr_bad", tc.recovery.ddpr_bad_count + 1);
      info.set("ddpr_fast_recover", main_res);
      info.set("ddpr_fast_worst_sat_res", worst);
      return apply_sanity_reset(tc, ep, pred_res);
    }
  }
  // Escalation step 2: count consecutive bad epochs, CP-hold on each.
  tc.recovery.ddpr_bad_count += 1;
  info.set("ddpr_bad", tc.recovery.ddpr_bad_count);
  trigger_cp_hold(tc, "ddpr_main_res", info, main_res);
  if (tc.recovery.ddpr_bad_count < tc.cfg.ddpr_sanity_persist) return std::nullopt;
  info.set("ddpr_recover", tc.recovery.ddpr_bad_count);
  return apply_sanity_reset(tc, ep, pred_res);
}

}  // namespace gnss_fgo
