// pipeline/: the Phase-2 epoch flow, in execution order:
//
//   imu_prediction -> quality_gate
//   -> solve (measurement_factors, update_smoother, check_postfit,
//             fix_ambiguities)
//   -> validate_fix -> report
#include "gnss_fgo/modules.h"

#include <gtsam/navigation/CarrierPhaseFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NhcFactor.h>
#include <gtsam/navigation/PseudorangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>

namespace gnss_fgo {

namespace {

bool is_ddpr(const gtsam::NonlinearFactor* f) {
  return dynamic_cast<const gtsam::DoubleDifferencePseudorangeFactorArm*>(f) != nullptr;
}
bool is_ddcp(const gtsam::NonlinearFactor* f) {
  return dynamic_cast<const gtsam::DoubleDifferenceCarrierPhaseFactorArm*>(f) != nullptr;
}
const HeldDdcpFactor* as_held(const gtsam::NonlinearFactor* f) {
  return dynamic_cast<const HeldDdcpFactor*>(f);
}
std::pair<gtsam::Key, gtsam::Key> key_pair(const gtsam::NonlinearFactor* f) {
  const auto& k = f->keys();
  return {k[0], k.size() > 1 ? k[1] : gtsam::Key(-1)};
}
/// Key tuple of a held-DDCP factor as named in last_custom_ddcp_global.
bool in_custom_meta(const ImuGnssTc& tc, const gtsam::NonlinearFactor* f) {
  const HeldDdcpFactor* h = as_held(f);
  return h && h->keys().size() == 2 && tc.last_custom_ddcp_global.count(key_pair(f));
}

}  // namespace

// ── smoother glue ───────────────────────────────────────────────────

std::shared_ptr<Smoother> ImuGnssTc::make_isam2(double lag) const {
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = cfg.isam2_relinearize_threshold;
  params.relinearizeSkip = cfg.isam2_relinearize_skip;
  // apply_fde addresses this epoch's factors by slot: new factors must
  // always append, so slot reuse stays off.
  params.findUnusedFactorSlots = false;
  return std::make_shared<Smoother>(lag, params);
}

void fls_update(ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
                const gtsam::Values& values, int key_idx,
                const std::vector<gtsam::Key>& keep_keys, bool advance_time,
                bool include_prev) {
  if (advance_time) tc.tc_time += tc.epoch_dt;
  gtsam::FixedLagSmootherKeyTimestampMap ts;
  const double t = tc.tc_time;
  ts[ImuGnssTc::Xpose(key_idx)] = t;
  ts[ImuGnssTc::Vel(key_idx)] = t;
  ts[ImuGnssTc::Bias(key_idx)] = t;
  if (include_prev && key_idx > 0) {
    ts[ImuGnssTc::Xpose(key_idx - 1)] = t;
    ts[ImuGnssTc::Vel(key_idx - 1)] = t;
    ts[ImuGnssTc::Bias(key_idx - 1)] = t;
  }
  for (gtsam::Key k : values.keys()) ts.emplace(k, t);
  for (gtsam::Key k : keep_keys) ts[k] = t;
  tc.isam2->update(graph, values, ts);
}

// ── residuals ───────────────────────────────────────────────────────

MainDdprResult main_ddpr_residuals(const ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
                                   const gtsam::Values& estimate) {
  // UNIT CAVEAT (see residuals.py): rescaled by the FLAT sigma_pr, so with
  // varerr these are elevation-normalised pseudo-metres.
  const double sigma_pr_m = tc.cfg.sigma_pr * std::sqrt(2.0);
  MainDdprResult out;
  std::map<std::size_t, std::tuple<int, int, int>> tag_map;
  std::optional<std::size_t> first_tag;
  for (const auto& [idx, ref, j, f] : tc.last_ddpr_sat_tags) {
    if (!first_tag) first_tag = idx;
    tag_map[idx] = {ref, j, f};
  }
  // Iterate just the tagged DDPR slots when they index this graph.
  std::vector<std::size_t> indices;
  bool skip_type_check = false;
  if (!tag_map.empty() && tag_map.rbegin()->first < graph.size() && graph[*first_tag] &&
      is_ddpr(graph[*first_tag].get())) {
    for (const auto& [idx, _] : tag_map) indices.push_back(idx);
    skip_type_check = true;
  } else {
    for (std::size_t i = 0; i < graph.size(); ++i) indices.push_back(i);
  }
  double ss = 0.0;
  int n = 0;
  for (std::size_t i : indices) {
    const auto& fac = graph[i];
    if (!fac) continue;
    if (!skip_type_check && !is_ddpr(fac.get())) continue;
    double err;
    try {
      err = fac->error(estimate);
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
      continue;
    }
    const double res_m = std::sqrt(2.0 * std::max(err, 0.0)) * sigma_pr_m;
    ss += res_m * res_m;
    ++n;
    auto it = tag_map.find(i);
    if (it != tag_map.end()) {
      const auto [ref, j, f] = it->second;
      // The pair MAX goes to the reference too: deliberate (doubles as an
      // any-bad-pair-swaps-the-ref policy).
      if (res_m > out.per_sat.get(ref, 0.0)) out.per_sat[ref] = res_m;
      if (res_m > out.per_sat.get(j, 0.0)) out.per_sat[j] = res_m;
      out.pairs.push_back({ref, j, f, res_m});
    }
  }
  out.rms = n ? std::sqrt(ss / n) : 0.0;
  return out;
}

std::optional<double> ddpr_res_at_fixed_pose(const ImuGnssTc& tc,
                                             const gtsam::NonlinearFactorGraph& graph,
                                             const gtsam::Values& estimate, gtsam::Key key_pose,
                                             const StateVec& xa) {
  try {
    const gtsam::Pose3 cur = estimate.at<gtsam::Pose3>(key_pose);
    const Mat3 R_be = tc.ecef_T_nav.compose(cur).rotation().matrix();
    const Vec3 body_ecef = xa.pos - R_be * tc.lever_arm_tc;
    const gtsam::Point3 body_nav = tc.ecef_T_nav.transformTo(gtsam::Point3(body_ecef));
    gtsam::Values v;
    v.insert(key_pose, gtsam::Pose3(cur.rotation(), body_nav));
    return main_ddpr_residuals(tc, graph, v).rms;
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Value, PyErr::Index);
    return std::nullopt;
  }
}

std::map<std::string, std::pair<double, int>> all_factor_residuals(
    const ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& estimate) {
  std::map<std::string, std::pair<double, int>> buckets;
  for (std::size_t i = 0; i < graph.size(); ++i) {
    const auto& fac = graph[i];
    if (!fac) continue;
    double err;
    try {
      err = fac->error(estimate);
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
      continue;
    }
    const gtsam::NonlinearFactor* f = fac.get();
    std::string tag = "Other";
    if (is_ddpr(f)) tag = "DDPR";
    else if (is_ddcp(f) || tc.last_custom_ddcp_local.count(i)) tag = "DDCP";
    else if (dynamic_cast<const gtsam::CombinedImuFactor*>(f)) tag = "CombinedImu";
    else if (dynamic_cast<const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>*>(f)) tag = "BetweenBias";
    else if (dynamic_cast<const gtsam::BetweenFactor<double>*>(f)) tag = "BetweenN";
    else if (dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(f)) tag = "PriorBias";
    else if (dynamic_cast<const gtsam::NhcFactor*>(f)) tag = "NHC";
    else if (dynamic_cast<const gtsam::PriorFactor<gtsam::Vector3>*>(f)) tag = "PriorVel";
    else if (dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(f)) tag = "PriorPose";
    else if (dynamic_cast<const gtsam::PriorFactor<double>*>(f)) tag = "PriorN";
    buckets[tag].first += err;
    buckets[tag].second += 1;
  }
  for (auto& [tag, v] : buckets) v.first = std::sqrt(2 * v.first / v.second);
  return buckets;
}

// ── FDE ─────────────────────────────────────────────────────────────

gtsam::Values apply_fde(ImuGnssTc& tc, const gtsam::NonlinearFactorGraph& graph, int key_idx,
                        std::optional<int> nv, gtsam::Values estimate, Info& info,
                        std::optional<std::size_t> fi_start, Smoother* smoother,
                        std::optional<gtsam::Key> pose_key) {
  const bool phase2 = smoother == nullptr;
  Smoother* sm = phase2 ? tc.isam2.get() : smoother;
  const gtsam::Key pkey = pose_key ? *pose_key : ImuGnssTc::Xpose(key_idx);
  const int max_iter = std::max(1, tc.cfg.fde_max_iter);
  const bool iterative = max_iter > 1;
  int total_rejected = 0;
  for (int it = 0; it < max_iter; ++it) {
    const gtsam::NonlinearFactorGraph& all = sm->getFactors();
    const std::size_t nf_total = all.size();
    // Current epoch's factors only: rejecting history un-anchors the
    // trajectory.
    if (!fi_start) fi_start = nf_total > graph.size() ? nf_total - graph.size() : 0;
    info.set("fde_slot_shift", int64_t(nf_total) - int64_t(graph.size()) - int64_t(*fi_start));
    const std::size_t fi_end = std::min(nf_total, *fi_start + graph.size());
    std::vector<std::pair<std::size_t, double>> pr, cp;
    for (std::size_t fi = *fi_start; fi < fi_end; ++fi) {
      const auto& fac = all[fi];
      if (!fac) continue;
      const bool custom_cp = in_custom_meta(tc, fac.get());
      double err;
      try {
        err = fac->error(estimate);
      } catch (const std::exception& e) {
        GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
        continue;
      }
      if (is_ddpr(fac.get()))
        pr.emplace_back(fi, std::sqrt(2.0 * err) * tc.cfg.sigma_pr * std::sqrt(2.0));
      else if (is_ddcp(fac.get()) || custom_cp)
        cp.emplace_back(fi, std::sqrt(2.0 * err) * tc.cfg.sigma_cp * std::sqrt(2.0));
    }
    if (!nv) nv = int(pr.size() + cp.size());
    gtsam::FactorIndices reject;
    if (iterative) {
      // The single largest outlier across PR and CP.
      double best_d = 0.0;
      std::optional<std::size_t> best;
      for (const auto& [fi, r] : pr)
        if (std::abs(r) > tc.cfg.fde_pr && std::abs(r) > best_d) best_d = std::abs(r), best = fi;
      for (const auto& [fi, r] : cp)
        if (std::abs(r) > tc.cfg.fde_cp && std::abs(r) > best_d) best_d = std::abs(r), best = fi;
      if (!best) break;
      reject.push_back(*best);
      if (total_rejected + 1 > tc.cfg.fde_max_frac * std::max(1, *nv)) {
        info.set("fde_skipped", total_rejected + 1);
        if (phase2) trigger_cp_hold(tc, "fde_safeguard", info, total_rejected + 1, true);
        break;
      }
    } else {
      for (const auto& [fi, r] : pr)
        if (std::abs(r) > tc.cfg.fde_pr) reject.push_back(fi);
      for (const auto& [fi, r] : cp)
        if (std::abs(r) > tc.cfg.fde_cp) reject.push_back(fi);
      if (reject.empty()) break;
      if (double(reject.size()) > tc.cfg.fde_max_frac * std::max(1, *nv)) {
        info.set("fde_skipped", reject.size());
        if (phase2) trigger_cp_hold(tc, "fde_safeguard", info, double(reject.size()), true);
        return estimate;
      }
    }
    // Release the holds behind every rejected held-DDCP factor (handing
    // the integers over as seeds); the float arcs stay.
    for (std::size_t fi : reject) {
      const auto& fac = all[fi];
      if (!fac || !in_custom_meta(tc, fac.get())) continue;
      const auto [ref_sat, j_sat, freq] = tc.last_custom_ddcp_global.at(key_pair(fac.get()));
      for (const SfKey& key : {SfKey{ref_sat, freq}, SfKey{j_sat, freq}}) {
        SatState* st = tc.sat_states.find(key);
        if (st && st->held_value) st->release_hold(true);
      }
    }
    total_rejected += int(reject.size());
    try {
      sm->update(gtsam::NonlinearFactorGraph(), gtsam::Values(),
                 gtsam::FixedLagSmootherKeyTimestampMap(), reject);
      const gtsam::Values est = sm->calculateEstimate();
      if (est.exists(pkey)) estimate = est;
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index);
      break;
    }
    if (!iterative) {
      info.set("fde_reject", total_rejected);
      return estimate;
    }
  }
  if (total_rejected) info.set("fde_reject", total_rejected);
  return estimate;
}

// ── stages ──────────────────────────────────────────────────────────

namespace {

// Stage A — IMU preintegration + prediction from the smoother prior.
std::optional<EpochResult> stage_imu_prediction(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  tc.tc_epoch += 1;
  ep.key_idx = tc.tc_epoch;
  info.set("tc_epoch", ep.key_idx);
  ep.imu_idx_prev = tc.imu_idx;
  ep.tow = gnss::time2gpst(ep.in.obs->t);
  const PimResult pr = build_pim_tc(tc, tc.bias_or_init(), ep.tow);
  ep.pim = pr.pim;
  ep.n_imu = pr.n;
  ep.gyro_mean = pr.gyro_mean;
  info.set("n_imu", ep.n_imu);
  info.set("pim_dt_mismatch", std::round((ep.n_imu * 0.01 - tc.epoch_dt) * 1e6) / 1e6);
  if (ep.n_imu == 0) return advance_epoch_and_pack(tc, tc.nav.x.pos, "FLT", 0, info, *ep.in.obs);

  ep.estimate = tc.isam2->calculateEstimate();
  if (!ep.estimate.exists(ImuGnssTc::Xpose(ep.key_idx - 1))) {
    info.set("prev_pose_missing", ep.key_idx - 1);
    const gtsam::Pose3 dummy(gtsam::Rot3(),
                             ep.R_enu2ecef.transpose() * (ep.init_ecef - tc.base_ecef));
    const auto ddpr =
        try_ddpr_reset(tc, ep.in, dummy, dummy.rotation(), info, "ddpr_prev_missing_recover");
    return advance_epoch_and_pack(tc, ddpr ? *ddpr : tc.nav.x.pos, "FLT", 0, info, *ep.in.obs);
  }
  ep.pose_p = ep.estimate.at<gtsam::Pose3>(ImuGnssTc::Xpose(ep.key_idx - 1));
  ep.vel_prev = ep.estimate.at<gtsam::Vector3>(ImuGnssTc::Vel(ep.key_idx - 1));
  ep.bias_prev = ep.estimate.at<gtsam::imuBias::ConstantBias>(ImuGnssTc::Bias(ep.key_idx - 1));
  ep.pred_nav = ep.pim->predict(gtsam::NavState(ep.pose_p, ep.vel_prev), ep.bias_prev);
  info.set("pred_heading_deg", heading_from_pose(ep.pred_nav.pose()));
  ep.values.insert(ImuGnssTc::Xpose(ep.key_idx), ep.pred_nav.pose());
  ep.values.insert(ImuGnssTc::Vel(ep.key_idx), gtsam::Vector3(ep.pred_nav.velocity()));
  ep.values.insert(ImuGnssTc::Bias(ep.key_idx), ep.bias_prev);
  add_imu_chain(tc, ep.graph, ep.values, ep.key_idx, *ep.pim, ep.pose_p, ep.vel_prev, info);
  return std::nullopt;
}

// Stage B — quality gate + slip / CP-hold decisions.
void stage_quality_gate(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  info.set("gdop", tc.compute_gdop_(ep.pred_nav, ep.ns, ep.in));
  info.set("nsat", ep.ns);
  info.set("n_slip", detect_slips_and_reset_ambiguities(tc, ep.in));
  ep.skip_cp_now = tc.recovery.recov_cp_hold > 0;
  OrderedMap<int, double> el_deg, snr;
  const ObsEpoch& obs = *ep.in.obs;
  for (std::size_t i = 0; i < ep.in.sat.size(); ++i) {
    el_deg[ep.in.sat[i]] = ep.in.el[i] * 180.0 / gnss::PI;
    if (obs.S.cols() > 0) snr[ep.in.sat[i]] = obs.S.row(ep.in.iu[i]).maxCoeff();
  }
  info.sat_el_deg = el_deg;
  if (!snr.empty()) info.sat_snr_dbhz = snr;
  if (ep.skip_cp_now) tc.recovery.tick_cp_hold(info);

  // Previous-epoch N values for the BetweenN chain, then key rotation.
  ep.prev_amb_values.clear();
  const auto amb = tc.sat_states.amb_keys_dict();
  for (const auto& [sf, k] : amb) {
    if (ep.skip_cp_now) tc.sat_states.get(sf.first, sf.second).amb_gen += 1;
    else if (ep.estimate.exists(k)) ep.prev_amb_values[sf] = {k, ep.estimate.at<double>(k)};
  }
  tc.sat_states.for_each([](const SfKey&, SatState& st) { st.amb_key.reset(); });

  const Vec3 pred_body = ep.R_enu2ecef * ep.pred_nav.pose().translation() + tc.base_ecef;
  ep.pred_ecef = tc.antenna_ecef(ep.pred_nav.pose(), pred_body);
}

// Stage C1 — DD factors, BetweenN chain, thin-epoch priors, Doppler, NHC, ZUPT.
void stage_measurement_factors(ImuGnssTc& tc, EpochData& ep, int prev_smode) {
  Info& info = ep.info;
  ep.nv = build_dd_factors(tc, ep.graph, ep.values, ep.in, ImuGnssTc::Xpose(ep.key_idx),
                           tc.lever_arm_tc, AmbDictRef(&tc.sat_states), ep.key_idx,
                           &ep.prev_amb_values, ep.skip_cp_now);
  // BetweenFactor(N_prev, N_new, 0): tight after a FIX, loose after FLT.
  info.set("prev_smode", prev_smode);
  const bool last_flt = prev_smode == 5;
  int n_between = 0;
  if (!ep.skip_cp_now && tc.cfg.betweenn_enable) {
    for (const auto& [sf, k_new] : tc.sat_states.amb_keys_dict()) {
      auto it = ep.prev_amb_values.find(sf);
      if (it == ep.prev_amb_values.end()) continue;
      const double sig = last_flt ? tc.cfg.sigma_n_between_flt : tc.cfg.sigma_n_between;
      ep.graph.emplace_shared<gtsam::BetweenFactor<double>>(it->second.first, k_new, 0.0,
                                                            ImuGnssTc::noise1(sig));
      ++n_between;
    }
  }
  info.set("n_dd", ep.nv);
  if (ep.nv < tc.cfg.min_dd_for_solve) {
    // Too thin to solve: leash to the IMU prediction, anchor continuing N.
    info.set("propagate_prior", ep.nv);
    ep.graph.addPrior(ImuGnssTc::Xpose(ep.key_idx), ep.pred_nav.pose(),
                      gtsam::noiseModel::Isotropic::Sigma(6, tc.cfg.propagate_pose_sigma));
    ep.graph.addPrior<gtsam::Vector3>(
        ImuGnssTc::Vel(ep.key_idx), ep.pred_nav.velocity(),
        gtsam::noiseModel::Isotropic::Sigma(3, tc.cfg.propagate_vel_sigma));
    ep.graph.addPrior(ImuGnssTc::Bias(ep.key_idx), ep.bias_prev,
                      gtsam::noiseModel::Isotropic::Sigma(6, tc.cfg.propagate_bias_sigma));
    if (!ep.skip_cp_now) {
      int n_anchored = 0;
      const auto amb_noise = ImuGnssTc::noise1(tc.cfg.propagate_amb_sigma);
      for (const auto& [sf, k_new] : tc.sat_states.amb_keys_dict()) {
        auto it = ep.prev_amb_values.find(sf);
        if (it == ep.prev_amb_values.end()) continue;
        ep.graph.addPrior(k_new, it->second.second, amb_noise);
        ++n_anchored;
      }
      info.set("n_anchored", n_anchored);
    }
  }
  info.set("n_between", n_between);

  add_sd_doppler_factors(tc, ep);

  double speed;
  try {
    speed = ep.estimate.at<gtsam::Vector3>(ImuGnssTc::Vel(ep.key_idx - 1)).head<2>().norm();
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
    speed = Vec3(ep.pred_nav.velocity()).head<2>().norm();
  }
  if (add_nhc_factor(tc, ep.graph, ep.key_idx, speed, ep.gyro_mean)) info.set("nhc", true);
  add_zupt_factors_for_stage(tc, ep);
}

// Stage C2 — smoother update.
std::optional<EpochResult> stage_update_smoother(ImuGnssTc& tc, EpochData& ep) {
  try {
    std::vector<gtsam::Key> extra;
    for (const auto& [sf, k] : tc.sat_states.amb_keys_dict()) extra.push_back(k);
    for (const auto& [sf, prev] : ep.prev_amb_values)
      if (tc.sat_states.at(sf.first, sf.second).amb_key) extra.push_back(prev.first);
    // This epoch's factors land at [nf_before, nf_before + G).
    ep.nf_before = tc.isam2->getFactors().size();
    fls_update(tc, ep.graph, ep.values, ep.key_idx, extra);
    ep.estimate = tc.isam2->calculateEstimate();
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index, PyErr::Value);
    return handle_solve_exception(tc, e, ep);
  }
  return std::nullopt;
}

// Stage C3 — post-fit diagnostics, FDE re-solve, pose snapshot.
void stage_check_postfit(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  MainDdprResult m = main_ddpr_residuals(tc, ep.graph, ep.estimate);
  info.set("main_ddpr_res", m.rms);
  info.main_ddpr_per_sat = m.per_sat;
  info.main_ddpr_pairs = m.pairs;
  info.ref_sats = tc.current_epoch.ref_sats;
  tc.cached_ddpr_res_pre = m.rms;
  tc.mres_signals.last_res = m.rms;
  tc.mres_signals.per_sat = m.per_sat;
  if (tc.cfg.diag_factor_residuals) {
    for (const auto& [tag, v] : all_factor_residuals(tc, ep.graph, ep.estimate)) {
      info.set("fres_" + tag, v.first);
      info.set("fcnt_" + tag, v.second);
    }
  }
  if (auto worst = first_argmax(m.per_sat)) info.main_ddpr_sat_worst = *worst;
  if (tc.cfg.fde_enable)
    ep.estimate = apply_fde(tc, ep.graph, ep.key_idx, ep.nv, ep.estimate, info, ep.nf_before);

  ep.pose_tc = ep.estimate.at<gtsam::Pose3>(ImuGnssTc::Xpose(ep.key_idx));
  info.set("post_heading_deg", heading_from_pose(ep.pose_tc));
  tc.tc_bias = ep.estimate.at<gtsam::imuBias::ConstantBias>(ImuGnssTc::Bias(ep.key_idx));
  ep.ecef_tc = ep.R_enu2ecef * ep.pose_tc.translation() + tc.base_ecef;
}

const std::map<std::string, int> kArOutcomeCodes = {
    {"not_called", 0},  {"armode_off", 1},   {"entered", 2},
    {"lambda_exception", 3}, {"lambda_zero", 4}, {"ar_context_reject", 7},
    {"success", 8},     {"gdop_gate", 12},   {"problem_unposed", 13},
    {"min_pairs_declined", 15},
};

void run_ar_with_marginals(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  tc.nav.x.pos = tc.antenna_ecef(ep.pose_tc, ep.ecef_tc);
  const AmbDict amb_snapshot = tc.sat_states.amb_keys_dict();
  publish_marginals(tc, ep.estimate, ImuGnssTc::Xpose(ep.key_idx), amb_snapshot);
  // AR-only geometry gate: pure geometry, not the marginal covariance.
  const double ar_gdop = tc.cfg.ar_gdop_max;
  if (ar_gdop > 0.0 && info.num("gdop", 0.0) > ar_gdop) {
    info.set("ar_gdop_skip", true);
    tc.ar_diag.outcome = "gdop_gate";
    return;
  }
  auto [nb, xa] = run_ar(tc, ep.in.sat, ep.in.el, amb_snapshot);
  ep.nb = nb;
  ep.xa = xa;
  const double xv_thr = tc.cfg.ar_ddpr_xvalidate_thresh;
  if (xv_thr > 0.0 && ep.nb > 0 && ep.xa) {
    const auto res_xa =
        ddpr_res_at_fixed_pose(tc, ep.graph, ep.estimate, ImuGnssTc::Xpose(ep.key_idx), *ep.xa);
    if (res_xa) {
      info.set("ar_ddpr_xvalidate_res_at_xa", *res_xa);
      if (tc.cached_ddpr_res_pre)
        info.set("ar_ddpr_xvalidate_delta", *res_xa - *tc.cached_ddpr_res_pre);
      if (*res_xa > xv_thr) {
        info.set("ar_ddpr_xvalidate_reject", true);
        // Seed the rejected integers instead of holding them.
        int n_seeded = 0;
        for (const auto& [sf, _] : amb_snapshot) {
          if (tc.nav.fix(sf.first - 1, sf.second) != 2) continue;
          SatState& st = tc.sat_states.get(sf.first, sf.second);
          st.last_held_value = ep.xa->at(sf.first, sf.second);
          st.release_seed_pending = true;
          ++n_seeded;
        }
        info.set("xvalidate_soft_seeded", n_seeded);
        ep.nb = 0;
        ep.xa.reset();
        tc.nav.smode = 5;
      }
    }
  }
  // Hold only what survived the verdict; a partial fix holds only its
  // exact-integer subset.
  if (ep.nb > 0 && ep.xa && tc.nav.armode == 3) {
    const bool partial = tc.ar_diag.partial_dropped.value_or(0) != 0;
    const std::optional<std::set<SfKey>> allow =
        partial ? tc.ar_partial_holdable : std::nullopt;
    if (!partial || (allow && !allow->empty())) apply_fix_and_hold(tc, amb_snapshot, *ep.xa, allow);
  }
}

void record_ar_diagnostics(ImuGnssTc& tc, Info& info) {
  const ArDiagnostics& d = tc.ar_diag;
  const std::pair<const std::optional<int>*, const char*> attrs[] = {
      {&d.orphan_cp_count, "orphan_cp_count"},
      {&d.amb_dict_size, "amb_dict_size"},
      {&d.held_size, "held_size"},
      {&d.cp_visible_size, "cp_visible_size"},
      {&d.amb_estimate_missing, "amb_estimate_missing"},
      {&d.amb_vsat1, "amb_vsat1"},
      {&d.amb_vsat0_young, "amb_vsat0_young"},
      {&d.amb_not_in_obs, "amb_not_in_obs"},
      {&d.held_not_in_obs, "held_not_in_obs"},
      {&d.sat_in_obs_size, "sat_in_obs_size"},
      {&d.resamb_raw_nb, "resamb_raw_nb"},
      {&d.exact_fit_accept, "ar_exact_fit_accept"},
      {&d.amb_el_min_deg, "amb_el_min_deg"},
      {&d.amb_el_median_deg, "amb_el_median_deg"},
      {&d.amb_el_above15, "amb_el_above15"},
      {&d.amb_el_above25, "amb_el_above25"},
  };
  for (const auto& [v, key] : attrs)
    if (*v) info.set(key, **v);
  if (d.exception) info.set("ar_exception", *d.exception);
  if (!d.outcome.empty()) {
    auto it = kArOutcomeCodes.find(d.outcome);
    info.set("ar_outcome_code", it != kArOutcomeCodes.end() ? it->second : -1);
  }
  if (d.partial_dropped.value_or(0)) info.set("ar_partial_dropped", *d.partial_dropped);
  if (tc.ar_context_reject) {
    info.set("ar_context_reject", true);
    info.set("ar_context_reject_nb", tc.ar_context_reject->nb);
    info.set("ar_context_reject_main_ddpr_res", tc.ar_context_reject->main_ddpr_res);
    info.set("ar_context_reject_worst_sat_res", tc.ar_context_reject->worst_sat_res);
  }
  if (tc.ar_subset_debug) {
    info.set("ar_subset_candidates", tc.ar_subset_debug->candidates);
    if (tc.ar_subset_debug->used) {
      info.set("ar_subset_used", true);
      info.ar_subset_drop_sats = tc.ar_subset_debug->drop_sats;
      info.set("ar_subset_nb", tc.ar_subset_debug->nb);
      info.set("ar_subset_ratio", tc.ar_subset_debug->ratio);
    }
  }
}

/// Purge cold-biased float arcs after prolonged ratio starvation with a
/// quiet float (no other alarm can see that basin).
void ar_starvation_reset(ImuGnssTc& tc, EpochData& ep) {
  const int n_max = tc.cfg.ar_starve_reset;
  if (n_max <= 0) return;
  const std::string& outcome = tc.ar_diag.outcome;
  if (outcome == "success") {
    tc.ar_starve_streak = 0;
    return;
  }
  if (outcome != "lambda_zero") return;
  tc.ar_starve_streak += 1;
  ep.info.set("ar_starve_streak", tc.ar_starve_streak);
  if (tc.ar_starve_streak < n_max || tc.recovery.recov_cp_hold > 0) return;
  if (tc.cached_ddpr_res_pre && *tc.cached_ddpr_res_pre > tc.cfg.ar_starve_max_res)
    return;  // NLOS storm — arcs are load-bearing
  ep.info.set("ar_starve_reset", reset_ambiguities_with_cp_hold(tc));
  tc.ar_starve_streak = 0;
}

// Stage C4 — the AR pass over the FDE-cleaned float solution.
void stage_fix_ambiguities(ImuGnssTc& tc, EpochData& ep) {
  tc.nav.smode = 5;
  ep.nb = 0;
  ep.xa.reset();
  if (ep.key_idx < tc.cfg.n_collect + 3) return;
  if (tc.recovery.recov_cp_hold > 0) return;
  run_ar_with_marginals(tc, ep);
  ar_starvation_reset(tc, ep);
  record_ar_diagnostics(tc, ep.info);
}

/// Release the single most suspicious held ambiguity after an FLT epoch.
void release_suspicious_held_on_flt(ImuGnssTc& tc, Info& info) {
  std::optional<int> worst_sat;
  double worst_res = 0.0;
  if (info.main_ddpr_sat_worst) {
    worst_sat = info.main_ddpr_sat_worst->first;
    worst_res = info.main_ddpr_sat_worst->second;
  }
  const double res_thr = std::max(2.0, 0.5 * tc.cfg.ar_context_worst_sat_max);
  std::vector<std::tuple<double, double, int, int>> cands;
  for (const auto& [sf, _] : tc.sat_states.held_items()) {
    const double sat_res =
        info.main_ddpr_per_sat ? info.main_ddpr_per_sat->get(sf.first, 0.0) : 0.0;
    double score = 0.0;
    if (sat_res >= res_thr) score += sat_res;
    if (worst_sat && sf.first == *worst_sat && worst_res >= res_thr)
      score += std::max(1.0, 0.25 * worst_res);
    if (score > 0.0) cands.emplace_back(score, sat_res, sf.first, sf.second);
  }
  if (cands.empty()) return;
  const auto [score, sat_res, s, f] = *std::max_element(cands.begin(), cands.end());
  SatState& st = tc.sat_states.get(s, f);
  st.release_hold(true);
  st.fix_streak = 0;
  info.set("held_release_flt_count", 1);
  info.set("held_release_flt_sat", s);
  info.set("held_release_flt_freq", f);
  info.set("held_release_flt_score", score);
  info.set("held_release_flt_res", sat_res);
}

// Stage D — post-solve accuracy policy.
std::optional<EpochResult> stage_validate_fix(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  const Vec3 pred_ecef = ep.R_enu2ecef * ep.pred_nav.pose().translation() + tc.base_ecef;
  info.set("innovation", (ep.ecef_tc - pred_ecef).norm());
  if (tc.cfg.ddpr_sanity_enable) {
    if (auto r = run_ddpr_sanity(tc, ep)) return r;
  }
  // FIX / FLT decision: lambda_correction and low-nb gates.
  const Vec3 pose_ant = tc.antenna_ecef(ep.pose_tc, ep.ecef_tc);
  if (tc.nav.smode == 4 && ep.nb > 0) {
    const double lc = (ep.xa->pos - pose_ant).norm();
    info.set("lambda_correction", lc);
    const bool prev_was_flt = int(info.num("prev_smode", 0)) == 5;
    int streak_max = 0;
    tc.sat_states.for_each(
        [&](const SfKey&, const SatState& st) { streak_max = std::max(streak_max, st.fix_streak); });
    info.set("prev_fix_streak_max", streak_max);
    bool low_nb_fresh = prev_was_flt;
    if (tc.cfg.low_nb_fix_reject_max_prev_fix_streak > 0)
      low_nb_fresh = low_nb_fresh || streak_max <= tc.cfg.low_nb_fix_reject_max_prev_fix_streak;
    const double main_res = info.num("main_ddpr_res", 0.0);
    if (tc.cfg.lambda_corr_hard_max > 0 && lc > tc.cfg.lambda_corr_hard_max) {
      info.set("lambda_corr_hard_reject", lc);
      ep.sol = pose_ant, ep.tag = "FLT", ep.nb = 0;
    } else if (tc.cfg.low_nb_fix_reject_nb_max > 0 && ep.nb <= tc.cfg.low_nb_fix_reject_nb_max &&
               (!tc.cfg.low_nb_fix_only_after_flt || low_nb_fresh)) {
      info.set("low_nb_fix_reject", true);
      info.set("low_nb_fix_reject_nb", ep.nb);
      info.set("low_nb_fix_reject_lc", lc);
      info.set("low_nb_fix_reject_main_ddpr_res", main_res);
      ep.sol = pose_ant, ep.tag = "FLT", ep.nb = 0;
    } else {
      ep.sol = ep.xa->pos, ep.tag = "FIX";
    }
  } else {
    ep.sol = pose_ant, ep.tag = "FLT";
  }
  // Per-sat fix-streak update.
  if (ep.tag == "FIX") {
    for (const auto& [sf, _] : tc.sat_states.amb_items()) {
      SatState& st = tc.sat_states.get(sf.first, sf.second);
      st.fix_streak = tc.nav.fix(sf.first - 1, sf.second) == 3 ? st.fix_streak + 1 : 0;
    }
  } else {
    tc.sat_states.for_each([](const SfKey&, SatState& st) { st.fix_streak = 0; });
    release_suspicious_held_on_flt(tc, info);
  }
  return std::nullopt;
}

// Stage E — emit the result and book-keep.
EpochResult stage_report(ImuGnssTc& tc, EpochData& ep) {
  Info& info = ep.info;
  try {
    if (ep.key_idx > 0) {
      const gtsam::Pose3 prev = ep.estimate.at<gtsam::Pose3>(ImuGnssTc::Xpose(ep.key_idx - 1));
      const Vec3 prev_ecef = ep.R_enu2ecef * prev.translation() + tc.base_ecef;
      const Vec3 prev_ant = tc.antenna_ecef(prev, prev_ecef);
      if (tc.last_sol_ecef) info.set("prev_pose_drift", (prev_ant - *tc.last_sol_ecef).norm());
    }
  } catch (const std::exception& e) {
    GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
  }
  tc.last_sol_ecef = ep.sol;
  info.set("bias_acc", Vec3(tc.tc_bias->accelerometer()));
  info.set("bias_gyro", Vec3(tc.tc_bias->gyroscope()));
  tc.last_per_sat_res = info.main_ddpr_per_sat ? *info.main_ddpr_per_sat : OrderedMap<int, double>();
  return advance_epoch_and_pack(tc, ep.sol, ep.tag, ep.nb, std::move(info), *ep.in.obs);
}

}  // namespace

EpochResult run_tc_epoch(ImuGnssTc& tc, const EpochInput& in, Info info, int ns,
                         const Vec3& init_ecef, const Mat3& R) {
  EpochData ep;
  ep.in = in;
  ep.info = std::move(info);
  ep.ns = ns;
  ep.init_ecef = init_ecef;
  ep.R_enu2ecef = R;
  if (auto r = stage_imu_prediction(tc, ep)) return *r;
  stage_quality_gate(tc, ep);
  const int prev_smode = tc.nav.smode;
  stage_measurement_factors(tc, ep, prev_smode);
  if (auto r = stage_update_smoother(tc, ep)) return *r;
  stage_check_postfit(tc, ep);
  stage_fix_ambiguities(tc, ep);
  if (auto r = stage_validate_fix(tc, ep)) return *r;
  return stage_report(tc, ep);
}

}  // namespace gnss_fgo
