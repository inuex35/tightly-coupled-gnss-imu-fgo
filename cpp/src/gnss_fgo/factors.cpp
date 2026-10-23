// factors/: measurement -> GTSAM factor builders.
#include "gnss_fgo/modules.h"

#include "gnss_fgo/gnss.h"

#include <gtsam/navigation/CarrierPhaseFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/DopplerFactor.h>
#include <gtsam/navigation/NhcFactor.h>
#include <gtsam/navigation/PseudorangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace gnss_fgo {

namespace {

constexpr double kClight = 299792458.0;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

gtsam::SharedNoiseModel diag(std::initializer_list<double> s) {
  gtsam::Vector v(s.size());
  int i = 0;
  for (double x : s) v[i++] = x;
  return gtsam::noiseModel::Diagonal::Sigmas(v);
}

}  // namespace

// ── HeldDdcpFactor ──────────────────────────────────────────────────

HeldDdcpFactor::HeldDdcpFactor(gtsam::Key key_pose, std::optional<gtsam::Key> key_float,
                               const gtsam::SharedNoiseModel& noise,
                               const Vec3& sat_ref_rov, const Vec3& sat_j_rov,
                               const Vec3& sat_ref_base, const Vec3& sat_j_base,
                               const Vec3& base_ecef, double dd_obs_cp, const Vec3& lever,
                               const gtsam::Pose3& ecef_T_nav, double offset_m,
                               double coeff_m)
    : gtsam::NoiseModelFactor(noise, key_float ? gtsam::KeyVector{key_pose, *key_float}
                                               : gtsam::KeyVector{key_pose}),
      has_float_(key_float.has_value()),
      sat_ref_rov_(sat_ref_rov),
      sat_j_rov_(sat_j_rov),
      lever_(lever),
      ecef_T_nav_(ecef_T_nav),
      coeff_m_(coeff_m) {
  // Fold the base ranges, the held offset and the observation into one
  // constant so the error only does the rover-side geometry.
  const double rho_base_ref = gnss::geodist(sat_ref_base, base_ecef, nullptr);
  const double rho_base_j = gnss::geodist(sat_j_base, base_ecef, nullptr);
  err_const_ = offset_m - (rho_base_ref - rho_base_j) - dd_obs_cp;
}

gtsam::Vector HeldDdcpFactor::unwhitenedError(const gtsam::Values& x,
                                              gtsam::OptionalMatrixVecType H) const {
  const gtsam::Pose3 pose = x.at<gtsam::Pose3>(keys()[0]);
  const gtsam::Pose3 pose_ecef = ecef_T_nav_.compose(pose);
  const Mat3 R = pose_ecef.rotation().matrix();
  const Vec3 antenna = pose_ecef.translation() + R * lever_;
  Vec3 e_ref, e_j;
  const double rho_ref = gnss::geodist(sat_ref_rov_, antenna, &e_ref);
  const double rho_j = gnss::geodist(sat_j_rov_, antenna, &e_j);
  double err = (rho_ref - rho_j) + err_const_;
  if (has_float_) err += coeff_m_ * x.at<double>(keys()[1]);

  if (H) {
    const double omc = gnss::OMGE / gnss::CLIGHT;
    // d(rho_ref - rho_j)/d(antenna), Sagnac term included.
    const Vec3 h(-(e_ref[0] - e_j[0]) - omc * (sat_ref_rov_[1] - sat_j_rov_[1]),
                 -(e_ref[1] - e_j[1]) + omc * (sat_ref_rov_[0] - sat_j_rov_[0]),
                 -(e_ref[2] - e_j[2]));
    const Vec3 hR = R.transpose() * h;   // h @ R
    gtsam::Matrix Hp(1, 6);
    const Vec3 rot = lever_.cross(hR);   // H_ant @ (-R skew(lever))
    Hp << rot[0], rot[1], rot[2], hR[0], hR[1], hR[2];
    (*H)[0] = Hp;
    if (has_float_) (*H)[1] = (gtsam::Matrix(1, 1) << coeff_m_).finished();
  }
  return (gtsam::Vector(1) << err).finished();
}

// ── prefit ──────────────────────────────────────────────────────────

double varerr_dd_sigma(const ImuGnssTc& tc, int code, double el_rad, double dt_s) {
  const TcConfig& cfg = tc.cfg;
  const double fact = code ? cfg.err_eratio_pr : 1.0;
  const double a = fact * cfg.err_a;
  const double b = fact * cfg.err_b;
  const double d = kClight * cfg.err_sclkstab * dt_s;
  const double sinel = std::max(std::sin(el_rad), 0.05);
  const double var_sd = 2.0 * (a * a + b * b / (sinel * sinel)) + d * d;
  return std::sqrt(2.0 * var_sd);
}

std::pair<int, int> pick_ref_sat_idx(const ImuGnssTc& tc, int sys_id,
                                     const std::vector<int>& idx_sys,
                                     const std::vector<int>& sat,
                                     const std::vector<double>& el) {
  const auto& ref_sats = tc.current_epoch.ref_sats;
  auto itp = ref_sats.find(sys_id);
  const std::optional<int> prev_ref =
      itp != ref_sats.end() ? std::optional<int>(itp->second) : std::nullopt;
  const double thresh = tc.cfg.per_sat_res_thresh;
  if (prev_ref) {
    const bool prev_is_geo = sys_id == gnss::BDS && gnss::is_bds_geo(*prev_ref);
    const bool prev_multipath = tc.last_per_sat_res.get(*prev_ref, 0.0) > thresh;
    for (int i : idx_sys)
      if (sat[i] == *prev_ref && !prev_is_geo && !prev_multipath) return {i, *prev_ref};
  }
  // Fresh pick: highest elevation, excluding BDS GEO and high-residual sats.
  auto is_geo = [&](int i) { return sys_id == gnss::BDS && gnss::is_bds_geo(sat[i]); };
  std::vector<int> pool;
  for (int i : idx_sys)
    if (!is_geo(i) && !(tc.last_per_sat_res.get(sat[i], 0.0) > thresh)) pool.push_back(i);
  if (pool.empty())
    for (int i : idx_sys)
      if (!is_geo(i)) pool.push_back(i);
  if (pool.empty()) pool = idx_sys;
  int best = pool[0];
  for (int i : pool)
    if (el[i] > el[best]) best = i;
  return {best, sat[best]};
}

// ── amb_seed ────────────────────────────────────────────────────────

namespace {

/// One N value + prior, three-mode policy (pending re-seed, continuing,
/// fresh). See factors/amb_seed.py.
void seed_one_amb_prior(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph,
                        gtsam::Values& values, SatState& st, gtsam::Key key_n,
                        double n0_seed, const PrevAmbValues* prev, const SfKey& key_id) {
  if (st.release_seed_pending && st.last_held_value) {
    const double n0 = *st.last_held_value;
    values.insert(key_n, n0);
    graph.addPrior(key_n, n0, ImuGnssTc::noise1(0.1));
    tc.current_epoch.seeded_amb_keys.insert(key_id);
    st.release_seed_pending = false;
    return;
  }
  if (prev) {
    auto it = prev->find(key_id);
    if (it != prev->end()) {
      const double n0 = it->second.second;
      values.insert(key_n, n0);
      graph.addPrior(key_n, n0, ImuGnssTc::noise1(tc.cfg.sigma_cont));
      return;
    }
  }
  const double sig = tc.phase == 2 ? tc.cfg.sigma_amb0 : 3.0;
  values.insert(key_n, n0_seed);
  graph.addPrior(key_n, n0_seed, ImuGnssTc::noise1(sig));
  tc.current_epoch.seeded_amb_keys.insert(key_id);
}

struct SeedInfo {
  int sat;
  gtsam::Key key_n;
  double cp_rover, cp_base, pr_rover, pr_base;
};

void init_dd_ambiguity_priors(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph,
                              gtsam::Values& values, const AmbDictRef& amb_dict,
                              std::map<SfKey, gtsam::Key>& new_amb,
                              const PrevAmbValues* prev, int freq, double lam,
                              const SeedInfo (&pair)[2]) {
  for (const SeedInfo& s : pair) {
    const SfKey key_id{s.sat, freq};
    SatState& st = tc.sat_states.get(s.sat, freq);
    // Clock-free level: SD phase minus SD code (NOT geometry).
    const double n0_seed = ((s.cp_rover - s.cp_base) - (s.pr_rover - s.pr_base)) / lam;
    if (st.held_value) continue;
    if (amb_dict.contains(key_id) || new_amb.count(key_id)) continue;
    if (values.exists(s.key_n)) continue;
    seed_one_amb_prior(tc, graph, values, st, s.key_n, n0_seed, prev, key_id);
    new_amb[key_id] = s.key_n;
  }
}

// ── DD factor builder ───────────────────────────────────────────────

class DdFactorBuilder {
 public:
  DdFactorBuilder(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                  const EpochInput& in, gtsam::Key key_pose, const Vec3& lever,
                  AmbDictRef amb_dict, int dd_epoch, const PrevAmbValues* prev,
                  bool skip_cp)
      : tc_(tc), graph_(graph), values_(values), in_(in), key_pose_(key_pose),
        lever_(lever), amb_dict_(amb_dict), dd_epoch_(dd_epoch), prev_(prev),
        skip_cp_(skip_cp), nf_(tc.nav.nf), rb_(tc.nav.rb) {
    // Per-epoch tags so the consumers see this build only.
    tc.last_ddpr_sat_tags.clear();
    tc.last_custom_ddcp_local.clear();
    tc.last_custom_ddcp_global.clear();
    tc.ar_cp_visible_sf.clear();
    el_min_rad_ = std::max(1.0, tc.cfg.el_mask_deg) * gnss::PI / 180.0;
    dt_s_ = tc.epoch_dt;
    use_varerr_ = bool(tc.cfg.varerr_enable);
  }

  int run() {
    int nv = 0;
    const int ns = int(in_.sat.size());
    for (int sys_id : in_.sig().sorted_sys_ids()) {
      std::vector<int> idx_sys;
      for (int i = 0; i < ns; ++i)
        if (gnss::sat_sys(in_.sat[i]) == sys_id) idx_sys.push_back(i);
      if (idx_sys.size() < 2) continue;
      nv += build_system(sys_id, idx_sys);
    }
    for (const auto& [k, key] : new_amb_) amb_dict_.set(k, key);
    return nv;
  }

 private:
  struct Geom {
    gtsam::Point3 pt, base_pt;
  };

  double pair_sigma(int code, double el_ref, double el_j) const {
    if (use_varerr_) {
      const double el_pair = std::max(std::min(el_ref, el_j), el_min_rad_);
      return varerr_dd_sigma(tc_, code, el_pair, dt_s_);
    }
    return (code ? tc_.cfg.sigma_pr : tc_.cfg.sigma_cp) * std::sqrt(2.0);
  }

  Geom geometry(int idx, int s) const {
    Geom g;
    g.pt = in_.rs.row(in_.iu[idx]).transpose();
    auto it = in_.ir_map.find(s);
    g.base_pt = it != in_.ir_map.end() ? gtsam::Point3(in_.rsb.row(it->second).transpose())
                                       : g.pt;
    return g;
  }

  double base_obs(const Eigen::MatrixXd& M, int s, int f) const {
    auto it = in_.ir_map.find(s);
    return it != in_.ir_map.end() ? M(it->second, f) : 0.0;
  }

  int build_system(int sys_id, const std::vector<int>& idx_sys) {
    const auto [ref_idx, ref_sat] = pick_ref_sat_idx(tc_, sys_id, idx_sys, in_.sat, in_.el);
    tc_.current_epoch.ref_sats[sys_id] = ref_sat;
    const std::vector<double> lams = in_.sig().wavelengths(ref_sat);
    const Geom ref = geometry(ref_idx, ref_sat);
    const bool skip_bds_geo = bool(tc_.cfg.exclude_bds_geo);
    int nv = 0;
    for (int j_idx : idx_sys) {
      if (j_idx == ref_idx) continue;
      const int j_sat = in_.sat[j_idx];
      if (skip_bds_geo && gnss::sat_sys(j_sat) == gnss::BDS && gnss::is_bds_geo(j_sat))
        continue;
      nv += build_pair(sys_id, ref_idx, ref_sat, ref, lams, j_idx, j_sat);
    }
    return nv;
  }

  int build_pair(int sys_id, int ref_idx, int ref_sat, const Geom& ref,
                 const std::vector<double>& lams, int j_idx, int j_sat) {
    const Geom j = geometry(j_idx, j_sat);
    int nv = 0;
    for (int f = 0; f < nf_; ++f) {
      // PR factor: added even without CP.
      double pr[4];
      if (!build_pr(ref_idx, j_idx, ref_sat, j_sat, f, ref, j, pr)) continue;
      ++nv;
      nv += build_cp(sys_id, ref_idx, j_idx, ref_sat, j_sat, f, lams, ref, j, pr);
    }
    return nv;
  }

  bool build_pr(int ref_idx, int j_idx, int ref_sat, int j_sat, int f, const Geom& ref,
                const Geom& j, double pr[4]) {
    const ObsEpoch& obs = *in_.obs;
    pr[0] = obs.P(in_.iu[ref_idx], f);
    pr[1] = base_obs(in_.obsb->P, ref_sat, f);
    pr[2] = obs.P(in_.iu[j_idx], f);
    pr[3] = base_obs(in_.obsb->P, j_sat, f);
    if (pr[0] == 0 || pr[1] == 0 || pr[2] == 0 || pr[3] == 0) return false;
    const double sigma = pair_sigma(1, in_.el[ref_idx], in_.el[j_idx]);
    tc_.last_ddpr_sat_tags.emplace_back(graph_.size(), ref_sat, j_sat, f);
    graph_.emplace_shared<gtsam::DoubleDifferencePseudorangeFactorArm>(
        key_pose_, pr[0], pr[1], pr[2], pr[3], ref.pt, j.pt, ref.base_pt, j.base_pt,
        gtsam::Point3(tc_.base_ecef), gtsam::Point3(lever_), tc_.ecef_T_nav,
        ImuGnssTc::noise1(sigma));
    return true;
  }

  int build_cp(int sys_id, int ref_idx, int j_idx, int ref_sat, int j_sat, int f,
               const std::vector<double>& lams, const Geom& ref, const Geom& j,
               const double pr[4]) {
    if (sys_id == gnss::GLO) return 0;
    if (f >= int(lams.size()) || lams[f] <= 0) return 0;
    const double lam = lams[f];
    const ObsEpoch& obs = *in_.obs;
    const double cp_ref_r = obs.L(in_.iu[ref_idx], f) * lam;
    const double cp_ref_b = base_obs(in_.obsb->L, ref_sat, f) * lam;
    const double cp_j_r = obs.L(in_.iu[j_idx], f) * lam;
    const double cp_j_b = base_obs(in_.obsb->L, j_sat, f) * lam;
    if (cp_ref_r == 0 || cp_ref_b == 0 || cp_j_r == 0 || cp_j_b == 0) return 0;
    tc_.ar_cp_visible_sf.insert({ref_sat, f});
    tc_.ar_cp_visible_sf.insert({j_sat, f});

    SatState& ref_state = tc_.sat_states.get(ref_sat, f);
    SatState& j_state = tc_.sat_states.get(j_sat, f);
    // amb_gen is part of the key VALUE (slip resets bump it before the
    // build); kept for key stability.
    const gtsam::Key key_n_ref = ImuGnssTc::N(ref_sat, f, int64_t(dd_epoch_) * 100 + ref_state.amb_gen);
    const gtsam::Key key_n_j = ImuGnssTc::N(j_sat, f, int64_t(dd_epoch_) * 100 + j_state.amb_gen);
    const SeedInfo seeds[2] = {{ref_sat, key_n_ref, cp_ref_r, cp_ref_b, pr[0], pr[1]},
                               {j_sat, key_n_j, cp_j_r, cp_j_b, pr[2], pr[3]}};
    init_dd_ambiguity_priors(tc_, graph_, values_, amb_dict_, new_amb_, prev_, f, lam, seeds);
    // Read held state AFTER seeding (the seeding may release a hold).
    const std::optional<double> ref_held = ref_state.held_value;
    const std::optional<double> j_held = j_state.held_value;

    if (skip_cp_) return 0;
    const double cp_sigma = pair_sigma(0, in_.el[ref_idx], in_.el[j_idx]);
    const auto cp_noise = ImuGnssTc::noise1(cp_sigma);
    const double dd_obs_cp = (cp_ref_r - cp_j_r) - (cp_ref_b - cp_j_b);
    const std::tuple<int, int, int> pair_id{ref_sat, j_sat, f};
    auto emit_held = [&](std::optional<gtsam::Key> key_float, double offset_m,
                         double coeff_m) {
      graph_.emplace_shared<HeldDdcpFactor>(key_pose_, key_float, cp_noise, ref.pt, j.pt,
                                            ref.base_pt, j.base_pt, rb_, dd_obs_cp, lever_,
                                            tc_.ecef_T_nav, offset_m, coeff_m);
      tc_.last_custom_ddcp_local.insert(graph_.size() - 1);
      // Pose-only factors (both held) stay outside post-fit FDE.
      if (key_float) tc_.last_custom_ddcp_global[{key_pose_, *key_float}] = pair_id;
    };
    if (ref_held && j_held) {
      emit_held(std::nullopt, lam * (*ref_held - *j_held), 0.0);
    } else if (ref_held) {
      emit_held(key_n_j, lam * *ref_held, -lam);
    } else if (j_held) {
      emit_held(key_n_ref, -lam * *j_held, lam);
    } else {
      graph_.emplace_shared<gtsam::DoubleDifferenceCarrierPhaseFactorArm>(
          key_pose_, key_n_ref, key_n_j, cp_ref_r, cp_ref_b, cp_j_r, cp_j_b, ref.pt, j.pt,
          ref.base_pt, j.base_pt, gtsam::Point3(tc_.base_ecef), lam, gtsam::Point3(lever_),
          tc_.ecef_T_nav, cp_noise);
    }
    return 1;
  }

  ImuGnssTc& tc_;
  gtsam::NonlinearFactorGraph& graph_;
  gtsam::Values& values_;
  const EpochInput& in_;
  gtsam::Key key_pose_;
  Vec3 lever_;
  AmbDictRef amb_dict_;
  int dd_epoch_;
  const PrevAmbValues* prev_;
  bool skip_cp_;
  int nf_;
  Vec3 rb_;
  double el_min_rad_ = 0, dt_s_ = 0;
  bool use_varerr_ = true;
  std::map<SfKey, gtsam::Key> new_amb_;
};

}  // namespace

int build_dd_factors(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                     const EpochInput& in, gtsam::Key key_pose, const Vec3& lever,
                     AmbDictRef amb_dict, int dd_epoch, const PrevAmbValues* prev,
                     bool skip_cp) {
  DdFactorBuilder b(tc, graph, values, in, key_pose, lever, amb_dict, dd_epoch, prev,
                    skip_cp);
  return b.run();
}

// ── SD Doppler ──────────────────────────────────────────────────────

namespace {

struct DopplerRow {
  int sat;
  Vec3 p_sat, v_sat;
  double d_obs, lam, el, snr;
};

std::vector<DopplerRow> doppler_rows(const ImuGnssTc& tc, const EpochData& ep) {
  std::vector<DopplerRow> rows;
  const ObsEpoch& obs = *ep.in.obs;
  for (std::size_t si = 0; si < ep.in.iu.size(); ++si) {
    const int i_obs = ep.in.iu[si];
    const int s = ep.in.sat[si];
    if (gnss::sat_sys(s) == gnss::GLO) continue;
    if (i_obs >= obs.D.rows()) continue;
    std::optional<std::vector<double>> lams;
    for (int f = 0; f < std::min<int>(tc.nav.nf, int(obs.D.cols())); ++f) {
      const double d_obs = obs.D(i_obs, f);
      if (d_obs == 0.0) continue;
      if (!lams) lams = ep.in.sig().wavelengths(s);
      if (f < int(lams->size()) && (*lams)[f] > 0) {
        const double snr = f < obs.S.cols() ? obs.S(i_obs, f) : 0.0;
        rows.push_back({s, ep.in.rs.row(i_obs).transpose(), ep.in.vs.row(i_obs).transpose(),
                        d_obs, (*lams)[f], ep.in.el[si], snr});
        break;
      }
    }
  }
  return rows;
}

/// Robust epoch screen: drop outliers and return the residual scale.
std::pair<std::vector<DopplerRow>, double> screen_rows(const ImuGnssTc& tc, EpochData& ep,
                                                      const std::vector<DopplerRow>& rows) {
  if (rows.size() < 5) return {rows, 0.0};
  const Vec3 v_pred = ep.R_enu2ecef * Vec3(ep.pred_nav.velocity());
  const Vec3 p_r = ep.pred_ecef;
  std::vector<std::array<double, 4>> A;
  std::vector<double> b;
  std::vector<DopplerRow> keep;
  for (const DopplerRow& row : rows) {
    const Vec3 d_vec = row.p_sat - p_r;
    const double rho = d_vec.norm();
    if (rho < 1.0) continue;
    const Vec3 e = d_vec / rho;
    const double pred = e.dot(row.v_sat - v_pred);
    A.push_back({-e[0], -e[1], -e[2], 1.0});
    b.push_back(-row.lam * row.d_obs - pred);
    keep.push_back(row);
  }
  if (b.size() < 5) return {rows, 0.0};
  Eigen::MatrixXd Am(A.size(), 4);
  Eigen::VectorXd bm(b.size());
  for (std::size_t i = 0; i < A.size(); ++i) {
    Am.row(i) << A[i][0], A[i][1], A[i][2], A[i][3];
    bm[i] = b[i];
  }
  const Eigen::VectorXd x =
      Am.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bm);
  const Eigen::VectorXd r = Am * x - bm;
  std::vector<double> rv(r.data(), r.data() + r.size());
  const double med = median(rv);
  std::vector<double> dev;
  for (double v : rv) dev.push_back(std::abs(v - med));
  const double scale = 1.4826 * median(dev);
  const double k = tc.cfg.doppler_fde_k;
  if (k > 0 && scale > 0) {
    std::vector<DopplerRow> kept;
    for (std::size_t i = 0; i < keep.size(); ++i)
      if (std::abs(r[i]) <= k * scale) kept.push_back(keep[i]);
    if (kept.size() >= 4) {
      const int n_drop = int(keep.size() - kept.size());
      if (n_drop) ep.info.set("doppler_fde", n_drop);
      keep = kept;
    }
  }
  ep.info.set("doppler_scale", scale);
  return {keep, scale};
}

}  // namespace

void add_sd_doppler_factors(ImuGnssTc& tc, EpochData& ep) {
  const double sigma = tc.cfg.doppler_sd_sigma;
  if (sigma <= 0 || ep.key_idx < 0) return;
  const double gdop_max = tc.cfg.doppler_gdop_max;
  if (gdop_max > 0 && ep.info.num("gdop", 0.0) > gdop_max) {
    ep.info.set("doppler_sd_skipped", "gdop");
    return;
  }
  if (tc.cfg.doppler_require_dd && ep.nv < tc.cfg.min_dd_for_solve) {
    ep.info.set("doppler_sd_skipped", ep.nv);
    return;
  }
  auto [rows, scale] = screen_rows(tc, ep, doppler_rows(tc, ep));
  if (rows.size() < 2) return;

  const double snr_ref_db = tc.cfg.snr_ref_dbhz;
  auto sigma_of = [&](double el, double snr) {
    double s = std::max(sigma, tc.cfg.doppler_adaptive_sigma ? scale : 0.0);
    s /= std::max(std::sin(el), 0.1);
    if (tc.cfg.doppler_snr_weight && snr > 0)
      s *= std::clamp(std::pow(10.0, (snr_ref_db - snr) / 20.0), 1.0, 10.0);
    return s;
  };
  std::vector<double> sigmas;
  for (const auto& r : rows) sigmas.push_back(sigma_of(r.el, r.snr));
  const int i_ref = int(std::min_element(sigmas.begin(), sigmas.end()) - sigmas.begin());
  const DopplerRow& ref = rows[i_ref];
  const double sigma_ref = sigmas[i_ref];

  Vec3 omega = Vec3::Zero();
  if (ep.gyro_mean) {
    const Vec3 bias_gyro = tc.tc_bias ? Vec3(tc.tc_bias->gyroscope()) : Vec3::Zero();
    omega = *ep.gyro_mean - bias_gyro;
  }
  const int key_idx = ep.key_idx;
  const double huber = tc.cfg.doppler_huber;
  int n = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (int(i) == i_ref) continue;
    const DopplerRow& row = rows[i];
    // Differences share the reference: diagonal approximation, as the DD
    // factors do.
    gtsam::SharedNoiseModel noise =
        gtsam::noiseModel::Isotropic::Sigma(1, std::hypot(sigmas[i], sigma_ref));
    if (huber > 0)
      noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(huber), noise);
    ep.graph.emplace_shared<gtsam::SingleDifferenceDopplerFactorArm>(
        ImuGnssTc::Xpose(key_idx), ImuGnssTc::Vel(key_idx), row.d_obs, ref.d_obs, row.lam,
        ref.lam, gtsam::Point3(row.p_sat), gtsam::Point3(row.v_sat), gtsam::Point3(ref.p_sat),
        gtsam::Point3(ref.v_sat), gtsam::Point3(ep.pred_ecef), gtsam::Point3(tc.lever_arm_tc),
        tc.ecef_T_nav, gtsam::Point3(omega), 0.0, 0.0, noise);
    ++n;
  }
  if (n) {
    ep.info.set("doppler_sd_n", n);
    ep.info.set("doppler_sd_ref", ref.sat);
  }
}

// ── NHC ─────────────────────────────────────────────────────────────

bool add_nhc_factor(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, int key_idx,
                    double speed, const std::optional<Vec3>& gyro_mean) {
  if (!tc.cfg.nhc_enable || speed < tc.cfg.nhc_min_speed) return false;
  if (!tc.nhc_lever_cache || tc.nhc_lever_src != tc.cfg.nhc_lever) {
    tc.nhc_lever_cache = parse_lever(tc.cfg.nhc_lever);
    tc.nhc_lever_src = tc.cfg.nhc_lever;
  }
  Vec3 omega = Vec3::Zero();
  if (gyro_mean) {
    const Vec3 bias_gyro = tc.tc_bias ? Vec3(tc.tc_bias->gyroscope()) : Vec3::Zero();
    omega = *gyro_mean - bias_gyro;
  }
  graph.emplace_shared<gtsam::NhcFactor>(
      ImuGnssTc::Xpose(key_idx), ImuGnssTc::Vel(key_idx), omega, *tc.nhc_lever_cache, speed,
      diag({1e3, tc.cfg.nhc_sigma_lat, tc.cfg.nhc_sigma_vert}), Vec3::Zero());
  return true;
}

// ── ZUPT ────────────────────────────────────────────────────────────

namespace {

void clear_zupt_anchor(RecoveryState& rec) {
  rec.zupt_anchor_pose.reset();
  rec.zupt_anchor_start_ep.reset();
}

std::optional<ZuptStats> zupt_should_fire(ImuGnssTc& tc, int n_imu, Info& info,
                                          int imu_idx_prev,
                                          const std::optional<Vec3>& vel_prev,
                                          bool gnss_available) {
  const TcConfig& cfg = tc.cfg;
  if (!cfg.zupt_enable) return std::nullopt;
  if (n_imu < cfg.zupt_min_samples) return std::nullopt;
  if (cfg.zupt_max_speed > 0 && vel_prev && gnss_available) {
    const double sp = vel_prev->norm();
    info.set("zupt_speed_prev", sp);
    if (sp > cfg.zupt_max_speed) return std::nullopt;
  }
  std::optional<Vec3> ref_acc, ref_gyro;
  if (tc.tc_bias_init) {
    ref_acc = Vec3(tc.tc_bias_init->accelerometer());
    ref_gyro = Vec3(tc.tc_bias_init->gyroscope());
  }
  const int lo = std::clamp(imu_idx_prev, 0, int(tc.imu_data.size()));
  const int hi = std::clamp(tc.imu_idx, lo, int(tc.imu_data.size()));
  const auto stats = compute_zupt_stats(tc.imu_data.data() + lo, tc.imu_data.data() + hi,
                                        ref_acc, ref_gyro);
  if (!stats) return std::nullopt;
  info.set("zupt_acc_std", stats->acc_std);
  info.set("zupt_gyro_std", stats->gyro_std);
  info.set("zupt_gyro_median", stats->gyro_median);
  const double g_dev = std::abs(stats->acc_norm_mean - 9.81);
  info.set("zupt_g_dev", g_dev);
  if (stats->acc_std > cfg.zupt_max_acc_std) return std::nullopt;
  if (stats->gyro_std > cfg.zupt_max_gyro_std) return std::nullopt;
  if (stats->gyro_median > cfg.zupt_max_gyro_median) return std::nullopt;
  if (cfg.zupt_g_dev_thr > 0 && g_dev > cfg.zupt_g_dev_thr) return std::nullopt;
  return stats;
}

}  // namespace

bool add_zupt_factors(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, int key_idx,
                      int imu_idx_prev, int n_imu, Info& info,
                      const std::optional<gtsam::Pose3>& pose_prev, bool gnss_available,
                      const std::optional<Vec3>& vel_prev) {
  RecoveryState& rec = tc.recovery;
  if (!zupt_should_fire(tc, n_imu, info, imu_idx_prev, vel_prev, gnss_available)) {
    clear_zupt_anchor(rec);
    return false;
  }
  const TcConfig& cfg = tc.cfg;
  bool any = false;
  if (cfg.zupt_sigma_zero_velocity > 0) {
    graph.addPrior<gtsam::Vector3>(
        ImuGnssTc::Vel(key_idx), gtsam::Vector3::Zero(),
        gtsam::noiseModel::Isotropic::Sigma(3, cfg.zupt_sigma_zero_velocity));
    info.set("zupt", true);
    any = true;
  }
  if (cfg.zupt_sigma_zero_rotation > 0 && key_idx > 0) {
    const double sr = cfg.zupt_sigma_zero_rotation;
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        ImuGnssTc::Xpose(key_idx - 1), ImuGnssTc::Xpose(key_idx), gtsam::Pose3(),
        diag({sr, sr, sr, 1e3, 1e3, 1e3}));
    info.set("zaru", true);
    any = true;
  }
  if (!gnss_available) {
    const double sig_t = cfg.zupt_anchor_sigma_translation;
    const double sig_r = cfg.zupt_anchor_sigma_rotation;
    if (sig_t > 0 && sig_r > 0) {
      if (!rec.zupt_anchor_pose) {
        if (pose_prev) {
          rec.zupt_anchor_pose = *pose_prev;
          rec.zupt_anchor_start_ep = key_idx;
          info.set("zupt_anchor_capture", key_idx);
          any = true;
        }
      } else {
        graph.addPrior(ImuGnssTc::Xpose(key_idx), *rec.zupt_anchor_pose,
                       diag({sig_r, sig_r, sig_r, sig_t, sig_t, sig_t}));
        info.set("zupt_anchor", rec.zupt_anchor_start_ep.value_or(0));
        any = true;
      }
    }
  }
  return any;
}

void add_zupt_factors_for_stage(ImuGnssTc& tc, EpochData& ep) {
  // gnss_available=false keeps the stationary anchor unconditional at stops.
  add_zupt_factors(tc, ep.graph, ep.key_idx, ep.imu_idx_prev, ep.n_imu, ep.info, ep.pose_p,
                   false, ep.vel_prev);
}

// ── IMU chain ───────────────────────────────────────────────────────

PimResult build_pim_tc(ImuGnssTc& tc, const gtsam::imuBias::ConstantBias& bias,
                       std::optional<double> target_tow) {
  PimResult r = build_pim(tc.imu_params, bias, tc.imu_data, tc.imu_idx, target_tow);
  tc.imu_idx = r.new_idx;
  return r;
}

void add_imu_chain(ImuGnssTc& tc, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                   int key_idx, const gtsam::PreintegratedCombinedMeasurements& pim,
                   const gtsam::Pose3& pose_p, const Vec3& vel_prev, Info& info) {
  if (tc.recovery.pim_discontinuity) {
    info.set("pim_discontinuity", true);
    const gtsam::imuBias::ConstantBias bias_for_pred = tc.bias_or_init();
    gtsam::Pose3 seed_pose;
    Vec3 seed_vel;
    try {
      const gtsam::NavState pred = pim.predict(gtsam::NavState(pose_p, vel_prev), bias_for_pred);
      seed_pose = pred.pose();
      seed_vel = pred.velocity();
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
      const Vec3 t = pose_p.translation() + vel_prev * tc.epoch_dt;
      seed_pose = gtsam::Pose3(pose_p.rotation(), t);
      seed_vel = vel_prev;
    }
    values.update(ImuGnssTc::Xpose(key_idx), seed_pose);
    values.update(ImuGnssTc::Vel(key_idx), gtsam::Vector3(seed_vel));
    const double ts = tc.cfg.pim_break_trans_sigma;
    graph.addPrior(ImuGnssTc::Xpose(key_idx), seed_pose, diag({0.1, 0.1, 0.3, ts, ts, ts}));
    graph.addPrior<gtsam::Vector3>(ImuGnssTc::Vel(key_idx), seed_vel,
                                   gtsam::noiseModel::Isotropic::Sigma(3, 2.0));
    graph.addPrior(ImuGnssTc::Bias(key_idx), tc.bias_or_init(),
                   gtsam::noiseModel::Isotropic::Sigma(6, 0.01));
    tc.recovery.pim_discontinuity = false;
    return;
  }
  graph.emplace_shared<gtsam::CombinedImuFactor>(
      ImuGnssTc::Xpose(key_idx - 1), ImuGnssTc::Vel(key_idx - 1), ImuGnssTc::Xpose(key_idx),
      ImuGnssTc::Vel(key_idx), ImuGnssTc::Bias(key_idx - 1), ImuGnssTc::Bias(key_idx), pim);
  // Extra bias regularisation against NLOS leakage into the bias states.
  const double ag = tc.cfg.bias_between_acc_sigma, gy = tc.cfg.bias_between_gyro_sigma;
  graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
      ImuGnssTc::Bias(key_idx - 1), ImuGnssTc::Bias(key_idx), gtsam::imuBias::ConstantBias(),
      diag({ag, ag, ag, gy, gy, gy}));
  // Absolute bias anchor (bias_prior_mode: 0 off, 1 init, 2 current).
  std::optional<gtsam::imuBias::ConstantBias> anchor;
  if (tc.cfg.bias_prior_mode == 1) anchor = tc.tc_bias_init;
  else if (tc.cfg.bias_prior_mode != 0) anchor = tc.tc_bias ? tc.tc_bias : tc.tc_bias_init;
  if (anchor) {
    const double pa = tc.cfg.bias_prior_acc_sigma, pg = tc.cfg.bias_prior_gyro_sigma;
    graph.addPrior(ImuGnssTc::Bias(key_idx), *anchor, diag({pa, pa, pa, pg, pg, pg}));
  }
}

}  // namespace gnss_fgo
