// ar/: integer ambiguity resolution over the smoother's own float
// estimate (problem -> resolver -> nav bridge), the subset retry, the
// context gate and fix-and-hold.
#include "gnss_fgo/modules.h"

#include "gnss_fgo/gnss.h"
#include "gnss_fgo/mlambda.h"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>

namespace gnss_fgo {

namespace {

std::string ar_exc_label(const std::exception& e) {
  if (dynamic_cast<const lambda::LambdaError*>(&e))
    return std::string("LambdaError: ") + e.what();
  return exc_label(e);
}

// ── problem ─────────────────────────────────────────────────────────

/// One epoch's float-ambiguity problem, self-contained.
struct ArProblem {
  std::vector<SfKey> keys;
  std::map<SfKey, double> values;
  Eigen::MatrixXd cov;     // joint covariance over keys
  Eigen::MatrixXd cross;   // cov(position ENU, ambiguity), 3 x n
  std::map<int, double> elevations;
};

/// Read the smoother into an ArProblem; nullopt = no fix this epoch.
std::optional<ArProblem> build_problem(ImuGnssTc& tc, const std::vector<int>& sat_list,
                                       const AmbDict& amb_dict) {
  Smoother* smoother = tc.phase == 2 ? tc.isam2.get() : tc.isam.get();
  if (!smoother) return std::nullopt;
  const gtsam::ISAM2& isam2 = smoother->getISAM2();
  const gtsam::Values est = smoother->calculateEstimate();

  ArProblem p;
  std::map<SfKey, double> held_var;
  std::set<int> present(sat_list.begin(), sat_list.end());
  for (const auto& [sf, value] : tc.sat_states.held_items()) {
    if (!present.count(sf.first) || tc.nav.vsat(sf.first - 1, sf.second) != 1) continue;
    p.keys.push_back(sf);
    p.values[sf] = value;
    held_var[sf] = std::max(tc.cfg.varholdamb, 1e-6);
  }
  for (const auto& [sf, k] : amb_dict) {
    if (p.values.count(sf)) continue;
    if (present.count(sf.first) && est.exists(k) && tc.nav.vsat(sf.first - 1, sf.second) == 1) {
      p.keys.push_back(sf);
      p.values[sf] = est.at<double>(k);
    }
  }
  if (p.keys.size() < 2) return std::nullopt;

  std::vector<SfKey> in_graph;
  for (const SfKey& sf : p.keys) {
    auto it = amb_dict.find(sf);
    if (it != amb_dict.end() && est.exists(it->second)) in_graph.push_back(sf);
  }
  if (!tc.ar_key_pose) return std::nullopt;
  const gtsam::Key key_pose = *tc.ar_key_pose;

  // Retry/subset attempts resolve over key subsets of the first attempt:
  // slice its cached extraction instead of re-reading the smoother.
  const auto& cache = tc.current_epoch.ar_joint_cache;
  bool cached = cache && cache->key_pose == key_pose;
  if (cached)
    for (const SfKey& sf : in_graph) cached = cached && cache->pos.count(sf);
  const int m = int(in_graph.size());
  Eigen::MatrixXd cov_g(m, m), cross_g(3, m);
  if (cached) {
    for (int i = 0; i < m; ++i) {
      const int gi = cache->pos.at(in_graph[i]);
      for (int j = 0; j < m; ++j) cov_g(i, j) = cache->cov(gi, cache->pos.at(in_graph[j]));
      cross_g.col(i) = cache->cross.col(gi);
    }
  } else {
    gtsam::KeyVector kv{key_pose};
    for (const SfKey& sf : in_graph) kv.push_back(amb_dict.at(sf));
    try {
      const gtsam::JointMarginal jm = isam2.jointMarginalCovariance(kv);
      for (int i = 0; i < m; ++i) {
        const gtsam::Key ka = amb_dict.at(in_graph[i]);
        for (int j = 0; j < m; ++j) cov_g(i, j) = jm.at(ka, amb_dict.at(in_graph[j]))(0, 0);
        cross_g.col(i) = jm.at(key_pose, ka).block<3, 1>(3, 0);
      }
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index);
      return std::nullopt;
    }
    // Pose3 marginals live in the body tangent frame: rotate the
    // position rows into the nav frame (identity in Phase 1).
    cross_g = est.at<gtsam::Pose3>(key_pose).rotation().matrix() * cross_g;
    if (!cache || m > int(cache->pos.size())) {
      ArJointCache c;
      c.key_pose = key_pose;
      for (int i = 0; i < m; ++i) c.pos[in_graph[i]] = i;
      c.cov = cov_g;
      c.cross = cross_g;
      tc.current_epoch.ar_joint_cache = c;
    }
  }

  const int n = int(p.keys.size());
  p.cov = Eigen::MatrixXd::Zero(n, n);
  p.cross = Eigen::MatrixXd::Zero(3, n);
  std::map<SfKey, int> gpos;
  for (int i = 0; i < m; ++i) gpos[in_graph[i]] = i;
  for (int i = 0; i < n; ++i) {
    auto gi = gpos.find(p.keys[i]);
    if (gi != gpos.end()) {
      for (int j = 0; j < n; ++j) {
        auto gj = gpos.find(p.keys[j]);
        if (gj != gpos.end()) p.cov(i, j) = cov_g(gi->second, gj->second);
      }
      p.cross.col(i) = cross_g.col(gi->second);
    }
    auto hv = held_var.find(p.keys[i]);
    if (hv != held_var.end()) p.cov(i, i) = hv->second;
  }
  if (!p.cov.allFinite() || !p.cross.allFinite()) return std::nullopt;
  for (const SfKey& sf : p.keys) p.elevations[sf.first] = tc.nav.el(sf.first);
  return p;
}

// ── resolver ────────────────────────────────────────────────────────

struct ResolverResult {
  int nb = 0;
  std::map<SfKey, double> fixed;
  double s0 = 0.0, s1 = 0.0;
  std::vector<std::pair<SfKey, SfKey>> pairs;
  double thres_used = 0.0;
  int dropped = 0;
  std::set<SfKey> holdable;
};

/// LAMBDA over a float ambiguity vector and its covariance.
class AmbiguityResolver {
 public:
  // Partial AR: fewest z-components a partial fix may keep, the
  // bootstrapped success rate that picks the first subset, and the
  // reduced searches allowed per epoch.
  static constexpr int min_fix = 20;
  static constexpr double par_p0 = 0.995;
  static constexpr int par_trials = 8;

  AmbiguityResolver(double thresar, int min_pairs, double el_mask, double thresar_min,
                    double thresar_max)
      : thresar_(thresar), min_pairs_(min_pairs), el_mask_(el_mask),
        thresar_min_(thresar_min), thresar_max_(thresar_max) {}

  int min_pairs() const { return min_pairs_; }

  /// demo5/FFRT dimension-adaptive ratio threshold, clamped.
  double ratio_threshold(int nb) const {
    if (thresar_min_ == thresar_max_) return thresar_;
    static const double coeffs[3][5] = {
        {-1.94058448e-01, -7.79023476e+00, 1.24231120e+02, -4.03126050e+02, 3.50413202e+02},
        {6.42237302e-01, -8.39813962e+00, 2.92107285e+01, -2.37577308e+01, -1.14307128e+00},
        {-2.22600390e-02, 3.23169103e-01, -1.39837429e+00, 2.19282996e+00, -5.34583971e-02}};
    const double p0 = thresar_;
    const int nb1 = std::min(nb, 50);
    double c[3];
    for (int r = 0; r < 3; ++r) {
      c[r] = coeffs[r][0];
      for (int kj = 1; kj < 5; ++kj) c[r] = c[r] * p0 + coeffs[r][kj];
    }
    double th = c[0];
    for (int ki = 1; ki < 3; ++ki) th = th / (nb1 + 1) + c[ki];
    return std::min(std::max(th, thresar_min_), thresar_max_);
  }

  /// Reference per (constellation, frequency) = lowest-numbered member
  /// above the mask; every other member pairs against it.
  std::vector<std::pair<SfKey, SfKey>> double_difference(
      const std::vector<SfKey>& keys, const std::map<int, double>& el) const {
    std::map<std::pair<int, int>, std::vector<SfKey>> groups;
    for (const SfKey& k : keys) {
      auto it = el.find(k.first);
      const double e = it != el.end() ? it->second : -std::numeric_limits<double>::infinity();
      if (e < el_mask_) continue;
      groups[{gnss::sat_sys(k.first), k.second}].push_back(k);
    }
    std::vector<std::pair<SfKey, SfKey>> pairs;
    for (auto& [g, members] : groups) {
      std::sort(members.begin(), members.end());
      if (members.size() < 2) continue;
      for (std::size_t i = 1; i < members.size(); ++i) pairs.emplace_back(members[0], members[i]);
    }
    return pairs;
  }

  ResolverResult resolve(const ArProblem& p, bool allow_partial) const {
    ResolverResult result;
    result.pairs = double_difference(p.keys, p.elevations);
    if (int(result.pairs.size()) < min_pairs_) return result;
    const Eigen::MatrixXd D = dd_matrix(result.pairs, p.keys);
    Eigen::VectorXd x(p.keys.size());
    for (std::size_t i = 0; i < p.keys.size(); ++i) x[i] = p.values.at(p.keys[i]);
    const Eigen::VectorXd y = D * x;
    const Eigen::MatrixXd Q = D * p.cov * D.transpose();

    const lambda::Result lr = lambda::mlambda(y, Q);
    double s0 = lr.s.size() > 0 ? lr.s[0] : 0.0;
    const double s1 = lr.s.size() > 1 ? lr.s[1] : 0.0;
    // Exact fit (held integers re-entering): s0 is FP noise.
    if (s0 <= 1e-12 * std::max(s1, 1.0)) s0 = 0.0;
    const double ratio = s0 <= 0.0 ? 0.0 : s1 / s0;
    const double thres = ratio_threshold(int(result.pairs.size()));
    result.s0 = s0;
    result.s1 = s1;
    result.thres_used = thres;
    if (lr.nfix <= 0) return result;
    if (!(s0 <= 0.0 || ratio >= thres)) {
      if (!allow_partial) return result;
      Partial par;
      if (!partial(y, Q, &par)) return result;
      result.s0 = par.s0;
      result.s1 = par.s1;
      result.thres_used = par.thres;
      result.dropped = par.k;
      result.nb = par.nb;
      result.fixed = restore_sd(result.pairs, p.values, par.bvec);
      // A DD whose iZt row touches only fixed z-components is an exact
      // integer: those pairs may be held like a full fix.
      for (std::size_t row = 0; row < result.pairs.size(); ++row)
        if (par.int_rows[row]) {
          result.holdable.insert(result.pairs[row].first);
          result.holdable.insert(result.pairs[row].second);
        }
      return result;
    }
    result.nb = int(result.pairs.size());
    result.fixed = restore_sd(result.pairs, p.values, lr.afix.col(0));
    return result;
  }

  static Eigen::MatrixXd dd_matrix(const std::vector<std::pair<SfKey, SfKey>>& pairs,
                                   const std::vector<SfKey>& keys) {
    std::map<SfKey, int> index;
    for (std::size_t i = 0; i < keys.size(); ++i) index[keys[i]] = int(i);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(pairs.size(), keys.size());
    for (std::size_t r = 0; r < pairs.size(); ++r) {
      D(r, index.at(pairs[r].first)) = 1.0;
      D(r, index.at(pairs[r].second)) = -1.0;
    }
    return D;
  }

 private:
  struct Partial {
    Eigen::VectorXd bvec;
    int nb = 0, k = 0;
    double s0 = 0, s1 = 0, thres = 0;
    std::vector<bool> int_rows;
  };

  /// Reference keeps its float value; each target follows from its DD.
  static std::map<SfKey, double> restore_sd(const std::vector<std::pair<SfKey, SfKey>>& pairs,
                                            const std::map<SfKey, double>& float_values,
                                            const Eigen::VectorXd& bvec) {
    std::map<SfKey, double> fixed;
    for (std::size_t r = 0; r < pairs.size(); ++r) {
      const auto& [ref, tgt] = pairs[r];
      fixed.emplace(ref, float_values.at(ref));
      fixed[tgt] = fixed[ref] - bvec[r];
    }
    return fixed;
  }

  /// Partial AR: fix the well-determined z-components, condition the
  /// rest on them, accept by the reduced problem's own FFRT ratio.
  bool partial(const Eigen::VectorXd& y, const Eigen::MatrixXd& Q, Partial* out) const {
    const int n = int(y.size());
    if (n <= min_fix) return false;
    Eigen::MatrixXd L;
    Eigen::VectorXd d;
    lambda::ldldecom(Q, &L, &d);
    const Eigen::MatrixXd Z = lambda::reduction(&L, &d);
    const Eigen::MatrixXd iZt = Z.transpose().inverse().unaryExpr(
        [](double v) { return std::nearbyint(v); });
    const Eigen::VectorXd z = Z.transpose() * y;
    const Eigen::MatrixXd Qz = L.transpose() * d.asDiagonal() * L;
    const int kmax = n - min_fix;
    // Drop from the front (worst-first) until the bootstrapped success
    // rate says the rest can be fixed.
    int k = 1;
    while (k < kmax && lambda::sr_boost(d.tail(n - k)) < par_p0) ++k;
    for (int t = 0; t < par_trials; ++t) {
      if (k > kmax) return false;
      Eigen::MatrixXd zpar;
      Eigen::VectorXd sq;
      lambda::estimILS(L.bottomRightCorner(n - k, n - k), d.tail(n - k), z.tail(n - k), 2,
                       &zpar, &sq);
      const double s0 = sq[0], s1 = sq[1];
      const int nb = n - k;
      const double thres = ratio_threshold(nb);
      if (s0 > 0.0 && s1 / s0 >= thres) {
        // QP = Qz[:k, k:] inv(Qz[k:, k:])
        const Eigen::MatrixXd Qkk = Qz.bottomRightCorner(n - k, n - k);
        const Eigen::MatrixXd QP =
            Qkk.transpose().partialPivLu().solve(Qz.topRightCorner(k, n - k).transpose()).transpose();
        const Eigen::VectorXd zc = z.head(k) - QP * (z.tail(n - k) - zpar.col(0));
        Eigen::VectorXd zall(n);
        zall << zc, zpar.col(0);
        out->bvec = iZt * zall;
        out->nb = nb;
        out->s0 = s0;
        out->s1 = s1;
        out->thres = thres;
        out->k = k;
        out->int_rows.assign(n, true);
        for (int r = 0; r < n; ++r)
          for (int c = 0; c < k; ++c)
            if (iZt(r, c) != 0.0) out->int_rows[r] = false;
        return true;
      }
      k += std::max(1, (kmax - k) / 4);
    }
    return false;
  }

  double thresar_;
  int min_pairs_;
  double el_mask_, thresar_min_, thresar_max_;
};

/// The fixed state: ambiguity slots take the fixed values; the position
/// takes the conditional update x - Qab Qb^-1 (y_float - y_fixed).
StateVec fixed_state(const ImuGnssTc& tc, const ArProblem& p, const ResolverResult& r) {
  StateVec xa = tc.nav.x;
  for (const auto& [sf, v] : r.fixed) xa.at(sf.first, sf.second) = v;
  const Eigen::MatrixXd D = AmbiguityResolver::dd_matrix(r.pairs, p.keys);
  Eigen::VectorXd x_float(p.keys.size()), x_fixed(p.keys.size());
  for (std::size_t i = 0; i < p.keys.size(); ++i) {
    x_float[i] = p.values.at(p.keys[i]);
    auto it = r.fixed.find(p.keys[i]);
    x_fixed[i] = it != r.fixed.end() ? it->second : x_float[i];
  }
  const Eigen::MatrixXd Qb = D * p.cov * D.transpose();
  const Eigen::MatrixXd Qab = p.cross * D.transpose();
  const Eigen::MatrixXd gain =
      Qb.transpose().partialPivLu().solve(Qab.transpose()).transpose();
  if (!gain.allFinite()) return xa;   // singular Qb
  const Vec3 d_enu = gain * (D * (x_float - x_fixed));
  xa.pos = tc.nav.x.pos - tc.R_enu2ecef * d_enu;
  return xa;
}

// ── nav bridge ──────────────────────────────────────────────────────

/// Side effects of one resolution attempt, accepted or not.
void publish_attempt(ImuGnssTc& tc, const std::vector<int>& sat_list,
                     const ResolverResult& r) {
  tc.nav.ddidx(sat_list);
  tc.last_s0 = r.s0;
  tc.last_s1 = r.s1;
}

// ── run_ar ──────────────────────────────────────────────────────────

using Attempt = std::pair<int, std::optional<StateVec>>;

Attempt resolve(ImuGnssTc& tc, const std::vector<int>& sat_list, const AmbDict& amb_dict,
                bool allow_partial = false) {
  const auto problem = build_problem(tc, sat_list, amb_dict);
  if (!problem) {
    tc.ar_diag.outcome = "problem_unposed";
    tc.last_s0 = tc.last_s1 = 0.0;
    return {0, tc.nav.x};
  }
  const AmbiguityResolver resolver(tc.nav.thresar, 2, tc.nav.elmaskar, tc.cfg.ar_thresar_min,
                                   tc.cfg.ar_thresar_max);
  const ResolverResult res = resolver.resolve(*problem, allow_partial);
  tc.last_ar_thres = res.thres_used != 0.0 ? res.thres_used : tc.nav.thresar;
  publish_attempt(tc, sat_list, res);
  if (res.nb <= 0) {
    // A lone-pair decline is a candidate-stage verdict, not starvation.
    if (!res.pairs.empty() && int(res.pairs.size()) < resolver.min_pairs())
      tc.ar_diag.outcome = "min_pairs_declined";
    return {0, tc.nav.x};
  }
  if (res.s0 <= 0.0) tc.ar_diag.exact_fit_accept = 1;
  tc.ar_diag.partial_dropped = res.dropped;
  tc.ar_partial_holdable =
      res.dropped ? std::optional<std::set<SfKey>>(res.holdable) : std::nullopt;
  return {res.nb, fixed_state(tc, *problem, res)};
}

Attempt single_attempt(ImuGnssTc& tc, const std::vector<int>& sat, const AmbDict& amb_dict,
                       const std::vector<int>& exclude, bool restore_state) {
  std::vector<int> sat_list = sat;
  const Eigen::MatrixXi vsat_snapshot = tc.nav.vsat;
  const Eigen::MatrixXi fix_snapshot = tc.nav.fix;
  auto restore = [&] {
    if (restore_state) {
      tc.nav.vsat = vsat_snapshot;
      tc.nav.fix = fix_snapshot;
    }
  };
  try {
    if (!exclude.empty()) {
      for (int s : exclude) tc.nav.vsat.row(s - 1).setZero();
      sat_list.erase(std::remove_if(sat_list.begin(), sat_list.end(),
                                    [&](int s) {
                                      return std::find(exclude.begin(), exclude.end(), s) !=
                                             exclude.end();
                                    }),
                     sat_list.end());
    }
    Attempt r = resolve(tc, sat_list, amb_dict);
    restore();
    return r;
  } catch (...) {
    restore();
    throw;
  }
}

double ratio_from_last_lambda(const ImuGnssTc& tc) {
  return tc.last_s0 <= 0.0 ? 0.0 : tc.last_s1 / tc.last_s0;
}

/// Rank candidate satellites to drop: post-fit residual, then low elevation.
std::vector<int> rank_subset_drop_sats(const ImuGnssTc& tc, const std::vector<int>& sat,
                                       const std::vector<double>& el, const AmbDict& amb_dict) {
  std::map<int, double> sat_el;
  for (std::size_t i = 0; i < sat.size(); ++i)
    sat_el[sat[i]] = std::max(sat_el.count(sat[i]) ? sat_el[sat[i]] : 0.0, el[i]);
  std::set<int> seen;
  std::vector<std::tuple<double, double, int>> rows;
  for (const auto& [sf, _] : amb_dict) {
    if (!seen.insert(sf.first).second) continue;
    rows.emplace_back(tc.mres_signals.per_sat.get(sf.first, 0.0),
                      -(sat_el.count(sf.first) ? sat_el[sf.first] : 0.0), sf.first);
  }
  std::sort(rows.begin(), rows.end(), std::greater<>());
  const int max_c = std::max(0, tc.cfg.subset_ar_max_candidates);
  std::vector<int> out;
  for (int i = 0; i < int(rows.size()) && i < max_c; ++i) out.push_back(std::get<2>(rows[i]));
  return out;
}

void combinations(const std::vector<int>& pool, int k, std::size_t start, std::vector<int>& cur,
                  const std::function<void(const std::vector<int>&)>& fn) {
  if (int(cur.size()) == k) {
    fn(cur);
    return;
  }
  for (std::size_t i = start; i < pool.size(); ++i) {
    cur.push_back(pool[i]);
    combinations(pool, k, i + 1, cur, fn);
    cur.pop_back();
  }
}

/// Retry with the likeliest bad satellites removed, keeping the best ratio.
Attempt try_subset_ar(ImuGnssTc& tc, const std::vector<int>& sat, const std::vector<double>& el,
                      const AmbDict& amb_dict) {
  const int dirty_max = tc.cfg.subset_ar_max_dirty_sats;
  if (dirty_max > 0) {
    int dirty_n = 0;
    for (const auto& [s, v] : tc.mres_signals.per_sat)
      dirty_n += v > tc.cfg.subset_ar_dirty_sat_res_m ? 1 : 0;
    if (dirty_n > dirty_max) {
      ArSubsetDebug dbg;
      dbg.skip_reason = "dirty_gate";
      dbg.dirty_n = dirty_n;
      tc.ar_subset_debug = dbg;
      return {0, std::nullopt};
    }
  }
  const int min_nb = std::max(1, tc.cfg.subset_ar_min_nb);
  const int max_drop = std::max(1, tc.cfg.subset_ar_max_drop);
  const std::vector<int> candidates = rank_subset_drop_sats(tc, sat, el, amb_dict);
  ArSubsetDebug dbg;
  dbg.candidates = int(candidates.size());
  dbg.max_drop = max_drop;
  tc.ar_subset_debug = dbg;
  struct Best {
    std::vector<int> drop;
    std::tuple<double, int, int> score;
  };
  std::optional<Best> best;
  for (int k = 1; k <= max_drop; ++k) {
    if (k > int(candidates.size())) break;
    std::vector<int> cur;
    combinations(candidates, k, 0, cur, [&](const std::vector<int>& drop) {
      const Attempt a = single_attempt(tc, sat, amb_dict, drop, true);
      const double ratio = ratio_from_last_lambda(tc);
      if (a.first < min_nb || !a.second) return;
      // Quantise so ULP noise cannot outrank the smaller-k tie-break.
      const std::tuple<double, int, int> score{std::round(ratio * 1e6) / 1e6, a.first, -k};
      if (!best || score > best->score) best = Best{drop, score};
    });
  }
  if (!best) return {0, std::nullopt};
  Attempt a = single_attempt(tc, sat, amb_dict, best->drop, false);
  if (a.first < min_nb || !a.second) {
    ArSubsetDebug d;
    d.candidates = int(candidates.size());
    d.max_drop = max_drop;
    tc.ar_subset_debug = d;
    return {0, std::nullopt};
  }
  ArSubsetDebug d;
  d.candidates = int(candidates.size());
  d.max_drop = max_drop;
  d.used = true;
  d.drop_sats = best->drop;
  d.nb = a.first;
  d.ratio = ratio_from_last_lambda(tc);
  tc.ar_subset_debug = d;
  return a;
}

void record_amb_diagnostics(ImuGnssTc& tc, const std::vector<int>& sat,
                            const AmbDict& amb_dict) {
  std::set<int> in_obs(sat.begin(), sat.end());
  int amb_not = 0, held_not = 0;
  for (const auto& [sf, _] : amb_dict) amb_not += in_obs.count(sf.first) ? 0 : 1;
  for (const auto& [sf, _] : tc.sat_states.held_items()) held_not += in_obs.count(sf.first) ? 0 : 1;
  tc.ar_diag.amb_not_in_obs = amb_not;
  tc.ar_diag.held_not_in_obs = held_not;
  tc.ar_diag.sat_in_obs_size = int(in_obs.size());
  tc.ar_diag.outcome = "entered";
}

Attempt run_lambda_attempts(ImuGnssTc& tc, const std::vector<int>& sat,
                            const std::vector<double>& el, const AmbDict& amb_dict) {
  tc.ar_diag.resamb_raw_nb = -1;
  Attempt r;
  try {
    r = resolve(tc, sat, amb_dict);
  } catch (const std::exception& ex) {
    // Non positive-definite Qah is a live path in the degenerate
    // held-fit regime.
    tc.ar_diag.outcome = "lambda_exception";
    tc.ar_diag.exception = ar_exc_label(ex);
    return {0, std::nullopt};
  }
  tc.ar_diag.resamb_raw_nb = r.first;
  if (r.first <= 0 && tc.cfg.subset_ar_enable &&
      int(amb_dict.size()) >= tc.cfg.subset_ar_min_nb + 1) {
    try {
      r = try_subset_ar(tc, sat, el, amb_dict);
    } catch (const std::exception& ex) {
      tc.ar_diag.exception = ar_exc_label(ex);
      r = {0, std::nullopt};
    }
  }
  if (r.first <= 0) {
    // Last resort: fix the well-determined z-subspace (published, never
    // held -- see fix_ambiguities).
    try {
      r = resolve(tc, sat, amb_dict, true);
    } catch (const std::exception& ex) {
      tc.ar_diag.outcome = "lambda_exception";
      tc.ar_diag.exception = ar_exc_label(ex);
      r = {0, std::nullopt};
    }
  }
  if (r.first <= 0) {
    if (tc.ar_diag.outcome != "min_pairs_declined") tc.ar_diag.outcome = "lambda_zero";
    return {0, std::nullopt};
  }
  return r;
}

/// A small fix in a burst-like context is more likely wrong than lucky.
bool validate_fix(ImuGnssTc& tc, int nb) {
  if (nb > 0) {
    const double cached = tc.cached_ddpr_res_pre.value_or(0.0);
    const double main_res = cached != 0.0 ? cached : tc.mres_signals.last_res;
    double worst = 0.0;
    bool any = false;
    for (const auto& [s, v] : tc.mres_signals.per_sat) {
      worst = any ? std::max(worst, v) : v;
      any = true;
    }
    const bool burst_like = main_res > tc.cfg.ar_context_main_ddpr_max ||
                            worst > tc.cfg.ar_context_worst_sat_max;
    if (burst_like && nb <= tc.cfg.ar_context_nb_max) {
      tc.ar_context_reject = ArContextReject{nb, main_res, worst};
      tc.ar_diag.outcome = "ar_context_reject";
      return false;
    }
  }
  tc.ar_context_reject.reset();
  tc.ar_diag.outcome = "success";
  return true;
}

}  // namespace

std::pair<int, std::optional<StateVec>> run_ar(ImuGnssTc& tc, const std::vector<int>& sat,
                                               const std::vector<double>& el,
                                               const AmbDict& amb_dict) {
  tc.ar_subset_debug.reset();
  tc.ar_diag.outcome = "not_called";
  if (tc.nav.armode == 0) {
    tc.ar_diag.outcome = "armode_off";
    return {0, std::nullopt};
  }
  record_amb_diagnostics(tc, sat, amb_dict);
  Attempt r = run_lambda_attempts(tc, sat, el, amb_dict);
  if (r.first <= 0) return {0, std::nullopt};
  if (!validate_fix(tc, r.first)) return {0, std::nullopt};
  // fix-and-hold is the caller's move, after its remaining verdicts.
  tc.nav.smode = 4;
  return r;
}

void publish_marginals(ImuGnssTc& tc, const gtsam::Values& estimate, gtsam::Key key_pose,
                       const AmbDict& amb_dict) {
  // The resolver reads the marginals against this very pose.
  tc.ar_key_pose = key_pose;
  tc.nav.vsat.setZero();
  std::fill(tc.nav.x.amb.begin(), tc.nav.x.amb.end(), 0.0);
  const std::set<SfKey> cp_visible = tc.ar_cp_visible_sf;

  // Float ambiguities; vsat marks AR eligibility (not seeded this epoch).
  int missing = 0, vsat1 = 0, young = 0;
  std::vector<double> el_deg;
  for (const auto& [sf, k] : amb_dict) {
    const auto [s, f] = sf;
    if (!estimate.exists(k)) {
      ++missing;
      continue;
    }
    tc.nav.x.at(s, f) = estimate.at<double>(k);
    if (!tc.current_epoch.seeded_amb_keys.count(sf)) {
      tc.nav.vsat(s - 1, f) = 1;
      ++vsat1;
      el_deg.push_back(tc.nav.el(s) * 180.0 / gnss::PI);
    } else {
      tc.nav.vsat(s - 1, f) = 0;
      ++young;
    }
  }
  if (!el_deg.empty()) {
    std::vector<double> sorted = el_deg;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    const double med = n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    tc.ar_diag.amb_el_min_deg = int(std::nearbyint(sorted.front()));
    tc.ar_diag.amb_el_median_deg = int(std::nearbyint(med));
  } else {
    tc.ar_diag.amb_el_min_deg = -1;
    tc.ar_diag.amb_el_median_deg = -1;
  }
  tc.ar_diag.amb_el_above15 = int(std::count_if(el_deg.begin(), el_deg.end(),
                                                [](double e) { return e >= 15; }));
  tc.ar_diag.amb_el_above25 = int(std::count_if(el_deg.begin(), el_deg.end(),
                                                [](double e) { return e >= 25; }));
  tc.ar_diag.amb_estimate_missing = missing;
  tc.ar_diag.amb_vsat1 = vsat1;
  tc.ar_diag.amb_vsat0_young = young;

  // Held integers enter at varholdamb variance; vsat only while visible.
  std::set<SfKey> held_sf;
  for (const auto& [sf, held] : tc.sat_states.held_items()) {
    tc.nav.x.at(sf.first, sf.second) = held;
    tc.nav.vsat(sf.first - 1, sf.second) = cp_visible.count(sf) ? 1 : 0;
    held_sf.insert(sf);
  }
  int orphan = 0;
  for (const SfKey& sf : cp_visible)
    if (!held_sf.count(sf) && !amb_dict.count(sf)) ++orphan;
  tc.ar_diag.orphan_cp_count = orphan;
  tc.ar_diag.amb_dict_size = int(amb_dict.size());
  tc.ar_diag.held_size = int(held_sf.size());
  tc.ar_diag.cp_visible_size = int(cp_visible.size());
}

void apply_fix_and_hold(ImuGnssTc& tc, const AmbDict& amb_dict, const StateVec& xa,
                        const std::optional<std::set<SfKey>>& allow) {
  tc.nav.holdamb_flags();
  std::vector<SfKey> hold_keys;
  for (const auto& [sf, _] : amb_dict)
    if (tc.nav.fix(sf.first - 1, sf.second) == 3) hold_keys.push_back(sf);
  if (allow) {  // partial fix: hold only the integer subset
    std::vector<SfKey> kept;
    for (const SfKey& sf : hold_keys) {
      if (allow->count(sf)) kept.push_back(sf);
      else tc.nav.fix(sf.first - 1, sf.second) = 2;  // fix==3 means held; these are not
    }
    hold_keys = kept;
  }
  if (tc.phase != 2) {
    // Phase 1: per-held-N PriorDouble at sigma = sqrt(varholdamb).
    gtsam::NonlinearFactorGraph hg;
    const auto hold_noise = ImuGnssTc::noise1(std::sqrt(tc.cfg.varholdamb));
    for (const SfKey& sf : hold_keys)
      hg.addPrior(amb_dict.at(sf), xa.at(sf.first, sf.second), hold_noise);
    if (hg.size() > 0) {
      gtsam::FixedLagSmootherKeyTimestampMap ts;
      for (const SfKey& sf : hold_keys) ts[amb_dict.at(sf)] = tc.phase1_t;
      try {
        tc.isam->update(hg, gtsam::Values(), ts);
      } catch (const std::exception& e) {
        GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime, PyErr::Index);
        // nav.fix already claims held — record the graph/nav mismatch.
        tc.ar_diag.exception = "hold_prior: " + exc_label(e);
      }
    }
    return;
  }
  // Phase 2: the held integers leave the graph.
  for (const SfKey& sf : hold_keys) {
    const double held = xa.at(sf.first, sf.second);
    SatState& st = tc.sat_states.get(sf.first, sf.second);
    st.activate_hold(held);
    st.fix_streak += 1;
    st.amb_key.reset();
    tc.nav.x.at(sf.first, sf.second) = held;
  }
}

}  // namespace gnss_fgo
