#include "gnss_fgo/frontend.h"

#include <algorithm>

namespace gnss_fgo {

// ── NavState ────────────────────────────────────────────────────────

std::vector<std::array<SfKey, 2>> NavState::ddidx(const std::vector<int>& sats) {
  std::vector<bool> present(gnss::MAXSAT + 2, false);
  for (int s : sats)
    if (s > 0 && s <= gnss::MAXSAT) present[s] = true;
  fix.setZero();
  std::vector<std::array<SfKey, 2>> ix;
  auto usable = [&](int s, int f) {
    return present[s] && x.at(s, f) != 0.0 && vsat(s - 1, f) != 0;
  };
  for (int m = 0; m < gnss::GNSSMAX; ++m) {
    for (int f = 0; f < nf; ++f) {
      int ref = -1;
      for (int s = 1; s <= gnss::MAXSAT; ++s) {
        if (gnss::sat_sys(s) != m || !usable(s, f)) continue;
        if (rcv.el[s - 1] >= elmaskar) {
          fix(s - 1, f) = 2;
          ref = s;
          break;
        }
        fix(s - 1, f) = 1;
      }
      if (ref < 0) continue;
      for (int s = 1; s <= gnss::MAXSAT; ++s) {
        if (gnss::sat_sys(s) != m || s == ref || !usable(s, f)) continue;
        if (rcv.el[s - 1] >= elmaskar) {
          ix.push_back({SfKey{ref, f}, SfKey{s, f}});
          fix(s - 1, f) = 2;
        }
      }
    }
  }
  return ix;
}

int NavState::holdamb_flags() {
  int n = 0;
  for (int i = 0; i < fix.rows(); ++i)
    for (int f = 0; f < nf; ++f)
      if (fix(i, f) == 2) {
        fix(i, f) = 3;
        ++n;
      }
  return n;
}

// ── DdFrontend ──────────────────────────────────────────────────────

std::vector<int> DdFrontend::edit(ObsEpoch& obs, const NavData::SatStates& st,
                                  const Vec3& rr, const NavState& nav,
                                  ReceiverState& rcv) const {
  const Vec3 pos = gnss::ecef2pos(rr);
  std::vector<int> admitted;
  for (std::size_t j = 0; j < obs.sat.size(); ++j) {
    const int sat = obs.sat[j];
    // Slots this satellite has ever produced (code and phase): its
    // observable signal plan.
    for (int f = 0; f < obs.nf; ++f)
      if (obs.P(j, f) != 0.0 && obs.L(j, f) != 0.0) rcv.band_seen(sat - 1, f) = true;
    bool keep = st.usable[j];
    if (keep) {
      Vec3 e;
      gnss::geodist(st.rs.row(j).transpose(), rr, &e);
      double az, el;
      gnss::satazel(pos, e, &az, &el);
      rcv.el[sat - 1] = el;
      keep = el >= nav.elmin;
    }
    // Per slot: unusable without code and phase or below the SNR mask.
    bool all_seen_ok = true;
    std::vector<bool> slot_ok(obs.nf);
    for (int f = 0; f < obs.nf; ++f) {
      slot_ok[f] = obs.P(j, f) != 0.0 && obs.L(j, f) != 0.0 &&
                   !(nav.cnr_min > 0 && obs.S(j, f) < nav.cnr_min);
      if (rcv.band_seen(sat - 1, f) && !slot_ok[f]) all_seen_ok = false;
    }
    if (nav.strict_sat_gate && !all_seen_ok) keep = false;
    bool any_pr = false;
    for (int f = 0; f < obs.nf; ++f) {
      const bool snr_masked = nav.cnr_min > 0 && obs.S(j, f) < nav.cnr_min;
      if (!keep || snr_masked) {
        obs.P(j, f) = obs.L(j, f) = obs.D(j, f) = 0.0;
        continue;
      }
      any_pr = any_pr || obs.P(j, f) != 0.0;
    }
    if (any_pr) admitted.push_back(sat);
  }
  return admitted;
}

std::optional<DDMeasurements> DdFrontend::prepare(const ObsEpoch& obs_in,
                                                  const ObsEpoch* obsb_in,
                                                  NavState& nav) {
  if (obs_in.size() == 0 || !obsb_in || obsb_in->size() == 0) return std::nullopt;
  const NavData::SatStates st = navdata_.satposs(obs_in);
  if (st.nsat < 4) return std::nullopt;
  const NavData::SatStates stb = navdata_.satposs(*obsb_in);

  auto obs = std::make_shared<ObsEpoch>(obs_in);
  auto obsb = std::make_shared<ObsEpoch>(*obsb_in);
  const std::vector<int> ok_r = edit(*obsb, stb, nav.rb, nav, base_rcv_);
  const std::vector<int> ok_u = edit(*obs, st, nav.x.pos, nav, nav.rcv);
  std::vector<int> common;
  std::set_intersection(ok_u.begin(), ok_u.end(), ok_r.begin(), ok_r.end(),
                        std::back_inserter(common));
  if (common.size() < 4) return std::nullopt;

  DDMeasurements dd;
  dd.obs = obs;
  dd.obsb = obsb;
  dd.rs = st.rs;
  dd.vs = st.vs;
  dd.rsb = stb.rs;
  std::size_t iu = 0, ir = 0;
  for (int s : common) {
    while (obs->sat[iu] != s) ++iu;
    while (obsb->sat[ir] != s) ++ir;
    dd.iu.push_back(int(iu));
    dd.ir.push_back(int(ir));
    dd.sat.push_back(s);
    dd.el.push_back(nav.el(s));
  }
  return dd;
}

}  // namespace gnss_fgo
