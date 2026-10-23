#include "rtklib.h"
// RTKLIB's constant macros collide with the gnss:: constants of the same
// name; this file uses RTKLIB's types and functions, not those macros.
#undef PI
#undef MAXSAT
#undef CLIGHT
#undef OMGE
#undef RE_WGS84
#undef FE_WGS84
// ...and its no-op trace macros with Eigen's trace() member.
#undef trace
#undef tracet
#undef tracemat
#undef traceobs
#undef tracenav
#undef tracegnav
#undef tracehnav
#undef tracepeph
#undef tracepclk
#undef traceopen
#undef traceclose
#undef tracelevel
#undef gettracelevel

#include "gnss_fgo/rinex.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

namespace gnss_fgo {

namespace {

// RINEX header system index (RTKLIB syscodes "GREJSCI") -> system.
constexpr int kRnxSys[] = {gnss::GPS, gnss::GLO, gnss::GAL, gnss::QZS,
                           gnss::SBS, gnss::BDS, gnss::IRN};

int rtk_sys_to_sys(int rtk_sys) {
  switch (rtk_sys) {
    case SYS_GPS: return gnss::GPS;
    case SYS_GAL: return gnss::GAL;
    case SYS_QZS: return gnss::QZS;
    case SYS_CMP: return gnss::BDS;
    case SYS_GLO: return gnss::GLO;
    case SYS_SBS: return gnss::SBS;
    case SYS_IRN: return gnss::IRN;
    default: return gnss::NONE;
  }
}

int sys_to_rtk_sys(int sys) {
  switch (sys) {
    case gnss::GPS: return SYS_GPS;
    case gnss::GAL: return SYS_GAL;
    case gnss::QZS: return SYS_QZS;
    case gnss::BDS: return SYS_CMP;
    case gnss::GLO: return SYS_GLO;
    case gnss::SBS: return SYS_SBS;
    case gnss::IRN: return SYS_IRN;
    default: return SYS_NONE;
  }
}

/// RTKLIB satellite number -> estimator satellite number (0 if unmapped).
int to_rtk_sat(int sat) {
  return satno(sys_to_rtk_sys(gnss::sat_sys(sat)), gnss::sat_prn(sat));
}

int from_rtk_sat(int rtk_sat) {
  int prn = 0;
  const int sys = rtk_sys_to_sys(satsys(rtk_sat, &prn));
  return sys == gnss::NONE ? 0 : gnss::prn2sat(sys, prn);
}

gnss::GTime from_rtk(const gtime_t& t) { return {int64_t(t.time), t.sec}; }
gtime_t to_rtk(const gnss::GTime& t) {
  gtime_t r;
  r.time = time_t(t.time);
  r.sec = t.sec;
  return r;
}

}  // namespace

// ── SignalPlan ──────────────────────────────────────────────────────

std::vector<double> SignalPlan::wavelengths(int sat) const {
  auto it = by_sys.find(gnss::sat_sys(sat));
  if (it == by_sys.end()) return {};
  std::vector<double> out;
  for (const auto& b : it->second) out.push_back(b.wavelength());
  return out;
}

std::vector<int> SignalPlan::sorted_sys_ids() const {
  std::vector<int> out;
  for (const auto& [sys, _] : by_sys) out.push_back(sys);
  return out;  // std::map keys are already ascending
}

SignalPlan rtklib_signal_plan(const std::vector<int>& systems, int nf) {
  SignalPlan plan;
  for (int sys : systems) {
    const int rsys = sys_to_rtk_sys(sys);
    std::vector<BandSignal> slots(nf);
    // Every code RTKLIB knows, mapped onto its frequency index.
    for (int code = 1; code <= MAXCODE; ++code) {
      const int idx = code2idx(rsys, uint8_t(code));
      if (idx < 0 || idx >= nf || slots[idx].freq > 0) continue;
      slots[idx].band = code2obs(uint8_t(code))[0] - '0';
      slots[idx].freq = code2freq(rsys, uint8_t(code), 0);
    }
    plan.by_sys[sys] = slots;
  }
  return plan;
}

// ── RinexObsReader ──────────────────────────────────────────────────

struct RinexObsReader::Impl {
  FILE* fp = nullptr;
  rnxctr_t* rnx = nullptr;
};

RinexObsReader::RinexObsReader(const std::string& path) : impl_(new Impl) {
  impl_->fp = std::fopen(path.c_str(), "r");
  if (!impl_->fp) throw std::runtime_error("cannot open " + path);
  impl_->rnx = new rnxctr_t();
  if (!init_rnxctr(impl_->rnx) || !open_rnxctr(impl_->rnx, impl_->fp))
    throw std::runtime_error("cannot read RINEX header of " + path);
  if (impl_->rnx->type != 'O')
    throw std::runtime_error(path + " is not a RINEX observation file");
  for (int i = 0; i < RNX_NUMSYS; ++i) {
    std::vector<std::string> types;
    for (int j = 0; j < MAXOBSTYPE && impl_->rnx->tobs[i][j][0]; ++j)
      types.emplace_back(impl_->rnx->tobs[i][j]);
    if (!types.empty()) hdr_[kRnxSys[i]] = types;
  }
  for (int k = 0; k < 3; ++k) pos_[k] = impl_->rnx->sta.pos[k];
}

RinexObsReader::~RinexObsReader() {
  if (impl_->rnx) {
    free_rnxctr(impl_->rnx);
    delete impl_->rnx;
  }
  if (impl_->fp) std::fclose(impl_->fp);
}

ObsPtr RinexObsReader::next() {
  if (!plan_) throw std::logic_error("RinexObsReader: signals not set");
  rnxctr_t* rnx = impl_->rnx;
  int stat;
  while ((stat = input_rnxctr(rnx, impl_->fp)) != 1) {
    if (stat == -2) return nullptr;
  }
  struct Row {
    int sat;
    const obsd_t* d;
  };
  std::vector<Row> rows;
  for (int i = 0; i < rnx->obs.n; ++i) {
    const obsd_t& d = rnx->obs.data[i];
    const int sat = from_rtk_sat(d.sat);
    if (!sat || !plan_->has_sys(gnss::sat_sys(sat))) continue;
    rows.push_back({sat, &d});
  }
  std::sort(rows.begin(), rows.end(),
            [](const Row& a, const Row& b) { return a.sat < b.sat; });

  auto obs = std::make_shared<ObsEpoch>();
  obs->t = from_rtk(rnx->time);
  obs->nf = nf_;
  obs->sig = plan_;
  const int n = int(rows.size());
  obs->P = Eigen::MatrixXd::Zero(n, nf_);
  obs->L = Eigen::MatrixXd::Zero(n, nf_);
  obs->D = Eigen::MatrixXd::Zero(n, nf_);
  obs->S = Eigen::MatrixXd::Zero(n, nf_);
  obs->lli = Eigen::MatrixXi::Zero(n, nf_);
  for (int i = 0; i < n; ++i) {
    const obsd_t& d = *rows[i].d;
    obs->sat.push_back(rows[i].sat);
    // RTKLIB slot f is frequency index f (its highest-priority code).
    for (int f = 0; f < nf_ && f < NFREQ; ++f) {
      obs->P(i, f) = d.P[f];
      obs->L(i, f) = d.L[f];
      obs->D(i, f) = d.D[f];
      obs->S(i, f) = d.SNR[f];
      obs->lli(i, f) = d.LLI[f];
    }
  }
  return obs;
}

// ── NavData ─────────────────────────────────────────────────────────

struct NavData::Impl {
  nav_t nav{};
};

NavData::NavData() : impl_(new Impl) {}
NavData::~NavData() { freenav(&impl_->nav, 0xFF); }

bool NavData::read(const std::string& path) {
  if (!readrnx(path.c_str(), 0, "", nullptr, &impl_->nav, nullptr)) return false;
  uniqnav(&impl_->nav);
  return true;
}

NavData::SatStates NavData::satposs(const ObsEpoch& obs) const {
  const int n = int(obs.size());
  SatStates out;
  out.rs = Eigen::MatrixXd::Zero(n, 3);
  out.vs = Eigen::MatrixXd::Zero(n, 3);
  out.dts.assign(n, 0.0);
  out.var.assign(n, 0.0);
  out.svh.assign(n, 0);
  out.usable.assign(n, false);
  std::vector<obsd_t> data(n);
  for (int i = 0; i < n; ++i) {
    obsd_t& d = data[i];
    std::memset(&d, 0, sizeof(d));
    d.time = to_rtk(obs.t);
    d.sat = uint8_t(to_rtk_sat(obs.sat[i]));
    for (int f = 0; f < obs.nf && f < NFREQ; ++f) d.P[f] = obs.P(i, f);
  }
  std::vector<double> rs(6 * n), dts(2 * n), var(n);
  std::vector<int> svh(n);
  const gtime_t teph = to_rtk(obs.t);
  for (int i0 = 0; i0 < n; i0 += MAXOBS) {
    const int m = std::min(MAXOBS, n - i0);
    ::satposs(teph, data.data() + i0, m, &impl_->nav, EPHOPT_BRDC, rs.data() + 6 * i0,
              dts.data() + 2 * i0, var.data() + i0, svh.data() + i0);
  }
  for (int i = 0; i < n; ++i) {
    const double* r = rs.data() + 6 * i;
    const bool has = r[0] != 0.0 || r[1] != 0.0 || r[2] != 0.0;
    out.rs.row(i) << r[0], r[1], r[2];
    out.vs.row(i) << r[3], r[4], r[5];
    out.dts[i] = dts[2 * i];
    out.var[i] = var[i];
    out.svh[i] = svh[i];
    out.usable[i] = has && !satexclude(data[i].sat, var[i], svh[i], nullptr);
    if (has) ++out.nsat;
  }
  return out;
}

// ── SyncObsHold ─────────────────────────────────────────────────────

SyncObsHold::SyncObsHold(RinexObsReader& rov, RinexObsReader& base, double maxage)
    : rov_(rov), base_(base), maxage_(maxage) {
  curr_ = base_.next();
  next_ = curr_ ? base_.next() : nullptr;
}

bool SyncObsHold::next(ObsPtr* obs, ObsPtr* obsb, double* dt) {
  ObsPtr o = rov_.next();
  if (!o) return false;
  while (next_) {
    if (!curr_) {
      curr_ = next_;
      next_ = base_.next();
      continue;
    }
    const double dc = std::abs(gnss::timediff(o->t, curr_->t));
    const double dn = std::abs(gnss::timediff(o->t, next_->t));
    if (dn < dc) {
      curr_ = next_;
      next_ = base_.next();
    } else {
      break;
    }
  }
  *obs = o;
  if (!curr_) {
    *obsb = nullptr;
    *dt = nan();
    return true;
  }
  *dt = gnss::timediff(o->t, curr_->t);
  *obsb = std::abs(*dt) <= maxage_ ? curr_ : nullptr;
  return true;
}

}  // namespace gnss_fgo
