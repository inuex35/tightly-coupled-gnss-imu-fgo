#include "rtklib.h"
// rtklib.h macros that collide with the gnss:: names below.
#undef PI
#undef MAXSAT
#undef CLIGHT
#undef OMGE
#undef RE_WGS84
#undef FE_WGS84
#undef trace

#include "gnss_fgo/gnss.h"

namespace gnss_fgo::gnss {

static_assert(MAXSAT == NSATGPS + NSATGLO + NSATGAL + NSATQZS + NSATCMP + NSATIRN + NSATSBS +
                            NSATLEO,
              "gnss::MAXSAT out of step with the RTKLIB build configuration");

int from_rtk_sys(int rtk_sys) {
  switch (rtk_sys) {
    case SYS_GPS: return GPS;
    case SYS_GLO: return GLO;
    case SYS_GAL: return GAL;
    case SYS_QZS: return QZS;
    case SYS_CMP: return BDS;
    case SYS_IRN: return IRN;
    case SYS_SBS: return SBS;
    default: return NONE;
  }
}

int to_rtk_sys(int sys) {
  switch (sys) {
    case GPS: return SYS_GPS;
    case GLO: return SYS_GLO;
    case GAL: return SYS_GAL;
    case QZS: return SYS_QZS;
    case BDS: return SYS_CMP;
    case IRN: return SYS_IRN;
    case SBS: return SYS_SBS;
    default: return SYS_NONE;
  }
}

int sat_sys(int sat) {
  if (sat <= 0 || sat > MAXSAT) return NONE;
  return from_rtk_sys(satsys(sat, nullptr));
}

int sat_prn(int sat) {
  int prn = 0;
  if (sat <= 0 || sat > MAXSAT) return 0;
  satsys(sat, &prn);
  return prn;
}

int prn2sat(int sys, int prn) { return satno(to_rtk_sys(sys), prn); }

std::string sat_id(int sat) {
  char id[16];
  satno2id(sat, id);
  return id;
}

bool is_bds_geo(int sat) {
  const int prn = sat_prn(sat);
  return prn <= 5 || (59 <= prn && prn <= 63);
}

namespace {
gtime_t to_rtk(const GTime& t) {
  gtime_t r;
  r.time = time_t(t.time);
  r.sec = t.sec;
  return r;
}
}  // namespace

double timediff(const GTime& a, const GTime& b) { return ::timediff(to_rtk(a), to_rtk(b)); }

GTime timeadd(GTime t, double s) {
  const gtime_t r = ::timeadd(to_rtk(t), s);
  return {int64_t(r.time), r.sec};
}

double time2gpst(const GTime& t, int* week) { return ::time2gpst(to_rtk(t), week); }

double geodist(const Vec3& rs, const Vec3& rr, Vec3* e) {
  double ev[3];
  const double r = ::geodist(rs.data(), rr.data(), ev);
  if (e) *e = Vec3(ev[0], ev[1], ev[2]);
  return r;
}

Vec3 ecef2pos(const Vec3& r) {
  Vec3 pos;
  ::ecef2pos(r.data(), pos.data());
  return pos;
}

Mat3 xyz2enu(const Vec3& pos) {
  double E[9];
  ::xyz2enu(pos.data(), E);
  Mat3 m;  // RTKLIB stores E column-major
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m(i, j) = E[i + 3 * j];
  return m;
}

void satazel(const Vec3& pos, const Vec3& e, double* az, double* el) {
  double azel[2];
  ::satazel(pos.data(), e.data(), azel);
  *az = azel[0];
  *el = azel[1];
}

std::optional<std::array<double, 4>> dops(const std::vector<double>& az,
                                          const std::vector<double>& el, double elmin) {
  std::vector<double> azel(2 * az.size());
  for (std::size_t i = 0; i < az.size(); ++i) {
    azel[2 * i] = az[i];
    azel[2 * i + 1] = el[i];
  }
  double dop[4];
  ::dops(int(az.size()), azel.data(), elmin, dop);
  if (dop[0] <= 0.0) return std::nullopt;  // fewer than 4 usable rows
  return std::array<double, 4>{dop[0], dop[1], dop[2], dop[3]};
}

}  // namespace gnss_fgo::gnss
