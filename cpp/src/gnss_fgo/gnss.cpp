#include "gnss_fgo/gnss.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>

namespace gnss_fgo::gnss {

int sat_sys(int sat) {
  if (sat <= 0 || sat > MAXSAT) return NONE;
  if (sat > IRNMIN) return IRN;
  if (sat > SBSMIN) return SBS;
  if (sat > GLOMIN) return GLO;
  if (sat > BDSMIN) return BDS;
  if (sat > QZSMIN) return QZS;
  if (sat > GALMIN) return GAL;
  return GPS;
}

int sat_prn(int sat) {
  switch (sat_sys(sat)) {
    case IRN: return sat - IRNMIN;
    case SBS: return sat + 119 - SBSMIN;
    case GLO: return sat - GLOMIN;
    case BDS: return sat - BDSMIN;
    case QZS: return sat + 192 - QZSMIN;
    case GAL: return sat - GALMIN;
    case GPS: return sat;
    default: return 0;
  }
}

int prn2sat(int sys, int prn) {
  int sat = 0;
  switch (sys) {
    case GPS: sat = (prn >= 1 && prn <= GPSMAX) ? prn : 0; break;
    case GAL: sat = (prn >= 1 && prn <= GALMAX) ? prn + GALMIN : 0; break;
    case QZS: sat = (prn >= 193 && prn < 193 + QZSMAX) ? prn - 192 + QZSMIN : 0; break;
    case GLO: sat = (prn >= 1 && prn <= GLOMAX) ? prn + GLOMIN : 0; break;
    case BDS: sat = (prn >= 1 && prn <= BDSMAX) ? prn + BDSMIN : 0; break;
    case SBS: sat = (prn >= 120 && prn < 120 + SBSMAX) ? prn - 119 + SBSMIN : 0; break;
    case IRN: sat = (prn >= 1 && prn <= IRNMAX) ? prn + IRNMIN : 0; break;
    default: break;
  }
  return sat;
}

std::string sat_id(int sat) {
  static const char tbl[] = {'G', 'E', 'J', 'C', 'R', 'S', 'I'};
  const int sys = sat_sys(sat);
  if (sys < 0) return "?";
  int prn = sat_prn(sat);
  if (sys == QZS) prn -= 192;
  else if (sys == SBS) prn -= 100;
  char buf[8];
  std::snprintf(buf, sizeof(buf), "%c%02d", tbl[sys], prn);
  return buf;
}

bool is_bds_geo(int sat) {
  const int prn = sat_prn(sat);
  return prn <= 5 || (59 <= prn && prn <= 63);
}

double timediff(const GTime& a, const GTime& b) {
  return double(a.time - b.time) + a.sec - b.sec;
}

GTime timeadd(GTime t, double s) {
  t.sec += s;
  const double tt = std::floor(t.sec);
  t.time += int64_t(tt);
  t.sec -= tt;
  return t;
}

double time2gpst(const GTime& t, int* week) {
  constexpr int64_t gpst0 = 315964800;  // 1980-01-06 00:00:00 as time_t
  const int64_t sec = t.time - gpst0;
  const int64_t w = sec / (86400 * 7);
  if (week) *week = int(w);
  return double(sec - w * 86400 * 7) + t.sec;
}

double geodist(const Vec3& rs, const Vec3& rr, Vec3* e) {
  Vec3 los = rs - rr;
  const double r = los.norm();
  if (r <= 0.0) {
    if (e) e->setZero();
    return 0.0;
  }
  if (e) *e = los / r;
  return r + OMGE * (rs[0] * rr[1] - rs[1] * rr[0]) / CLIGHT;
}

Vec3 ecef2pos(const Vec3& r) {
  const double e2 = FE_WGS84 * (2.0 - FE_WGS84);
  const double r2 = r[0] * r[0] + r[1] * r[1];
  double v = RE_WGS84, z = r[2];
  for (int i = 0; i < 1000; ++i) {
    const double zk = z;
    const double sp = z / std::sqrt(r2 + z * z);
    v = RE_WGS84 / std::sqrt(1.0 - e2 * sp * sp);
    z = r[2] + v * e2 * sp;
    if (std::abs(z - zk) < 1e-4) break;
  }
  return Vec3(std::atan(z / std::sqrt(r2)), std::atan2(r[1], r[0]),
              std::sqrt(r2 + z * z) - v);
}

Mat3 xyz2enu(const Vec3& pos) {
  const double sp = std::sin(pos[0]), cp = std::cos(pos[0]);
  const double sl = std::sin(pos[1]), cl = std::cos(pos[1]);
  Mat3 E;
  E << -sl, cl, 0.0,
       -sp * cl, -sp * sl, cp,
       cp * cl, cp * sl, sp;
  return E;
}

void satazel(const Vec3& pos, const Vec3& e, double* az, double* el) {
  const Vec3 enu = ecef2enu(pos, e);
  *az = std::atan2(enu[0], enu[1]);
  *el = std::asin(enu[2] / enu.norm());
}

std::optional<std::array<double, 4>> dops(const std::vector<double>& az,
                                          const std::vector<double>& el,
                                          double elmin) {
  Eigen::MatrixXd H(az.size(), 4);
  int n = 0;
  for (std::size_t i = 0; i < az.size(); ++i) {
    if (el[i] < elmin) continue;
    const double cel = std::cos(el[i]), sel = std::sin(el[i]);
    H.row(n++) << cel * std::sin(az[i]), cel * std::cos(az[i]), sel, 1.0;
  }
  if (n < 4) return std::nullopt;
  const Eigen::MatrixXd Hn = H.topRows(n);
  const Eigen::Matrix4d Q = (Hn.transpose() * Hn).inverse();
  const Eigen::Vector4d d = Q.diagonal();
  return std::array<double, 4>{std::sqrt(d.sum()), std::sqrt(d.head<3>().sum()),
                               std::sqrt(d[0] + d[1]), std::sqrt(d[2])};
}

}  // namespace gnss_fgo::gnss
