// GNSS primitives the estimator works in: satellite numbering, time,
// geometry. Thin wrappers over RTKLIB (rtklibexplorer demo5), so the
// estimator uses RTKLIB's satellite numbers (satno) and models; rtklib.h
// itself stays out of the headers (its macros collide with Eigen/GTSAM).
#pragma once

#include "gnss_fgo/common.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gnss_fgo::gnss {

/// Systems in RTKLIB satellite-number order (GPS, GLONASS, Galileo, QZSS,
/// BeiDou, NavIC, SBAS): iterating systems by id follows satno.
enum Sys : int { NONE = -1, GPS = 0, GLO = 1, GAL = 2, QZS = 3, BDS = 4, IRN = 5, SBS = 6 };
constexpr int GNSSMAX = 7;

/// RTKLIB MAXSAT for the build configuration in CMakeLists.txt (GPS 32,
/// GLONASS 27, Galileo 36, QZSS 10, BeiDou 63, NavIC 14, SBAS 39);
/// checked against rtklib.h at compile time.
constexpr int MAXSAT = 221;

constexpr double CLIGHT = 299792458.0;
constexpr double OMGE = 7.2921151467e-5;
constexpr double RE_WGS84 = 6378137.0;
constexpr double FE_WGS84 = 1.0 / 298.257223563;
constexpr double PI = 3.1415926535897932;

/// sat -> system (NONE when out of range).
int sat_sys(int sat);
/// sat -> PRN (QZSS 193.., SBAS 120..).
int sat_prn(int sat);
/// system + PRN -> RTKLIB satellite number (0 when invalid).
int prn2sat(int sys, int prn);
/// "G05" style identifier (RTKLIB satno2id).
std::string sat_id(int sat);
/// RTKLIB SYS_* bit <-> Sys.
int from_rtk_sys(int rtk_sys);
int to_rtk_sys(int sys);
/// BeiDou GEO slots (C01-C05, C59-C63).
bool is_bds_geo(int sat);

/// GPST time: integer seconds since 1970-01-01 + fraction (RTKLIB gtime_t).
struct GTime {
  int64_t time = 0;
  double sec = 0.0;
};
double timediff(const GTime& a, const GTime& b);
GTime timeadd(GTime t, double s);
/// GPS time of week [s]; week number into *week when non-null.
double time2gpst(const GTime& t, int* week = nullptr);

/// Geometric range with Sagnac term, and the receiver->satellite unit LOS
/// (RTKLIB geodist: negative when the satellite is not above the Earth).
double geodist(const Vec3& rs, const Vec3& rr, Vec3* e);
/// ECEF -> geodetic (lat, lon, h) [rad, rad, m].
Vec3 ecef2pos(const Vec3& r);
/// ECEF -> ENU rotation at a geodetic position.
Mat3 xyz2enu(const Vec3& pos);
/// Relative ECEF vector -> ENU.
inline Vec3 ecef2enu(const Vec3& pos, const Vec3& r) { return xyz2enu(pos) * r; }
/// Azimuth / elevation of a LOS vector [rad].
void satazel(const Vec3& pos, const Vec3& e, double* az, double* el);
/// (gdop, pdop, hdop, vdop) from az/el (RTKLIB dops); nullopt with fewer
/// than 4 usable rows.
std::optional<std::array<double, 4>> dops(const std::vector<double>& az,
                                          const std::vector<double>& el,
                                          double elmin = 0.0);

}  // namespace gnss_fgo::gnss
