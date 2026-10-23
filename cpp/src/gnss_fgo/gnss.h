// GNSS primitives the estimator works in: satellite numbering, time,
// geometry. RTKLIB numbers satellites differently; the RINEX front-end
// converts at the boundary so that ambiguity keys, the system iteration
// order and every "lowest-numbered satellite" rule keep the numbering
// the pipeline was tuned with (the cssrlib one, below).
#pragma once

#include "gnss_fgo/common.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gnss_fgo::gnss {

enum Sys : int { NONE = -1, GPS = 0, GAL = 1, QZS = 2, BDS = 3, GLO = 4, SBS = 5, IRN = 6 };
constexpr int GNSSMAX = 7;

constexpr int GPSMAX = 32, GALMAX = 36, QZSMAX = 17, BDSMAX = 63,
              GLOMAX = 32, SBSMAX = 39, IRNMAX = 14;
constexpr int GPSMIN = 0;
constexpr int GALMIN = GPSMIN + GPSMAX;
constexpr int QZSMIN = GALMIN + GALMAX;
constexpr int BDSMIN = QZSMIN + QZSMAX;
constexpr int GLOMIN = BDSMIN + BDSMAX;
constexpr int SBSMIN = GLOMIN + GLOMAX;
constexpr int IRNMIN = SBSMIN + SBSMAX;
constexpr int MAXSAT = IRNMIN + IRNMAX;

constexpr double CLIGHT = 299792458.0;
constexpr double OMGE = 7.2921151467e-5;
constexpr double RE_WGS84 = 6378137.0;
constexpr double FE_WGS84 = 1.0 / 298.257223563;
constexpr double PI = 3.1415926535897932;

/// sat -> system (NONE when out of range).
int sat_sys(int sat);
/// sat -> PRN (QZSS 193.., SBAS 120..).
int sat_prn(int sat);
/// system + PRN -> sat (0 when invalid).
int prn2sat(int sys, int prn);
/// "G05" style identifier.
std::string sat_id(int sat);
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

/// Geometric range with Sagnac term, and the receiver->satellite unit LOS.
double geodist(const Vec3& rs, const Vec3& rr, Vec3* e);
/// ECEF -> geodetic (lat, lon, h) [rad, rad, m].
Vec3 ecef2pos(const Vec3& r);
/// ECEF -> ENU rotation at a geodetic position.
Mat3 xyz2enu(const Vec3& pos);
/// Relative ECEF vector -> ENU.
inline Vec3 ecef2enu(const Vec3& pos, const Vec3& r) { return xyz2enu(pos) * r; }
/// Azimuth / elevation of a LOS vector [rad].
void satazel(const Vec3& pos, const Vec3& e, double* az, double* el);
/// (gdop, pdop, hdop, vdop) from az/el; nullopt with fewer than 4 rows.
std::optional<std::array<double, 4>> dops(const std::vector<double>& az,
                                          const std::vector<double>& el,
                                          double elmin = 0.0);

}  // namespace gnss_fgo::gnss
