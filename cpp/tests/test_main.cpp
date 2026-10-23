// Unit tests for the C++ port (no framework: each check prints and counts).
// Reference values for MLAMBDA come from the cssrlib kernels the Python
// pipeline ran.
#include "gnss_fgo/config.h"
#include "gnss_fgo/gnss.h"
#include "gnss_fgo/mlambda.h"
#include "gnss_fgo/modules.h"
#include "gnss_fgo/nav_state.h"

#include <gtsam/base/numericalDerivative.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>

using namespace gnss_fgo;

namespace {

int g_fail = 0, g_pass = 0;

void check(bool ok, const std::string& what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string toStr(const gtsam::Matrix& m) { std::ostringstream o; o << m; return o.str(); }

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

void test_sat_numbering() {
  check(gnss::prn2sat(gnss::GPS, 5) == 5, "GPS G05 -> 5");
  check(gnss::prn2sat(gnss::GAL, 1) == 33, "GAL E01 -> 33");
  check(gnss::prn2sat(gnss::QZS, 193) == 69, "QZS J01 -> 69");
  check(gnss::prn2sat(gnss::BDS, 1) == 86, "BDS C01 -> 86");
  for (int s = 1; s <= gnss::MAXSAT; ++s)
    check(gnss::prn2sat(gnss::sat_sys(s), gnss::sat_prn(s)) == s, "round trip sat " + std::to_string(s));
  check(gnss::sat_id(69) == "J01", "sat_id J01");
  check(gnss::sat_id(gnss::prn2sat(gnss::BDS, 59)) == "C59", "sat_id C59");
  check(gnss::is_bds_geo(gnss::prn2sat(gnss::BDS, 3)), "C03 is GEO");
  check(!gnss::is_bds_geo(gnss::prn2sat(gnss::BDS, 19)), "C19 not GEO");
}

void test_mlambda() {
  Eigen::Matrix3d Q;
  Q << 6.2900, 5.9780, 0.5440, 5.9780, 6.2920, 2.3400, 0.5440, 2.3400, 6.2880;
  const lambda::Result r = lambda::mlambda(Eigen::Vector3d(5.45, 3.10, 2.97), Q);
  check(r.afix.col(0) == Eigen::Vector3d(5, 3, 4), "mlambda 3x3 best");
  check(r.afix.col(1) == Eigen::Vector3d(6, 4, 4), "mlambda 3x3 second");
  check(near(r.s[0], 0.2183311, 1e-6) && near(r.s[1], 0.30727258, 1e-6), "mlambda 3x3 norms");
  check(near(r.Ps, 0.032479737124315795, 1e-12), "mlambda 3x3 Ps");

  const double q[8][8] = {
      {0.7889726845860648, -0.37351535401125885, 0.03894858482911134, 0.048001320816114856, 0.16806297195612388, -0.19070254988205382, -0.0890044250898327, 0.4675621573157891},
      {-0.37351535401125885, 0.6978862074242997, -0.11701109425751997, 0.14673120647656587, 0.13635982660041834, 0.03102803731696533, 0.18927547028528513, -0.36807883044533696},
      {0.03894858482911134, -0.11701109425751997, 0.21944742706417736, 0.1448907685517575, 0.10353640298661602, 0.003144287288564199, 0.05047208300175926, 0.020757070687094564},
      {0.048001320816114856, 0.14673120647656587, 0.1448907685517575, 0.31388618198095314, 0.15993501842861543, -0.08827995029279204, 0.06388984082166908, 0.045372544364515155},
      {0.16806297195612388, 0.13635982660041834, 0.10353640298661602, 0.15993501842861543, 0.7259938108684065, -0.14176218347174643, 0.03162181018841725, 0.17533127546002758},
      {-0.19070254988205382, 0.03102803731696533, 0.003144287288564199, -0.08827995029279204, -0.14176218347174643, 0.2330721353049702, 0.05089363637739846, -0.16774138619974893},
      {-0.0890044250898327, 0.18927547028528513, 0.05047208300175926, 0.06388984082166908, 0.03162181018841725, 0.05089363637739846, 0.1856119241447901, -0.2315883147957002},
      {0.4675621573157891, -0.36807883044533696, 0.020757070687094564, 0.045372544364515155, 0.17533127546002758, -0.16774138619974893, -0.2315883147957002, 0.5691112006787565}};
  Eigen::MatrixXd Q8(8, 8);
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) Q8(i, j) = q[i][j];
  Eigen::VectorXd a8(8);
  a8 << -14.239170914103996, -6.401461765962121, 4.967059914194473, 8.094807339025099,
      -15.106037660883455, -3.4862789132591234, 2.2961406477980386, -4.264512965649095;
  const lambda::Result r8 = lambda::mlambda(a8, Q8);
  Eigen::VectorXd b0(8), b1(8);
  b0 << -14, -7, 5, 8, -15, -4, 2, -4;
  b1 << -14, -7, 5, 8, -16, -3, 2, -4;
  check(r8.afix.col(0) == b0 && r8.afix.col(1) == b1, "mlambda 8x8 candidates");
  check(near(r8.s[0], 2.745052143152182, 1e-9) && near(r8.s[1], 3.413056646352481, 1e-9),
        "mlambda 8x8 norms");

  bool threw = false;
  try {
    lambda::mlambda(Eigen::Vector2d(1, 2), Eigen::Matrix2d::Zero());
  } catch (const lambda::LambdaError&) {
    threw = true;
  }
  check(threw, "mlambda rejects a singular Q");
}

void test_held_ddcp_jacobian() {
  const Vec3 base(-3961904.0, 3348993.0, 3698211.0);
  const Vec3 sr(-12000e3, 14000e3, 18000e3), sj(-20000e3, 2000e3, 16000e3);
  const Vec3 srb = sr + Vec3(1.0, -2.0, 0.5), sjb = sj + Vec3(-0.3, 0.7, 1.1);
  const gtsam::Pose3 ecef_T_nav(gtsam::Rot3::RzRyRx(0.3, -0.7, 1.1), base);
  const Vec3 lever(0.31, 0.0, 0.55);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(1, 0.003);
  const gtsam::Key kp = gtsam::Symbol('P', 1), kn = gtsam::Symbol('n', 7);
  const HeldDdcpFactor f(kp, kn, noise, sr, sj, srb, sjb, base, 12.5, lever, ecef_T_nav, 3.0,
                         -0.19);
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.05, -0.02, 1.3), gtsam::Point3(12.0, -4.0, 1.5));
  gtsam::Values v;
  v.insert(kp, pose);
  v.insert(kn, 4.2);
  std::vector<gtsam::Matrix> H(2);
  f.unwhitenedError(v, &H);
  const std::function<gtsam::Vector1(const gtsam::Pose3&)> fp = [&](const gtsam::Pose3& p) {
    gtsam::Values vv = v;
    vv.update(kp, p);
    return gtsam::Vector1(f.unwhitenedError(vv)[0]);
  };
  const gtsam::Matrix Hnum = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose3>(fp, pose, 1e-3);
  if ((H[0] - Hnum).cwiseAbs().maxCoeff() >= 1e-5) std::printf("H=%s\nHnum=%s\n", toStr(H[0]).c_str(), toStr(Hnum).c_str());
  check((H[0] - Hnum).cwiseAbs().maxCoeff() < 1e-5, "held DDCP pose Jacobian");
  check(near(H[1](0, 0), -0.19, 0), "held DDCP ambiguity Jacobian");

  const HeldDdcpFactor f0(kp, std::nullopt, noise, sr, sj, srb, sjb, base, 12.5, lever,
                          ecef_T_nav, 3.0, 0.0);
  check(f0.keys().size() == 1 && !f0.has_float(), "held-held DDCP is pose-only");
}

void test_ddidx() {
  NavState nav(2);
  nav.elmaskar = 20.0 * gnss::PI / 180.0;
  const int g3 = 3, g7 = 7, g9 = 9, e1 = gnss::prn2sat(gnss::GAL, 1), e4 = gnss::prn2sat(gnss::GAL, 4);
  for (int s : {g3, g7, g9, e1, e4}) {
    nav.x.at(s, 0) = 100.0 + s;
    nav.vsat(s - 1, 0) = 1;
    nav.rcv.el[s - 1] = 0.8;
  }
  nav.rcv.el[g3 - 1] = 0.1;  // below the AR mask: candidate before the reference
  const auto ix = nav.ddidx({g3, g7, g9, e1, e4});
  check(ix.size() == 2, "ddidx pair count");
  check(ix[0][0] == SfKey{g7, 0} && ix[0][1] == SfKey{g9, 0}, "ddidx GPS pair");
  check(ix[1][0] == SfKey{e1, 0} && ix[1][1] == SfKey{e4, 0}, "ddidx GAL pair");
  check(nav.fix(g3 - 1, 0) == 1 && nav.fix(g7 - 1, 0) == 2 && nav.fix(g9 - 1, 0) == 2,
        "ddidx fix marks");
  check(nav.holdamb_flags() == 4 && nav.fix(g9 - 1, 0) == 3, "holdamb_flags");
}

void test_config_env() {
  setenv("SIG_PR", "0.7", 1);
  setenv("AR_STARVE_RESET", "12", 1);
  setenv("NHC_LEVER", "1,2,3", 1);
  const TcConfig cfg = TcConfig::from_env();
  check(cfg.sigma_pr == 0.7, "SIG_PR override");
  check(cfg.ar_starve_reset == 12, "int env field");
  check(cfg.nhc_lever == "1,2,3", "string env field");
  check(cfg.accel_noise == 2.84e-3, "tactical preset");
  setenv("IMU_GRADE", "consumer", 1);
  check(TcConfig::from_env().gyro_noise == 4.0e-4, "consumer preset");
  unsetenv("SIG_PR");
  unsetenv("AR_STARVE_RESET");
  unsetenv("NHC_LEVER");
  unsetenv("IMU_GRADE");
  check(std::isnan(TcConfig().init_pitch_deg), "init_pitch_deg default NaN");
}

void test_geometry() {
  const Vec3 r(-3961767.5545, 3349008.6167, 3698309.7479);
  const Vec3 llh = gnss::ecef2pos(r);
  check(near(llh[0] * 180 / gnss::PI, 35.6675, 1e-3) && near(llh[1] * 180 / gnss::PI, 139.7911, 1e-3),
        "ecef2pos lat/lon");
  Vec3 e;
  const Vec3 rs(-12000e3, 14000e3, 18000e3);
  const double rho = gnss::geodist(rs, r, &e);
  check(near(rho, (rs - r).norm() + gnss::OMGE * (rs[0] * r[1] - rs[1] * r[0]) / gnss::CLIGHT, 1e-6),
        "geodist Sagnac");
  check(near(e.norm(), 1.0, 1e-12), "geodist unit LOS");
}

}  // namespace

int main() {
  test_sat_numbering();
  test_mlambda();
  test_held_ddcp_jacobian();
  test_ddidx();
  test_config_env();
  test_geometry();
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
