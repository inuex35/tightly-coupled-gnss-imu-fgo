#include "gnss_fgo/utils.h"

#include "gnss_fgo/gnss.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/PseudorangeFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gnss_fgo {

namespace {
Mat3 make(double a, double b, double c, double d, double e, double f, double g,
          double h, double i) {
  Mat3 m;
  m << a, b, c, d, e, f, g, h, i;
  return m;
}
double round4(double v) { return std::round(v * 1e4) / 1e4; }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

const Mat3 R_ENU2NED = make(0, 1, 0, 1, 0, 0, 0, 0, -1);
const Mat3 R_NED2ENU = R_ENU2NED.transpose();
const Mat3 R_FRD2FLU = Vec3(1.0, -1.0, -1.0).asDiagonal();
const Mat3 R_FLU2FRD = R_FRD2FLU.transpose();

Mat3 euler_to_R_body2ned(double roll, double pitch, double heading) {
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double ch = std::cos(heading), sh = std::sin(heading);
  const Mat3 R_ned2body = make(cp * ch, cp * sh, -sp,
                               sr * sp * ch - cr * sh, sr * sp * sh + cr * ch, sr * cp,
                               cr * sp * ch + sr * sh, cr * sp * sh - sr * ch, cr * cp);
  return R_ned2body.transpose() * R_FLU2FRD;
}

Mat3 euler_to_R_body2enu(double roll, double pitch, double heading) {
  return R_NED2ENU * euler_to_R_body2ned(roll, pitch, heading);
}

Vec3 parse_lever(const std::string& s) {
  Vec3 v = Vec3::Zero();
  std::stringstream ss(s);
  std::string tok;
  int i = 0;
  while (std::getline(ss, tok, ',') && i < 3) v[i++] = std::stod(tok);
  if (i != 3) throw std::invalid_argument("lever '" + s + "' is not x,y,z");
  return v;
}

std::vector<ImuSample> load_imu_csv(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<ImuSample> out;
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> v;
    while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
    if (v.size() < 8) continue;
    ImuSample s;
    s.tow = v[0];
    s.acc = Vec3(v[2], v[3], v[4]);
    s.gyro = Vec3(v[5], v[6], v[7]) * gnss::PI / 180.0;
    out.push_back(s);
  }
  return out;
}

double heading_from_vel(const Vec3& vel, double fallback, const std::optional<Vec3>& disp,
                        double vel_speed_min, double disp_min) {
  if (vel.head<2>().norm() > vel_speed_min) return std::atan2(vel[0], vel[1]);
  if (disp && disp->head<2>().norm() > disp_min) return std::atan2((*disp)[0], (*disp)[1]);
  return fallback;
}

double heading_from_pose(const gtsam::Pose3& pose) {
  const Mat3 r = pose.rotation().matrix();
  return std::atan2(r(0, 0), r(1, 0)) * 180.0 / gnss::PI;
}

double compute_gdop(const Vec3& pred_trans_enu, int ns, const Eigen::MatrixXd& rs,
                    const std::vector<int>& iu, const Mat3& R_enu2ecef,
                    const Vec3& base_ecef) {
  if (ns < 4) return 999.0;
  const Vec3 pp = R_enu2ecef * pred_trans_enu + base_ecef;
  if (pp.norm() > 1e7 || !pp.allFinite()) return 999.0;
  const Vec3 pos = gnss::ecef2pos(pp);
  std::vector<double> az(ns), el(ns);
  for (int i = 0; i < ns; ++i) {
    const Vec3 diff = rs.row(iu[i]).transpose() - pp;
    const double nrm = diff.norm();
    if (nrm < 1.0) return 999.0;
    gnss::satazel(pos, diff / nrm, &az[i], &el[i]);
  }
  const auto d = gnss::dops(az, el);
  return d ? (*d)[0] : 999.0;
}

std::shared_ptr<gtsam::PreintegrationCombinedParams> make_imu_params(
    double accel_noise, double gyro_noise, double accel_bias_sigma,
    double gyro_bias_sigma, double integ_cov) {
  // ENU navigation + FLU body; samples are already body FLU.
  auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
  p->setBodyPSensor(gtsam::Pose3());
  p->setAccelerometerCovariance(accel_noise * accel_noise * Mat3::Identity());
  p->setGyroscopeCovariance(gyro_noise * gyro_noise * Mat3::Identity());
  p->setIntegrationCovariance(integ_cov * Mat3::Identity());
  p->setBiasAccCovariance(accel_bias_sigma * accel_bias_sigma * Mat3::Identity());
  p->setBiasOmegaCovariance(gyro_bias_sigma * gyro_bias_sigma * Mat3::Identity());
  return p;
}

PimResult build_pim(const std::shared_ptr<gtsam::PreintegrationCombinedParams>& params,
                    const gtsam::imuBias::ConstantBias& bias,
                    const std::vector<ImuSample>& imu, int imu_idx,
                    std::optional<double> target_tow, int max_samples, double dt) {
  PimResult r;
  r.pim = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(params, bias);
  Vec3 gyro_sum = Vec3::Zero();
  int idx = imu_idx;
  while (idx < int(imu.size())) {
    const ImuSample& im = imu[idx];
    if (target_tow) {
      if (im.tow > *target_tow + 1e-6) break;
    } else if (r.n >= max_samples) {
      break;
    }
    // Real per-sample timestep, rounded to 0.1 ms (dropped samples show
    // up as 0.02, 0.03, ...); first sample: nominal dt.
    double dt_i = dt;
    if (idx > 0) {
      const double d = round4(im.tow - imu[idx - 1].tow);
      if (d > 0.0 && d <= 10 * dt) dt_i = d;
    }
    r.pim->integrateMeasurement(im.acc, im.gyro, dt_i);
    gyro_sum += im.gyro;
    ++r.n;
    ++idx;
  }
  if (r.n > 0) r.gyro_mean = gyro_sum / r.n;
  r.new_idx = idx;
  return r;
}

PimResult build_pim_from_samples(
    const std::shared_ptr<gtsam::PreintegrationCombinedParams>& params,
    const gtsam::imuBias::ConstantBias& bias, const std::vector<ImuSample>& samples,
    double dt) {
  PimResult r;
  r.pim = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(params, bias);
  Vec3 gyro_sum = Vec3::Zero();
  std::optional<double> prev_tow;
  for (const ImuSample& im : samples) {
    double dt_i = dt;
    if (prev_tow) {
      const double d = round4(im.tow - *prev_tow);
      if (d > 0.0 && d <= 10 * dt) dt_i = d;
    }
    prev_tow = im.tow;
    r.pim->integrateMeasurement(im.acc, im.gyro, dt_i);
    gyro_sum += im.gyro;
    ++r.n;
  }
  if (r.n > 0) r.gyro_mean = gyro_sum / r.n;
  return r;
}

std::pair<Vec3, Vec3> estimate_stationary_bias(const std::vector<ImuSample>& imu,
                                               int n_max, int n_min, double g_expected) {
  if (n_max <= n_min) return {Vec3::Zero(), Vec3::Zero()};
  const int n = std::min<int>(n_max, int(imu.size()));
  Vec3 acc = Vec3::Zero(), gyro = Vec3::Zero();
  for (int i = 0; i < n; ++i) {
    acc += imu[i].acc;
    gyro += imu[i].gyro;
  }
  acc /= n;
  gyro /= n;
  const double g = acc.norm();
  const Vec3 bias_acc = g > 1.0 ? Vec3(acc - g_expected * acc / g) : Vec3::Zero();
  return {bias_acc, gyro};
}

std::optional<ZuptStats> compute_zupt_stats(const ImuSample* begin, const ImuSample* end,
                                            const std::optional<Vec3>& bias_acc_ref,
                                            const std::optional<Vec3>& bias_gyro_ref) {
  const int n = int(end - begin);
  if (n <= 0) return std::nullopt;
  Vec3 acc_mean = Vec3::Zero(), gyro_mean = Vec3::Zero();
  for (const ImuSample* p = begin; p != end; ++p) {
    acc_mean += p->acc;
    gyro_mean += p->gyro;
  }
  acc_mean /= n;
  gyro_mean /= n;
  double acc_var = 0, gyro_var = 0;
  std::vector<double> gyro_resid;
  gyro_resid.reserve(n);
  for (const ImuSample* p = begin; p != end; ++p) {
    acc_var += (p->acc - acc_mean).squaredNorm();
    gyro_var += (p->gyro - gyro_mean).squaredNorm();
    gyro_resid.push_back(
        (bias_gyro_ref ? Vec3(p->gyro - *bias_gyro_ref) : p->gyro).norm());
  }
  acc_var /= n;
  gyro_var /= n;
  const Vec3 acc_corr = bias_acc_ref ? Vec3(acc_mean - *bias_acc_ref) : acc_mean;
  ZuptStats s;
  s.n = n;
  s.acc_std = std::sqrt(std::max(acc_var, 0.0));
  s.gyro_std = std::sqrt(std::max(gyro_var, 0.0));
  s.gyro_median = median(gyro_resid);
  s.acc_norm_mean = acc_corr.norm();
  return s;
}

std::vector<ImuSample> collect_imu_samples(const std::vector<ImuSample>& imu, int* imu_idx,
                                           int n_samples, std::optional<double> target_tow) {
  std::vector<ImuSample> out;
  int idx = *imu_idx;
  while (idx < int(imu.size())) {
    const ImuSample& im = imu[idx];
    if (target_tow) {
      if (im.tow > *target_tow + 1e-6) break;
    } else if (int(out.size()) >= n_samples) {
      break;
    }
    out.push_back(im);
    ++idx;
  }
  *imu_idx = idx;
  return out;
}

// ── DDPR-only least squares ─────────────────────────────────────────

namespace {

struct DdprSpec {
  double a1, a2, a3, a4;
  gtsam::Point3 sr, st, srb, stb;
};

void ddpr_build_specs(const EpochInput& in, const DDPRContext& ctx,
                      std::vector<DdprSpec>* specs, std::vector<SfKey>* spec_sats) {
  const ObsEpoch& obs = *in.obs;
  const ObsEpoch& obsb = *in.obsb;
  const int ns = int(in.sat.size());
  for (int sys_id : in.sig().sorted_sys_ids()) {
    std::vector<int> idx_sys;
    for (int i = 0; i < ns; ++i)
      if (gnss::sat_sys(in.sat[i]) == sys_id) idx_sys.push_back(i);
    if (idx_sys.size() < 2) continue;
    const auto [ri, ref_sat] = ctx.pick_ref_sat_idx(sys_id, idx_sys, in.sat, in.el);
    auto itr = in.ir_map.find(ref_sat);
    for (int ji : idx_sys) {
      if (ji == ri) continue;
      const int j_sat = in.sat[ji];
      auto itj = in.ir_map.find(j_sat);
      const gtsam::Point3 sr = in.rs.row(in.iu[ri]).transpose();
      const gtsam::Point3 st = in.rs.row(in.iu[ji]).transpose();
      const gtsam::Point3 srb =
          itr != in.ir_map.end() ? gtsam::Point3(in.rsb.row(itr->second).transpose()) : sr;
      const gtsam::Point3 stb =
          itj != in.ir_map.end() ? gtsam::Point3(in.rsb.row(itj->second).transpose()) : st;
      for (int f = 0; f < ctx.nav_nf; ++f) {
        const double a1 = obs.P(in.iu[ri], f);
        const double a2 = itr != in.ir_map.end() ? obsb.P(itr->second, f) : 0.0;
        const double a3 = obs.P(in.iu[ji], f);
        const double a4 = itj != in.ir_map.end() ? obsb.P(itj->second, f) : 0.0;
        if (a1 == 0 || a2 == 0 || a3 == 0 || a4 == 0) continue;
        specs->push_back({a1, a2, a3, a4, sr, st, srb, stb});
        spec_sats->push_back({j_sat, f});
      }
    }
  }
}

}  // namespace

DdprLsResult ddpr_only_position(const EpochInput& in, const gtsam::Pose3& pose_init,
                                const DDPRContext& ctx) {
  DdprLsResult out;
  std::vector<DdprSpec> specs;
  std::vector<SfKey> spec_sats;
  ddpr_build_specs(in, ctx, &specs, &spec_sats);
  if (specs.size() < 4) return out;

  const gtsam::Key key = gtsam::Symbol('z', 0);
  const double sigma_pr_m = ctx.sigma_pr * std::sqrt(2.0);
  const auto pr_noise = gtsam::noiseModel::Isotropic::Sigma(1, sigma_pr_m);
  const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6() << 0.05, 0.05, 0.1, 50.0, 50.0, 50.0).finished());
  const gtsam::Point3 base_pt(ctx.base_ecef);

  std::vector<int> active(specs.size());
  for (std::size_t i = 0; i < specs.size(); ++i) active[i] = int(i);
  std::optional<gtsam::Values> est;
  std::vector<double> res_kept;
  for (int it = 0; it < 3; ++it) {
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    v.insert(key, pose_init);
    g.addPrior(key, pose_init, prior_noise);
    for (int idx : active) {
      const DdprSpec& s = specs[idx];
      g.emplace_shared<gtsam::DoubleDifferencePseudorangeFactorArm>(
          key, s.a1, s.a2, s.a3, s.a4, s.sr, s.st, s.srb, s.stb, base_pt,
          gtsam::Point3(ctx.lever), ctx.ecef_T_nav, pr_noise);
    }
    try {
      gtsam::LevenbergMarquardtParams params;
      params.setMaxIterations(10);
      est = gtsam::LevenbergMarquardtOptimizer(g, v, params).optimize();
    } catch (const std::exception& e) {
      GNSS_FGO_CATCH_ONLY(e, PyErr::Runtime);
      out.n_active = int(active.size());
      return out;
    }
    std::vector<int> new_active;
    int dropped = 0;
    res_kept.clear();
    for (std::size_t j = 0; j < active.size(); ++j) {
      const double err = g.at(j + 1)->error(*est);  // slot 0 is the pose prior
      const double res_m = std::sqrt(std::max(err, 0.0) * 2.0) * sigma_pr_m;
      if (res_m > ctx.fde_pr && int(active.size()) - dropped > 4) {
        ++dropped;
        continue;
      }
      new_active.push_back(active[j]);
      res_kept.push_back(res_m);
    }
    if (dropped == 0) break;
    active = new_active;
  }
  out.n_active = int(active.size());
  if (!est || active.size() < 4) return out;
  const gtsam::Pose3 pose_out = est->at<gtsam::Pose3>(key);
  out.ecef = ctx.R_enu2ecef * pose_out.translation() + ctx.base_ecef;
  double ss = 0.0;
  for (double r : res_kept) ss += r * r;
  out.res_rms = res_kept.empty() ? 0.0 : std::sqrt(ss / res_kept.size());
  return out;
}

}  // namespace gnss_fgo
