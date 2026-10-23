#include "gnss_fgo/config.h"

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace gnss_fgo {

namespace {

struct ImuPreset {
  double accel_noise, gyro_noise, accel_bias_sigma, gyro_bias_sigma;
};

// 'tactical' == the field defaults (accel/gyro noise carry the measured
// x10/x100 inflation); listed so the table tells the truth.
const std::map<std::string, ImuPreset> kImuPresets = {
    {"tactical", {2.84e-3, 4.01e-3, 3.14e-4, 9.70e-6}},
    {"consumer", {2.0e-3, 4.0e-4, 1.0e-3, 5.0e-5}},
    {"industrial", {6.0e-4, 1.0e-4, 5.0e-4, 2.0e-5}},
    {"nav_grade", {5.7e-5, 3.5e-6, 2.0e-5, 5.0e-7}},
};

const std::map<std::string, std::string> kEnvOverrides = {
    {"sigma_pr", "SIG_PR"},
    {"sigma_cp", "SIG_CP"},
    {"sigma_amb0", "SIG_AMB"},
    {"sigma_cont", "SIG_CONT"},
    {"sigma_n_between", "SIG_N_BETWEEN"},
    {"sigma_n_between_flt", "SIG_N_BETWEEN_FLT"},
    {"zupt_g_dev_thr", "ZUPT_G_DEV"},
};

std::string env_name(const std::string& field) {
  auto it = kEnvOverrides.find(field);
  if (it != kEnvOverrides.end()) return it->second;
  std::string out = field;
  for (char& c : out) c = char(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

void load(const std::string& field, double& v) {
  const std::string name = env_name(field);
  const char* raw = std::getenv(name.c_str());
  if (!raw) return;
  try {
    std::size_t pos = 0;
    v = std::stod(raw, &pos);
    if (pos != std::string(raw).size()) throw std::invalid_argument(raw);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + "='" + raw + "' is not a float");
  }
}

void load(const std::string& field, int& v) {
  const std::string name = env_name(field);
  const char* raw = std::getenv(name.c_str());
  if (!raw) return;
  try {
    std::size_t pos = 0;
    v = std::stoi(raw, &pos);
    if (pos != std::string(raw).size()) throw std::invalid_argument(raw);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + "='" + raw + "' is not an int");
  }
}

void load(const std::string& field, std::string& v) {
  const char* raw = std::getenv(env_name(field).c_str());
  if (raw) v = raw;
}

}  // namespace

TcConfig TcConfig::from_env() {
  TcConfig cfg;
#define GNSS_FGO_LOAD(type, name, dflt) load(#name, cfg.name);
  GNSS_FGO_CONFIG_FIELDS(GNSS_FGO_LOAD)
#undef GNSS_FGO_LOAD
  cfg.apply_imu_preset();
  return cfg;
}

void TcConfig::apply_imu_preset() {
  const TcConfig d;
  const bool is_default = accel_noise == d.accel_noise && gyro_noise == d.gyro_noise &&
                          accel_bias_sigma == d.accel_bias_sigma &&
                          gyro_bias_sigma == d.gyro_bias_sigma;
  if (!is_default) return;
  auto it = kImuPresets.find(imu_grade);
  if (it == kImuPresets.end())
    throw std::invalid_argument("Unknown imu_grade: " + imu_grade +
                                ". Choose from tactical, consumer, industrial, nav_grade");
  accel_noise = it->second.accel_noise;
  gyro_noise = it->second.gyro_noise;
  accel_bias_sigma = it->second.accel_bias_sigma;
  gyro_bias_sigma = it->second.gyro_bias_sigma;
}

}  // namespace gnss_fgo
