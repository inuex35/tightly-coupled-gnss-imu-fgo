// RINEX observation / navigation input on RTKLIB.
//
// RTKLIB reads the files, picks the signals (its frequency slots and code
// priorities) and computes satellite states; this layer only converts its
// records into the per-epoch arrays the estimator indexes by frequency
// slot f.
#pragma once

#include "gnss_fgo/common.h"
#include "gnss_fgo/gnss.h"

#include <Eigen/Core>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gnss_fgo {

/// One frequency slot of one system.
struct BandSignal {
  int band = 0;        // RINEX band digit (1, 2, 5, 7, 6, ...)
  double freq = 0.0;   // carrier frequency [Hz]
  double wavelength() const { return freq > 0 ? gnss::CLIGHT / freq : 0.0; }
};

/// The frequency slots observations are decoded into: system -> slots.
struct SignalPlan {
  std::map<int, std::vector<BandSignal>> by_sys;

  bool has_sys(int sys) const { return by_sys.count(sys) > 0; }
  /// Phase wavelengths per frequency slot ([] when the system is absent).
  std::vector<double> wavelengths(int sat) const;
  /// Systems in ascending id order (the estimator's iteration order).
  std::vector<int> sorted_sys_ids() const;
};

/// RINEX header observation types per system, in header order ("C1C", ...).
using HeaderObsTypes = std::map<int, std::vector<std::string>>;

/// RTKLIB's frequency slots for ``systems``: slot f is RTKLIB frequency
/// index f (GPS/QZSS L1/L2/L5, Galileo E1/E5b/E5a, BeiDou B1I/B2I-B2b/B2a).
/// Within a slot RTKLIB keeps its highest-priority code.
SignalPlan rtklib_signal_plan(const std::vector<int>& systems, int nf);

/// One epoch of observations, satellites ascending, nf frequency slots.
struct ObsEpoch {
  gnss::GTime t;
  std::vector<int> sat;
  Eigen::MatrixXd P, L, D, S;
  Eigen::MatrixXi lli;
  std::shared_ptr<const SignalPlan> sig;
  int nf = 0;
  std::size_t size() const { return sat.size(); }
};
using ObsPtr = std::shared_ptr<const ObsEpoch>;

/// Streaming RINEX observation reader (RTKLIB rnxctr).
class RinexObsReader {
 public:
  explicit RinexObsReader(const std::string& path);
  ~RinexObsReader();
  RinexObsReader(const RinexObsReader&) = delete;
  RinexObsReader& operator=(const RinexObsReader&) = delete;

  const HeaderObsTypes& header_types() const { return hdr_; }
  Vec3 approx_pos() const { return pos_; }
  void set_signals(std::shared_ptr<const SignalPlan> plan, int nf) {
    plan_ = std::move(plan);
    nf_ = nf;
  }
  /// Next epoch, or nullptr at end of file.
  ObsPtr next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  HeaderObsTypes hdr_;
  Vec3 pos_ = Vec3::Zero();
  std::shared_ptr<const SignalPlan> plan_;
  int nf_ = 0;
};

/// Broadcast navigation data (RTKLIB nav_t) and satellite states.
class NavData {
 public:
  NavData();
  ~NavData();
  NavData(const NavData&) = delete;
  NavData& operator=(const NavData&) = delete;

  /// Read a RINEX navigation file (may be called for several files).
  bool read(const std::string& path);

  /// RTKLIB satposs: position/velocity/clock at transmission time for
  /// every observation of ``obs``. ``usable[i]`` is false when RTKLIB has
  /// no state for it or excludes it (satexclude: health, URA).
  struct SatStates {
    Eigen::MatrixXd rs, vs;      // n x 3
    std::vector<double> dts, var;
    std::vector<int> svh;
    std::vector<bool> usable;
    int nsat = 0;                // satellites with a computed state
  };
  SatStates satposs(const ObsEpoch& obs) const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Rover-driven synchronisation with base hold (RTKLIB maxtdiff style):
/// for every rover epoch the nearest base epoch, reused until a closer one
/// arrives, or null when it is older than ``maxage``.
class SyncObsHold {
 public:
  SyncObsHold(RinexObsReader& rov, RinexObsReader& base, double maxage);
  /// false at the end of the rover file.
  bool next(ObsPtr* obs, ObsPtr* obsb, double* dt);

 private:
  RinexObsReader& rov_;
  RinexObsReader& base_;
  double maxage_;
  ObsPtr curr_, next_;
};

}  // namespace gnss_fgo
