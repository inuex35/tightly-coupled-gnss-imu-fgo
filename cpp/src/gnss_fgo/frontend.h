// Double-difference measurement preparation, RTKLIB style: satellite
// states (satposs), exclusion (satexclude), elevation and SNR masks per
// receiver, and the common-satellite set. The estimator owns the state;
// this only edits and pairs observations.
#pragma once

#include "gnss_fgo/nav_state.h"
#include "gnss_fgo/rinex.h"

#include <optional>
#include <vector>

namespace gnss_fgo {

/// Inputs of one epoch for the factor builders.
struct DDMeasurements {
  ObsPtr obs, obsb;            // edited copies: masked slots are zero
  Eigen::MatrixXd rs, vs;      // rover satellite pos/vel, rows = obs.sat
  Eigen::MatrixXd rsb;         // base satellite pos, rows = obsb.sat
  std::vector<int> iu;         // rows of obs for the common satellites
  std::vector<int> ir;         // rows of obsb for the common satellites
  std::vector<int> sat;        // common satellites, ascending
  std::vector<double> el;      // rover elevations of ``sat`` [rad]
};

class DdFrontend {
 public:
  DdFrontend(const NavData& navdata, int nf) : navdata_(navdata), base_rcv_(nf) {}

  /// nullopt with fewer than 4 satellites or no base.
  std::optional<DDMeasurements> prepare(const ObsEpoch& obs, const ObsEpoch* obsb,
                                        NavState& nav);

 private:
  /// Edit one receiver's observations in place: a satellite RTKLIB
  /// excludes or below the elevation mask loses every slot, a slot below
  /// the SNR mask loses its code, phase and Doppler. Returns the
  /// satellites keeping at least one pseudorange (ascending) and records
  /// the elevations on ``rcv``.
  std::vector<int> edit(ObsEpoch& obs, const NavData::SatStates& st, const Vec3& rr,
                        const NavState& nav, ReceiverState& rcv) const;

  const NavData& navdata_;
  ReceiverState base_rcv_;
};

}  // namespace gnss_fgo
