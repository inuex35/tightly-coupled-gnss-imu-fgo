// The navigation state shared between the estimator and the GNSS
// front-end: what the Python port kept on cssrlib's ``nav``.
//
//   field          written by                 read by
//   -----          ----------                 -------
//   x.pos          stages / postfit           front-end (rover el), gates
//   x.amb          publish_marginals          ddidx (nonzero = estimate)
//   vsat           publish_marginals          ddidx selection, subset AR
//   el             front-end (rover)          ddidx mask, weights
//   fix            ddidx / holdamb_flags      hold policy
#pragma once

#include "gnss_fgo/common.h"
#include "gnss_fgo/gnss.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <vector>

namespace gnss_fgo {

/// Position + one ambiguity slot per (sat, freq): the cssrlib state
/// vector restricted to what this pipeline uses (``nav.x`` / ``xa``).
struct StateVec {
  Vec3 pos = Vec3::Zero();
  std::vector<double> amb;   // [f * MAXSAT + sat - 1]
  int nf = 0;

  StateVec() = default;
  explicit StateVec(int nf_) : amb(std::size_t(gnss::MAXSAT) * nf_, 0.0), nf(nf_) {}
  double& at(int sat, int f) { return amb[std::size_t(f) * gnss::MAXSAT + sat - 1]; }
  double at(int sat, int f) const { return amb[std::size_t(f) * gnss::MAXSAT + sat - 1]; }
};

/// Per-receiver editing memory.
struct ReceiverState {
  std::vector<double> el;   // last elevation per satellite [rad]
  /// Slots each satellite has produced (code + phase) this session.
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> band_seen;
  explicit ReceiverState(int nf = 0)
      : el(gnss::MAXSAT, 0.0),
        band_seen(Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
            gnss::MAXSAT, nf, false)) {}
};

struct NavState {
  int nf = 3;
  StateVec x;
  Eigen::MatrixXi vsat, fix;   // MAXSAT x nf
  ReceiverState rcv;           // rover; ``el`` is its elevation memory
  int smode = 0;               // 4 FIX, 5 FLT
  int armode = 3;
  double thresar = 3.0;
  double elmaskar = 20.0 * gnss::PI / 180.0;
  double elmin = 15.0 * gnss::PI / 180.0;
  double cnr_min = 25.0;          // SNR mask [dB-Hz], every slot, both receivers
  /// Admit a satellite only while every slot it has produced is usable
  /// (off: RTKLIB's per-slot editing alone).
  bool strict_sat_gate = true;
  int minfixsats = 4;
  Vec3 rb = Vec3::Zero();      // base position [ECEF m]
  gnss::GTime t;

  explicit NavState(int nf_ = 3)
      : nf(nf_), x(nf_),
        vsat(Eigen::MatrixXi::Zero(gnss::MAXSAT, nf_)),
        fix(Eigen::MatrixXi::Zero(gnss::MAXSAT, nf_)),
        rcv(nf_) {}

  double el(int sat) const { return rcv.el[sat - 1]; }

  /// SD->DD index: per (system, band) the lowest-numbered present
  /// satellite with an estimate, ``vsat`` set and elevation >= elmaskar is
  /// the reference; ``fix`` is rewritten (2: in a DD, 1: masked candidate
  /// met before the reference). Returns the (ref, target) slot pairs.
  std::vector<std::array<SfKey, 2>> ddidx(const std::vector<int>& sats);

  /// fix 2 -> 3 (held) everywhere; returns the count.
  int holdamb_flags();
};

}  // namespace gnss_fgo
