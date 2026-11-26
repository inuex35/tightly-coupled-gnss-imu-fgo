#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <optional>

namespace py = pybind11;

namespace gnss_fgo_factors {

constexpr double kOmge = 7.2921151467e-5;
constexpr double kClight = 299792458.0;

inline double geodist(const gtsam::Point3& rs, const gtsam::Point3& rr,
                      gtsam::Point3& e) {
  const gtsam::Vector3 diff = gtsam::Vector3(rs - rr);
  const double r = diff.norm();
  e = gtsam::Point3(diff / r);
  return r + kOmge * (rs.x() * rr.y() - rs.y() * rr.x()) / kClight;
}

inline void initGeometry(const gtsam::Point3& satPosTarget,
                         const gtsam::Point3& satVelTarget,
                         const gtsam::Point3& satPosRef,
                         const gtsam::Point3& satVelRef,
                         const gtsam::Point3& receiverPosition,
                         double satClkDriftTarget, double satClkDriftRef,
                         double measRangeRateDiff, double& offset,
                         gtsam::Point3& velCoeff) {
  gtsam::Point3 eTarget, eRef;
  geodist(satPosTarget, receiverPosition, eTarget);
  geodist(satPosRef, receiverPosition, eRef);

  const double k = kOmge / kClight;
  const double sagnacOffsetTarget =
      k * (satVelTarget.y() * receiverPosition.x() -
           satVelTarget.x() * receiverPosition.y());
  const double sagnacOffsetRef =
      k * (satVelRef.y() * receiverPosition.x() -
           satVelRef.x() * receiverPosition.y());
  const gtsam::Point3 velSagnacTarget(k * satPosTarget.y(),
                                      -k * satPosTarget.x(), 0.0);
  const gtsam::Point3 velSagnacRef(k * satPosRef.y(), -k * satPosRef.x(), 0.0);

  offset = eTarget.dot(satVelTarget) - eRef.dot(satVelRef) +
           sagnacOffsetTarget - sagnacOffsetRef -
           kClight * (satClkDriftTarget - satClkDriftRef) - measRangeRateDiff;
  velCoeff = (velSagnacTarget - velSagnacRef) - (eTarget - eRef);
}

class SingleDifferenceDopplerFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector3> {
 private:
  using Base = gtsam::NoiseModelFactorN<gtsam::Vector3>;
  double offset_ = 0.0;
  gtsam::Point3 velCoeff_{0, 0, 0};

 public:
  using Base::evaluateError;
  using shared_ptr = std::shared_ptr<SingleDifferenceDopplerFactor>;

  SingleDifferenceDopplerFactor(
      gtsam::Key velocityKey, double measuredDopplerTarget,
      double measuredDopplerRef, double wavelengthTarget, double wavelengthRef,
      const gtsam::Point3& satPosTarget, const gtsam::Point3& satVelTarget,
      const gtsam::Point3& satPosRef, const gtsam::Point3& satVelRef,
      const gtsam::Point3& receiverPosition, double satClkDriftTarget,
      double satClkDriftRef, const gtsam::SharedNoiseModel& model)
      : Base(model, velocityKey) {
    const double measRangeRateDiff =
        -wavelengthTarget * measuredDopplerTarget +
        wavelengthRef * measuredDopplerRef;
    initGeometry(satPosTarget, satVelTarget, satPosRef, satVelRef,
                 receiverPosition, satClkDriftTarget, satClkDriftRef,
                 measRangeRateDiff, offset_, velCoeff_);
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<SingleDifferenceDopplerFactor>(*this);
  }

  gtsam::Vector evaluateError(
      const gtsam::Vector3& velocity,
      gtsam::OptionalMatrixType Hvelocity) const override {
    if (Hvelocity) *Hvelocity = gtsam::Vector3(velCoeff_).transpose();
    return gtsam::Vector1(offset_ + gtsam::Vector3(velCoeff_).dot(velocity));
  }
};

class SingleDifferenceDopplerFactorArm
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3> {
 private:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3>;
  double offset_ = 0.0;
  gtsam::Point3 velCoeff_{0, 0, 0};
  gtsam::Point3 leverArm_{0, 0, 0};
  std::optional<gtsam::Pose3> ecefTnav_;
  gtsam::Point3 leverVel_{0, 0, 0};

 public:
  using Base::evaluateError;
  using shared_ptr = std::shared_ptr<SingleDifferenceDopplerFactorArm>;

  SingleDifferenceDopplerFactorArm(
      gtsam::Key poseKey, gtsam::Key velocityKey, double measuredDopplerTarget,
      double measuredDopplerRef, double wavelengthTarget, double wavelengthRef,
      const gtsam::Point3& satPosTarget, const gtsam::Point3& satVelTarget,
      const gtsam::Point3& satPosRef, const gtsam::Point3& satVelRef,
      const gtsam::Point3& receiverPosition, const gtsam::Point3& leverArm,
      const gtsam::Point3& angularVelocity, double satClkDriftTarget,
      double satClkDriftRef, const gtsam::SharedNoiseModel& model)
      : Base(model, poseKey, velocityKey), leverArm_(leverArm) {
    const double measRangeRateDiff =
        -wavelengthTarget * measuredDopplerTarget +
        wavelengthRef * measuredDopplerRef;
    initGeometry(satPosTarget, satVelTarget, satPosRef, satVelRef,
                 receiverPosition, satClkDriftTarget, satClkDriftRef,
                 measRangeRateDiff, offset_, velCoeff_);
    leverVel_ = angularVelocity.cross(leverArm);
  }

  SingleDifferenceDopplerFactorArm(
      gtsam::Key poseKey, gtsam::Key velocityKey, double measuredDopplerTarget,
      double measuredDopplerRef, double wavelengthTarget, double wavelengthRef,
      const gtsam::Point3& satPosTarget, const gtsam::Point3& satVelTarget,
      const gtsam::Point3& satPosRef, const gtsam::Point3& satVelRef,
      const gtsam::Point3& receiverPosition, const gtsam::Point3& leverArm,
      const gtsam::Pose3& ecef_T_nav, const gtsam::Point3& angularVelocity,
      double satClkDriftTarget, double satClkDriftRef,
      const gtsam::SharedNoiseModel& model)
      : Base(model, poseKey, velocityKey),
        leverArm_(leverArm),
        ecefTnav_(ecef_T_nav) {
    const double measRangeRateDiff =
        -wavelengthTarget * measuredDopplerTarget +
        wavelengthRef * measuredDopplerRef;
    initGeometry(satPosTarget, satVelTarget, satPosRef, satVelRef,
                 receiverPosition, satClkDriftTarget, satClkDriftRef,
                 measRangeRateDiff, offset_, velCoeff_);
    leverVel_ = angularVelocity.cross(leverArm);
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<SingleDifferenceDopplerFactorArm>(*this);
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
      gtsam::OptionalMatrixType Hpose,
      gtsam::OptionalMatrixType Hvelocity) const override {
    gtsam::Matrix3 Hrot;
    gtsam::Matrix3 Rvel = gtsam::I_3x3;
    gtsam::Point3 leverVelEcef;
    gtsam::Vector3 velEcef;
    if (ecefTnav_) {
      gtsam::Matrix3 Hinner;
      const gtsam::Point3 vNav = pose.rotation().rotate(leverVel_, Hinner);
      Rvel = ecefTnav_->rotation().matrix();
      leverVelEcef = ecefTnav_->rotation().rotate(vNav);
      Hrot = Rvel * Hinner;
      velEcef = Rvel * velocity;
    } else {
      leverVelEcef = pose.rotation().rotate(leverVel_, Hrot);
      velEcef = velocity;
    }
    const gtsam::Vector3 vAnt = velEcef + gtsam::Vector3(leverVelEcef);
    const gtsam::Vector3 g = gtsam::Vector3(velCoeff_);
    if (Hpose) {
      gtsam::Matrix16 H = gtsam::Matrix16::Zero();
      H.block<1, 3>(0, 0) = g.transpose() * Hrot;
      *Hpose = H;
    }
    if (Hvelocity) *Hvelocity = g.transpose() * Rvel;
    return gtsam::Vector1(offset_ + g.dot(vAnt));
  }
};

inline gtsam::Vector2 lateralVerticalVelocity(
    const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
    const gtsam::Rot3& mountRotation, const gtsam::Vector3& angularVelocity,
    const gtsam::Vector3& leverArm, gtsam::OptionalMatrixType H_pose,
    gtsam::OptionalMatrixType H_velocity,
    gtsam::OptionalMatrixType H_mountRotation) {
  const gtsam::Matrix3 bRw = pose.rotation().matrix().transpose();
  const gtsam::Vector3 bodyVelocity = bRw * velocity;
  const gtsam::Vector3 pointVelocity =
      bodyVelocity + angularVelocity.cross(leverArm);

  gtsam::Matrix3 H_unrotate;
  const gtsam::Vector3 vehicleVelocity =
      mountRotation.unrotate(pointVelocity, H_unrotate);
  const gtsam::Matrix3 vRb = mountRotation.matrix().transpose();

  if (H_pose) {
    gtsam::Matrix26 H;
    H << (vRb * gtsam::skewSymmetric(bodyVelocity)).bottomRows<2>(),
        gtsam::Matrix23::Zero();
    *H_pose = H;
  }
  if (H_velocity) *H_velocity = gtsam::Matrix23((vRb * bRw).bottomRows<2>());
  if (H_mountRotation)
    *H_mountRotation = gtsam::Matrix23(H_unrotate.bottomRows<2>());
  return vehicleVelocity.tail<2>();
}

class NhcFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3> {
 private:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3>;
  gtsam::Vector3 angularVelocity_;
  gtsam::Vector3 leverArm_;
  gtsam::Rot3 mountRotation_;

 public:
  using Base::evaluateError;
  using shared_ptr = std::shared_ptr<NhcFactor>;

  NhcFactor(gtsam::Key poseKey, gtsam::Key velocityKey,
            const gtsam::Vector3& angularVelocity,
            const gtsam::Vector3& leverArm,
            const gtsam::SharedNoiseModel& model,
            const gtsam::Rot3& mountRotation = gtsam::Rot3())
      : Base(model, poseKey, velocityKey),
        angularVelocity_(angularVelocity),
        leverArm_(leverArm),
        mountRotation_(mountRotation) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NhcFactor>(*this);
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
      gtsam::OptionalMatrixType H_pose,
      gtsam::OptionalMatrixType H_velocity) const override {
    return lateralVerticalVelocity(pose, velocity, mountRotation_,
                                   angularVelocity_, leverArm_, H_pose,
                                   H_velocity, nullptr);
  }
};

class NhcFactorCalib
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3,
                                      gtsam::Rot3> {
 private:
  using Base =
      gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Rot3>;
  gtsam::Vector3 angularVelocity_;
  gtsam::Vector3 leverArm_;

 public:
  using Base::evaluateError;
  using shared_ptr = std::shared_ptr<NhcFactorCalib>;

  NhcFactorCalib(gtsam::Key poseKey, gtsam::Key velocityKey,
                 gtsam::Key mountRotationKey,
                 const gtsam::Vector3& angularVelocity,
                 const gtsam::Vector3& leverArm,
                 const gtsam::SharedNoiseModel& model)
      : Base(model, poseKey, velocityKey, mountRotationKey),
        angularVelocity_(angularVelocity),
        leverArm_(leverArm) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NhcFactorCalib>(*this);
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
      const gtsam::Rot3& mountRotation, gtsam::OptionalMatrixType H_pose,
      gtsam::OptionalMatrixType H_velocity,
      gtsam::OptionalMatrixType H_mountRotation) const override {
    return lateralVerticalVelocity(pose, velocity, mountRotation,
                                   angularVelocity_, leverArm_, H_pose,
                                   H_velocity, H_mountRotation);
  }
};

}  // namespace gnss_fgo_factors

PYBIND11_MODULE(_extra_factors, m) {
  py::module_::import("gtsam");

  py::class_<gnss_fgo_factors::SingleDifferenceDopplerFactor,
             gtsam::NoiseModelFactor,
             std::shared_ptr<gnss_fgo_factors::SingleDifferenceDopplerFactor>>(
      m, "SingleDifferenceDopplerFactor")
      .def(py::init<gtsam::Key, double, double, double, double,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, double, double,
                    const gtsam::SharedNoiseModel&>())
      .def("evaluateError",
           [](const gnss_fgo_factors::SingleDifferenceDopplerFactor& self,
              const gtsam::Vector3& v) {
             return self.evaluateError(v, nullptr);
           });

  py::class_<
      gnss_fgo_factors::SingleDifferenceDopplerFactorArm,
      gtsam::NoiseModelFactor,
      std::shared_ptr<gnss_fgo_factors::SingleDifferenceDopplerFactorArm>>(
      m, "SingleDifferenceDopplerFactorArm")
      .def(py::init<gtsam::Key, gtsam::Key, double, double, double, double,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, double, double,
                    const gtsam::SharedNoiseModel&>())
      .def(py::init<gtsam::Key, gtsam::Key, double, double, double, double,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Point3&, const gtsam::Point3&,
                    const gtsam::Pose3&, const gtsam::Point3&, double, double,
                    const gtsam::SharedNoiseModel&>())
      .def("evaluateError",
           [](const gnss_fgo_factors::SingleDifferenceDopplerFactorArm& self,
              const gtsam::Pose3& p, const gtsam::Vector3& v) {
             return self.evaluateError(p, v, nullptr, nullptr);
           });

  py::class_<gnss_fgo_factors::NhcFactor, gtsam::NoiseModelFactor,
             std::shared_ptr<gnss_fgo_factors::NhcFactor>>(m, "NhcFactor")
      .def(py::init<gtsam::Key, gtsam::Key, const gtsam::Vector3&,
                    const gtsam::Vector3&, const gtsam::SharedNoiseModel&,
                    const gtsam::Rot3&>(),
           py::arg("poseKey"), py::arg("velocityKey"),
           py::arg("angularVelocity"), py::arg("leverArm"), py::arg("model"),
           py::arg("mountRotation") = gtsam::Rot3())
      .def("evaluateError",
           [](const gnss_fgo_factors::NhcFactor& self, const gtsam::Pose3& p,
              const gtsam::Vector3& v) {
             return self.evaluateError(p, v, nullptr, nullptr);
           });

  py::class_<gnss_fgo_factors::NhcFactorCalib, gtsam::NoiseModelFactor,
             std::shared_ptr<gnss_fgo_factors::NhcFactorCalib>>(
      m, "NhcFactorCalib")
      .def(py::init<gtsam::Key, gtsam::Key, gtsam::Key, const gtsam::Vector3&,
                    const gtsam::Vector3&, const gtsam::SharedNoiseModel&>())
      .def("evaluateError",
           [](const gnss_fgo_factors::NhcFactorCalib& self,
              const gtsam::Pose3& p, const gtsam::Vector3& v,
              const gtsam::Rot3& r) {
             return self.evaluateError(p, v, r, nullptr, nullptr, nullptr);
           });
}
