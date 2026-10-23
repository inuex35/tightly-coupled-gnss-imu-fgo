// Integer least squares by MLAMBDA (Chang et al. 2005) with the
// search-and-shrink ILS estimator (Massarweh et al. 2025). Loop-for-loop
// port of the cssrlib kernels the Python pipeline ran.
#pragma once

#include <Eigen/Core>

#include <stdexcept>

namespace gnss_fgo::lambda {

/// Non positive-definite covariance and similar failures.
struct LambdaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Q = L' diag(d) L; throws LambdaError when Q is not positive definite.
void ldldecom(const Eigen::MatrixXd& Q, Eigen::MatrixXd* L, Eigen::VectorXd* d);
/// LLL-type decorrelation; L, d updated in place, Z returned.
Eigen::MatrixXd reduction(Eigen::MatrixXd* L, Eigen::VectorXd* d);
/// Bootstrapped success rate.
double sr_boost(const Eigen::VectorXd& d);
/// Best ``ncands`` integer vectors (columns) and their squared norms.
void estimILS(const Eigen::MatrixXd& L, const Eigen::VectorXd& d,
              const Eigen::VectorXd& ahat, int ncands, Eigen::MatrixXd* afixed,
              Eigen::VectorXd* sqnorm);

struct Result {
  Eigen::MatrixXd afix;   // n x ncands, best first
  Eigen::VectorXd s;      // squared norms, ascending
  int nfix = 0;
  double Ps = 0.0;
};
/// Full-ILS MLAMBDA (parmode 1).
Result mlambda(const Eigen::VectorXd& ahat, const Eigen::MatrixXd& Qahat, int ncands = 2);

}  // namespace gnss_fgo::lambda
