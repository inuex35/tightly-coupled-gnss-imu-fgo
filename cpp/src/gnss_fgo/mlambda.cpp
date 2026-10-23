#include "gnss_fgo/mlambda.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace gnss_fgo::lambda {

namespace {

constexpr int kLoopMax = 10000;

int round_to_int(double v) { return int(std::nearbyint(v)); }  // np.rint

int signed_step(double v) { return v > 0.0 ? 1 : (v < 0.0 ? -1 : 0); }

}  // namespace

void ldldecom(const Eigen::MatrixXd& Q, Eigen::MatrixXd* Lout, Eigen::VectorXd* dout) {
  const int n = int(Q.rows());
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd A = Q;
  for (int i = n - 1; i >= 0; --i) {
    d[i] = A(i, i);
    if (d[i] <= 0.0) continue;
    const double sq = std::sqrt(d[i]);
    L.row(i).head(i + 1) = A.row(i).head(i + 1) / sq;
    for (int j = 0; j < i; ++j) A.row(j).head(j + 1) -= L.row(i).head(j + 1) * L(i, j);
    L.row(i).head(i + 1) /= L(i, i);
  }
  for (int i = 0; i < n; ++i)
    if (d[i] < 1e-10) throw LambdaError("Qah should be positive definite.");
  *Lout = L;
  *dout = d;
}

Eigen::MatrixXd reduction(Eigen::MatrixXd* Lp, Eigen::VectorXd* dp) {
  Eigen::MatrixXd& L = *Lp;
  Eigen::VectorXd& d = *dp;
  const int n = int(d.size());
  Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n, n);
  int j = n - 2, k = n - 2, loops = 0;
  while (j >= 0 && loops < kLoopMax) {
    ++loops;
    if (j <= k) {
      for (int i = j + 1; i < n; ++i) {
        const double mu = std::nearbyint(L(i, j));
        if (mu != 0.0) {
          for (int r = i; r < n; ++r) L(r, j) -= mu * L(r, i);
          for (int r = 0; r < n; ++r) Z(r, j) -= mu * Z(r, i);
        }
      }
    }
    const double delta = d[j] + L(j + 1, j) * L(j + 1, j) * d[j + 1];
    if (delta + 1e-6 < d[j + 1]) {
      const double eta = d[j] / delta;
      const double lam = d[j + 1] * L(j + 1, j) / delta;
      d[j] = eta * d[j + 1];
      d[j + 1] = delta;
      const double lj1j = L(j + 1, j);
      for (int col = 0; col < j; ++col) {
        const double Lj = L(j, col), Lj1 = L(j + 1, col);
        L(j, col) = -lj1j * Lj + Lj1;
        L(j + 1, col) = eta * Lj + lam * Lj1;
      }
      L(j + 1, j) = lam;
      for (int row = j + 2; row < n; ++row) std::swap(L(row, j), L(row, j + 1));
      for (int row = 0; row < n; ++row) std::swap(Z(row, j), Z(row, j + 1));
      k = j;
      j = n - 2;
    } else {
      --j;
    }
  }
  return Z;
}

double sr_boost(const Eigen::VectorXd& d) {
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  double prod = 1.0;
  for (int i = 0; i < d.size(); ++i) {
    const double val = 0.5 / std::sqrt(d[i]);
    const double cdf = 0.5 * (1.0 + std::erf(val * inv_sqrt2));
    prod *= 2.0 * cdf - 1.0;
  }
  return prod;
}

void estimILS(const Eigen::MatrixXd& L, const Eigen::VectorXd& d,
              const Eigen::VectorXd& ahat, int ncands, Eigen::MatrixXd* afixed_out,
              Eigen::VectorXd* sqnorm_out) {
  const int n = int(d.size());
  double Chi2 = 1e18;
  int loop_count = 0;
  bool aborted = false;
  const int k0 = (ncands == 1 && n > 1) ? 1 : 0;

  Eigen::MatrixXd afixed = Eigen::MatrixXd::Zero(n, ncands);
  Eigen::VectorXd sqnorm = Eigen::VectorXd::Constant(ncands, 1e18);
  std::vector<double> acond(n, 0.0), left(n, 0.0), dist(n, 0.0);
  std::vector<int> zcond(n, 0), step(n, 0), path(n, n - 1);
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, n);

  acond[n - 1] = ahat[n - 1];
  zcond[n - 1] = round_to_int(acond[n - 1]);
  left[n - 1] = acond[n - 1] - zcond[n - 1];
  step[n - 1] = signed_step(left[n - 1]);
  if (step[n - 1] == 0) step[n - 1] = 1;

  int count = -1, imax = ncands - 1;
  bool endSearch = false;
  int k = n - 1;

  while (!endSearch) {
    double newdist = dist[k] + left[k] * left[k] / d[k];
    while (newdist < Chi2) {
      if (++loop_count > kLoopMax) {
        aborted = true;
        endSearch = true;
        break;
      }
      if (k != 0) {
        --k;
        dist[k] = newdist;
        for (int jj = path[k]; jj > k; --jj) S(jj - 1, k) = S(jj, k) - left[jj] * L(jj, k);
        acond[k] = ahat[k] + S(k, k);
        zcond[k] = round_to_int(acond[k]);
        left[k] = acond[k] - zcond[k];
        step[k] = signed_step(left[k]);
        if (step[k] == 0) step[k] = 1;
      } else {
        if (count < ncands - 2) {
          ++count;
          for (int r = 0; r < n; ++r) afixed(r, count) = zcond[r];
          sqnorm[count] = newdist;
        } else {
          for (int r = 0; r < n; ++r) afixed(r, imax) = zcond[r];
          sqnorm[imax] = newdist;
          Eigen::Index im;
          sqnorm.maxCoeff(&im);  // first maximum, as np.argmax
          imax = int(im);
          Chi2 = sqnorm[imax];
        }
        k = k0;
        zcond[k] += step[k];
        left[k] = acond[k] - zcond[k];
        step[k] = -step[k] - signed_step(step[k]);
      }
      newdist = dist[k] + left[k] * left[k] / d[k];
    }
    if (aborted) break;

    const int ilevel = k;
    while (newdist >= Chi2) {
      if (++loop_count > kLoopMax) {
        aborted = true;
        endSearch = true;
        break;
      }
      if (k == n - 1) {
        endSearch = true;
        break;
      }
      ++k;
      zcond[k] += step[k];
      left[k] = acond[k] - zcond[k];
      step[k] = -step[k] - signed_step(step[k]);
      newdist = dist[k] + left[k] * left[k] / d[k];
    }
    if (aborted) break;

    for (int jj = ilevel; jj < k; ++jj) path[jj] = k;
    for (int jj = ilevel - 1; jj >= 0; --jj) {
      if (path[jj] < k) path[jj] = k;
      else break;
    }
  }

  std::vector<int> order(ncands);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return sqnorm[a] < sqnorm[b]; });
  afixed_out->resize(n, ncands);
  sqnorm_out->resize(ncands);
  for (int i = 0; i < ncands; ++i) {
    (*sqnorm_out)[i] = sqnorm[order[i]];
    afixed_out->col(i) = afixed.col(order[i]);
  }
}

Result mlambda(const Eigen::VectorXd& ahat, const Eigen::MatrixXd& Qahat, int ncands) {
  Eigen::MatrixXd L;
  Eigen::VectorXd d;
  ldldecom(Qahat, &L, &d);
  const Eigen::MatrixXd Z = reduction(&L, &d);
  const Eigen::MatrixXd iZt = Z.transpose().inverse().unaryExpr(
      [](double v) { return std::nearbyint(v); });
  const Eigen::VectorXd zhat = Z.transpose() * ahat;
  Result r;
  r.Ps = sr_boost(d);
  Eigen::MatrixXd zfix;
  estimILS(L, d, zhat, ncands, &zfix, &r.s);
  r.nfix = int(zhat.size());
  r.afix = iZt * zfix;
  return r;
}

}  // namespace gnss_fgo::lambda
