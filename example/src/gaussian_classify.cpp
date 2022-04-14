
#include "gaussian_classify.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Eigenvalues>

GaussianParams::GaussianParams(int dim)
  : mean(Vec(dim, 1)), cov_sqrt(Mat::Identity(dim, dim)) {}

GaussianParams::GaussianParams(const Vec &mean_, const Mat cov)
  : mean(mean_), cov_sqrt() {
  Eigen::SelfAdjointEigenSolver<Mat> solver(cov);
  cov_sqrt = solver.operatorSqrt();
}

GaussianDist::GaussianDist(int dim) : gauss(dim), n_dist(), rng(155) {}
GaussianDist::GaussianDist(const Vec &mean, const Mat cov)
    : gauss(mean, cov), n_dist(), rng(155) {}

GaussianMix::GaussianMix(std::vector<std::pair<double, GaussianParams>> params)
    : gauss_mix(), urd_dist(0.0, 1.0), n_dist(), rng(155) {
  double tot_weight = 0.0;
  for (const auto &[w, _] : params) {
    tot_weight += w;
  }
  for (auto [w, g] : params) {
    gauss_mix.push_back({w / tot_weight, g});
  }
}
