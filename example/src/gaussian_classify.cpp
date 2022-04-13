
#include "gaussian_classify.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Eigenvalues>

GaussianParams::GaussianParams(int dim)
    : mean(Vec(dim, 1)), cov_eigvecs(dim, dim) {}

GaussianParams::GaussianParams(const Vec &mean_, const Mat cov)
    : mean(mean_), cov_eigvecs() {
  Eigen::EigenSolver<Mat> solver(cov);
  cov_eigvecs = solver.eigenvectors().real();
  const Vec eigvals = solver.eigenvalues().real();
  for (int i = 0; i < size(); ++i) {
    auto &&col = cov_eigvecs.col(i);
    col.normalize();
    col *= eigvals(i);
  }
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
