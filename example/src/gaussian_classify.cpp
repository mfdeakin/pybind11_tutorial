
#include "gaussian_classify.hpp"

#include <cmath>
#include <stdexcept>

GaussianParams::GaussianParams(int dim)
    : mean(arma::vec(dim, arma::fill::zeros)),
      cov_eigvecs(dim, dim, arma::fill::eye) {}

GaussianParams::GaussianParams(const arma::vec &mean_, const arma::mat cov)
    : mean(mean_), cov_eigvecs() {
  if (!cov.is_symmetric()) {
    throw std::runtime_error("Covariance matrix is not symmetric");
  }
  arma::vec eigvals;
  bool err = arma::eig_sym(eigvals, cov_eigvecs, cov);
  if (err) {
    throw std::runtime_error("Error decomposing the covariance matrix");
  }
  for (int i = 0; i < mean.n_elem; ++i) {
    arma::vec col = cov_eigvecs.col(i);
    const double norm = arma::norm(col);
    col *= eigvals(i) / norm;
  }
}

GaussianDist::GaussianDist(int dim) : gauss(dim), n_dist() {}
GaussianDist::GaussianDist(const arma::vec &mean, const arma::mat cov)
    : gauss(mean, cov), n_dist() {}

GaussianMix::GaussianMix(std::vector<std::pair<double, GaussianParams>> params)
    : gauss_mix(), urd_dist(0.0, 1.0), n_dist() {
  double tot_weight = 0.0;
  for (auto [w, _] : params) {
    tot_weight += w;
  }
  for (auto [w, g] : params) {
    gauss_mix.push_back({w, g});
  }
}
