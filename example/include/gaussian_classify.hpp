
#ifndef GAUSSIAN_CLASSIFY_HPP
#define GAUSSIAN_CLASSIFY_HPP

#include <random>
#include <utility>

#include <armadillo>

struct GaussianParams {
  GaussianParams(int dim);
  GaussianParams(const arma::vec &mean_, const arma::mat cov);
  arma::vec mean;
  // We don't directly store the eigenvalues; the lengths of these eigenvectors
  // specify the variance in their directions
  arma::mat cov_eigvecs;
};

class GaussianDist {
public:
  GaussianDist(int dim);
  GaussianDist(const arma::vec &mean, const arma::mat cov);

  template <class Generator> arma::vec operator()(Generator &g) {
    arma::vec m(gauss.mean.n_elem, arma::fill::zeros);
    for (int i = 0; i < gauss.mean.n_elem; ++i) {
      m(i) = n_dist(g);
    }
    return gauss.cov_eigvecs * m + gauss.mean;
  }

private:
  GaussianParams gauss;
  std::normal_distribution<> n_dist;
};

class GaussianMix {
public:
  GaussianMix(std::vector<std::pair<double, GaussianParams>> params);

  template <class Generator> arma::vec operator()(Generator &g) {
    const double g_selection = urd_dist(g);
    double seen_weight = 0.0;
    for (auto [weight, gauss] : gauss_mix) {
      seen_weight += weight;
      if (seen_weight > g_selection) {
        arma::vec m(gauss.mean.n_elem, arma::fill::zeros);
        for (int i = 0; i < gauss.mean.n_elem; ++i) {
          m(i) = n_dist(g);
        }
        return gauss.cov_eigvecs * m + gauss.mean;
      }
    }
  }

private:
  std::vector<std::pair<double, GaussianParams>> gauss_mix;

  std::uniform_real_distribution<double> urd_dist;
  std::normal_distribution<> n_dist;
};

#endif // GAUSSIAN_CLASSIFY_HPP
