
#ifndef GAUSSIAN_CLASSIFY_HPP
#define GAUSSIAN_CLASSIFY_HPP

#include <random>
#include <utility>

#include <Eigen/Core>

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

struct GaussianParams {
  GaussianParams(int dim);
  GaussianParams(const Vec &mean_, const Mat cov);
  size_t size() const { return mean.rows(); }

  Vec mean;
  // We don't directly store the eigenvalues; the lengths of these eigenvectors
  // specify the variance in their directions
  Mat cov_eigvecs;
};

class GaussianDist {
public:
  GaussianDist(int dim);
  GaussianDist(const Vec &mean, const Mat cov);

  template <class Generator> Vec operator()(Generator &g) {
    Vec m(gauss.size());
    for (int i = 0; i < gauss.size(); ++i) {
      m(i) = n_dist(g);
    }
    return gauss.cov_eigvecs * m + gauss.mean;
  }

  Vec operator()() { return (*this)(rng); }

private:
  GaussianParams gauss;
  std::normal_distribution<> n_dist;
  std::mt19937_64 rng;
};

class GaussianMix {
public:
  GaussianMix(std::vector<std::pair<double, GaussianParams>> params);

  GaussianMix(const GaussianMix &) = delete;
  GaussianMix &operator=(const GaussianMix &) = delete;

  template <class Generator> Vec operator()(Generator &g) {
    const double g_selection = urd_dist(g);
    double seen_weight = 0.0;
    for (auto [weight, gauss] : gauss_mix) {
      seen_weight += weight;
      if (seen_weight > g_selection) {
        Vec m(gauss.size());
        for (int i = 0; i < gauss.size(); ++i) {
          m(i) = n_dist(g);
        }
        return gauss.cov_eigvecs * m + gauss.mean;
      }
    }
    auto [weight, gauss] = gauss_mix.back();
    Vec m(gauss.size());
    for (int i = 0; i < gauss.size(); ++i) {
      m(i) = n_dist(g);
    }
    return gauss.cov_eigvecs * m + gauss.mean;
  }

  Vec operator()() { return (*this)(rng); }

  const std::vector<std::pair<double, GaussianParams>> &gaussians() const {
    return gauss_mix;
  }

private:
  std::vector<std::pair<double, GaussianParams>> gauss_mix;

  std::uniform_real_distribution<double> urd_dist;
  std::normal_distribution<> n_dist;
  std::mt19937_64 rng;
};

#endif // GAUSSIAN_CLASSIFY_HPP
