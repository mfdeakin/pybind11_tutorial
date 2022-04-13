
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include "gaussian_classify.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gda_bindings, module) {
  module.doc() = "Gaussian Discriminant Analysis library";
  py::class_<GaussianParams> gauss_params(module, "GaussianParams");
  gauss_params.def(py::init<int>(), "Constructs an isotropic Guassian");
  gauss_params.def(py::init<Vec, Mat>(),
                   "Generic parameters for a multivariate Guassian");
  py::class_<GaussianDist> gauss_dist(module, "GaussianDist");
  gauss_dist.def(py::init<int>(), "Constructs an isotropic Guassian");
  gauss_dist.def(py::init<Vec, Mat>(),
                 "Generic parameters for a multivariate Guassian");
  gauss_dist.def(
      "__call__", [](GaussianDist &gd) { return gd(); },
      "Samples the distribution");
  py::class_<GaussianMix> gauss_mix(module, "GaussianMix");
  gauss_mix.def(py::init<std::vector<std::pair<double, GaussianParams>>>(),
                "Constructs a Gaussian mixture model with these Gaussians and "
                "their associated weights");
  gauss_mix.def(
      "__call__", [](GaussianMix &gd) { return gd(); },
      "Samples the distribution");
}
