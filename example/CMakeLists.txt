
cmake_minimum_required(VERSION 3.13.0)

project(gda)

find_package(Eigen3 REQUIRED)

add_library(GaussianClassify SHARED src/gaussian_classify.cpp)
target_include_directories(GaussianClassify PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})

find_package(pybind11 CONFIG)

pybind11_add_module(gda_bindings pybindings/pybindings.cxx)
target_link_libraries(gda_bindings PUBLIC GaussianClassify)
