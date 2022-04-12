
cmake_minimum_required(VERSION 3.13.0)

project(gda)

find_package(Armadillo REQUIRED)
# find_package(pybind11 CONFIG)

add_library(GaussianDist SHARED src/gaussian_classify.cpp)
target_include_directories(GaussianDist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ARMADILLO_INCLUDE_DIRS})
