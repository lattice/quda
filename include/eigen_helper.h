#pragma once

#ifdef OPENBLAS_LIB
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#endif

#if defined(__NVCOMPILER) // WAR for nvc++ until we update to latest Eigen
#define EIGEN_DONT_VECTORIZE
#endif

#include <math.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/LU>

using namespace Eigen;
