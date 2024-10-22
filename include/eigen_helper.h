#pragma once

#ifdef OPENBLAS_LIB
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#endif

#if defined(__NVCOMPILER) // WAR for nvc++ until we update to latest Eigen
#define EIGEN_DONT_VECTORIZE
#endif

#include <math.h>

// hide annoying warning
#if !defined(__clang__) && !defined(_NVHPC_CUDA)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/LU>

#if !defined(__clang__) && !defined(_NVHPC_CUDA)
#pragma GCC diagnostic pop
#endif

using namespace Eigen;
