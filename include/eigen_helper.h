#pragma once

#ifdef OPENBLAS_LIB
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#endif

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

#ifdef QUDA_USE_QUAD_SCALAR
#include <eigen_quad_scalar.h>
#endif

#include <quda_internal.h>

namespace quda
{

  // Fully qualify Eigen::Matrix / Eigen::Dynamic: in translation units that also
  // pull in quda's own Matrix type (e.g. multigrid), unqualified Matrix<...> is
  // ambiguous between Eigen::Matrix and quda::Matrix.
  using MatrixX = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXc = Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
  using VectorXc = Eigen::Matrix<complex_t, Eigen::Dynamic, 1>;

} // namespace quda
