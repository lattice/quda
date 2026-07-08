#pragma once

#ifdef OPENBLAS_LIB
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
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

#ifdef QUDA_USE_QUAD_SCALAR
#include <eigen_quad_scalar.h>
#include <quda_internal.h>

// Fully qualify Eigen::Matrix / Eigen::Dynamic here: in translation units that
// also pull in quda's own Matrix type (e.g. multigrid block orthogonalization),
// unqualified Matrix<...> is ambiguous between Eigen::Matrix and quda::Matrix.
using EigMatrixXcd = Eigen::Matrix<quda::complex_t, Eigen::Dynamic, Eigen::Dynamic>;
using EigVectorXcd = Eigen::Matrix<quda::complex_t, Eigen::Dynamic, 1>;
using EigMatrixXd = Eigen::Matrix<quda::real_t, Eigen::Dynamic, Eigen::Dynamic>;
using EigVectorXd = Eigen::Matrix<quda::real_t, Eigen::Dynamic, 1>;

#else

using EigMatrixXcd = MatrixXcd;
using EigVectorXcd = VectorXcd;
using EigMatrixXd = MatrixXd;
using EigVectorXd = VectorXd;

#endif

// Precision used for the small dense decompositions inside the eigensolvers
// (arrow/projected matrix eigen-decompositions, Schur, LU used for Ritz
// rotations). Eigen's dense eigensolvers compute correct eigenVALUES but
// INCORRECT eigenVECTORS for extended precision (__float128 /
// complex<__float128>): eigenpair residuals are ~1e-2 versus ~1e-16 in double.
// These projected matrices only carry <= double precision (they are assembled
// from BLAS reductions over single/double device fields) and their
// eigenvectors merely produce rotation coefficients that are themselves
// truncated to device precision (float/double) before use, so we always solve
// these problems in double. In the standard (non-quad) build real_t == double,
// so this is exactly the pre-existing behavior.
#include <complex>

namespace quda
{
  using eig_solve_real_t = double;
  using eig_solve_complex_t = std::complex<double>;

  // Component-wise conversions between the host complex scalar (complex_t, which
  // is std::complex<__float128> in the quad build) and the double solve type:
  // std::complex has no viable converting constructor between complex<__float128>
  // and complex<double>. Templated so this header stays free of a hard
  // dependency on quda's real_t/complex_t (it is included by translation units
  // that do not pull in quda_internal.h).
  template <typename R> inline eig_solve_complex_t to_eig_solve(const std::complex<R> &z)
  {
    return eig_solve_complex_t(static_cast<double>(z.real()), static_cast<double>(z.imag()));
  }
  template <typename R> inline std::complex<R> from_eig_solve(const eig_solve_complex_t &z)
  {
    return std::complex<R>(static_cast<R>(z.real()), static_cast<R>(z.imag()));
  }
} // namespace quda
