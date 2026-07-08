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
