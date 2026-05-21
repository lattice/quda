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

using EigMatrixXcd = Matrix<quda::complex_t, Dynamic, Dynamic>;
using EigVectorXcd = Matrix<quda::complex_t, Dynamic, 1>;
using EigMatrixXd = Matrix<quda::real_t, Dynamic, Dynamic>;
using EigVectorXd = Matrix<quda::real_t, Dynamic, 1>;

#else

using EigMatrixXcd = MatrixXcd;
using EigVectorXcd = VectorXcd;
using EigMatrixXd = MatrixXd;
using EigVectorXd = VectorXd;

#endif
