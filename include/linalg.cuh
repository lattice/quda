#pragma once
#include <color_spinor.h> // vector container
#include <math_helper.cuh>

/**
   @file linalg.cuh

   @section DESCRIPTION

   This file contains implementations of basic dense linear algebra
   methods that can be called by either a CPU thread or a GPU thread.
   At present, only a Cholesky decomposition, together with backward
   and forward substitution is implemented.
 */

namespace quda {

  namespace linalg {

    /**
       @brief Compute Cholesky decomposition of A.  By default, we use a
       modified Cholesky which avoids the division and sqrt, and instead
       only needs rsqrt.  In which case we must use a modified forward
       and backward difference substitution.

       @tparam Mat The Matrix container class type.  This can either
       be a general Matrix (quda::Matrix) or a Hermitian matrix
       (quda::HMatrix).
       @tparam T The underlying type.  For Hermitian matrices this
       should be real type and for general matrices this should be the
       complex type.
       @tparam N The size of the linear system we are solving
       @tparam fast Whether to use the optimized Cholesky algorithm
       that stores the reciprocal sqrt on the diagonal of L, instead
       of the sqrt.  This flag must be consistent with the equivalent
       flag used with the forward and backward substitutions.
    */
    template <template<typename,int> class Mat, typename T, int N, bool fast=true>
    class Cholesky {

      //! The Cholesky factorization
      Mat<T,N> L_;

    public:
      /**
	 @brief Constructor that computes the Cholesky decomposition
	 @param[in] A Input matrix we are decomposing
      */

      template <typename matrix_t> __device__ __host__ inline Cholesky(const matrix_t &A)
      {
	const Mat<T,N> &L = L_;

#pragma unroll
	for (int i=0; i<N; i++) {
#pragma unroll
	  for (int j=0; j<N; j++) if (j<i+1) {
	    complex<T> s = 0;
#pragma unroll
	    for (int k=0; k<N; k++) {
	      if (k==0) {
		s.x  = L(i,k).real()*L(j,k).real();
		s.x += L(i,k).imag()*L(j,k).imag();
		s.y  = L(i,k).imag()*L(j,k).real();
		s.y -= L(i,k).real()*L(j,k).imag();
	      } else if (k<j) {
		s.x += L(i,k).real()*L(j,k).real();
		s.x += L(i,k).imag()*L(j,k).imag();
		s.y += L(i,k).imag()*L(j,k).real();
		s.y -= L(i,k).real()*L(j,k).imag();
	      }
	    }
            complex<T> A_ij(A(i, j));
	    if (!fast) { // traditional Cholesky with sqrt and division
	      L_(i, j) = (i == j) ? sqrt((A_ij - s).real()) : (A_ij - s) / L(j,j).real();
	    } else { // optimized - since fwd/back subsitition only need inverse diagonal elements, avoid division and use rsqrt
	      L_(i, j) = (i == j) ? quda::rsqrt((A_ij - s).real()) : (A_ij - s) * L(j, j).real();
	    }
	  }
	}
      }

      /**
	 @brief Return the diagonal element of the Cholesky decomposition L(i,i)
	 @param[in] i Index
	 @return Element at L(i,i)
      */
      __device__ __host__ inline const T D(int i) const {
	const auto &L = L_;
	if (!fast) return L(i,i).real();
	else return static_cast<T>(1.0) / L(i,i).real();
      }

      /**
	 @brief Forward substitution to solve Lx = b
	 @tparam Vector The Vector container class, e.g., quda::colorspinor
	 @param[in] b Source vector
	 @return solution vector
      */
      template <class Vector>
      __device__ __host__ inline Vector forward(const Vector &b) {
	const Mat<T,N> &L = L_;
	Vector x;
#pragma unroll
	for (int i=0; i<N; i++) {
	  x(i) = b(i);
#pragma unroll
	  for (int j=0; j<N; j++) if (j<i) {
	    x(i).x -= L(i,j).real()*x(j).real();
	    x(i).x += L(i,j).imag()*x(j).imag();
	    x(i).y -= L(i,j).real()*x(j).imag();
	    x(i).y -= L(i,j).imag()*x(j).real();
	  }
	  if (!fast) x(i) /= L(i,i).real(); // traditional
	  else x(i) *= L(i,i).real();       // optimized
	}
	return x;
      }

      /**
	 @brief Backward substitution to solve L^dagger x = b
	 @tparam Vector The Vector container class, e.g., quda::colorspinor
	 @param[in] b Source vector
	 @return solution vector
      */
      template <class Vector>
      __device__ __host__ inline Vector backward(const Vector &b) {
	const Mat<T,N> &L = L_;
	Vector x;
#pragma unroll
	for (int i=N-1; i>=0; i--) {
	  x(i) = b(i);
#pragma unroll
	  for (int j=0; j<N; j++) if (j>=i+1) {
	    x(i).x -= L(i,j).real()*x(j).real();
	    x(i).x += L(i,j).imag()*x(j).imag();
	    x(i).y -= L(i,j).real()*x(j).imag();
	    x(i).y -= L(i,j).imag()*x(j).real();
	  }
	  if (!fast) x(i) /=L(i,i).real(); // traditional
	  else x(i) *= L(i,i).real();      // optimized
	}
	return x;
      }

      /**
         @brief Solve Ax = b using the Cholesky factorization.  This
         is just a wrapper around the forward and backward
         substitution functions with the ability to change precision
         in the solve.
	 @tparam Vector The Vector container class, e.g., quda::colorspinor
	 @param[in] b Source vector
	 @return solution vector
      */
      template <class Vector>
      __device__ __host__ inline Vector solve(const Vector &b)
      {
        // copy source vector into factorization precision
	ColorSpinor<T,1,N> b_;
#pragma unroll
        for (int i = 0; i < N; i++) b_(i) = b(i);

        auto x_ = backward(forward(b_));

        // copy solution vector into desired precision
        Vector x;
#pragma unroll
        for (int i = 0; i < N; i++) x(i) = x_(i);

        return x;
      }

      /**
	 @brief Compute the inverse of A (the matrix used to construct the Cholesky decomposition).
	 @return Matrix inverse
      */
      template <typename matrix_t> __device__ __host__ inline matrix_t invert() {
	const Mat<T,N> &L = L_;
	matrix_t Ainv;
	ColorSpinor<T,1,N> v;

#pragma unroll
	for (int k=0;k<N;k++) {

	  // forward substitute
	  if (!fast) v(k) = complex<T>(static_cast<T>(1.0)/L(k,k).real());
	  else v(k) = L(k,k).real();

#pragma unroll
	  for (int i=0; i<N; i++) if (i>k) {
	    v(i) = complex<T>(0.0);
#pragma unroll
	    for (int j=0; j<N; j++) if (j>=k && j<i) {
	      v(i).x -= L(i,j).real() * v(j).real();
	      v(i).x += L(i,j).imag() * v(j).imag();
	      v(i).y -= L(i,j).real() * v(j).imag();
	      v(i).y -= L(i,j).imag() * v(j).real();
	    }
	    if (!fast) v(i) *= static_cast<T>(1.0) / L(i,i);
	    else v(i) *= L(i,i);
	  }

	  // backward substitute
	  if (!fast) v(N-1) *= static_cast<T>(1.0) / L(N-1,N-1);
	  else v(N-1) *= L(N-1,N-1);

#pragma unroll
	  for (int i=N-2; i>=0; i--) if (i>=k) {
#pragma unroll
	    for (int j=0; j<N; j++) if (j>i) {
	      v(i).x -= L(i,j).real() * v(j).real();
	      v(i).x += L(i,j).imag() * v(j).imag();
	      v(i).y -= L(i,j).real() * v(j).imag();
	      v(i).y -= L(i,j).imag() * v(j).real();
	    }
	    if (!fast) v(i) *= static_cast<T>(1.0) / L(i,i);
	    else v(i) *= L(i,i);
	  }

	  // Overwrite column k
	  Ainv(k,k) = v(k);

#pragma unroll
	  for(int i=0;i<N;i++) if (i>k) Ainv(i,k) = v(i);
	}

	return Ainv;
      }

    };

  } // namespace linalg

} // namespace quda
