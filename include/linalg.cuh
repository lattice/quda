#pragma once

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

       @param[in] A Input matrix we are decomposing
       @return Cholesky decomposition containing the lower triangular matrix
    */
    template < template<typename,int> class Mat, typename T, int N, bool fast=true>
    __device__ __host__ inline Mat<T,N> cholesky(const Mat<T,N> &A) {
      Mat<T,N> L;
      const auto &L_ = L;

#pragma unroll
      for (int i=0; i<N; i++) {
#pragma unroll
	for (int j=0; j<N; j++) if (j<i+1) {
	    complex<T> s = 0;
#pragma unroll
	    for (int k=0; k<N; k++) {
	      if (k==0) {
		s.x  = L_(i,k).real()*L_(j,k).real();
		s.x += L_(i,k).imag()*L_(j,k).imag();
		s.y  = L_(i,k).imag()*L_(j,k).real();
		s.y -= L_(i,k).real()*L_(j,k).imag();
	      } else if (k<j) {
		s.x += L_(i,k).real()*L_(j,k).real();
		s.x += L_(i,k).imag()*L_(j,k).imag();
		s.y += L_(i,k).imag()*L_(j,k).real();
		s.y -= L_(i,k).real()*L_(j,k).imag();
	      }
	    }
	    if (!fast) { // traditional Cholesky with sqrt and division
	      L(i,j) = (i == j) ? sqrt((A(i,i)-s).real()) : (A(i,j) - s) / L_(j,j).real();
	    } else { // optimized - since fwd/back subsitition only need inverse diagonal elements, avoid division and use rsqrt
	      L(i,j) = (i == j) ? rsqrt((A(i,i)-s).real()) : (A(i,j)-s) * L_(j,j).real();
	    }
	  }
      }

      return L;
    }

    /**
       @brief Forward substition to solve Lx = b.  L should have been
       previously generated, e.g., using a Cholesky decomposition.

       @tparam Vector The Vector container class, e.g., quda::colorspinor
       @tparam Mat The Matrix container class type.  This can either
       be a general Matrix (quda::Matrix) or a Hermitian matrix
       (quda::HMatrix).
       @tparam T The underlying type.  For Hermitian matrices this
       should be real type and for general matrices this should be the
       complex type.
       @tparam N The size of the linear system we are solving
       @tparam fast Whether to use the optimized forward-substitution algorithm
       that avoids division.  This flag must be consistent with the equivalent
       flag used with the prior decomposition.

       @param[in] L Lower triangular matrix for forward substitution
       @param[in] b Source vector
       @return solution vector
    */
    template <class Vector, template<typename,int> class Mat, typename T, int N, bool fast=true>
    __device__ __host__ inline Vector forward(const Mat<T,N> &L, const Vector &b) {
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
       @brief Backward ubstition to solve Ux = b.  U should have been
       previously generated, e.g., using a Cholesky decomposition.

       @tparam Vector The Vector container class, e.g., quda::colorspinor
       @tparam Mat The Matrix container class type.  This can either
       be a general Matrix (quda::Matrix) or a Hermitian matrix
       (quda::HMatrix).
       @tparam T The underlying type.  For Hermitian matrices this
       should be real type and for general matrices this should be the
       complex type.
       @tparam N The size of the linear system we are solving
       @tparam fast Whether to use the optimized backward-substitution algorithm
       that avoids division.  This flag must be consistent with the equivalent
       flag used with the prior decomposition.

       @param[in] U Upper triangular matrix for backward substitution
       @param[in] b Source vector
       @return solution vector
    */
    template <class Vector, template<typename,int> class Mat, typename T, int N, bool fast=true>
    __device__ __host__ inline Vector backward(const Mat<T,N> &U, const Vector &b) {
      Vector x;
#pragma unroll
      for (int i=N-1; i>=0; i--) {
	x(i) = b(i);
#pragma unroll
	for (int j=0; j<N; j++) if (j>=i+1) {
	  x(i).x -= U(i,j).real()*x(j).real();
	  x(i).x += U(i,j).imag()*x(j).imag();
	  x(i).y -= U(i,j).real()*x(j).imag();
	  x(i).y -= U(i,j).imag()*x(j).real();
	}
	if (!fast) x(i) /= U(i,i).real(); // traditional
	else x(i) *= U(i,i).real();       // optimized 
      }
      return x;
    }

  } // namespace linalg

} // namespace quda
