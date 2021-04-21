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
      __device__ __host__ inline Cholesky(const Mat<T,N> &A) {
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
	    if (!fast) { // traditional Cholesky with sqrt and division
	      L_(i,j) = (i == j) ? sqrt((A(i,i)-s).real()) : (A(i,j) - s) / L(j,j).real();
	    } else { // optimized - since fwd/back subsitition only need inverse diagonal elements, avoid division and use rsqrt
	      L_(i,j) = (i == j) ? quda::rsqrt((A(i,i)-s).real()) : (A(i,j)-s) * L(j,j).real();
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
	 @brief Forward substition to solve Lx = b
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
	 @brief Backward substition to solve L^dagger x = b
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
	 @brief Compute the inverse of A (the matrix used to construct the Cholesky decomposition).
	 @return Matrix inverse
      */
      __device__ __host__ inline Mat<T,N> invert() {
	const Mat<T,N> &L = L_;
	Mat<T,N> Ainv;
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

    /**
       @brief Compute eigen decomposition of A.  
       @tparam Mat The Matrix container class type.  This can either
       be a general Matrix (quda::Matrix) or a Hermitian matrix
       (quda::HMatrix).

       @tparam T The underlying type.
       @tparam N The size of the linear system we are solving
       @tparam tol The tolerance of the QR solver
       
    */
    template <template<typename,int> class Mat, typename T, int N, bool from_upper=false>
    class Eigensolve {
      
      //! The eigen decomposition
      Mat<T,N> evecs_;
      //! The triangular decomposition
      Mat<T,N> tri_;
      
    public:
      /**
	 @brief Constructor that computes the eigen decomposition
	 @param[in] A Input matrix we are decomposing
      */
      __device__ __host__ inline Eigensolve(const Mat<T,N> &A) {
	const Mat<T,N> &evecs = evecs_;
	
      }

      /**
	 @brief Return the eigenvalue element of the eigen decomposition Tri(i,i)
	 @param[in] i Index
	 @return Element at Tri(i,i)
      */
      __device__ __host__ inline const T eval(int i) const {
	const auto &tri = tri_;
	return tri(i,i);
      }

      /**
	 @brief Compute the upper Hessenberg reduction of A 
	 @return Matrix inverse
      */
      __device__ __host__ inline Mat<T,N> upperHessReduction(const Mat<T,N> &A) {
	typedef decltype(A(0,0).x) Real;
	
	Mat<T,N> &tri = tri_;
	tri = A; // Copy A into tri, upper hess reduce, then make triangular

	Mat<T,N> P; // Stores the reflector
	T rho;
	Real col_norm;
	int m = N;
	ColorSpinor<T,1,N> v; // vector of length (1 x N)
	
	for(int i = 0; i < m - 2; i++) {
	  
	  // get (partial) dot product of ith column vector
	  col_norm = static_cast<T>(0.0);
#pragma unroll
	  for(int j = i+1; j < m; j++) {
	    col_norm += tri(j,i).real() * tri(j,i).real();
	    col_norm += tri(j,i).imag() * tri(j,i).imag();
	  }
	  col_norm = sqrt(col_norm);
	  
	  rho = tri(i+1,i) / sqrt(tri(i+1,i).real() * tri(i+1,i).real() + tri(i+1,i).imag() * tri(i+1,i).imag());
	  v(i+1) = tri(i+1,i) - rho * col_norm;

	  // reuse col_norm
	  col_norm = v(i+1).real() * v(i+1).real() + v(i+1).imag() * v(i+1).imag();

	  // copy the rest of the column
#pragma unroll
	  for(int j = i + 2; j < m; j++ ) {
	    v(j) = tri(j,i);
	    col_norm += v(j).real() * v(j).real() + v(j).imag() * v(j).imag();
	  }
	  col_norm = sqrt(col_norm);
	  
	  // Normalise the column
#pragma unroll
	  for(int j = i + 1; j < m; j++) v(j) = col_norm;
	  
	  // Construct the householder matrix P = I - 2 U U*T
	  setIdentity(&P);
#pragma unroll
	  for(int j = i + 1; j < m; j++ ) {
	    for(int k = i + 1; k < m; k++ ) {
	      T P_elem;
	      P_elem.x  = v(j).real() * v(k).real();
	      P_elem.x += v(j).imag() * v(k).conj();
	      P_elem.y  = v(j).imag() * v(k).real();
	      P_elem.y -= v(j).real() * v(k).conj();
	      P(j,k) -= static_cast<T>(2.0) * P_elem;
	    }	    
	  }
	  
	  // Transform as PHP
	  tri = P * tri * P;
	}	
      }
    };
    
  } // namespace linalg

} // namespace quda
