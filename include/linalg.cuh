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
    */
    template <template<typename,int> class Mat, typename T, int N>
    class Eigensolve {

      //! The eigenvalues
      ColorSpinor<T,1,N> eval_; 
      //! The eigen decomposition of input matrix A
      Mat<T,N> E_;
      //! The upper Hessenberg decomposition of A
      Mat<T,N> H_;
      //! The Householder reflectors
      Mat<T,N> P_;
      //! The Q rotations
      Mat<T,N> Q_;
      //! QR tolerance
      double tol_;
      
    public:
      /**
	 @brief Constructor that computes the eigen decomposition
	 @param[in] A Input matrix we are decomposing
	 @param[in] tol Tolerance on the QR iterations
	 @param[in] from_upper_hessenberg If true, omit the upper 
	 Hessenberg decomposition
      */
      __device__ __host__ inline Eigensolve(const Mat<T,N> &A, double tol, bool from_upper_hessenberg=false)
      {
	Mat<T,N> &H = H_;
	Mat<T,N> &P = P_;
	Mat<T,N> &Q = Q_;
	Mat<T,N> &E = E_;
	ColorSpinor<T,1,N> &eval = eval_;
	tol_ = tol;
	
	// Reduce A to upper Hessenberg in H
	H = A;
	setIdentity(&P);
	if(!from_upper_hessenberg) upperHessReduction(A, H, P);
	
	// QR the upper Hessenberg matrix into upper triangular form;
	qrUpperHess(Q, H);
	// Accumulate rotations
	P = P * Q;
	
	// eigensolve the upper triangular matrix by back substitution
	eigensolveUpperTri(E, H);	
	// Rotate the eigenvectors
	E = P * E;
	
	// Update the eigenvalue array
	for(int i=0; i<N; i++) eval(i) = H(i,i);	
      }
      
      /**
	 @brief Return the ith eigenvalue of the eigen decomposition
	 @param[in] i Index
	 @return Element at H(i,i)
      */
      __device__ __host__ inline const T eval(int i) const {
	const auto &eval = eval_;
	return eval(i);
      }
      
      /**
	 @brief Return the eigenvalue array of the eigen decomposition
	 @return eval The array of eigenvalues 
      */
      template <class Vector>
      __device__ __host__ inline const Vector evals() const {
	const auto &eval = eval_;
	return eval;
      }

      /**
	 @brief Return the eigendecomposition of A
	 @return E the matrix of eigenvectors
      */
      __device__ __host__ inline const Mat<T,N> evecs() const {
	const auto &E = E_;
	return E;
      }

      /**
	 @brief Return the exponent S of A: a = exp(iS)
	 If the input matrix is a unitary gauge field link and we want
	 the coefficients of the fundamental representation
	 we must take the log of the matrix.
	 @return S The matrix exponent
      */
      __device__ __host__ inline const Mat<T,N> exponent() const
      {
	// Reuse P and Q.
	Mat<T,N> &P = P_;
	Mat<T,N> &E = E_;
	Mat<T,N> ret;	
	typedef decltype(P(0,0).x) Real;
	
	// Place evals of S in P.
	setIdentity(&P);
	for(int i=0; i<N; i++) {
	  Real sigma = atan(eval(i).imag() / eval(i).real());	  
	  eval(i).real() = sigma;
	  eval(i).imag() = 0.0;
	  P(i,i) = eval(i);
	}
	// S = V * Sigma * V^{\dag}
	ret = E * P * E.conj();
	return ret;	
      }
      
      /**
	 @brief Compute the upper Hessenberg reduction of A 
	 @param[in] A The problem matrix
	 @param[in] H The upper Hessenberg matrix
	 @param[in] P The accumulated reflectors
      */
      __device__ __host__ inline void upperHessReduction(const Mat<T,N> &A, Mat<T,N> &H, Mat<T,N> &P)
      {
	typedef decltype(A(0,0).x) Real;	
	Mat<T,N> Pi;      // reflector at iteration i
	
	Real col_norm, col_norm_inv;
	ColorSpinor<T,1,N> v; // vector of length (1 x N)
	
	for(int i = 0; i < N - 2; i++) {
	  
	  // get (partial) dot product of ith column vector
	  col_norm = 0.0;
#pragma unroll
	  for(int j = i+1; j < N; j++) {
	    col_norm += H(j,i).real() * H(j,i).real();
	    col_norm += H(j,i).imag() * H(j,i).imag();
	  }
	  col_norm = sqrt(col_norm);
	  
	  v(i+1) = H(i+1,i) - (static_cast<T>(col_norm) * H(i+1,i)) / sqrt(H(i+1,i).real() * H(i+1,i).real() + H(i+1,i).imag() * H(i+1,i).imag());
	  
	  // reuse col_norm
	  col_norm = v(i+1).real() * v(i+1).real() + v(i+1).imag() * v(i+1).imag();

	  // copy the rest of the column
#pragma unroll
	  for(int j = i + 2; j < N; j++ ) {
	    v(j) = H(j,i);
	    col_norm += v(j).real() * v(j).real() + v(j).imag() * v(j).imag();
	  }
	  col_norm_inv = 1.0/sqrt(col_norm);
	  
	  // Normalise the column
#pragma unroll
	  for(int j = i + 1; j < N; j++) v(j) *= col_norm_inv;
	  
	  // Construct the householder matrix P = I - 2 U U*T
	  setIdentity(&Pi);
#pragma unroll
	  for(int j = i + 1; j < N; j++ ) {
	    for(int k = i + 1; k < N; k++ ) {
	      T P_elem;
	      P_elem.x  = v(j).real() * v(k).real();
	      P_elem.x += v(j).imag() * v(k).conj();
	      P_elem.y  = v(j).imag() * v(k).real();
	      P_elem.y -= v(j).real() * v(k).conj();
	      Pi(j,k) -= static_cast<T>(2.0) * P_elem;
	    }	    
	  }
	  
	  // Similarity transform
	  H = Pi * H * Pi;
	  // Store reflector
	  P = P * Pi;
	}
      }

      /**
	 @brief Compute a QR iteration
	 @param[in] Q The accumulated reflectors
	 @param[in] R The R matrix
      */
      __device__ __host__ inline void qrIteration(Mat<T,N> &Q, Mat<T,N> &R)
      {
	typedef decltype(Q(0,0).x) Real;
	T T11, T12, T21, T22, U1, U2;
	Real dV;
	Real tol = tol_;
	
	// Allocate the rotation matrices.
	ColorSpinor<T,1,N-1> R11;
	ColorSpinor<T,1,N-1> R12;
	ColorSpinor<T,1,N-1> R21;
	ColorSpinor<T,1,N-1> R22;
	
	for (int i = 0; i < N - 1; i++) {
	  
	  // If the sub-diagonal element is numerically
	  // small enough, floor it to 0
	  if (abs(R(i + 1,i)) < tol) {
	    R(i + 1,i) = static_cast<T>(0.0);
	    continue;
	  }
	  
	  U1 = R(i,i);
	  dV = sqrt(norm(R(i,i)) + norm(R(i + 1,i)));
	  dV = (U1.real() > 0) ? dV : -dV;
	  U1 += dV;
	  U2 = R(i + 1,i);
	  
	  T11 = conj(U1) / dV;
	  R11(i) = conj(T11);
	  
	  T12 = conj(U2) / dV;
	  R12(i) = conj(T12);
	  
	  T21 = conj(T12) * conj(U1) / U1;
	  R21(i) = conj(T21);
	  
	  T22 = T12 * U2 / U1;
	  R22(i) = conj(T22);
	  
	  // Do the H_kk and set the H_k+1k to zero
	  R(i,i) -= (T11 * R(i,i) + T12 * R(i + 1,i));
	  R(i + 1,i) = 0;
	  
	  // Continue for the other columns
#pragma unroll
	  for (int j = i + 1; j < N; j++) {
	    T temp = R(i,j);
	    R(i,j) -= (T11 * temp + T12 * R(i + 1,j));
	    R(i + 1,j) -= (T21 * temp + T22 * R(i + 1,j));
	  }
	}
	
	// Rotate R and V, i.e. H->RQ. V->VQ
	// Loop over columns of upper Hessenberg
	for (int j = 0; j < N - 1; j++) {
	  if (abs(R11(j)) > tol) {
	    // Loop over the rows, up to the sub diagonal element i=j+1
#pragma unroll
	    for (int i = 0; i < j + 2; i++) {
	      T temp = R(i,j);
	      R(i,j) -= (R11(j) * temp + R12(j) * R(i,j + 1));
	      R(i,j + 1) -= (R21(j) * temp + R22(j) * R(i,j + 1));
	    }
	    
#pragma unroll
	    for (int i = 0; i < N; i++) {
	      T temp = Q(i,j);
	      Q(i,j) -= (R11(j) * temp + R12(j) * Q(i,j + 1));
	      Q(i,j + 1) -= (R21(j) * temp + R22(j) * Q(i,j + 1));
	    }
	  }
	}
      }
      
      /**
	 @brief Compute the in place QR reduction of an upper Hessenberg UH
	 @param[in] Q The accumulated reflectors
	 @param[in] UH The upper Hessenberg matrix
      */
      __device__ __host__ inline void qrUpperHess(Mat<T,N> &Q, Mat<T,N> &UH)
      {
	setIdentity(&Q);
	double tol = tol_;
	int max_iter = 100000;
	int iter = 0;
	
	T temp, discriminant, sol1, sol2, eval;
	for (int i = N - 2; i >= 0; i--) {
	  while (iter < max_iter) {
	    if (abs(UH(i + 1,i)) < tol) {
	      UH(i + 1,i) = 0.0;
	      break;
	    } else {
	      
	      // Compute the 2 eigenvalues via the quadratic formula
	      //----------------------------------------------------
	      // The discriminant
	      temp = (UH(i,i) - UH(i + 1,i + 1)) * (UH(i,i) - UH(i + 1,i + 1)) / 4.0;
	      discriminant = sqrt(UH(i + 1,i) * UH(i,i + 1) + temp);
	      
	      // Reuse temp
	      temp = (UH(i,i) + UH(i + 1,i + 1)) / 2.0;
	      
	      sol1 = temp - UH(i + 1,i + 1) + discriminant;
	      sol2 = temp - UH(i + 1,i + 1) - discriminant;
	      //----------------------------------------------------
	      
	      // Deduce the better eval to shift
	      eval = UH(i + 1,i + 1) + (norm(sol1) < norm(sol2) ? sol1 : sol2);
	      
	      // Shift the eigenvalue
#pragma unroll
	      for (int j = 0; j < N; j++) UH(j,j) -= eval;
	      
	      // Do the QR iteration
	      //qrIteration(Q, UH);
	      
	      // Shift back
#pragma unroll
	      for (int j = 0; j < N; j++) UH(j,j) += eval;
	    }
	    iter++;
	  }
	}	
      }
      
      /**
	 @brief Compute eigen decomposition of a triangular matrix T
	 @param[in] E The eigenvectors 
	 @param[in] UT The upper triangular matrix
      */
      __device__ __host__ inline void eigensolveUpperTri(Mat<T,N> &E, const Mat<T,N> &UT) {
	typedef decltype(E(0,0).x) Real;
	Real tol = tol_;
	Real vnorm, vnorm_inv;
	
	// Temp matrix storage
	Mat<T,N> UTt;
	
	for (int i = N - 1; i >= 0; i--) {
	  ColorSpinor<T,1,N> V;
	  T lambda = UT(i,i);
#pragma unroll
	  for (int j = 0; j < N; j++ ) UTt(j,j) = UT(j,j) - lambda; 
	  
	  // free choice of this component
	  // back-substitute for other components
	  V(i) = 1.0;
	  
	  for (int j = i - 1; j >= 0; j--) {         
	    V(j) = 0.0;
#pragma unroll
	    for (int k = j + 1; k <= i; k++) V(j) -= UTt(j,k) * V(k);
	    
	    if (abs(UTt(j,j)) < tol) {
	      // Degeneracy detected in upper triangular eigensolve
	      // What do...?
	    } else V(j) = V(j) / UTt(j,j);
	  }
	  // Normalise
	  vnorm = 0.0;
#pragma unroll
	  for (int j=0; j<N; j++) vnorm += (V(j).x * V(j).x + V(j).y * V(j).y);
	  vnorm_inv = sqrt(vnorm);
#pragma unroll
	  for (int j = 0; j <= i; j++ ) E(j,i) = V(j) * static_cast<T>(vnorm_inv);
	}
      }
    };
    
  } // namespace linalg

} // namespace quda
