#pragma once

/**
 * @file suN_project.cuh
 * 
 * @section Description
 * 
 * This header file defines an iterative SU(N) projection algorithm
 */

#include <quda_matrix.h>

namespace quda {

  /**
   * @brief Check the unitarity of the input matrix to a given
   * tolerance
   *
   * @param inv The inverse of the input matrix
   * @param in The input matrix to which we're reporting its unitarity
   * @param tol Tolerance to which this check is applied
   */
  template <typename Matrix, typename Float>
  __host__ __device__ inline bool checkUnitary(const Matrix &inv, const Matrix &in, const Float tol)
  {
    // first check U - U^{-1} = 0
#pragma unroll
    for (int i = 0; i < in.rows(); i++) {
#pragma unroll
      for (int j = 0; j < in.rows(); j++) {
        if (fabs(in(i,j).real() - inv(j,i).real()) > tol ||
            fabs(in(i,j).imag() + inv(j,i).imag()) > tol) return false;
      }
    }

    // now check 1 - U^dag * U = 0
    // this check is more expensive so delay until we have passed first check
    const Matrix identity = conj(in)*in;
#pragma unroll
    for (int i = 0; i < in.rows(); i++) {
      if (fabs(identity(i,i).real() - static_cast<Float>(1.0)) > tol ||
          fabs(identity(i,i).imag()) > tol)
        return false;
#pragma unroll
      for (int j = 0; j < in.rows(); j++) {
        if (i>j) { // off-diagonal identity check
	  if (fabs(identity(i,j).real()) > tol || fabs(identity(i,j).imag()) > tol ||
	      fabs(identity(j,i).real()) > tol || fabs(identity(j,i).imag()) > tol )
	    return false;
        }
      }
    }
    
    return true;
  }

  /**
     @brief Print out deviation for each component (used for
     debugging only).
     
     @param inv The inverse of the input matrix
     @param in The input matrix to which we're reporting its unitarity
  */
  template <typename Matrix>
  __host__ __device__ void checkUnitaryPrint(const Matrix &inv, const Matrix &in)
  {
    for (int i = 0; i < in.rows(); i++) {
      for (int j = 0; j < in.rows(); j++) {
        printf("TESTR: %+.13le %+.13le %+.13le\n",
               in(i,j).real(), inv(j,i).real(), fabs(in(i,j).real() - inv(j,i).real()));
	printf("TESTI: %+.13le %+.13le %+.13le\n",
               in(i,j).imag(), inv(j,i).imag(), fabs(in(i,j).imag() + inv(j,i).imag()));
      }
    }
  }

  /**
     @brief Project the input matrix on the SU(N) group.  First
     unitarize the matrix and then project onto the special unitary
     group.
     
     @param in The input matrix to which we're projecting
     @param tol Tolerance to which this check is applied
  */
  template <typename Float, int N>
  __host__ __device__ inline void polarSUN(Matrix<complex<Float>,N> &in, Float tol)
  {

    constexpr Float negative_one_on_N = -1.0/N;
    constexpr Float negative_one_on_2N = -1.0/(2*N);
    Matrix<complex<Float>,N> out = in;
    Matrix<complex<Float>,N> inv = inverse(in);

    constexpr int max_iter = 100;
    int i = 0;
    do { // iterate until matrix is unitary
      out = static_cast<Float>(0.5)*(out + conj(inv));
      inv = inverse(out);
    } while (!checkUnitary(inv, out, tol) && ++i < max_iter);

    // now project onto special unitary group
    complex<Float> det = getDeterminant(out);
    Float mod = pow(norm(det), negative_one_on_2N);
    Float angle = arg(det);

    Float re, im;
    quda::sincos(negative_one_on_N * angle, &im, &re);

    in = complex<Float>(mod * re, mod * im) * out;
  }
} // namespace quda
