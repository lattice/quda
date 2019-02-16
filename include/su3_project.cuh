#pragma once

/**
 * @file su3_project.cuh
 * 
 * @section Description
 * 
 * This header file defines an interative SU(3) projection algorithm
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
    for (int i=0; i<in.size(); i++) {
#pragma unroll
      for (int j=0; j<in.size(); j++) {
        if (fabs(in(i,j).real() - inv(j,i).real()) > tol ||
            fabs(in(i,j).imag() + inv(j,i).imag()) > tol) return false;
      }
    }

    // now check 1 - U^dag * U = 0
    // this check is more expensive so delay until we have passed first check
    const Matrix identity = conj(in)*in;
#pragma unroll
    for (int i=0; i<in.size(); i++) {
      if (fabs(identity(i,i).real() - static_cast<Float>(1.0)) > tol ||
          fabs(identity(i,i).imag()) > tol)
        return false;
#pragma unroll
      for (int j=0; j<in.size(); j++) {
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
   * @brief Print out deviation for each component (used for
   * debugging only).
   *
   * @param inv The inverse of the input matrix
   * @param in The input matrix to which we're reporting its unitarity
   */
  template <typename Matrix>
  __host__ __device__ void checkUnitaryPrint(const Matrix &inv, const Matrix &in)
  {
    for (int i=0; i<in.size(); i++) {
      for (int j=0; j<in.size(); j++) {
        printf("TESTR: %+.13le %+.13le %+.13le\n",
               in(i,j).real(), inv(j,i).real(), fabs(in(i,j).real() - inv(j,i).real()));
	printf("TESTI: %+.13le %+.13le %+.13le\n",
               in(i,j).imag(), inv(j,i).imag(), fabs(in(i,j).imag() + inv(j,i).imag()));
      }
    }
  }

  /**
   * @brief Project the input matrix on the SU(3) group.  First
   * unitarize the matrix and then project onto the special unitary
   * group.
   *
   * @param in The input matrix to which we're projecting
   * @param tol Tolerance to which this check is applied
   */
  template <typename Float>
  __host__ __device__ inline void polarSu3(Matrix<complex<Float>,3> &in, Float tol)
  {
    constexpr Float negative_third = -1.0/3.0;
    constexpr Float negative_sixth = -1.0/6.0;
    Matrix<complex<Float>,3> out = in;
    Matrix<complex<Float>,3> inv = inverse(in);

    constexpr int max_iter = 100;
    int i = 0;
    do { // iterate until matrix is unitary
      out = 0.5*(out + conj(inv));
      inv = inverse(out);
    } while (!checkUnitary(inv, out, tol) && ++i < max_iter);

    // now project onto special unitary group
    complex<Float> det = getDeterminant(out);
    Float mod = pow(norm(det), negative_sixth);
    Float angle = arg(det);

    complex<Float> cTemp;
    sincos(negative_third * angle, &cTemp.y, &cTemp.x);

    in = (mod*cTemp)*out;
  }

  
} // namespace quda
