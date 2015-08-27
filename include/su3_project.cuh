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
  template <typename Float2, typename Float>
  __host__ __device__ int checkUnitary(Matrix<Float2,3> &inv,  Matrix<Float2,3> in, const Float tol)
  {
    computeMatrixInverse(in, &inv);

    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        if (fabs(in(i,j).x - inv(j,i).x) > tol)
          return 1;
        if (fabs(in(i,j).y + inv(j,i).y) > tol)
          return 1;
      }
    return 0;
  }

  /**
   * @brief Check the unitarity of the input matrix to a given
   * tolerance (1e-14) and print out deviation for each component (used for
   * debugging only).
   *
   * @param inv The inverse of the input matrix
   * @param in The input matrix to which we're reporting its unitarity
   */  template <typename Float2>
  __host__ __device__ int checkUnitaryPrint(Matrix<Float2,3> &inv, Matrix<Float2,3> in)
  {
    computeMatrixInverse(in, &inv);
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        printf("TESTR: %+.3le %+.3le %+.3le\n", in(i,j).x, (*inv)(j,i).x, fabs(in(i,j).x - (*inv)(j,i).x));
	printf("TESTI: %+.3le %+.3le %+.3le\n", in(i,j).y, (*inv)(j,i).y, fabs(in(i,j).y + (*inv)(j,i).y));
        cudaDeviceSynchronize();
        if (fabs(in(i,j).x - inv(j,i).x) > 1e-14)
          return 1;
        if (fabs(in(i,j).y + inv(j,i).y) > 1e-14)
          return 1;
      }
    return 0;  
  }

  /**
   * @brief Project the input matrix on the SU(3) group.  First unitarize the matrix and then project onto the special unitary group.
   *
   * @param in The input matrix to which we're projecting
   * @param tol Tolerance to which this check is applied
   */
  template <typename Float2,typename Float>
  __host__ __device__ void polarSu3(Matrix<Float2,3> &in, Float tol)
  {
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    Matrix<Cmplx,3> inv, out;

    out = in;
    computeMatrixInverse(out, &inv);

    // iterate until matrix is unitary
    do {
      out = out + conj(inv);
      out = out*0.5;
    } while(checkUnitary(inv, out, tol));


    // now project onto special unitary group
    Cmplx  det = getDeterminant(out);
    double mod = det.x*det.x + det.y*det.y;
    mod = pow(mod, (1./6.));
    double angle = atan2(det.y, det.x);
    angle /= -3.;
    
    Cmplx cTemp;

    cTemp.x = cos(angle)/mod;
    cTemp.y = sin(angle)/mod;

    in = out*cTemp;
/*    if (checkUnitary(inv, out))
    {
    	cTemp = getDeterminant(out);
	printf ("DetX: %+.3lf  %+.3lfi, %.3lf %.3lf\nDetN: %+.3lf  %+.3lfi", det.x, det.y, mod, angle, cTemp.x, cTemp.y);
        cudaDeviceSynchronize();
	checkUnitaryPrint(out, &inv);
	setIdentity(in);
        *in = *in * 0.5;
    }
    else
    {
      cTemp = getDeterminant(out);
//      printf("Det: %+.3lf %+.3lf\n", cTemp.x, cTemp.y);
      cudaDeviceSynchronize();

      if (fabs(cTemp.x - 1.0) > 1e-8)
	setIdentity(in);
      else if (fabs(cTemp.y) > 1e-8)
      {
	setIdentity(in);
        printf("DadadaUnitary failed\n");
        *in = *in * 0.1;
      }
      else
        *in = out;
    }*/
  }

  
} // namespace quda
