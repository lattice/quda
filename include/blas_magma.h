#ifndef _BLAS_MAGMA_H
#define _BLAS_MAGMA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <complex>
#include <cuComplex.h>
#include <stdio.h>
#include <enum_quda.h>

//MAGMA library interface

  //Initialization:
  void OpenMagma();
  //
  void CloseMagma();

  /**
     Solves a system of linear equations
     @param sol (in/out) array containing source (in). Overwritten by solution (out)
     @param ldn Array leading dimension
     @param n Dimension of the problem
     @param Mat Matrix field containing the input matrices on the CPU
     @param ldm Matrix leading dimension
     @param prec Matrix precision
  */
  void magma_Xgesv(void* sol, const int ldn, const int n, void* Mat, const int ldm, const int prec);

  /**
     Computes for an n-by-n complex nonsymmetric matrix Mat, the
     eigenvalues and right eigenvectors.
     @param Mat Matrix field containing the input matrices on the CPU
     @param n Dimension of the problem
     @param ldm Matrix leading dimension
     @param vr (out) array containing right eigenvectors
     @param evalues (out) array containing eigenvalues
     @param ldv Array leading dimension
     @param prec Matrix precision
  */
  void magma_Xgeev(void *Mat, const int n, const int ldm, void *vr, void *evalues, const int ldv, const int prec);

  /**
     Solves the overdetermined (rows > = cols), least squares problem
     @param Mat Matrix field containing the input matrices on the CPU
     @param c Array containing source/solution vector
     @param rows Number of rows of the matrix Mat
     @param cols Number of columns of the matrix Mat
     @param ldm Matrix leading dimension
     @param prec Matrix precision
  */
  void magma_Xgels(void *Mat, void *c, int rows, int cols, int ldm, const int prec);

  /**
     Computes for an n-by-n complex symmetric matrix Mat, the
     eigenvalues and eigenvectors.
     @param Mat Matrix field containing the input matrices on the CPU, and eigenvectors on exit
     @param n Dimension of the problem
     @param ldm Matrix leading dimension
     @param evalues (out) array containing eigenvalues
     @param ldv Array leading dimension
     @param prec Matrix precision
  */
  void magma_Xheev(void *Mat, const int n, const int ldm, void *evalues, const int prec);

#endif // _BLAS_MAGMA_H
