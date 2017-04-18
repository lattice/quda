#ifndef _BLAS_MAGMA_H
#define _BLAS_MAGMA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <complex>
#include <cuComplex.h>
#include <stdio.h>

//MAGMA library interface

  //Initialization:
  void OpenMagma();
  //
  void CloseMagma();

  /**
     Batch inversion the matrix field using an LU decomposition method.
     @param Ainv_h Matrix field containing the inverse matrices on the CPU
     @param A_h Matrix field containing the input matrices on the CPU
     @param Temporary storate on the GPU of size = A_h
     @param n Dimension each matrix
     @param batch Problem batch size
     @param prec Matrix precision
  */

  void magma_batchInvertMatrix(void *Ainv_h, void* A_h, const int n, const int batch, const int prec);

  void magma_Xgesv(void* sol, const int ldn, const int n, void* Mat, const int ldm, const int prec);

  void magma_Xgeev(void *Mat, const int m, const int ldm, void *vr, void *evalues, const int ldv, const int prec);

  void magma_Xgels(void *Mat, void *c, int rows, int cols, int ldm, const int prec);

  void magma_Xheev(void *Mat, const int m, const int ldm, void *evalues, const int prec);


#endif // _BLAS_MAGMA_H
