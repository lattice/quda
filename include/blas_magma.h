#ifndef _BLAS_MAGMA_H
#define _BLAS_MAGMA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <complex>
#include <cuComplex.h>
#include <stdio.h>

//MAGMA library interface
//required for (incremental) EigCG solver



   class BlasMagmaArgs{

      typedef std::complex<double> Complex;

    private:

      int prec;
      //general magma library parameters:
      int info;

    public:

      BlasMagmaArgs() : prec(8), info(-1), init(true), alloc(false) {  }

      BlasMagmaArgs(const int prec){
#ifdef MAGMA_LIB
        if(magma_finalize() != MAGMA_SUCCESS) errorQuda("\nError: cannot close MAGMA library\n");
#else
        errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif
      }

      ~BlasMagmaArgs() {}

      //Initialization methods:
      static void OpenMagma();
      //
      static void CloseMagma();

      /**
	 Batch inversion the matrix field using an LU decomposition method.
	 @param Ainv_h Matrix field containing the inverse matrices on the CPU
	 @param A_h Matrix field containing the input matrices on the CPU
	 @param Temporary storate on the GPU of size = A_h
	 @param n Dimension each matrix
	 @param batch Problem batch size
       */
      void BatchInvertMatrix(void *Ainv_h, void* A_h, const int n, const int batch);

   };

  void magma_Xgesv(void* sol, const int ldn, const int n, void* Mat, const int ldm, const int prec);

  void magma_Xgeev(void *Mat, const int m, const int ldm, void *vr, void *evalues, const int ldv, const int prec);

  void magma_Xgels(void *Mat, void *c, int rows, int cols, int ldm, const int prec);

  void magma_Xheev(void *Mat, const int m, const int ldm, void *evalues, const int prec);


#endif // _BLAS_MAGMA_H
