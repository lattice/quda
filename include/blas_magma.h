#ifndef _BLAS_MAGMA_H
#define _BLAS_MAGMA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <complex>
#include <cuComplex.h>
#include <stdio.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

#ifdef __cplusplus
extern "C" {
#endif

   typedef struct blasMagmaArgs_s {
      //problem sizes:
      int m;
      int nev;

      //general magma library parameters:	
      int info;

      bool init;

      //magma params/objects:
      int ldTm;//hTm (host/device)ld (may include padding)

      int nb;

      int llwork; 
      int lrwork;
      int liwork;

      int sideLR;

      int htsize;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
      int dtsize;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

      int lwork_max; 

      cuFloatComplex *W;
      cuFloatComplex *hTau;
      cuFloatComplex *dTau;

      cuFloatComplex *lwork;
      float          *rwork;
      int        *iwork;

   }blasMagmaArgs;


   void init_magma(blasMagmaArgs *param, const int m, const int nev);
   void shutdown_magma(blasMagmaArgs *param);

   int runRayleighRitz(cuFloatComplex *dTm, 
                       cuFloatComplex *dTvecm0,  
                       cuFloatComplex *dTvecm1, 
                       std::complex<float> *hTvecm, 
                       float *hTvalm, 
                       const blasMagmaArgs *param, const int i);

   void restart_2nev_vectors(cuFloatComplex *dVm, cuFloatComplex *dQ, const blasMagmaArgs *param, const int len);

#ifdef __cplusplus
}
#endif

#endif // _BLAS_MAGMA_H
