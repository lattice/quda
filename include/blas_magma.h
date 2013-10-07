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

   typedef struct blasMagmaParam_s {
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

      cuDoubleComplex *W;
      cuDoubleComplex *hTau;
      cuDoubleComplex *dTau;

      cuDoubleComplex *lwork;
      double             *rwork;
      int        *iwork;

   }blasMagmaParam;
/*
   typedef struct blasMagmaParam_s {
      //problem sizes:
      int m;
      int nev;

      //general magma library parameters:	
      magma_int_t info;

      bool init;

      //magma params/objects:
      magma_int_t ldTm;//hTm (host/device)ld (may include padding)

      magma_int_t nb;

      magma_int_t llwork; 
      magma_int_t lrwork;
      magma_int_t liwork;

      int sideLR;

      magma_int_t htsize;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
      magma_int_t dtsize;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

      magma_int_t lwork_max; 

      magmaDoubleComplex *W;
      magmaDoubleComplex *hTau;
      magmaDoubleComplex *dTau;

      magmaDoubleComplex *lwork;
      double             *rwork;
      magma_int_t        *iwork;

   }blasMagmaParam;
*/

   void init_magma(blasMagmaParam *param, const int m, const int nev);
   void shutdown_magma(blasMagmaParam *param);

   int runRayleighRitz(cuDoubleComplex *dTm, 
                       cuDoubleComplex *dTvecm0,  
                       cuDoubleComplex *dTvecm1, 
                       std::complex<double> *hTvecm, 
                       double *hTvalm, 
                       const blasMagmaParam *param, const int i);

   void restart_2nev_vectors(cuDoubleComplex *dVm, cuDoubleComplex *dQ, const blasMagmaParam *param, const int len);

#ifdef __cplusplus
}
#endif

#endif // _BLAS_MAGMA_H
