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

//magma library interface 
//required for Incremental EigCG solver

   class BlasMagmaArgs{
    private:

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

      cuComplex *W;
      cuComplex *hTau;
      cuComplex *dTau;

      cuComplex *lwork;
      float     *rwork;
      int       *iwork;

    public:
      BlasMagmaArgs(const int m, const int nev);
      ~BlasMagmaArgs();

      //Collection of methods for Incremental EigCG
      int RayleighRitz(cuComplex *dTm, cuComplex *dTvecm0, cuComplex *dTvecm1,  std::complex<float> *hTvecm,  float *hTvalm);

      void Restart_2nev_vectors(cuComplex *dVm, cuComplex *dQ, const int len);
      
      //this accepts host routines, and employ either CPU or GPU, depending on problem size etc.
      void SolveProjMatrix(void* rhs, const int n, void* H, const int ldH, const int prec);

      //GPU version of the above
      void SolveGPUProjMatrix(void* rhs, const int n, void* H, const int ldH, const int prec);
   };


#endif // _BLAS_MAGMA_H
