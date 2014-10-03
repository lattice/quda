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

    protected:
      //problem sizes:
      int m;
      int nev;
      int prec;
      int ldm;//(leading dimension : may include padding)

      //general magma library parameters:	
      int info;

      bool init;
      bool alloc;

      //magma params/objects:
      int llwork; 
      int lrwork;
      int liwork;

      int sideLR;

      int htsize;//MIN(l,k)-number of Householder reflectors, but we always have k <= MIN(m,n)
      int dtsize;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

      int lwork_max; 

      void *W;
      void *W2;
      void *hTau;
      void *dTau;

      void *lwork;
      void *rwork;
      int  *iwork;

    public:

      BlasMagmaArgs() : prec(8), info(-1), init(true), alloc(false) {  }

      BlasMagmaArgs(const int prec);

      BlasMagmaArgs(const int m, const int nev, const int ldm, const int prec);

      BlasMagmaArgs(const int m, const int ldm, const int prec);

      ~BlasMagmaArgs();

      //Initialization methods:
      static void OpenMagma();
      //
      static void CloseMagma();

      //Collection of methods for EigCG/GMRES-DR solvers:
      void MagmaHEEVD(void *symM, void *hvals, const int problem_size, bool host = false);
      //
      void MagmaGEEVR(void *genM, void *vecr, const int ldv, void *hvals, const int problem_size);//computes right eigenvectors of the matrix M
      //
      void MagmaGELS(void *genM, const int ldm, const int m, const int n, void *vecr, const int ldv);//least squares problem
      //
      int  MagmaORTH_2nev(void *dTvecm, void *dTm);
      //
      void RestartV(void *dV, const int vld, const int vlen, const int vprec, void *dTevecm, void *dTm);

      //Collection of methods used for the initial guess vector deflation:

      //this accepts host routines, and employ either CPU or GPU, depending on the problem size etc.
      void SolveProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH);

      //GPU version of the above
      void SolveGPUProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH);
      
      //Spinor matrix vector product:
      void SpinorMatVec(void *spinorOut, const void *spinorSetIn, const int sld, const int slen, const void *vec, const int vlen);
   };


#endif // _BLAS_MAGMA_H
