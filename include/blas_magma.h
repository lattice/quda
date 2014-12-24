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
    private:

      //problem sizes:
      int m;
      int nev;
      int prec;
      int ldm;//(may include padding)

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

      //Collection of methods for EigCG solver:
      void MagmaHEEVD(void *dTvecm, void *hTvalm, const int problem_size, bool host = false);
      //
      int  MagmaORTH_2nev(void *dTvecm, void *dTm);
      //
      void RestartV(void *dV, const int vld, const int vlen, const int vprec, void *dTevecm, void *dTm);

      //Collection of methods used for the initial guess vector deflation:

      //this accepts host routines, and employ either CPU or GPU, depending on problem size etc.
      void SolveProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH);

      //GPU version of the above
      void SolveGPUProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH);
      
      //Spinor matrix vector product:
      void SpinorMatVec(void *spinorOut, const void *spinorSetIn, const int sld, const int slen, const void *vec, const int vlen);

      void MagmaRightNotrUNMQR(const int clen, const int qrlen, const int nrefls, void *QR, const int ldqr, void *tau, void *Vm, const int cldn);

      //Pure LAPACK routines (when problem size is very small, no need for MAGMA routines):

      void LapackGESV(void* rhs, const int ldn, const int n, void* H, const int ldH, int* index_arr);//ipiv
      //Compute right eigenvectors and eigenvalues of a complex non-symm. matrix
      void LapackRightEV(void *Mat, const int m,  const int ldm, void *harVecs, const int *ldv, void *harVals);
      //
      void LapackGEQR(const int n, void *Mat, const int m, const int ldm, void *tau);//QR decomposion of a (m by n) matrix, ldm is the leading dimension
      //
      void LapackLeftConjUNMQR(const int m, const int ncolsMat, const int nref, void *QRM, const int ldqr, void *tau, void *Mat, const int ldm);//Apply from the left conjugate QR-decomposed matrix QRM, of size m by n.
      void LapackRightNotrUNMQR(const int nrowsMat, const int ncolsMat, const int nref, void *QRM, const int ldqr, void *tau, void *Mat, const int ldm);//Apply from the left conjugate QR-decomposed matrix QRM, of size m by n.
      //
      void Sort(const int m, const int ldm, void *eVecs, const int nev, void *eVals);//Sort nev smallest eigenvectors

   };


#endif // _BLAS_MAGMA_H
