#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <string.h>

#include <face_quda.h>

#include <iostream>

#include <blas_magma.h>
#include <algorithm>


/*
GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) p. 20-37
See also: A.Frommer et al, "Deflation and Flexible SAP-Preconditioning of GMRES in Lattice QCD simulations" ArXiv hep-lat/1204.5463
*/

namespace quda {
//Notes:
//GMResDR does not require large m (and esp. nev), so this will use normal LAPACK routines.

    static GMResDRDeflationParam *defl_param = 0;

    struct SortEvals{

      double eval_nrm;
      int    eval_idx;

      SortEvals(double val, int idx) : eval_nrm(val), eval_idx(idx) {}; 

      static bool CmpEigenNrms (SortEvals v1, SortEvals v2) { return (v1.eval_nrm < v2.eval_nrm);}

    };

    struct GMResDRDeflationParam 
    {

      //Objects required for the Galerkin projections
      Complex *projMat;  //host   projection matrix:
      cudaColorSpinorField    *projVecs;      //device buffer for HRitz vectors

      //needed for MinRes Projection:
      Complex *projQRMat; 
      Complex *projRMat; 
      Complex *projTau;  
      //
      QudaProjectionType projType;
      int projFreq;
      
      int ld;                 //projection matrix leading dimension
      int nv;                //(nv+1) => number of projection vectors, note also that projMat is a (nv+1) by (nv) matrix

      bool alloc_flag;
      bool init_flag;

      GMResDRDeflationParam(const int ldm, const int nev, bool alloc = false, int proj_freq = 1, QudaProjectionType proj_type = QUDA_INVALID_PROJECTION) : projMat(0), projVecs(0), 
      projQRMat(0), projRMat(0), projTau(0), projType(proj_type), projFreq(proj_freq), ld(ldm), nv(nev), alloc_flag(alloc), init_flag(false)
      {
        if(nev == 0) errorQuda("\nIncorrect deflation space parameters...\n");
        //Create deflation objects:
        if(alloc)
        {
          projMat  = new Complex[ld*nv];

          if(projType == QUDA_MINRES_PROJECTION)
          {
            projQRMat  = new Complex[ld*nv];
            //
            projRMat  = new Complex[ld*nv];
            //
            projTau   = new Complex[ld];
          }
        }

        return;
      }

      void LoadData(cudaColorSpinorField  *Vm, Complex *srcMat, const int nevs, const int ldn)
      {

         if(init_flag)                 errorQuda("Error: data was already initialized.");

         if(ldn != ld)                 errorQuda("Error: leading dimension of the source matrix must match that of the projection matrix.");

         if(Vm->EigvDim() < (nevs+1))  errorQuda("Error: it seems that the provided eigenvector content is inconsistent with the projection matrix dimensions: %d vs %d", Vm->EigvDim(), (nevs+1));

         if(nevs != nv) nv = nevs;

         if(alloc_flag == false)
         { 
           projMat  = new Complex[ld*nv];

           if(projType == QUDA_MINRES_PROJECTION)
           {
             projQRMat  = new Complex[ld*nv];
             //
             projRMat  = new Complex[ld*nv];
             //
             projTau   = new Complex[ld];
           }

           alloc_flag = true;
         }

         memcpy(projMat, srcMat, nv*ld*sizeof(Complex));

         ColorSpinorParam csParam(Vm->Eigenvec(0));
         csParam.create = QUDA_ZERO_FIELD_CREATE;

         csParam.eigv_dim  = (nv+1);
         csParam.eigv_id   = -1;

         projVecs  = new cudaColorSpinorField(csParam);//temporary field

         for(int i = 0; i < (nv+1); i++) copyCuda(projVecs->Eigenvec(i), Vm->Eigenvec(i)); 

         //perform QR decomposition of the projection matrix:
         if(projType == QUDA_MINRES_PROJECTION)
         { 
            BlasMagmaArgs magma_args(sizeof(double));

            memcpy(projQRMat, projMat, nv*ld*sizeof(Complex));//use this also for intermediate QR matrix
      
            magma_args.ComputeQR( nv, projQRMat, (nv+1), ld, projTau );
            //extract triangular part to the givens matrix:
            for(int i = 0; i < nv; i++) memcpy(&projRMat[ld*i], &projQRMat[ld*i], (i+1)*sizeof(Complex));
         }

         init_flag = true;

         return;
      }

      void CleanResourses()
      {
        if(alloc_flag)    
        {
          delete projVecs;
          //if(projType == QUDA_MINRES_PROJECTION) delete[] projTau;
          delete[] projMat;

          if(projType == QUDA_MINRES_PROJECTION)
          {
            delete[] projTau;
            //
            delete[] projRMat;
            //
            delete[] projQRMat;
          }
        }//end of if(alloc_flag)

        alloc_flag = false; 
        init_flag  = false;

        return;
      }

      ~GMResDRDeflationParam(){
        if(alloc_flag)    
        {
          delete projVecs;
          //if(projType == QUDA_MINRES_PROJECTION) delete[] projTau;
          delete[] projMat;

          if(projType == QUDA_MINRES_PROJECTION)
          {
            delete[] projTau;
            //
            delete[] projRMat;
            //
            delete[] projQRMat;
          }
        }//end of if(alloc_flag)
      }

    };


    class GMResDRArgs{
     
      private:
      BlasMagmaArgs *GMResDR_magma_args;
      //
      Complex *harVecs;//array of harmonic eigenvectors
      Complex *harVals;//array of harmonic Ritz values
      //
      Complex *sortedHarVecs;//array of harmonic eigenvectors: (nev+1) length
      Complex *sortedHarVals;//array of harmonic Ritz values: nev+1 length
      //
      Complex *harMat;//harmonic matrix
      //aux array to keep QR decomposed "restarted Hessenberg":
      Complex *qrH;
      Complex *tauH;//nev->m : for DEBUGING only!
      //auxilary object
      Complex *srtRes;//the "short residual" is in fact an alias pointer to &sortedHarVecs[nev*m] !
   
      public:

      Complex *H;//Hessenberg matrix
      Complex *cH;//conjugate Hess. matrix
      Complex *lsqSol;//the least squares problem solution

      int m;
      int ldm;//leading dimension (must be >= m+1)
      int nev;//number of harmonic eigenvectors used for the restart

      bool init_flag; 
      //public:

      GMResDRArgs( ) { };

      GMResDRArgs(int m, int ldm, int nev);

      ~GMResDRArgs();
      //more implementations here:
      void InitGMResDRArgs();//allocate GMResDR specific objects, set init_flag to true

      void ResetHessenberg();
 
      void ConjugateH();//rows = cols+1

      void SetHessenbergElement(const int i, const int j, const Complex h_ij);
      //
      void ComputeHarmonicRitzPairs();

      void RestartVH(cudaColorSpinorField *Vm);

      void PrepareDeflatedRestart(Complex *givensH, Complex *g, const bool use_deflated_cycles = true);

      void PrepareGivens(Complex *givensH, const int col, const bool use_deflated_cycles = true);

      void UpdateSolution(cudaColorSpinorField *x, cudaColorSpinorField *Vm, Complex *givensH, Complex *g, const int j);
   };

  GMResDRArgs::GMResDRArgs(int m, int ldm, int nev): GMResDR_magma_args(0), harVecs(0), harVals(0), sortedHarVecs(0), sortedHarVals(0), harMat(0), qrH(0), tauH(0), srtRes(0),
  m(m), ldm(ldm), nev(nev), init_flag(false) {

     int mp1 = m+1;    

     if(ldm < mp1) errorQuda("\nError: the leading dimension is not correct.\n");   

     if(nev == 0) errorQuda("\nError: incorrect deflation size..\n");

     H        = new Complex[ldm*m];//Hessenberg matrix
     cH       = new Complex[ldm*mp1];//conjugate Hessenberg matrix
     lsqSol   = new Complex[(m+1)];
   
     return;
  }

  void GMResDRArgs::InitGMResDRArgs()
  {
     if(init_flag) errorQuda("\nGMResDR resources were allocated.\n");
     //magma library initialization:
     GMResDR_magma_args = new BlasMagmaArgs(m, nev+1, ldm, sizeof(double));

     harVecs  = new Complex[ldm*m];//(m+1)xm (note that ldm >= m+1)
     harVals  = new Complex[m];//

     sortedHarVecs  = new Complex[ldm*(nev+1)];//
     sortedHarVals  = new Complex[(nev+1)];//

     //ALL this in column major format (for using with LAPACK or MAGMA)
     harMat   = new Complex[ldm*m];

     qrH      = new Complex[ldm*m];
     tauH     = new Complex[m];//nev->m : for DEBUGING only!

     srtRes   = &sortedHarVecs[ldm*(nev)];
    
     init_flag = true;
 
     return;
  }

  void GMResDRArgs::ResetHessenberg()
  {
    memset(H,  0, ldm*m*sizeof(Complex));
    //
    memset(cH, 0, ldm*(m+1)*sizeof(Complex));

    return;
  }

  void GMResDRArgs::ConjugateH()//rows = cols+1
  {
    for(int c = 0; c < nev; c++ )
    {
      for(int r = 0; r < (nev+1); r++ ) cH[r*ldm+c] = conj(H[c*ldm+r]);
    }

    return;
  }

  //helper method
  void GMResDRArgs::SetHessenbergElement(const int row, const int col, const Complex el)
  {
    H[col*ldm+row]  = el;

    cH[row*ldm+col] = conj(el); 

    return;
  }


  GMResDRArgs::~GMResDRArgs() {

    delete[] lsqSol;

    delete[] cH;
    delete[] H;

    if(init_flag)
    {
      delete[] harMat;

      delete[] harVals;
      delete[] harVecs;

      delete[] sortedHarVals;
      delete[] sortedHarVecs;

      delete[] qrH;
      delete[] tauH;

      delete GMResDR_magma_args;
    }

    return;
  }


//Note: in fact, x is an accumulation field initialized to zero (if a sloppy field is used!) and has the same precision as Vm
 void GMResDRArgs::UpdateSolution(cudaColorSpinorField *x, cudaColorSpinorField *Vm, Complex *givensH, Complex *g, const int j)
 {
   //if (mixed_precision_GMResDR) zeroCuda(*x);//it's done in the main loop, no need for this
   memset(lsqSol, 0, (m+1)*sizeof(Complex));

   //Get LS solution:
   for(int l = (j-1); l >= 0; l--)
   {
     Complex accum = 0.0;

     for(int k = (l+1); k <= (j-1); k++)
     {
        Complex cdtp = givensH[ldm*k+l]*lsqSol[k];
        accum = accum+cdtp; 
     }
  
     lsqSol[l] = (g[l] - accum) / givensH[ldm*l+l]; 
   }
   //
   //compute x = x0+Vm*lsqSol
   for(int l = 0; l < j; l++)  caxpyCuda(lsqSol[l], Vm->Eigenvec(l), *x);

   return;
 }

 void GMResDRArgs::ComputeHarmonicRitzPairs()
 {
   if(!init_flag) errorQuda("\nGMResDR resources were not allocated.\n");

   memcpy(harMat, H, ldm*m*sizeof(Complex));

   const double beta2 = norm(H[ldm*(m-1)+m]);

   GMResDR_magma_args->Construct_harmonic_matrix(harMat, cH, beta2, m, ldm);

   // Computed eigenpairs for harmonic matrix:
   // Save harmonic matrix :
   memset(cH, 0, ldm*m*sizeof(Complex));

   memcpy(cH, harMat, ldm*m*sizeof(Complex));

   GMResDR_magma_args->Compute_harmonic_matrix_eigenpairs(harMat, m, ldm, harVecs, harVals, ldm);//check it!

//   for(int e = 0; e < m; e++) printf("\nEigenval #%d: %le, %le, %le\n", e, harVals[e].real(), harVals[e].imag(), abs(harVals[e]));
   //do sort:
   std::vector<SortEvals> sorted_evals_cntr;

   sorted_evals_cntr.reserve(m);

   for(int e = 0; e < m; e++) sorted_evals_cntr.push_back( SortEvals( abs(harVals[e]), e ));

   std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);
 
   for(int e = 0; e < nev; e++) memcpy(&sortedHarVecs[ldm*e], &harVecs[ldm*( sorted_evals_cntr[e].eval_idx)], (ldm)*sizeof(Complex));

//   for(int e = 0; e < 16; e++) printfQuda("\nEigenval #%d: real %le imag %le abs %le\n", sorted_evals_cntr[e].eval_idx, harVals[(sorted_evals_cntr[e].eval_idx)].real(), harVals[(sorted_evals_cntr[e].eval_idx)].imag(),  abs(harVals[(sorted_evals_cntr[e].eval_idx)]));

   return;
 }

 void GMResDRArgs::RestartVH(cudaColorSpinorField *Vm)
 {
   if(!init_flag) errorQuda("\nGMResDR resources were not allocated.\n");

   int cldn = Vm->EigvTotalLength() >> 1; //complex leading dimension
   int clen = Vm->EigvLength()      >> 1; //complex vector length

   for(int j = 0; j <= m; j++) //row index
   {
     Complex accum = 0.0;

     for(int i = 0; i < m; i++) //column index
     {
        accum += (H[ldm*i + j]*lsqSol[i]);
     }
     
     srtRes[j] -= accum;
   }

   /**** Update Vm and H ****/
   GMResDR_magma_args->RestartVH(Vm->V(), clen, cldn, Vm->Precision(), sortedHarVecs, H, ldm);//check ldm with internal (magma) ldm

   checkCudaError();

   /*****REORTH V_{nev+1}:****/

   for(int j = 0; j < nev; j++)
   {
     Complex calpha = cDotProductCuda(Vm->Eigenvec(j), Vm->Eigenvec(nev));//
     caxpyCuda(-calpha, Vm->Eigenvec(j), Vm->Eigenvec(nev));
   }
         
   double dalpha = norm2(Vm->Eigenvec(nev));

   double tmpd=1.0e+00/sqrt(dalpha);

   axCuda(tmpd, Vm->Eigenvec(nev));

   //done.

   return;
 }


 void GMResDRArgs::PrepareDeflatedRestart(Complex *givensH, Complex *g, const bool use_deflated_cycles)
 {
   if(!init_flag) errorQuda("\nGMResDR resources were not allocated.\n");

   //update initial values for the "short residual"
   memcpy(srtRes, g, (m+1)*sizeof(Complex));

   if(use_deflated_cycles)
   { 
     memset(cH, 0, ldm*(m+1)*sizeof(Complex));
     //
     memset(harMat, 0, ldm*m*sizeof(Complex));
     //
     //now update conjugate matrix
     for(int j = 0 ; j < nev; j++)
     {
       for(int i = 0 ; i < nev+1; i++)
       {
         cH[ldm*i+j] = conj(H[ldm*j+i]);
       }
     }

     memcpy(qrH, H, ldm*nev*sizeof(Complex));

     GMResDR_magma_args->ComputeQR(nev, qrH, (nev+1), ldm, tauH);

     //extract triangular part to the givens matrix:
     for(int i = 0; i < nev; i++) memcpy(&givensH[ldm*i], &qrH[ldm*i], (i+1)*sizeof(Complex));

     GMResDR_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, g, (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);
   }

   return;
 }

 void GMResDRArgs::PrepareGivens(Complex *givensH, const int col, const bool use_deflated_cycles)
 {
   if(!use_deflated_cycles) return; //we are in the "projected" cycle nothing to do..

   if(!init_flag) errorQuda("\nGMResDR resources were not allocated.\n"); 

   memcpy(&givensH[ldm*col], &H[ldm*col], (nev+1)*sizeof(Complex) );
   //
   GMResDR_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, &givensH[ldm*col], (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);

   return;
 }


 GMResDR::GMResDR(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matDefl, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), gmres_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), profile(profile), args(0), gmres_alloc(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMResDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     //create GMResDR objects:
     int mp1    = param.m + 1;

     int ldm    = mp1;//((mp1+15)/16)*16;//leading dimension

     args = new GMResDRArgs(param.m, ldm, param.nev);//use_deflated_cycles flag is true by default  

     return;
 }

 GMResDR::GMResDR(SolverParam &param) :
    DeflatedSolver(param, NULL), mat(NULL), matSloppy(NULL), matDefl(NULL), gmres_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), profile(NULL), args(0), gmres_alloc(false) { }


 GMResDR::~GMResDR() {

    if(args) delete args;

    if(gmres_alloc) 
    {
      delete Vm;
      Vm = NULL;
    }

 }

 void GMResDR::AllocateKrylovSubspace(ColorSpinorParam &csParam)
 {
   if(gmres_alloc) errorQuda("\nKrylov subspace was allocated.\n");

   printfQuda("\nAllocating resources for the GMResDR solver...\n");

   csParam.eigv_dim = param.m+1; // basis dimension (abusive notations!)

   Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors
//BUG ALLERT! if one sets csParam.eigv_dim = 1 then any other allocation of regular spinors with csParam hangs the program!
   csParam.eigv_dim = 0;

   checkCudaError();

   printfQuda("\n..done.\n");

   gmres_alloc = true;

   return; 
 }

 void GMResDR::PerformProjection(cudaColorSpinorField &x_sloppy,  cudaColorSpinorField &r_sloppy, GMResDRDeflationParam *dpar)
 {
    if(dpar->projType == QUDA_INVALID_PROJECTION)
    {
      warningQuda("\nWarning: projection method was not defined (using default nop).\n");
      return;
    }

    if(!dpar->alloc_flag) errorQuda("\nError: projection matrix was not allocated.\n");

    Complex *c    = new Complex[(dpar->nv+1)];
    Complex *d    = new Complex[dpar->ld];

    BlasMagmaArgs magma_args(sizeof(double));

    //if(getVerbosity() >= QUDA_DEBUG_VERBOSE) 
       printfQuda("\nUsing projection method %d\n", static_cast<int>(dpar->projType));

    if(dpar->projType == QUDA_GALERKIN_PROJECTION)
    {
      //Compute c = VT^{pr}_k r0 
      for(int i = 0; i < dpar->nv; i++ ) d[i] = cDotProductCuda(dpar->projVecs->Eigenvec(i), r_sloppy);//
      //Solve H^{pr}_k d = c: this nvxnv problem..
      magma_args.SolveProjMatrix((void*)d, dpar->ld,  dpar->nv, (void*)dpar->projMat, dpar->ld);
    }
    else if(dpar->projType == QUDA_MINRES_PROJECTION)
    {
      //Compute c = VT^{pr}_k+1 r0 
      for(int i = 0; i < (dpar->nv + 1); i++ ) d[i] = cDotProductCuda(dpar->projVecs->Eigenvec(i), r_sloppy);//
      //
      magma_args.LeftConjZUNMQR(dpar->nv /*number of reflectors*/, 1 /*number of columns of mat*/, d, (dpar->nv+1) /*number of rows*/, dpar->ld, dpar->projQRMat, dpar->ld, dpar->projTau);
       //Solve H^{pr}_k d = c: this nvxnv problem..
      magma_args.SolveProjMatrix((void*)d, dpar->ld,  dpar->nv, (void*)dpar->projRMat, dpar->ld);
    }
    else
    {
      errorQuda("\nProjection type is not supported.\n");
    }
   
    //Compute the new approximate solution:
    for(int l = 0; l < dpar->nv; l++)  caxpyCuda(d[l], dpar->projVecs->Eigenvec(l), x_sloppy); 

    //Compute the new residual vector: 
    //memset(c, 0, (dpar->nv+1)*sizeof(Complex));//now use as a temp array

    for(int j = 0; j < (dpar->nv+1); j++)
      for (int i = 0; i < (dpar->nv); i++)
      {
         c[j] +=  (dpar->projMat[i*dpar->ld + j] * d[i]);//column major format
      }   

    for(int l = 0; l < (dpar->nv+1); l++)  caxpyCuda(-c[l], dpar->projVecs->Eigenvec(l), r_sloppy);

    delete[] d;
    delete[] c;

    return;
 }

 void GMResDR::CleanResources()
 {
   if(defl_param) 
   {
     delete defl_param;
     defl_param = 0;
   }

   return;
 }

 void GMResDR::RunDeflatedCycles(cudaColorSpinorField *out, cudaColorSpinorField *in, GMResDRDeflationParam *dpar, const double tol_threshold /*=2.0*/)
 {
    bool use_deflated_cycles = true;

    bool mixed_precision_GMResDR = (param.precision != param.precision_sloppy);

    profile->TPSTART(QUDA_PROFILE_INIT);

    DiracMatrix *sloppy_mat = matSloppy;

    cudaColorSpinorField r(*in); //high precision residual 
    // 
    cudaColorSpinorField &x = *out;
    //
    cudaColorSpinorField y(*in); //high precision aux field 

    //Sloppy precision fields:
    ColorSpinorParam csParam(*in);//create spinor parameters

    csParam.create = QUDA_ZERO_FIELD_CREATE;

    csParam.setPrecision(gmres_space_prec);

    cudaColorSpinorField *r_sloppy, *x_sloppy;

    if (mixed_precision_GMResDR) 
    {
       r_sloppy = new cudaColorSpinorField(r, csParam);
       x_sloppy = new cudaColorSpinorField(x, csParam);
    } 
    else 
    {
       r_sloppy = &r;
       x_sloppy = &x;
    }

    cudaColorSpinorField tmp(*in, csParam);

    cudaColorSpinorField *tmp2_p;

    tmp2_p = new cudaColorSpinorField(*in, csParam);

    cudaColorSpinorField &tmp2 = *tmp2_p;

    //Allocate Vm array:
    if(gmres_alloc == false) AllocateKrylovSubspace(csParam);

    profile->TPSTOP(QUDA_PROFILE_INIT);
    profile->TPSTART(QUDA_PROFILE_PREAMBLE);

    const int m         = args->m;
    const int ldH       = args->ldm;
    const int nev       = args->nev;

    int tot_iters = 0;

    //GMRES objects:
    //Givens rotated matrix (m+1, m):
    Complex *givensH = new Complex[ldH*m];//complex
    
    //Auxilary objects: 
    Complex *g = new Complex[(m+1)];

    //Givens coefficients:
    Complex *Cn = new Complex[m];
    //
    double *Sn = (double *) safe_malloc(m*sizeof(double));
    memset(Sn, 0, m * sizeof(double));
 
    //Compute initial residual:
    const double normb = norm2(*in);  
    double stop = param.tol*param.tol* normb;	/* Relative to b tolerance */

    (*mat)(r, *out, y);
    //
    double r2 = xmyNormCuda(*in, r);//compute residual

    const double b2 = r2;

    printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);

    //copy the first vector:
    double beta = sqrt(r2);
    //
    double r2_inv = 1.0 / beta;//check beta!

    axCuda(r2_inv, r);
    //
    copyCuda(Vm->Eigenvec(0), r);

    int j = 0;//here also a column index of H

    //set g-vector:
    g[0] = beta;

    args->PrepareDeflatedRestart(givensH, g, use_deflated_cycles);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    PrintStats("GMResDR:", tot_iters, r2, b2, heavy_quark_res);

    profile->TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile->TPSTART(QUDA_PROFILE_COMPUTE);
    blas_flops = 0;

    while(j < m)//we allow full cycle
    {
      cudaColorSpinorField *Av = &Vm->Eigenvec(j+1);

      (*sloppy_mat)(*Av, Vm->Eigenvec(j), tmp, tmp2);

      ///////////
      Complex h0 = cDotProductCuda(Vm->Eigenvec(0), *Av);//
      args->SetHessenbergElement(0, j, h0);
      //scale w:
      caxpyCuda(-h0, Vm->Eigenvec(0), *Av);
      //
      for (int i = 1; i <= j; i++)//i is a row index
      {
        Complex h1 = cDotProductCuda(Vm->Eigenvec(i), *Av);//
        //load columns:
        args->SetHessenbergElement(i, j, h1);
        //scale:
        caxpyCuda(-h1, Vm->Eigenvec(i), *Av);

        //lets do Givens rotations:
        givensH[ldH*j+(i-1)]   =  conj(Cn[i-1])*h0 + Sn[i-1]*h1;
        //compute (j, i+1) element:
        h0          = -Sn[i-1]*h0 + Cn[i-1]*h1;
      }
      //Compute h(j+1,j):
      double h_jp1j = sqrt(norm2(*Av)); 

      //Update H-matrix (j+1) element:
      args->SetHessenbergElement((j+1), j, Complex(h_jp1j, 0.0));
      //
      axCuda(1. / h_jp1j, *Av);
    
      double inv_denom = 1.0 / sqrt(norm(h0)+h_jp1j*h_jp1j);
      //
      Cn[j] = h0 * inv_denom; 
      //
      Sn[j] = h_jp1j * inv_denom; 
      //produce diagonal element in G:
      givensH[ldH*j+j] = conj(Cn[j])*h0 + Sn[j]*h_jp1j; 
   
      //compute this in opposite direction:
      g[j+1] = -Sn[j]*g[j];  
      //
      g[j]   =  g[j] * conj(Cn[j]);
      //
      r2 = norm(g[j+1]);//stopping criterio
      //
      j += 1;

      tot_iters += 1;

      //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j); 
      PrintStats("GMResDR:", tot_iters, r2, b2, heavy_quark_res);

    }//end of GMRES loop

    printfQuda("\nDone for: %d iters, %1.15e last residual squared.\n", j, r2); 

    //j -> m+1
    //update solution and final residual:
    args->UpdateSolution(x_sloppy, Vm, givensH, g, j);

    if (mixed_precision_GMResDR)
    {
      copyCuda(y, *x_sloppy);
      xpyCuda(y, x);
      zeroCuda(*x_sloppy);
    }

   (*mat)(r, x, y);
   //
   r2 = xmyNormCuda(*in, r);//compute full precision residual

   printfQuda("\nDone for cycle 0, true residual squared %1.15e\n", r2);
   //
   PrintStats("GMResDR:", tot_iters, r2, b2, heavy_quark_res);

//****************BEGIN RESTARTS:

   const int max_cycles = param.deflation_grid;

   int cycle_idx = 1;

   bool last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

   //For the deflated cycles: just pointer aliases, for the projected cycles: new (half precision) objects
   cudaColorSpinorField *ctmp1_p = &tmp;
   cudaColorSpinorField *ctmp2_p = tmp2_p;

   cudaColorSpinorField *x_sloppy2 = x_sloppy;

   while(cycle_idx < max_cycles && !last_cycle)
   {
     printfQuda("\nRestart #%d\n", cycle_idx);

     if (mixed_precision_GMResDR) copyCuda(*r_sloppy, r);

     memset(givensH, 0, ldH*m*sizeof(Complex));
     memset(g, 0 , (m+1)*sizeof(Complex));
     //
     memset(Cn, 0 , m*sizeof(Complex));
     memset(Sn, 0 , m*sizeof(double));

     int j = nev;//here also a column index of H

     if(use_deflated_cycles)
     {
       args->ComputeHarmonicRitzPairs();

       args->RestartVH( Vm );

       for(int i = 0; i <= nev ; i++ ) g[i] = cDotProductCuda(Vm->Eigenvec(i), *r_sloppy);//
     }
     else //use Galerkin projection instead: 
     {
       j = 0; //we will launch "projected" GMRES cycles

       if(defl_param->projType == QUDA_INVALID_PROJECTION) defl_param->projType = QUDA_GALERKIN_PROJECTION;

       PerformProjection(*x_sloppy, *r_sloppy, defl_param);//note : full precision residual

       if (mixed_precision_GMResDR)
       {
         copyCuda(y, *x_sloppy);
         xpyCuda(y, x);
         zeroCuda(*x_sloppy);
         copyCuda(r, *r_sloppy);//ok
       }

       r2 = norm2(r);
       
       beta = sqrt(r2);
       //
       axCuda(1.0 / beta, r);
       //
       copyCuda(Vm->Eigenvec(0), r);

       g[0] = beta;

       printfQuda("\nNew residual %1.15e\n", beta);
     }

     args->PrepareDeflatedRestart(givensH, g, use_deflated_cycles);

     const int jlim = j;//=nev for deflated restarts and 0 for "projected" restarts (abused terminology!)

     while(j < m) //we allow full cycle
     {
       //pointer aliasing:
       cudaColorSpinorField *Av = &Vm->Eigenvec(j+1);

       (*sloppy_mat)(*Av, Vm->Eigenvec(j), *ctmp1_p, *ctmp2_p);
       //
       Complex h0(0.0, 0.0);
       //
       for (int i = 0; i <= jlim; i++)//i is a row index
       {
          h0 = cDotProductCuda(Vm->Eigenvec(i), *Av);//
          //Warning: column major format!!! (m+1) size of a column.
          //load columns:
          args->SetHessenbergElement(i, j, h0);
          //
          caxpyCuda(-h0, Vm->Eigenvec(i), *Av);
       }

       //Let's do Givens rotation:
       args->PrepareGivens(givensH, j, use_deflated_cycles);//check it! 

       if(use_deflated_cycles) h0 = givensH[ldH*j+jlim];

       for(int i = (jlim+1); i <= j; i++)
       {
          Complex h1 = cDotProductCuda(Vm->Eigenvec(i), *Av);//
          //Warning: column major format!!! (m+1) size of a column.
          //load columns:
          args->SetHessenbergElement(i, j, h1);
          //
          caxpyCuda(-h1, Vm->Eigenvec(i), *Av);

          //Lets do Givens rotations:
          givensH[ldH*j+(i-1)]   =  conj(Cn[i-1])*h0 + Sn[i-1]*h1;
          //compute (j, i+1) element:
          h0  = -Sn[i-1]*h0 + Cn[i-1]*h1;//G[(m+1)*j+i+1]
       }
       //Compute h(j+1,j):
       //
       double h_jp1j = sqrt(norm2(*Av)); 

       //Update H-matrix (j+1) element:
       args->SetHessenbergElement((j+1), j, Complex(h_jp1j,0.0));
       //
       axCuda(1. / h_jp1j, *Av);
 
       double inv_denom = 1.0 / sqrt(norm(h0)+h_jp1j*h_jp1j);
       //
       Cn[j] = h0 * inv_denom; 
       //
       Sn[j] = h_jp1j * inv_denom; 
       //produce diagonal element in G:
       givensH[ldH*j+j] = conj(Cn[j])*h0 + Sn[j]*h_jp1j; 
   
       //compute this in opposite direction:
       g[j+1] = -Sn[j]*g[j];  
       //
       g[j]   =  g[j] * conj(Cn[j]);
       //
       r2 = norm(g[j+1]);//stop criterio
       //
       j += 1;

       tot_iters += 1;
 
       //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j);
       PrintStats("GMResDR:", tot_iters, r2, b2, heavy_quark_res); 
     }//end of main loop.

     args->UpdateSolution(x_sloppy2, Vm, givensH, g, j);

     if (mixed_precision_GMResDR)
     {
       copyCuda(y, *x_sloppy2);
       xpyCuda(y, x);//don't mind arguments;)
       zeroCuda(*x_sloppy2);
     }

     (*mat)(r, x, y);
     //
     double ext_r2 = xmyNormCuda(*in, r);//compute full precision residual

     if(use_deflated_cycles && (mixed_precision_GMResDR && ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold)))
     {
       printfQuda("\nLaunch projection stage (%le)\n", sqrt(ext_r2 / r2));

       args->ComputeHarmonicRitzPairs();

       args->RestartVH( Vm );

       defl_param->LoadData(Vm, args->H, nev, ldH);

       //Allocate low precision fields:
       csParam.setPrecision(QUDA_HALF_PRECISION);

       delete Vm; 
       gmres_alloc = false;

       AllocateKrylovSubspace(csParam);

       x_sloppy2 = new cudaColorSpinorField(x, csParam);

       ctmp1_p   = new cudaColorSpinorField(csParam);

       ctmp2_p   = new cudaColorSpinorField(csParam);

       sloppy_mat = const_cast<DiracMatrix*> (matDefl);

       use_deflated_cycles = false;
     }

     printfQuda("\nDone for cycle:  %d, true residual squared %1.15e\n", cycle_idx, ext_r2);

     last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

     cycle_idx += 1;

   }//end of deflated restarts

   profile->TPSTOP(QUDA_PROFILE_COMPUTE);
   profile->TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile->Last(QUDA_PROFILE_COMPUTE);
   double gflops = (quda::blas_flops + mat->flops())*1e-9;
   reduceDouble(gflops);
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   (*mat)(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNormCuda(*in, r) / b2);

   PrintSummary("GMResDR:", tot_iters, r2, b2);

   // reset the flops counters
   quda::blas_flops = 0;
   mat->flops();

   profile->TPSTOP(QUDA_PROFILE_EPILOGUE);
   profile->TPSTART(QUDA_PROFILE_FREE);

   delete [] givensH;

   delete [] g;

   delete [] Cn;

   host_free(Sn);

   printfQuda("\n..done.\n");

   if (mixed_precision_GMResDR)
   {
     delete r_sloppy;
     delete x_sloppy;

     if(use_deflated_cycles == false)
     {
       printfQuda("\nDealocating resources from the projector cycles...\n");
       delete x_sloppy2;
       delete ctmp1_p;
       delete ctmp2_p;
     } 
   }

   delete tmp2_p;

   profile->TPSTOP(QUDA_PROFILE_FREE);

   return;
 } 

 void GMResDR::RunProjectedCycles(cudaColorSpinorField *out, cudaColorSpinorField *in, GMResDRDeflationParam *dpar, const bool enforce_mixed_precision)
 {
     bool mixed_precision_gmres = enforce_mixed_precision || (param.precision != param.precision_sloppy);

     profile->TPSTART(QUDA_PROFILE_INIT);

     DiracMatrix *sloppy_mat;

     cudaColorSpinorField r(*in); //high precision residual 
     // 
     cudaColorSpinorField &x = *out;
     //
     cudaColorSpinorField y(*in); //high precision aux field 

     //Sloppy precision fields:
     ColorSpinorParam csParam(*in);//create spinor parameters

     csParam.create = QUDA_ZERO_FIELD_CREATE;

     cudaColorSpinorField *r_sloppy = NULL, *x_sloppy = NULL;

     csParam.setPrecision(gmres_space_prec);//this is a solo precision solver

     cudaColorSpinorField *r_sloppy_proj = NULL;
     cudaColorSpinorField *x_sloppy_proj = NULL;

     if (mixed_precision_gmres) 
     {
        if(gmres_space_prec != QUDA_HALF_PRECISION)
        {
           r_sloppy_proj = new cudaColorSpinorField(r, csParam);
           x_sloppy_proj = new cudaColorSpinorField(x, csParam);
        }

        csParam.setPrecision(QUDA_HALF_PRECISION);

        sloppy_mat = const_cast<DiracMatrix*> (matDefl);
        r_sloppy = new cudaColorSpinorField(r, csParam);
        x_sloppy = new cudaColorSpinorField(x, csParam);
     } 
     else 
     {
        sloppy_mat = matSloppy;
        r_sloppy = &r;
        x_sloppy = &x;
     }

     if(!r_sloppy_proj) r_sloppy_proj = r_sloppy;
     if(!x_sloppy_proj) x_sloppy_proj = x_sloppy;

     cudaColorSpinorField tmp(*in, csParam);

     cudaColorSpinorField *tmp2_p;

     tmp2_p = new cudaColorSpinorField(*in, csParam);

     cudaColorSpinorField &tmp2 = *tmp2_p;

     //Allocate Vm array:
     if(gmres_alloc == false)
     {
       AllocateKrylovSubspace(csParam);
     }

     profile->TPSTOP(QUDA_PROFILE_INIT);
     profile->TPSTART(QUDA_PROFILE_PREAMBLE);

     const int m         = args->m;
     const int ldH       = args->ldm;

     int tot_iters = 0;

     //GMRES objects:
     //Givens rotated matrix (m+1, m):
     Complex *givensH = new Complex[ldH*m];//complex
    
     //Auxilary objects: 
     Complex *g = new Complex[(m+1)];

     //Givens coefficients:
     Complex *Cn = new Complex[m];
     //
     double *Sn = (double *) safe_malloc(m*sizeof(double));
     memset(Sn, 0, m * sizeof(double));

     //Compute initial residual:
     const double normb = norm2(*in);  
     double stop = param.tol*param.tol* normb;	/* Relative to b tolerance */

     (*mat)(r, *out, y);
     //
     double r2 = xmyNormCuda(*in, r);//compute residual

     const double b2 = r2;

     printfQuda("\nInitial residual: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);

     //copy the first vector:
     double beta = sqrt(r2);
     //
     const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

     double heavy_quark_res = 0.0; // heavy quark residual

     if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);

     int heavy_quark_check = 10; // how often to check the heavy quark residual

     PrintStats("GMRES-Proj:", tot_iters, r2, b2, heavy_quark_res);

     profile->TPSTOP(QUDA_PROFILE_PREAMBLE);
     profile->TPSTART(QUDA_PROFILE_COMPUTE);
     blas_flops = 0;

//BEGIN PROJECTED RESTARTS:
     const int max_cycles = param.deflation_grid;

     int cycle_idx = 0;

     bool last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

     while(cycle_idx < max_cycles && !last_cycle)
     {
       printfQuda("\nRestart #%d\n", cycle_idx);

       cycle_idx += 1;

       if (mixed_precision_gmres) copyCuda(*r_sloppy_proj, r);

       memset(givensH, 0, ldH*m*sizeof(Complex));
       memset(g, 0 , (m+1)*sizeof(Complex));
       //
       memset(Cn, 0 , m*sizeof(Complex));
       memset(Sn, 0 , m*sizeof(double));

       bool do_projection = defl_param->projFreq == 1 ? true : (cycle_idx % defl_param->projFreq) == 0 ? true : false; 

       if(do_projection) PerformProjection(*x_sloppy_proj, *r_sloppy_proj, defl_param);//note : full precision residual

       if (mixed_precision_gmres)
       {
         copyCuda(y, *x_sloppy_proj);
         xpyCuda(y, x);
         zeroCuda(*x_sloppy_proj);
         copyCuda(r, *r_sloppy_proj);
       }

       r2 = norm2(r);
       
       beta = sqrt(r2);
       //
       axCuda(1.0 / beta, r);
       //
       copyCuda(Vm->Eigenvec(0), r);

       g[0] = beta;

       printfQuda("\nResidual after projection %1.15e\n", beta);

       int j = 0;//here also a column index of H

      while(j < m)//we allow full cycle
      {
        cudaColorSpinorField *Av = &Vm->Eigenvec(j+1);

        (*sloppy_mat)(*Av, Vm->Eigenvec(j), tmp, tmp2);//must be matDefl

        ///////////
        Complex h0 = cDotProductCuda(Vm->Eigenvec(0), *Av);//
        args->SetHessenbergElement(0, j, h0);
        //scale w:
        caxpyCuda(-h0, Vm->Eigenvec(0), *Av);
        //
        for (int i = 1; i <= j; i++)//i is a row index
        {
          Complex h1 = cDotProductCuda(Vm->Eigenvec(i), *Av);//
          //load columns:
          args->SetHessenbergElement(i, j, h1);
          //scale:
          caxpyCuda(-h1, Vm->Eigenvec(i), *Av);

          //lets do Givens rotations:
          givensH[ldH*j+(i-1)]   =  conj(Cn[i-1])*h0 + Sn[i-1]*h1;
          //compute (j, i+1) element:
          h0          = -Sn[i-1]*h0 + Cn[i-1]*h1;
        }
        //Compute h(j+1,j):
        double h_jp1j = sqrt(norm2(*Av)); 
        //Update H-matrix (j+1) element:
        args->SetHessenbergElement((j+1), j, Complex(h_jp1j, 0.0));
        //
        axCuda(1. / h_jp1j, *Av);
    
        double inv_denom = 1.0 / sqrt(norm(h0)+h_jp1j*h_jp1j);
        //
        Cn[j] = h0 * inv_denom; 
        //
        Sn[j] = h_jp1j * inv_denom; 
        //produce diagonal element in G:
        givensH[ldH*j+j] = conj(Cn[j])*h0 + Sn[j]*h_jp1j; 
   
        //compute this in opposite direction:
        g[j+1] = -Sn[j]*g[j];  
        //
        g[j]   =  g[j] * conj(Cn[j]);
        //
        r2 = norm(g[j+1]);//stopping criterio
        //
        j += 1;

        tot_iters += 1;

        //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j); 
        PrintStats("GMRES-Proj:", tot_iters, r2, b2, heavy_quark_res);

     }//end of GMRES loop

     args->UpdateSolution(x_sloppy, Vm, givensH, g, j);

     if (mixed_precision_gmres)
     {
       copyCuda(y, *x_sloppy);
       xpyCuda(y, x);//don't mind arguments;)
       zeroCuda(*x_sloppy);
     }

     (*mat)(r, x, y);
     //
     double ext_r2 = xmyNormCuda(*in, r);//compute full precision residual

     printfQuda("\nDone for cycle:  %d, true residual squared %1.15e\n", cycle_idx, ext_r2);

     last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

   }//end of deflated restarts

   profile->TPSTOP(QUDA_PROFILE_COMPUTE);
   profile->TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile->Last(QUDA_PROFILE_COMPUTE);
   double gflops = (quda::blas_flops + mat->flops())*1e-9;
   reduceDouble(gflops);
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   (*mat)(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNormCuda(*in, r) / b2);

   PrintSummary("GMRES-Proj:", tot_iters, r2, b2);

   // reset the flops counters
   quda::blas_flops = 0;
   mat->flops();

   profile->TPSTOP(QUDA_PROFILE_EPILOGUE);
   profile->TPSTART(QUDA_PROFILE_FREE);

    delete [] givensH;

   delete [] g;

   delete [] Cn;

   host_free(Sn);

   printfQuda("\n..done.\n");

   if (mixed_precision_gmres)
   {
     delete r_sloppy;
     delete x_sloppy; 

     if(gmres_space_prec != QUDA_HALF_PRECISION)
     {
       delete r_sloppy_proj;
       delete x_sloppy_proj;
     }
   }

   delete tmp2_p;

   profile->TPSTOP(QUDA_PROFILE_FREE);

   return;

 }

 void GMResDR::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in)
 {
   if(!defl_param)
   {
     defl_param = new GMResDRDeflationParam(args->ldm, args->nev);
   }

   const double tol_threshold = 6.0;//for mixed precision version only.
   const bool   use_gmresproj_mixed_precision = true;

   if(param.inv_type == QUDA_GMRESDR_INVERTER || (param.inv_type == QUDA_GMRESDR_PROJ_INVERTER && param.rhs_idx == 0))
   {
     //run GMResDR cycles:
     args->InitGMResDRArgs();
     //
     RunDeflatedCycles(out, in, defl_param, tol_threshold);
   }
   else
   {
     //run GMRES-Proj cycles:
     RunProjectedCycles(out, in, defl_param, use_gmresproj_mixed_precision); //last argument enforces mixed precision for this stage
   }

   if(param.inv_type == QUDA_GMRESDR_PROJ_INVERTER && defl_param->init_flag == false) 
   {
     printfQuda("\nLoad eigenvectors for the projection stage.\n");
     defl_param->projType = QUDA_MINRES_PROJECTION;  
//     defl_param->projType = QUDA_GALERKIN_PROJECTION;

     args->ComputeHarmonicRitzPairs();
     args->RestartVH( Vm );

     defl_param->LoadData(Vm, args->H, args->nev, args->ldm);
   }

   param.rhs_idx += 1;

   return;
 }

} // namespace quda
