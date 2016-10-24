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
FGCRO-DR algorithm:
L.M.Carvalho et al, "A flexible Generalized Conjugate Residual method with inner orthogonalization and Deflated restarting", 

*/

namespace quda {
//Notes:
//FGCRODR does not require large m (and esp. nev), so this will use normal LAPACK routines.

    using namespace blas;

    static FGCRODRDeflationParam *defl_param = 0;

    struct SortEvals{

      double eval_nrm;
      int    eval_idx;

      SortEvals(double val, int idx) : eval_nrm(val), eval_idx(idx) {};

      static bool CmpEigenNrms (SortEvals v1, SortEvals v2) { return (v1.eval_nrm < v2.eval_nrm);}

    };

    // set the required parameters for the inner solver
    void fillInnerSolveParam(SolverParam &inner, const SolverParam &outer) {
      inner.tol = outer.tol_precondition;
      inner.maxiter = outer.maxiter_precondition;
      inner.delta = 1e-20; // no reliable updates within the inner solver
  
      inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
      inner.precision_sloppy = outer.precision_precondition;
  
      inner.iter = 0;
      inner.gflops = 0;
      inner.secs = 0;

      inner.inv_type_precondition = QUDA_INVALID_INVERTER;
      inner.is_preconditioner = true; // tell inner solver it is a preconditionis_re

      inner.global_reduction = false;

      inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

      if (outer.inv_type == QUDA_GCR_INVERTER && outer.precision_sloppy != outer.precision_precondition) 
        inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
      else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
    }

    struct FGCRODRDeflationParam
    {

      //Objects required for the Galerkin projections
      Complex *projMat;  //host   projection matrix:
      ColorSpinorFieldSet    *projVecs;      //device buffer for HRitz vectors

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

      FGCRODRDeflationParam(const int ldm, const int nev, bool alloc = false, int proj_freq = 1, QudaProjectionType proj_type = QUDA_INVALID_PROJECTION) : projMat(0), projVecs(0),
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

      void LoadData(ColorSpinorField  *Vm, Complex *srcMat, const int nevs, const int ldn)
      {

         if(init_flag)                 errorQuda("Error: data was already initialized.");

         if(ldn != ld)                 errorQuda("Error: leading dimension of the source matrix must match that of the projection matrix.");

         if(Vm->CompositeDim() < (nevs+1))  errorQuda("Error: it seems that the provided eigenvector content is inconsistent with the projection matrix dimensions: %d vs %d", Vm->CompositeDim(), (nevs+1));

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

         ColorSpinorParam csParam(Vm->Component(0));
         csParam.create = QUDA_ZERO_FIELD_CREATE;
         csParam.is_composite   = true;
         csParam.composite_dim  = (nv+1);

         projVecs  = ColorSpinorFieldSet::Create(csParam);//temporary field

         for(int i = 0; i < (nv+1); i++) copy(projVecs->Component(i), Vm->Component(i));

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

      ~FGCRODRDeflationParam(){
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


    class FGCRODRArgs{

      private:
      BlasMagmaArgs *FGCRODR_magma_args;
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

      Complex *givensH;//
      Complex *g;
      //Givens coefficients:
      Complex *Cn;
      //
      double *Sn;

      int m;
      int ldm;//leading dimension (must be >= m+1)
      int nev;//number of harmonic eigenvectors used for the restart

      bool init_flag;
      //public:

      FGCRODRArgs( ) { };

      FGCRODRArgs(int m, int ldm, int nev);

      ~FGCRODRArgs();
      //more implementations here:
      void InitFGCRODRArgs();//allocate FGCRODR specific objects, set init_flag to true

      void ResetHessenberg();

      void ConjugateH();//rows = cols+1

      void UpdateHessenberg(const int row, const int col, const Complex &el) { H[col*ldm+row]  = el; cH[row*ldm+col] = conj(el); }
      //
      void UpdateGivens(const int row, const int col, Complex &h0, Complex &h1);
      //
      double UpdateGivensCoeff(const int row, const int col, Complex &h0, double &h1);
      //
      void ComputeHarmonicRitzPairs();

      void RestartZVH(ColorSpinorFieldSet *Vm, ColorSpinorFieldSet *Zm);

      void PrepareCycle(ColorSpinorFieldSet *Vm, ColorSpinorFieldSet *Zm, ColorSpinorField &r, const bool use_deflated_cycles = false);

      void PrepareGivens(const int col, const bool use_deflated_cycles = false);

      void UpdateSolution(ColorSpinorField *x, ColorSpinorFieldSet *Wm, const int j);
   };

  FGCRODRArgs::FGCRODRArgs(int m, int ldm, int nev): FGCRODR_magma_args(0), harVecs(0), harVals(0), sortedHarVecs(0), sortedHarVals(0), harMat(0), qrH(0), tauH(0), srtRes(0),
  m(m), ldm(ldm), nev(nev), init_flag(false) {

     int mp1 = m+1;

     if(ldm < mp1) errorQuda("\nError: the leading dimension is not correct.\n");

     if(nev == 0) errorQuda("\nError: incorrect deflation size..\n");

     H        = new Complex[ldm*m];//Hessenberg matrix
     cH       = new Complex[ldm*mp1];//conjugate Hessenberg matrix
     lsqSol   = new Complex[(m+1)];

     givensH = new Complex[ldm*m];//complex
     g = new Complex[(m+1)];
     //Givens coefficients:
     Cn = new Complex[m];
     Sn = (double *) safe_malloc(m*sizeof(double));
     memset(Sn, 0, m * sizeof(double));

     return;
  }

  void FGCRODRArgs::InitFGCRODRArgs()
  {
     if(init_flag) errorQuda("\nFGCRODR resources were allocated.\n");
     //magma library initialization:
     FGCRODR_magma_args = new BlasMagmaArgs(m, nev+1, ldm, sizeof(double));

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

  void FGCRODRArgs::ResetHessenberg()
  {
    memset(H,  0, ldm*m*sizeof(Complex));
    //
    memset(cH, 0, ldm*(m+1)*sizeof(Complex));

    return;
  }

  void FGCRODRArgs::ConjugateH()//rows = cols+1
  {
    for(int c = 0; c < nev; c++ )
      for(int r = 0; r < (nev+1); r++ ) cH[r*ldm+c] = conj(H[c*ldm+r]);

    return;
  }

  double FGCRODRArgs::UpdateGivensCoeff(const int row, const int col, Complex &h0, double &h1)
  {
     double inv_denom = 1.0 / sqrt(norm(h0)+h1*h1);
     //
     Cn[row] = h0 * inv_denom;
     //
     Sn[row] = h1 * inv_denom;
     //lets do Givens rotations:
     givensH[ldm*col+row]   =  conj(Cn[row])*h0 + Sn[row]*h1;
     //compute this in opposite direction:
     g[row+1] = -Sn[row]*g[row];
     //
     g[row]   =  g[row] * conj(Cn[row]);
     //
     return norm(g[row+1]);//stopping criterio
  }


  void FGCRODRArgs::UpdateGivens(const int row, const int col, Complex &h0, Complex &h1)
  {
     //lets do Givens rotations:
     givensH[ldm*col+row]   =  conj(Cn[row])*h0 + Sn[row]*h1;
     //compute (j, i+1) element:
     h0                     = -Sn[row]*h0 + Cn[row]*h1;

     return;
  }

  FGCRODRArgs::~FGCRODRArgs() {

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

      delete[] givensH;
      delete[] g;
      delete[] Cn;
      host_free(Sn);      

      delete FGCRODR_magma_args;
    }

    return;
  }


//Note: in fact, x is an accumulation field initialized to zero (if a sloppy field is used!) and has the same precision as Vm
 void FGCRODRArgs::UpdateSolution(ColorSpinorField *x, ColorSpinorFieldSet *Wm, const int j)
 {
   //if (mixed_precision_FGCRODR) zero(*x);//it's done in the main loop, no need for this
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
   //compute x = x0+Wm*lsqSol, where Wm is Vm or Zm (if preconditioned)
   for(int l = 0; l < j; l++)  caxpy(lsqSol[l], Wm->Component(l), *x);

   return;
 }

 void FGCRODRArgs::ComputeHarmonicRitzPairsStrategyA()
 {
   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");
   //Compute H^{bar}^{H}_{m}H_m y = \theta (H^{bar}_{m} V_m+1 Zm) y;  
   return;
 }


 void FGCRODRArgs::ComputeHarmonicRitzPairsStrategyB()
 {
   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");

   memcpy(harMat, H, ldm*m*sizeof(Complex));

   const double beta2 = norm(H[ldm*(m-1)+m]);

   FGCRODR_magma_args->Construct_harmonic_matrix(harMat, cH, beta2, m, ldm);

   // Computed eigenpairs for harmonic matrix:
   // Save harmonic matrix :
   memset(cH, 0, ldm*m*sizeof(Complex));

   memcpy(cH, harMat, ldm*m*sizeof(Complex));

   FGCRODR_magma_args->Compute_harmonic_matrix_eigenpairs(harMat, m, ldm, harVecs, harVals, ldm);//check it!

//   for(int e = 0; e < m; e++) printf("\nEigenval #%d: %le, %le, %le\n", e, harVals[e].real(), harVals[e].imag(), abs(harVals[e]));
   //do sort:
   std::vector<SortEvals> sorted_evals_cntr;

   sorted_evals_cntr.reserve(m);

   for(int e = 0; e < m; e++) sorted_evals_cntr.push_back( SortEvals( abs(harVals[e]), e ));

   std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);

   for(int e = 0; e < nev; e++) memcpy(&sortedHarVecs[ldm*e], &harVecs[ldm*( sorted_evals_cntr[e].eval_idx)], (ldm)*sizeof(Complex));

   return;
 }

 void FGCRODRArgs::ComputeHarmonicRitzPairsStrategyC()
 {
   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");
   //Compute H^{bar}^{H}_{m}H_m y = \theta (H^{bar}_{m} V_m+1 Wm) y; 
   return;
 }

 void FGCRODRArgs::RestartZVH(ColorSpinorFieldSet *Vm, ColorSpinorFieldSet *Zm )
 {
   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");

   int cldn = Vm->ComponentLength() >> 1; //complex leading dimension
   int clen = Vm->ComponentLength() >> 1; //complex vector length

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
//@@@
   if( K ) FGCRODR_magma_args->RestartZVH(Zm->V(), Vm->V(), clen, cldn, Vm->Precision(), sortedHarVecs, H, ldm);//check ldm with internal (magma) ldm
   else    FGCRODR_magma_args->RestartVH (Vm->V(), clen, cldn, Vm->Precision(), sortedHarVecs, H, ldm);//check ldm with internal (magma) ldm

   checkCudaError();

   /*****REORTH V_{nev+1}:****/

   for(int j = 0; j < nev; j++)
   {
     Complex calpha = cDotProduct(Vm->Component(j), Vm->Component(nev));//
     caxpy(-calpha, Vm->Component(j), Vm->Component(nev));
   }

   double dalpha = norm2(Vm->Component(nev));

   double tmpd=1.0e+00/sqrt(dalpha);

   ax(tmpd, Vm->Component(nev));

   //done.

   return;
 }


 void FGCRODRArgs::PrepareCycle(ColorSpinorFieldSet *Vm, ColorSpinorFieldSet *Zm, ColorSpinorField &r, const bool run_deflated_cycle)
 {
   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");

   memset(givensH, 0, ldm*m*sizeof(Complex));
   memset(g,       0, (m+1)*sizeof(Complex));
   memset(Cn,      0, m*sizeof(Complex));
   memset(Sn,      0, m*sizeof(double));

   if(run_deflated_cycle)
   {
     ComputeHarmonicRitzPairs();
     RestartZVH( Vm, Zm );

     for(int i = 0; i <= nev ; i++ ) g[i] = cDotProduct(Vm->Component(i), r);//

     memset(cH, 0, ldm*(m+1)*sizeof(Complex));
     memset(harMat, 0, ldm*m*sizeof(Complex));
     //
     //now update conjugate matrix
     for(int j = 0 ; j < nev; j++)
       for(int i = 0 ; i < nev+1; i++)
         cH[ldm*i+j] = conj(H[ldm*j+i]);

     memcpy(qrH, H, ldm*nev*sizeof(Complex));

     FGCRODR_magma_args->ComputeQR(nev, qrH, (nev+1), ldm, tauH);

     //extract triangular part to the givens matrix:
     for(int i = 0; i < nev; i++) memcpy(&givensH[ldm*i], &qrH[ldm*i], (i+1)*sizeof(Complex));

     FGCRODR_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, g, (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);
   } else {
     double r2 = norm2(r);//compute residual
     printfQuda("\nInitial/restarted residual squared: %1.16e\n", sqrt(r2));
     //copy the first vector:
     double beta = sqrt(r2);
     //scale residual vector:
     ax(1.0 / beta, r);
     //
     copy(Vm->Component(0), r);
     //set g-vector:
     g[0] = beta;
     //update initial values for the "short residual"
     memcpy(srtRes, g, (m+1)*sizeof(Complex));
   }

   return;
 }

 void FGCRODRArgs::PrepareGivens(const int col, const bool use_deflated_cycles)
 {
   if(!use_deflated_cycles) return; //we are in the "projected" cycle nothing to do..

   if(!init_flag) errorQuda("\nFGCRODR resources were not allocated.\n");

   memcpy(&givensH[ldm*col], &H[ldm*col], (nev+1)*sizeof(Complex) );
   //
   FGCRODR_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, &givensH[ldm*col], (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);

   return;
 }


 FGCRODR::FGCRODR(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matDefl, DiracMatrix *matPrecon, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), matPrecon(matPrecon), K(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr), gmres_alloc(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision FGCRODR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     //create FGCRODR objects:
     int mp1    = param.m + 1;
     int ldm    = mp1;//((mp1+15)/16)*16;//leading dimension

     args = new FGCRODRArgs(param.m, ldm, param.nev);//use_deflated_cycles flag is true by default

     fillInnerSolveParam(Kparam, param);

     if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
       K = new CG(matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
       K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
       K = new MR(matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner MR preconditioner
       K = new SD(matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unknown preconditioner
       K = nullptr;
     else 
       errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);

     return;
 }

 FGCRODR::FGCRODR(DiracMatrix *mat, Solver &K, DiracMatrix *matSloppy, DiracMatrix *matDefl, DiracMatrix *matPrecon, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), matPrecon(matPrecon), K(&K), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr), gmres_alloc(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision FGCRODR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     //create FGCRODR objects:
     int mp1    = param.m + 1;
     int ldm    = mp1;//((mp1+15)/16)*16;//leading dimension

     args = new FGCRODRArgs(param.m, ldm, param.nev);//use_deflated_cycles flag is true by default

     return;
 }

 FGCRODR::FGCRODR(SolverParam &param) :
    DeflatedSolver(param, nullptr), mat(nullptr), matSloppy(nullptr), matDefl(nullptr), matPrecon(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), profile(nullptr), args(nullptr), gmres_alloc(false) { }


 FGCRODR::~FGCRODR() {

    if(args) delete args;

    if(gmres_alloc)
    {
      delete Vm;
      Vm = nillptr;
      if(K) delete Zm;
    }

 }

 void FGCRODR::AllocateKrylovSubspace(ColorSpinorParam &csParam)
 {
   if(gmres_alloc) errorQuda("\nKrylov subspace was allocated.\n");

   printfQuda("\nAllocating resources for the FGCRODR solver...\n");

   csParam.is_composite  = true;
   csParam.composite_dim = param.m+1;
   Vm       = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

   csParam.composite_dim = param.m;
   if(K) Zm = ColorSpinorFieldSet::Create(csParam);
   else  Zm = Vm;

   csParam.is_composite  = false;
   csParam.composite_dim = 0;

   checkCudaError();
   printfQuda("\n..done.\n");

   gmres_alloc = true;

   return;
 }

 void FGCRODR::PerformProjection(ColorSpinorField &x_sloppy,  ColorSpinorField &r_sloppy, FGCRODRDeflationParam *dpar)
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
      for(int i = 0; i < dpar->nv; i++ ) d[i] = cDotProduct(dpar->projVecs->Component(i), r_sloppy);//
      //Solve H^{pr}_k d = c: this nvxnv problem..
      magma_args.SolveProjMatrix((void*)d, dpar->ld,  dpar->nv, (void*)dpar->projMat, dpar->ld);
    }
    else if(dpar->projType == QUDA_MINRES_PROJECTION)
    {
      //Compute c = VT^{pr}_k+1 r0
      for(int i = 0; i < (dpar->nv + 1); i++ ) d[i] = cDotProduct(dpar->projVecs->Component(i), r_sloppy);//
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
    for(int l = 0; l < dpar->nv; l++)  caxpy(d[l], dpar->projVecs->Component(l), x_sloppy);

    //Compute the new residual vector:
    //memset(c, 0, (dpar->nv+1)*sizeof(Complex));//now use as a temp array

    for(int j = 0; j < (dpar->nv+1); j++)
      for (int i = 0; i < (dpar->nv); i++)
         c[j] +=  (dpar->projMat[i*dpar->ld + j] * d[i]);//column major format

    for(int l = 0; l < (dpar->nv+1); l++)  caxpy(-c[l], dpar->projVecs->Component(l), r_sloppy);

    delete[] d;
    delete[] c;

    return;
 }

 void FGCRODR::CleanResources()
 {
   if(defl_param)
   {
     delete defl_param;
     defl_param = 0;
   }

   return;
 }

 void FGCRODR::RunDeflatedCycles(ColorSpinorField *out, ColorSpinorField *in, FGCRODRDeflationParam *dpar, const double tol_threshold /*=2.0*/)
 {
    bool mixed_precision_FGCRODR = (param.precision != param.precision_sloppy);

    profile->TPSTART(QUDA_PROFILE_INIT);

    DiracMatrix *sloppy_mat = matSloppy;

    //Full precision fields:
    ColorSpinorField *yp = ColorSpinorField::Create(*in); //high precision accumulation field
    ColorSpinorField *rp = ColorSpinorField::Create(*in); //high precision residual
    ColorSpinorField &r = *rp;
    ColorSpinorField &x = *out;
    ColorSpinorField &y = *yp;

    //Sloppy precision fields:
    ColorSpinorParam csParam(*in);//create spinor parameters

    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(gmres_space_prec);

    ColorSpinorField *r_sloppy = (mixed_precision_FGCRODR) ? ColorSpinorField::Create(csParam) : &r;
    ColorSpinorField *x_sloppy = (mixed_precision_FGCRODR) ? ColorSpinorField::Create(csParam) : &x;
    ColorSpinorField *tmp1_p   = ColorSpinorField::Create(csParam);

    ColorSpinorField &tmp = *tmp1_p;

    bool precMatch = ( param.precision_precondition != param.precision_sloppy ) ? false : true;

    ColorSpinorField *r_pre = nullptr, *p_pre = nullptr;

    if ( !precMatch && K) {
      csParam.setPrecision(param.precision_precondition);
      p_pre = ColorSpinorField::Create(csParam);
      r_pre = ColorSpinorField::Create(csParam);
    }

    //Allocate Vm array:
    if(gmres_alloc == false) AllocateKrylovSubspace(csParam);

    profile->TPSTOP(QUDA_PROFILE_INIT);
    profile->TPSTART(QUDA_PROFILE_PREAMBLE);

    int tot_iters = 0;

    //Compute initial residual:
    const double normb = norm2(*in);
    double stop = param.tol*param.tol* normb;	/* Relative to b tolerance */

    (*mat)(r, *out, y);
    //
    double r2 = xmyNorm(*in, r);//compute residual
    const double b2 = r2;

    printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);

    args->PrepareCycle(Vm, Zm, r);//pass initial residual

    bool use_deflated_cycles = true;

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    if (use_heavy_quark_res) errorQuda("Heavy-quark residual not supported in this solver");
    double heavy_quark_res = 0.0;

    PrintStats("FGCRODR:", tot_iters, r2, b2, heavy_quark_res);

    profile->TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile->TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    //run initial cycle:
    int j  = 0;

    while( j < args->m )//we allow full cycle
    {
      ColorSpinorField *Av = dynamic_cast<ColorSpinorField*>( &Vm->Component(j+1));

      if(K) {
        if( !precMatch ) copy(*r_pre, Vm->Component(j));
        else
        {             
          r_pre = &Vm->Component(j);
          p_pre = &Zm->Component(j);
        }

        (*K)( *p_pre ,*r_pre );

        if( !precMatch ) copy(Zm->Component(j), p_pre);
      }

      (*sloppy_mat)(*Av, Zm->Component(j), tmp);

      ///////////
      Complex h0 = cDotProduct(Vm->Component(j), *Av);//
      args->UpdateHessenberg(0, j, h0);
      //scale w:
      caxpy(-h0, Vm->Component(j), *Av);
      //
      for (int i = 1; i <= j; i++)//i is a row index
      {
        Complex h1 = cDotProduct(Vm->Component(i), *Av);//
        //load columns:
        args->UpdateHessenberg(i, j, h1);
        //scale:
        caxpy(-h1, Vm->Component(i), *Av);
        //lets do Givens rotations:
        args->UpdateGivens(j, (i-1), h0, h1);//stopping criterio
      }
      //Compute h(j+1,j):
      double h_jp1j = sqrt(norm2(*Av));

      //Update H-matrix (j+1) element:
      args->UpdateHessenberg((j+1), j, Complex(h_jp1j, 0.0));
      //
      ax(1. / h_jp1j, *Av);
      //
      double r2 = args->UpdateGivensCoeff(j, j, h0, h_jp1j);//return current stopping residual norm sq.
      //
      j += 1;

      tot_iters += 1;

      //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j);
      PrintStats("FGCRODR:", tot_iters, r2, b2, heavy_quark_res);

    }//end of GMRES loop

    printfQuda("\nDone for: %d iters, %1.15e last residual squared.\n", j, r2);

    //j -> m+1
    //update solution and final residual:
    args->UpdateSolution(x_sloppy, Zm, j);

    if (mixed_precision_FGCRODR)
    {
      copy(y, *x_sloppy);
      xpy(y, x);
      zero(*x_sloppy);
    }

   (*mat)(r, x, y);
   //
   r2 = xmyNorm(*in, r);//compute full precision residual

   printfQuda("\nDone for cycle 0, true residual squared %1.15e\n", r2);
   //
   PrintStats("FGCRODR:", tot_iters, r2, b2, heavy_quark_res);

//****************BEGIN RESTARTS:

   const int max_cycles = param.deflation_grid;

   int cycle_idx = 1;

   bool last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

   //For the deflated cycles: just pointer aliases, for the projected cycles: new (half precision) objects
   ColorSpinorField *ctmp1_p = &tmp;

   ColorSpinorField *x_sloppy2 = x_sloppy;

   while(cycle_idx < max_cycles && !last_cycle)
   {
     printfQuda("\nRestart #%d\n", cycle_idx);

     if (mixed_precision_FGCRODR) copy(*r_sloppy, r);

     j = args->nev;//here also a column index of H

     if( use_deflated_cycles ) //use Galerkin projection instead:
     {
       j = args->nev;//here also a column index of H
       args->PrepareCycle(Vm, Zm, *r_sloppy, true);

     } else {
       j = 0; //we will launch "projected" GMRES cycles

       if(defl_param->projType == QUDA_INVALID_PROJECTION) defl_param->projType = QUDA_GALERKIN_PROJECTION;

       PerformProjection(*x_sloppy, *r_sloppy, defl_param);//note : full precision residual

       if (mixed_precision_FGCRODR)
       {
         copy(y, *x_sloppy);
         xpy(y, x);
         zero(*x_sloppy);
         copy(r, *r_sloppy);//ok
       }
       args->PrepareCycle(Vm, nullptr, r);
     }

     const int jlim = j;//=nev for deflated restarts and 0 for "projected" restarts (abused terminology!)

     while(j < args->m) //we allow full cycle
     {
       //pointer aliasing:
       ColorSpinorField *Av = dynamic_cast<ColorSpinorField*>(&Vm->Component(j+1));

       if(K) {
         if( !precMatch ) copy(*r_pre, Vm->Component(j));
         else
         {             
           r_pre = &Vm->Component(j);
           p_pre = &Zm->Component(j);
         }
         (*K)( *p_pre ,*r_pre );

         if( !precMatch ) copy(Zm->Component(j), p_pre);
       }

       (*sloppy_mat)(*Av, Zm->Component(j), *ctmp1_p);
       //
       Complex h0(0.0, 0.0);
       //
       for (int i = 0; i <= jlim; i++)//i is a row index
       {
          h0 = cDotProduct(Vm->Component(i), *Av);//
          //Warning: column major format!!! (m+1) size of a column.
          //load columns:
          args->UpdateHessenberg(i, j, h0);
          //
          caxpy(-h0, Vm->Component(i), *Av);
       }

       //Let's do Givens rotation:
       args->PrepareGivens(j, use_deflated_cycles);//check it!

       if(use_deflated_cycles) h0 = args->givensH[args->ldm*j+jlim];

       for(int i = (jlim+1); i <= j; i++)
       {
          Complex h1 = cDotProduct(Vm->Component(i), *Av);//
          //Warning: column major format!!! (m+1) size of a column.
          //load columns:
          args->UpdateHessenberg(i, j, h1);
          //
          caxpy(-h1, Vm->Component(i), *Av);
          //lets do Givens rotations:
          args->UpdateGivens(j, (i-1), h0, h1);
       }
       //Compute h(j+1,j):
       //
       double h_jp1j = sqrt(norm2(*Av));

       //Update H-matrix (j+1) element:
       args->UpdateHessenberg((j+1), j, Complex(h_jp1j,0.0));
       //
       ax(1. / h_jp1j, *Av);
       //
       double r2 = args->UpdateGivensCoeff(j, j, h0, h_jp1j);//stopping criterio
       //
       j += 1;

       tot_iters += 1;

       //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j);
       PrintStats("FGCRODR:", tot_iters, r2, b2, heavy_quark_res);
     }//end of main loop.

     args->UpdateSolution(x_sloppy2, Zm, j);

     if (mixed_precision_FGCRODR)
     {
       copy(y, *x_sloppy2);
       xpy(y, x);//don't mind arguments;)
       zero(*x_sloppy2);
     }

     (*mat)(r, x, y);
     //
     double ext_r2 = xmyNorm(*in, r);//compute full precision residual

     if(use_deflated_cycles && (mixed_precision_FGCRODR && ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold)))
     {
       printfQuda("\nLaunch projection stage (%le)\n", sqrt(ext_r2 / r2));

       args->ComputeHarmonicRitzPairs();

       args->RestartZVH( Vm, Zm );

       defl_param->LoadData(Vm, Zm, args->H, args->nev, args->ldm);//Vm or Zm

       //Allocate low precision fields:
       csParam.setPrecision(QUDA_HALF_PRECISION);

       delete Vm;
       if(K) delete Zm;

       gmres_alloc = false;

       AllocateKrylovSubspace(csParam);

       x_sloppy2 = ColorSpinorField::Create(x, csParam);

       ctmp1_p   = ColorSpinorField::Create(csParam);

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
   double gflops = (blas::flops + mat->flops())*1e-9;
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   (*mat)(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNorm(*in, r) / b2);

   PrintSummary("FGCRODR:", tot_iters, r2, b2);

   // reset the flops counters
   blas::flops = 0;
   mat->flops();

   profile->TPSTOP(QUDA_PROFILE_EPILOGUE);
   profile->TPSTART(QUDA_PROFILE_FREE);

   if (mixed_precision_FGCRODR)
   {
     delete r_sloppy;
     delete x_sloppy;

     if(use_deflated_cycles == false)
     {
       printfQuda("\nDealocating resources from the projector cycles...\n");
       delete x_sloppy2;
       delete ctmp1_p;
     }
   }

   if(K && !precMatch) 
   {
      delete r_pre;
      delete p_pre;
   }

   delete tmp1_p;
   delete yp;
   delete rp;

   profile->TPSTOP(QUDA_PROFILE_FREE);

   return;
 }

 void FGCRODR::RunProjectedCycles(ColorSpinorField *out, ColorSpinorField *in, FGCRODRDeflationParam *dpar, const bool enforce_mixed_precision)
 {
     if(K) errorQuda("\nFGRESDR-Proj is not implemented.\n");

     bool mixed_precision_gmres = enforce_mixed_precision || (param.precision != param.precision_sloppy);

     profile->TPSTART(QUDA_PROFILE_INIT);

     DiracMatrix *sloppy_mat;

     ColorSpinorField *rp = ColorSpinorField::Create(*in); //high precision residual
     ColorSpinorField *yp = ColorSpinorField::Create(*in); //high precision aux field

     ColorSpinorField &r  = *rp;
     ColorSpinorField &x  = *out;
     ColorSpinorField &y  = *yp;

     //Sloppy precision fields:
     ColorSpinorParam csParam(*in);//create spinor parameters

     csParam.create = QUDA_ZERO_FIELD_CREATE;
     csParam.setPrecision(gmres_space_prec);//this is a solo precision solver

     ColorSpinorField *r_sloppy = nullptr, *x_sloppy = nullptr;

     ColorSpinorField *r_sloppy_proj = nullptr;
     ColorSpinorField *x_sloppy_proj = nullptr;

     if (mixed_precision_gmres)
     {
        if(gmres_space_prec != QUDA_HALF_PRECISION)
        {
           r_sloppy_proj = ColorSpinorField::Create(r, csParam);
           x_sloppy_proj = ColorSpinorField::Create(x, csParam);
        }

        csParam.setPrecision(QUDA_HALF_PRECISION);

        sloppy_mat = const_cast<DiracMatrix*> (matDefl);
        r_sloppy = ColorSpinorField::Create(r, csParam);
        x_sloppy = ColorSpinorField::Create(x, csParam);
     }
     else
     {
        sloppy_mat = matSloppy;
        r_sloppy = &r;
        x_sloppy = &x;
     }

     if(!r_sloppy_proj) r_sloppy_proj = r_sloppy;
     if(!x_sloppy_proj) x_sloppy_proj = x_sloppy;

     ColorSpinorField *tmp1_p = ColorSpinorField::Create(*in, csParam);
     ColorSpinorField &tmp = *tmp1_p;

     //Allocate Vm array:
     if(gmres_alloc == false)  AllocateKrylovSubspace(csParam);

     profile->TPSTOP(QUDA_PROFILE_INIT);
     profile->TPSTART(QUDA_PROFILE_PREAMBLE);

     int tot_iters = 0;

     //Compute initial residual:
     const double normb = norm2(*in);
     double stop = param.tol*param.tol* normb;	/* Relative to b tolerance */

     (*mat)(r, *out, y);
     //
     double r2 = xmyNorm(*in, r);//compute residual

     const double b2 = r2;

     printfQuda("\nInitial residual: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);
     //
     const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
     if (use_heavy_quark_res) errorQuda("Heavy-quark residual not supported in this solver");
     double heavy_quark_res = 0.0;

     PrintStats("GMRES-Proj:", tot_iters, r2, b2, heavy_quark_res);

     profile->TPSTOP(QUDA_PROFILE_PREAMBLE);
     profile->TPSTART(QUDA_PROFILE_COMPUTE);
     flops = 0;

//BEGIN PROJECTED RESTARTS:
     const int max_cycles = param.deflation_grid;

     int cycle_idx = 0;

     bool last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

     while(cycle_idx < max_cycles && !last_cycle)
     {
       printfQuda("\nRestart #%d\n", cycle_idx);

       cycle_idx += 1;

       if (mixed_precision_gmres) copy(*r_sloppy_proj, r);

       bool do_projection = defl_param->projFreq == 1 ? true : (cycle_idx % defl_param->projFreq) == 0 ? true : false;

       if(do_projection) PerformProjection(*x_sloppy_proj, *r_sloppy_proj, defl_param);//note : full precision residual

       if (mixed_precision_gmres)
       {
         copy(y, *x_sloppy_proj);
         xpy(y, x);
         zero(*x_sloppy_proj);
         copy(r, *r_sloppy_proj);
       }

      args->PrepareCycle(Vm, r);//pass initial residual
      int j = 0;//here also a column index of H

      while(j < args->m)//we allow full cycle
      {
        ColorSpinorField *Av = dynamic_cast<ColorSpinorField*>(&Vm->Component(j+1));

        (*sloppy_mat)(*Av, Vm->Component(j), tmp);//must be matDefl

        ///////////
        Complex h0 = cDotProduct(Vm->Component(0), *Av);//
        args->UpdateHessenberg(0, j, h0);
        //scale w:
        caxpy(-h0, Vm->Component(0), *Av);
        //
        for (int i = 1; i <= j; i++)//i is a row index
        {
          Complex h1 = cDotProduct(Vm->Component(i), *Av);//
          //load columns:
          args->UpdateHessenberg(i, j, h1);
          //scale:
          caxpy(-h1, Vm->Component(i), *Av);

          //lets do Givens rotations:
          args->UpdateGivens(j, (i-1), h0, h1);//stopping criterio
        }
        //Compute h(j+1,j):
        double h_jp1j = sqrt(norm2(*Av));
        //Update H-matrix (j+1) element:
        args->UpdateHessenberg((j+1), j, Complex(h_jp1j, 0.0));
        //
        ax(1. / h_jp1j, *Av);
        //
        double r2 = args->UpdateGivensCoeff(j, j, h0, h_jp1j);//return current stopping residual norm sq.
        //
        j += 1;

        tot_iters += 1;

        //printfQuda("GMRES residual: %1.15e ( iteration = %d)\n", sqrt(r2), j);
        PrintStats("GMRES-Proj:", tot_iters, r2, b2, heavy_quark_res);

     }//end of GMRES loop

     args->UpdateSolution(x_sloppy, Vm, j);

     if (mixed_precision_gmres)
     {
       copy(y, *x_sloppy);
       xpy(y, x);//don't mind arguments;)
       zero(*x_sloppy);
     }

     (*mat)(r, x, y);
     //
     double ext_r2 = xmyNorm(*in, r);//compute full precision residual

     printfQuda("\nDone for cycle:  %d, true residual squared %1.15e\n", cycle_idx, ext_r2);

     last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop);

   }//end of deflated restarts

   profile->TPSTOP(QUDA_PROFILE_COMPUTE);
   profile->TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile->Last(QUDA_PROFILE_COMPUTE);
   double gflops = (flops + mat->flops())*1e-9;
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   (*mat)(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNorm(*in, r) / b2);

   PrintSummary("GMRES-Proj:", tot_iters, r2, b2);

   // reset the flops counters
   flops = 0;
   mat->flops();

   profile->TPSTOP(QUDA_PROFILE_EPILOGUE);
   profile->TPSTART(QUDA_PROFILE_FREE);


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

   delete tmp1_p; 

   delete yp;
   delete rp;

   profile->TPSTOP(QUDA_PROFILE_FREE);

   return;

 }

 void FGCRODR::operator()(ColorSpinorField *out, ColorSpinorField *in)
 {
   if(!defl_param)
   {
     defl_param = new FGCRODRDeflationParam(args->ldm, args->nev);
   }

   const double tol_threshold = 6.0;//for mixed precision version only.
   const bool   use_gmresproj_mixed_precision = true;

   if(param.inv_type == QUDA_FGCRODR_INVERTER && param.rhs_idx == 0))
   {
     //run FGCRODR cycles:
     args->InitFGCRODRArgs();
     //
     RunDeflatedCycles(out, in, defl_param, tol_threshold);
   }
   else
   {
   }

   if(param.inv_type == QUDA_FGCRODR_PROJ_INVERTER && defl_param->init_flag == false)
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
