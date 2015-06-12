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

#define DEBUG_MODE

#define MAX_EIGENVEC_WINDOW 64

/*
GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) p. 20-37
Code design based on: A.Frommer et al, "Deflation and Flexible SAP-Preconditioning of GMRES in Lattice QCD simulations" ArXiv hep-lat/1204.5463
*/

namespace quda {
//Notes:
//GmresDR does not require large m (and esp. nev), so this will use normal LAPACK routines.

    static GmresdrDeflationParam *defl_param = 0;

    struct SortEvals{

      double eval_nrm;
      int    eval_idx;

      SortEvals(double val, int idx) : eval_nrm(val), eval_idx(idx) {}; 

      static bool CmpEigenNrms (SortEvals v1, SortEvals v2) { return (v1.eval_nrm < v2.eval_nrm);}

    };

    struct GmresdrDeflationParam 
    {

      //Objects required for the Galerkin projections
      Complex *projMat;  //host   projection matrix:
      //
      cudaColorSpinorField    *projVecs;      //device buffer for HRitz vectors
      
      int ld;                 //projection matrix leading dimension
      int nv;                //(nv+1) => number of projection vectors, note also that projMat is a (nv+1) by (nv) matrix

      bool alloc_flag;

      GmresdrDeflationParam(cudaColorSpinorField  *Vm, const int ldm, const int nev) : ld(ldm), nv(nev), alloc_flag(true)
      {
        if(nev == 0) errorQuda("\nIncorrect deflation space parameters...\n");
       
        //Create deflation objects:
        projMat  = new Complex[ld*nv];

        ColorSpinorParam csParam(Vm->Eigenvec(0));
        csParam.create = QUDA_ZERO_FIELD_CREATE;

        csParam.eigv_dim  = (nv+1);
        csParam.eigv_id   = -1;

        projVecs  = new cudaColorSpinorField(csParam);//temporary field

        for(int i = 0; i < (nv+1); i++) copyCuda(projVecs->Eigenvec(i), Vm->Eigenvec(i));    

        return;
      }

      ~GmresdrDeflationParam(){
        if(alloc_flag)    
        {
          delete projVecs;
          delete[] projMat;
        }
      }

    };


    class GmresDRArgs{
     
      private:
      BlasMagmaArgs *gmresdr_magma_args;
      //
      Complex *harVecs;//array of harmonic eigenvectors
      Complex *harVals;//array of harmonic Ritz values
      //
      Complex *sortedHarVecs;//array of harmonic eigenvectors: (nev+1) length
      Complex *sortedHarVals;//array of harmonic Ritz values: nev+1 length
      //
      Complex *harMat;//harmonic matrix
      Complex *H;//Hessenberg matrix
      Complex *cH;//conjugate Hess. matrix

      //aux array to keep QR decomposed "restarted Hessenberg":
      Complex *qrH;
      Complex *tauH;//nev->m : for DEBUGING only!

      //auxilary object
      Complex *srtRes;//the "short residual" is in fact an alias pointer to &sortedHarVecs[nev*m] !
      Complex *lsqSol;//the least squares problem solution
   
      public:

      int m;
      int ldm;//leading dimension (must be >= m+1)
      int nev;//number of harmonic eigenvectors used for the restart

      bool mixed_precision_gmresdr;

      bool deflated_cycle; //if true - we use deflated restart of the cycle, otherwise the cycle is undeflated..

      //public:

      GmresDRArgs( ) { };

      GmresDRArgs(int m, int ldm, int nev, bool mixed_prec, bool deflated_flag = false);

      ~GmresDRArgs();
      //more implementations here:

      void ResetHessenberg();
 
      void ConjugateH();//rows = cols+1

      void SetHessenbergElement(const int i, const int j, const Complex h_ij);
      //
      void ComputeHarmonicRitzPairs();

      void RestartVH(cudaColorSpinorField *Vm);

      void PrepareDeflatedRestart(Complex *givensH, Complex *g);

      void PrepareGivens(Complex *givensH, const int col);

      void UpdateSolution(cudaColorSpinorField *x, cudaColorSpinorField *Vm, Complex *givensH, Complex *g, const int j);

      void StoreProjectionMatrix(Complex *srcMat);
   };

  GmresDRArgs::GmresDRArgs(int m, int ldm, int nev, bool mixed_prec, bool deflated_flag): m(m), ldm(ldm), nev(nev), mixed_precision_gmresdr(mixed_prec), deflated_cycle(deflated_flag){

     int mp1 = m+1;    

     if(ldm < mp1) errorQuda("\nError: the leading dimension is not correct.\n");   

     if(nev == 0) errorQuda("\nError: incorrect deflation size..\n");
     //magma library initialization:

     gmresdr_magma_args = new BlasMagmaArgs(m, nev+1, ldm, sizeof(double));

     harVecs  = new Complex[ldm*m];//(m+1)xm (note that ldm >= m+1)
     harVals  = new Complex[m];//

     sortedHarVecs  = new Complex[ldm*(nev+1)];//
     sortedHarVals  = new Complex[(nev+1)];//

     //ALL this in column major format (for using with LAPACK or MAGMA)
     harMat   = new Complex[ldm*m];
     H        = new Complex[ldm*m];//Hessenberg matrix
     cH       = new Complex[ldm*mp1];//conjugate Hessenberg matrix

     qrH      = new Complex[ldm*m];
     tauH     = new Complex[m];//nev->m : for DEBUGING only!

     srtRes   = &sortedHarVecs[ldm*(nev)];
     lsqSol   = new Complex[(m+1)];
   
     return;
  }

  void GmresDRArgs::ResetHessenberg()
  {
    memset(H,  0, ldm*m*sizeof(Complex));
    //
    memset(cH, 0, ldm*(m+1)*sizeof(Complex));

    return;
  }

  void GmresDRArgs::ConjugateH()//rows = cols+1
  {
    for(int c = 0; c < nev; c++ )
    {
      for(int r = 0; r < (nev+1); r++ ) cH[r*ldm+c] = conj(H[c*ldm+r]);
    }

    return;
  }

  //helper method
  void GmresDRArgs::SetHessenbergElement(const int row, const int col, const Complex el)
  {
    H[col*ldm+row]  = el;

    cH[row*ldm+col] = conj(el); 

    return;
  }

  void GmresDRArgs::StoreProjectionMatrix(Complex *srcMat)
  {
    memcpy(srcMat, H, nev*ldm*sizeof(Complex));

    return;
  }


  GmresDRArgs::~GmresDRArgs() {

    delete gmresdr_magma_args;

    delete[] cH;
    delete[] H;
    delete[] harMat;

    delete[] harVals;
    delete[] harVecs;

    delete[] sortedHarVals;
    delete[] sortedHarVecs;

    delete[] qrH;
    delete[] tauH;

    delete[] lsqSol;

    return;
  }


//Note: in fact, x is an accumulation field initialized to zero (if a sloppy field is used!) and has the same precision as Vm
 void GmresDRArgs::UpdateSolution(cudaColorSpinorField *x, cudaColorSpinorField *Vm, Complex *givensH, Complex *g, const int j)
 {
   //if (mixed_precision_gmresdr) zeroCuda(*x);//it's done in the main loop, no need for this
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

 void GmresDRArgs::ComputeHarmonicRitzPairs()
 {
   memcpy(harMat, H, ldm*m*sizeof(Complex));

   const double beta2 = norm(H[ldm*(m-1)+m]);

   gmresdr_magma_args->Construct_harmonic_matrix(harMat, cH, beta2, m, ldm);

   // Computed eigenpairs for harmonic matrix:
   // Save harmonic matrix :
   memset(cH, 0, ldm*m*sizeof(Complex));

   memcpy(cH, harMat, ldm*m*sizeof(Complex));

   gmresdr_magma_args->Compute_harmonic_matrix_eigenpairs(harMat, m, ldm, harVecs, harVals, ldm);//check it!

//   for(int e = 0; e < m; e++) printf("\nEigenval #%d: %le, %le, %le\n", e, harVals[e].real(), harVals[e].imag(), abs(harVals[e]));
   //do sort:
   std::vector<SortEvals> sorted_evals_cntr;

   sorted_evals_cntr.reserve(m);

   for(int e = 0; e < m; e++) sorted_evals_cntr.push_back( SortEvals( abs(harVals[e]), e ));

   std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);
 
   for(int e = 0; e < nev; e++) memcpy(&sortedHarVecs[ldm*e], &harVecs[ldm*( sorted_evals_cntr[e].eval_idx)], (ldm)*sizeof(Complex));

   for(int e = 0; e < 16; e++) printfQuda("\nEigenval #%d: real %le imag %le abs %le\n", sorted_evals_cntr[e].eval_idx, harVals[(sorted_evals_cntr[e].eval_idx)].real(), harVals[(sorted_evals_cntr[e].eval_idx)].imag(),  abs(harVals[(sorted_evals_cntr[e].eval_idx)]));

   return;
 }

 void GmresDRArgs::RestartVH(cudaColorSpinorField *Vm)
 {
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
   gmresdr_magma_args->RestartVH(Vm->V(), clen, cldn, Vm->Precision(), sortedHarVecs, H, ldm);//check ldm with internal (magma) ldm

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


 void GmresDRArgs::PrepareDeflatedRestart(Complex *givensH, Complex *g)
 {
   //update initial values for the "short residual"
   memcpy(srtRes, g, (m+1)*sizeof(Complex));

   if(deflated_cycle)
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

     gmresdr_magma_args->ComputeQR(nev, qrH, (nev+1), ldm, tauH);

     //extract triangular part to the givens matrix:
     for(int i = 0; i < nev; i++) memcpy(&givensH[ldm*i], &qrH[ldm*i], (i+1)*sizeof(Complex));

     gmresdr_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, g, (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);
   }

   return;
 }

 void GmresDRArgs::PrepareGivens(Complex *givensH, const int col)
 {
   if(deflated_cycle == false) return; //we are in the "projected" cycle nothing to do..

   memcpy(&givensH[ldm*col], &H[ldm*col], (nev+1)*sizeof(Complex) );
   //
   gmresdr_magma_args->LeftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, &givensH[ldm*col], (nev+1) /*number of rows*/, ldm, qrH, ldm, tauH);

   return;
 }



 GmresDR::GmresDR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), gmres_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), profile(profile), args(0), gmres_alloc(false)
 {
     if(param.nev > MAX_EIGENVEC_WINDOW )
     { 
          warningQuda("\nWarning: the eigenvector window is too big, using default value %d.\n", MAX_EIGENVEC_WINDOW);
          param.nev = MAX_EIGENVEC_WINDOW;
     }

     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMRESDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     //create GMRESDR objects:
     int mp1    = param.m + 1;

     int ldm    = mp1;//((mp1+15)/16)*16;//leading dimension

     args = new GmresDRArgs(param.m, ldm, param.nev, (param.precision != param.precision_sloppy));//deflated_cycle flag is false by default  

     return;
 }

 GmresDR::~GmresDR() {

    delete args;

    if(gmres_alloc) 
    {
      delete Vm;
      Vm = NULL;
    }

 }

 void GmresDR::PerformGalerkinProjection(cudaColorSpinorField &x_sloppy,  cudaColorSpinorField &r_sloppy, GmresdrDeflationParam *dpar)
 {
    Complex *c    = new Complex[(dpar->nv+1)];
    Complex *d    = new Complex[dpar->ld];

    BlasMagmaArgs magma_args(sizeof(double));

    //Compute c = VT^{pr}_k r0 
    for(int i = 0; i < dpar->nv; i++ ) c[i] = cDotProductCuda(dpar->projVecs->Eigenvec(i), r_sloppy);//
   
    //Solve H^{pr}_k d = c: this nvxnv problem..
    magma_args.SolveProjMatrix((void*)d, dpar->ld,  dpar->nv, (void*)dpar->projMat, dpar->ld);

    //Compute the new approximate solution:
    for(int l = 0; l < dpar->nv; l++)  caxpyCuda(d[l], dpar->projVecs->Eigenvec(l), x_sloppy); 

    //Compute the new residual vector: 
    memset(c, 0, (dpar->nv+1)*sizeof(Complex));//now use as a temp array

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

//Make this option available outside!
//#define USE_PLAIN_GMRES

 void GmresDR::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
 {
     profile.Start(QUDA_PROFILE_INIT);

     cudaColorSpinorField r(*in); //high precision residual 
     // 
     cudaColorSpinorField &x = *out;
     //
     cudaColorSpinorField y(*in); //high precision aux field 

     //Sloppy precision fields:
     ColorSpinorParam csParam(*in);//create spinor parameters

     csParam.create = QUDA_ZERO_FIELD_CREATE;

     csParam.setPrecision(gmres_space_prec);

     cudaColorSpinorField *r_sloppy;

     if (args->mixed_precision_gmresdr) 
     {
        r_sloppy = new cudaColorSpinorField(r, csParam);
     } 
     else 
     {
        r_sloppy = &r;
     }

     cudaColorSpinorField *x_sloppy;

     if (args->mixed_precision_gmresdr) 
     {
       x_sloppy = new cudaColorSpinorField(x, csParam);
     } 
     else 
     {
       x_sloppy = &x;
     }

     cudaColorSpinorField tmp(*in, csParam);

     cudaColorSpinorField *tmp2_p;

     tmp2_p = new cudaColorSpinorField(*in, csParam);

     cudaColorSpinorField &tmp2 = *tmp2_p;

     //Allocate Vm array:
     if(gmres_alloc == false)
     {
       printfQuda("\nAllocating resources for the GMRESDR solver...\n");

       csParam.eigv_dim = param.m+1; // basis dimension (abusive notations!)

       Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors

       checkCudaError();

       printfQuda("\n..done.\n");
       
       gmres_alloc = true;
     }

    profile.Stop(QUDA_PROFILE_INIT);
    profile.Start(QUDA_PROFILE_PREAMBLE);

   const int m         = args->m;
   const int ldH       = args->ldm;
   const int nev       = args->nev;

   const double tol_threshold = 2e+0;//think about other options: original value 4.0e+0, 2.0e+0 works better!

   int tot_iters = 0;

   //GMRES objects:
   //Givens rotated matrix (m+1, m):
   Complex *givensH = new Complex[ldH*m];//complex
    
   //Auxilary objects: 
   Complex *g = new Complex[(m+1)];

   //Givens coefficients:
   Complex *Cn = new Complex[m];
   //
   double *Sn = (double *) calloc(m, sizeof(double));//in fact, it's real

   //Compute initial residual:
   const double normb = norm2(*in);  
   double stop = param.tol*param.tol* normb;	/* Relative to b tolerance */

   mat(r, *out, y);
   //
   double r2 = xmyNormCuda(*in, r);//compute residual

   const double b2 = r2;

   printfQuda("\nInitial residual squred: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);

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

   args->PrepareDeflatedRestart(givensH, g);

   const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

   double heavy_quark_res = 0.0; // heavy quark residual
   if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
   int heavy_quark_check = 10; // how often to check the heavy quark residual

   PrintStats("GMRESDR:", tot_iters, r2, b2, heavy_quark_res);

   profile.Stop(QUDA_PROFILE_PREAMBLE);
   profile.Start(QUDA_PROFILE_COMPUTE);
   blas_flops = 0;

//   while((j < m) && (r2 > stop))
   while(j < m)//we allow full cycle
   {
     cudaColorSpinorField *Av = &Vm->Eigenvec(j+1);

     matSloppy(*Av, Vm->Eigenvec(j), tmp, tmp2);

     ///////////
     Complex h0 = cDotProductCuda(Vm->Eigenvec(0), *Av);//
     args->SetHessenbergElement(0, j, h0);
     //scale w:
     caxpyCuda(-h0, Vm->Eigenvec(0), *Av);
     //
#if 0 //to unify this later like below:
    //k = 0; //(nev = 0 for the first cycle)
    Complex h0;
    for (int i = 0; i <= k; i++)//i is a row index
    {
       h0 = cDotProductCuda(Vm->Eigenvec(i), *Av);//
       //Warning: column major format!!! (m+1) size of a column.
       //load columns:
       H[ldH*j+i] = h0;
       //
       conjH[ldH*i+j] = conj(h0);
       //
       caxpyCuda(-h0, Vm->Eigenvec(i), *Av);
    }

    if(not first_cycle, i.e. k != 0 )   
    {
    //Let's do Givens rotation:
       memcpy(&givensH[ldH*j], &H[ldH*j], (nev+1)*sizeof(Complex) );
       //
       leftConjZUNMQR(nev /*number of reflectors*/, 1 /*number of columns of mat*/, &givensH[ldH*j], (nev+1) /*number of rows*/, ldH, qrH, ldH, tauH);

       h0 = givensH[ldH*j+nev];
    }
#endif
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
     PrintStats("GMRESDR:", tot_iters, r2, b2, heavy_quark_res);

   }//end of GMRES loop

   printfQuda("\nDone for: %d iters, %1.15e last residual squared.\n", j, r2); 

   //j -> m+1
   //update solution and final residual:
   args->UpdateSolution(x_sloppy, Vm, givensH, g, j);

   if (args->mixed_precision_gmresdr)
   {
     copyCuda(y, *x_sloppy);
     xpyCuda(y, x);
     zeroCuda(*x_sloppy);
   }

   mat(r, x, y);
   //
   r2 = xmyNormCuda(*in, r);//compute full precision residual

   printfQuda("\nDone for cycle 0, true residual squared %1.15e\n", r2);
   //
   PrintStats("GMRESDR:", tot_iters, r2, b2, heavy_quark_res);

//BEGIN RESTARTS:
   const int max_cycles = param.deflation_grid;

   int cycle_idx = 1;

#ifndef USE_PLAIN_GMRES
   args->deflated_cycle = true;
#else
   args->deflated_cycle = false;
#endif

   bool last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq);

   while(cycle_idx < max_cycles && !last_cycle)
   {
     printfQuda("\nRestart #%d\n", cycle_idx);

     if (args->mixed_precision_gmresdr) copyCuda(*r_sloppy, r);

     memset(givensH, 0, ldH*m*sizeof(Complex));
     memset(g, 0 , (m+1)*sizeof(Complex));
     //
     memset(Cn, 0 , m*sizeof(Complex));
     memset(Sn, 0 , m*sizeof(double));

     int j = nev;//here also a column index of H
#ifndef USE_PLAIN_GMRES
     args->ComputeHarmonicRitzPairs();

     args->RestartVH( Vm );
#endif

     if(args->deflated_cycle)
     {
       for(int i = 0; i <= nev ; i++ ) g[i] = cDotProductCuda(Vm->Eigenvec(i), *r_sloppy);//
     }
     else //use Galerkin projection instead: 
     {
       j = 0; //we will launch a normal GMRESDR cycle
#ifndef USE_PLAIN_GMRES
       if(!defl_param)
       {
         defl_param = new GmresdrDeflationParam(Vm, args->ldm, args->nev);
         //
         args->StoreProjectionMatrix(defl_param->projMat);//merge this into the constructor..
       }

       PerformGalerkinProjection(*x_sloppy, *r_sloppy, defl_param);//note : full precision residual

       if (args->mixed_precision_gmresdr)
       {
         copyCuda(y, *x_sloppy);
         xpyCuda(y, x);
         zeroCuda(*x_sloppy);
         copyCuda(r, *r_sloppy);
       }
#endif
       r2 = norm2(r);
       
       beta = sqrt(r2);
       //
       axCuda(1.0 / beta, r);
       //
       copyCuda(Vm->Eigenvec(0), r);

       g[0] = beta;

       printfQuda("\nNew residual %1.15e\n", beta);
     }

     args->PrepareDeflatedRestart(givensH, g);

     const int jlim = j;//=nev for deflated restarts and 0 for "projected" restarts (abused terminology!)

//     while((j < m) && (r2 > stop))
     while(j < m) //we allow full cycle
     {
       //pointer aliasing:
       cudaColorSpinorField *Av = &Vm->Eigenvec(j+1);

       matSloppy(*Av, Vm->Eigenvec(j), tmp, tmp2);
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
       args->PrepareGivens(givensH, j);//check it! 

       if(args->deflated_cycle) h0 = givensH[ldH*j+jlim];

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
       PrintStats("GMRESDR:", tot_iters, r2, b2, heavy_quark_res); 
     }//end of main loop.

     args->UpdateSolution(x_sloppy, Vm, givensH, g, j);

     if (args->mixed_precision_gmresdr)
     {
       copyCuda(y, *x_sloppy);
       xpyCuda(y, x);//don't mind arguments;)
       zeroCuda(*x_sloppy);
     }

     mat(r, x, y);
     //
     double ext_r2 = xmyNormCuda(*in, r);//compute full precision residual

     if(args->mixed_precision_gmresdr && ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold || (cycle_idx > 64)))
     {
       printfQuda("\nLaunch projection stage (%le)\n", sqrt(ext_r2 / r2));
       args->deflated_cycle = false;
     }

     printfQuda("\nDone for cycle:  %d, true residual squared %1.15e\n", cycle_idx, ext_r2);

     last_cycle = convergence(r2, heavy_quark_res, stop, param.tol_hq);

     cycle_idx += 1;

   }//end of deflated restarts

   profile.Stop(QUDA_PROFILE_COMPUTE);
   profile.Start(QUDA_PROFILE_EPILOGUE);

   param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
   double gflops = (quda::blas_flops + mat.flops())*1e-9;
   reduceDouble(gflops);
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   mat(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNormCuda(*in, r) / b2);

   PrintSummary("GMRESDR:", tot_iters, r2, b2);

   // reset the flops counters
   quda::blas_flops = 0;
   mat.flops();

   profile.Stop(QUDA_PROFILE_EPILOGUE);
   profile.Start(QUDA_PROFILE_FREE);

   if(defl_param) 
   {
     delete defl_param;
     defl_param = 0;
   }

   delete [] givensH;

   delete [] g;

   delete [] Cn;

   free(Sn);

   printfQuda("\n..done.\n");

   if (args->mixed_precision_gmresdr)
   {
     delete r_sloppy;
     delete x_sloppy; 
   }

   delete tmp2_p;

   profile.Stop(QUDA_PROFILE_FREE);

   return;
 }//end of operator()

} // namespace quda
