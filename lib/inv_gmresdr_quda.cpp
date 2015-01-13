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

#define DEBUG_MODE

#define MAX_EIGENVEC_WINDOW 16

/*
Based on  GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting"
Code design based on: A.Frommer et al, ArXiv hep-lat/1204.5463
*/

namespace quda {
//Notes:
//GmresDR does not require large m (and esp. nev), so this will use normal LAPACK routines.

    class GmresDRArgs{
     
      private:
      BlasMagmaArgs *gmresdr_magma_args;
      //
      Complex *harVecs;//array of harmonic eigenvectors
      Complex *harVals;//array of harmonic Ritz values
      //
      Complex *harMat;//harmonic matrix
      Complex *H;//Hessenberg matrix
      Complex *cH;//conjugate Hess. matrix

      int m;

      int nev;//number of harmonic eigenvectors used for the restart

      int ldm;//leading dimension

      //int k; //current number of the basis vectors (usually: nev, note that Vm contains k+1 vectors, including the last residual) 
      bool first_cycle_flag;
      
      public:

      GmresDRArgs( ) { };

      GmresDRArgs(int m, int ldm, int nev, bool fc_flag = true);

      ~GmresDRArgs();
      //more implementations here:

      void ResetHessenberg();
 
      void ConjugateH();//rows = cols+1

      void SetHessenbergElement(const int i, const int j, const Complex h_ij);
      //
      void ComputeQR(Complex *triangH, Complex *Omega_k,  *Complex tau);

      void ConstructHarmonicMatrix();

      void ApplyOmegaK(Complex *u, Complex *Omega_k, Complex *tau);

      Complex ApplyGivens(Complex *triangH, Complex *Omega_k, Complex *tau, Complex *Cn, Complex *Sn, const int col);

      void ComputeHarmonicEigenpairs();

      void RestartVH(cudaColorSpinorField *Vm, Complex *u);//restart with nev harmonic eigenvectors + residual (orthogonolized to the eigenvectors)

   };

   GmresDRArgs::GmresDRArgs(int m, int ldm, int nev, bool fc_flag): m(m), ldm(ldm), nev(nev), first_cylce_flag(fc_flag){

    int mp1 = m+1;    

    if(ldm < mp1) errorQuda("\nError: the leading dimension is not correct.\n");   

    if(first_cycle_flag && nev == 0) errorQuda("\nError: incorrect deflation size..\n");
    //magma library initialization:

    gmresdr_magma_args = new BlasMagmaArgs(m, nev, ldm, sizeof(double));

    harVecs  = new Complex[ldm*m];//
    harVals  = new Complex[m];//
    //ALL this in column major format (for using with LAPACK or MAGMA)
    harMat   = new Complex[ldm*m];
    H        = new Complex[ldm*m];//Hessenberg matrix
    cH       = new Complex[ldm*mp1];//conjugate Hessenberg matrix
   
    return;
  }

  void GmresDRArgs::ResetHessenberg()
  {
    memset(H,  0, ldm*m*sizeof(Complex));
    //
    memset(cH, 0, ldm*mp1*sizeof(Complex));

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

  void GmresDRArgs::ComputeQR(Complex *triangH, Complex *Omega_k,  *Complex tau)
  {
    memcpy( Omega_k, H, ldm*nev*sizeof(Complex) );//first copy (restarted) H to triangH

    gmresdr_magma_args->LapackGEQR(nev, Omega_k, (nev+1), ldm, tau); //H is a (nev+1, nev) matrix

    //extract upper traingular part:
    for(int i = 0; i < nev; i++) memset(&triangH[ldm*i], &Omega_k[ldm*i], (i+1)*sizeof(Complex));

    return;
  }

  void GmresDRArgs::ApplyOmegaK(Complex *g, Complex *Omega_k, Complex *tau)
  {
    //Apply Omega_k^{H}*u
    //gmresdr_magma_args->LapackLeftConjUNMQR((nev+1), 1, nev, Omega_k, ldm, tau, u, ldm);
    gmresdr_magma_args->LapackLeftConjUNMQR(nev/*# of reflectors*/, 1, g, (nev+1)/*# of rows*/, ldm, Omega_k, ldm, tau);//as in the prototype code

    return;
  }

  void GmresDRArgs::ConstructHarmonicMatrix()
  {
    const double beta2 = norm(H[ldm*(m-1)+m]);//compute ||h_(m+1,m)|| squared
 
    Complex *em = new Complex[ldm];

    em[m-1] = Complex(beta2, 0.0);

    gmresdr_magma_args->LapackGESV(em, ldm, m, cH, ldm);

    memcpy( harMat, H, ldm*m*sizeof(Complex) );

    for(int i = 0; i < m; i++ ) harMat[ldm*(m-1)+i] += em[i]; 

    delete [] em;

    return;
  }

  Complex GmresDRArgs::ApplyGivens(Complex *triangH, Complex *Omega_k, Complex *tau, Complex *Cn, Complex *Sn, const int col)
  {

    Complex tmp = Complex(0.0, 0.0);

    if(first_cylce_flag)
    {
       tmp = H[(col)*ldm+0];

       //now just Givens rotations:
       for(int i = 0; i < col; i++)
       {
         triangH[ldm*col+i]   =  Cn[i]*tmp + Sn[i]*H[ldm*col+i+1];

         tmp    = -Sn[i]*tmp + Cn[i]*H[ldm*col+i+1];
       } 
    }
    else
    {
       memcpy(&triangH[(col)*ldm], &H[(col)*ldm], (nev+1)*sizeof(Complex));
       //Apply Omega_k^{H}*H[(col)*ldm]
       //gmresdr_magma_args->LapackLeftConjUNMQR((nev+1)/*number of rows of Omega or elements in the column of H*/, 
       //                                        1 /*number of columns of H*/, 
       //                                        nev/*number of reflctors*/, 
       //                                        Omega_k, ldm, tau, &H[ldm*(col)], ldm);
       gmresdr_magma_args->LapackLeftConjUNMQR(nev/*# of reflectors*/, 1, &H[ldm*(col)], (nev+1)/*# of rows*/, ldm, Omega_k, ldm, tau);//as in the prototype code
   
       tmp = H[(col)*ldm+(nev)];

       //now just Givens rotations:
       for(int i = nev; i < col; i++)
       {
         triangH[ldm*col+i]   =  Cn[i]*tmp + Sn[i]*H[ldm*col+i+1];

         tmp    = -Sn[i]*tmp + Cn[i]*H[ldm*col+i+1];
       }
    }    

    return tmp;
  }

  void GmresDRArgs::ComputeHarmonicEigenpairs()
  {

    Complex *unsorted_harVecs = new Complex[ldm*m];

    ConstructHarmonicMatrix();

    //Compute right eigenvectors, and eigenvalues:
    gmresdr_magma_args->LapackRightEV(m, ldm, harmMat, harVals, harVecs, ldm);

    //Sort out the smallest nev eigenvectors:

    gmresdr_magma_args->Sort(m, ldm, harVecs, nev, unsorted_harVecs, harVals); 

    memset(cH, 0, ldm*mp1*sizeof(Complex)); 
    memset(harMat, 0, ldm*m*sizeof(Complex)); 

    delete[] unsorted_harVecs;

    return;
  }

  void GmresDRArgs::RestartVH(cudaColorSpinorField *Vm, Complex *u)
  {
    Complex *tau = new Complex[nev+1];

    memcpy(&harVecs[nev*ldm], u, ldm*sizeof(Complex)); //load nev+1 vector
    //note: u is a (m+1) vector but be sure that (m+1) element of harVecs is zero!

    gmresdr_magma_args->LapackGEQR((nev+1), harVecs, (m+1), ldm, tau); 

    int cldn = Vm->EigvTotalLength() >> 1; //complex leading dimension
    int clen = Vm->EigvLength()      >> 1; //complex vector length    

    gmresdr_magma_args->MagmaRightNotrUNMQR(clen, (nev+1), nev, harVecs, ldm, tau, Vm, cldn);

    //re-orthogonalize Vm->Eigenvec(nev) against Vm->Eigenvec(i), i = 0...nev-1  

    gmresdr_magma_args->LapackRightNotrUNMQR((m+1), m, nev, harVecs, ldm, tau, H, ldm);

    gmresdr_magma_args->LapackLeftConjUNMQR((m+1), nev, (nev+1), harVecs, ldm, tau, H, ldm);

    for(int i = 0; i < nev; i++) memset(&H[ldm*i+nev+1], 0, (ldm-nev-1)*sizeof(Complex));

    for(int e = (nev+1); e < (m+1); e++) zeroCuda(Vm->Eigenvec(e));

    memset(&H[ldm*nev], 0, (m-nev)*sizeof(Complex));

    delete [] tau;

    return;
  }

  GmresDRArgs::~GmresDRArgs() {

    delete gmresdr_magma_args;

    delete[] cH;
    delete[] H;
    delete[] harMat;

    delete[] harVals;
    delete[] harVecs;

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

     if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMRESDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//in fact it currently uses full precision

     //create GMRESDR objects:
     int mp1    = param.m + 1;

     int ldm    = ((mp1+15)/16)*16;//leading dimension

     args = new GmresDRArgs(param.m, ldm, param.nev);//first_cycle_flag=true by default  

     return;
  }

  GmresDR::~GmresDR() {

    delete args;

    if(gmres_alloc)   delete Vm;

  }

  double GmresDR::GmresDRCycle(cudaColorSpinorField &x, double r2, Complex *u, const double stop) //no need for the residual!
  {

    const int m   = args->m;

    const int ldm = args->ldm;

    const int k   = args->first_cycle_flag ? 0 : args->nev;

    profile.Start(QUDA_PROFILE_PREAMBLE);

    Complex *traingH   = new Complex[ldm*m];//QR decomposed Hessenberg matrix (to keep R-matrix)
    //
    Complex *g = new Complex[ldm];
    //
    //Cos/Sin arrays for Givens rotations:
    //
    Complex *Cn = new Complex[m];
    //
    Complex *Sn = new Complex[m];

    ColorSpinorParam csParam(x);//r is a residual vector. Note that it can be in sloppy precision.
    //
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    //make this external.
    cudaColorSpinorField *Av = NULL;
    //
    cudaColorSpinorField tmp(x, csParam);

    cudaColorSpinorField *tmp2_p;

    // tmp only needed for multi-gpu Wilson-like kernels
    if (matSloppy.Type() != typeid(DiracStaggeredPC).name() && matSloppy.Type() != typeid(DiracStaggered).name()) 
    {
      tmp2_p = new cudaColorSpinorField(x, csParam);
    }
    else
    {
      tmp2_p = &tmp;
    }

    cudaColorSpinorField &tmp2 = *tmp2_p;

    Complex *Omega_k = NULL;

    Complex *tau     = NULL;

    profile.Stop(QUDA_PROFILE_PREAMBLE);

    profile.Start(QUDA_PROFILE_COMPUTE);

//k = nev(+1):
    memcpy(g, u, ldm*sizeof(Complex));//ok

    if(args->first_cycle_flag)//that is , we need to build the whole Krylov subspace
    {
       args->ResetHessenberg();//not required..
    }
    else
    {
       Omega_k = new Complex[ldm*k];

       tau     = new Complex[k];

       args->ComputeQR(triangH, Omega_k, tau);//perform QR decomposition of the (dense!) (nev+1) x nev H matrix 
       //
       //Apply Omega_k on vector u: 
       //Note: Omega_k is usually a small matrix, no need for GPU magma routines, LAPACK is fine.
       args->ApplyOmegaK(g, Omega_k, tau);
       //
       args->ConjugateH();//conjugate the matrix H
    }

    int j = k;

    //Start Arnoldi iterations:

    while(j < m && !convergence(r2, stop))//column index
    {
       Av = &Vm->Eigenvec(j+1);

       matSloppy(*Av, Vm->Eigenvec(j), tmp, tmp2);

       for(int i = 0; i <= j; i++)
       {
          Complex h_ij = cDotProductCuda(*Av, Vm->Eigenvec(i));
          //
          args->SetHessenbergElement(i, j, h_ij);//set i-row, j-col element in the Hessenberg matrix (and its conjugate)
          //
          caxpyCuda(-h_ij, Vm->Eigenvec(i), *Av);
       } 

       //compute h(j+1, j)
       double h_jp1j = sqrt(norm2(*Av));
       //
       if(h_jp1j < 1e-15){

         warningQuda("\nInvariant subspace was built\n");

         return r2;
       }

       args->SetHessenbergElement((j+1), j,  Complex(h_jp1j, 0.0));
       //
       axCuda(1. / h_jp1j, *Av);
       //
       //Now do QR decomposition needed for the least squared problem:
       Complex e = args->ApplyGivens(triangH, Omega_k, tau, Cn, Sn, j);//applies Omega_k to the last column of H (copied to givensH), returns the last element 
       //
       //Compute last Cn, Sn and diagonal element for traingH:
       
       double inv_denorm = 1.0 / sqrt(norm(e)+h_jp1j*h_jp1j);//again no check here...
       // 
       Cn[j] = inv_denom * e;
       
       Sn[j] = h_jp1j * inv_denom;
       
       triangH[ldm*j+j] = e*Cn[j]+Sn[j]*h_jp1j;

       g[j+1] = - Sn[j]*g[j];
       g[j]   = g[j] * Cn[j];
       //update the residual norm squared:

       r2 = norm(g[j+1]);

       if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GMRES residual: %f ( iteration = %d)\n", sqrt(r2), j);

       j += 1; 
    }//end while loop

    //solve ls problem:
    Complex *sol = new Complex[ldm];

    for(int l = (j-1); l >= 0; l--)
    {
       Complex accum = 0.0;

       for(int i = (l+1); i <= (j-1); i++) accum = accum + triangH[ldm*i+l]*sol[i]; 
  
       sol[l] = (g[l] - accum) / triangH[ldm*l+l]; 
    }

    //update solution: compute x = x0+Vm*g
    for(int l = 0; l < j; l++)  caxpyCuda(sol[l], Vm->Eigenvec(l), x);
     

    //Compute c = u - H*sol
    for(int j = 0; j <= m; j++) //row index
    {
       _Complex double accum = 0.0;

       for(int i = 0; i < m; i++) //column index
       {
        accum += (H[ldH*i + j]*sol[i]);
       }
     
       u[j] -= accum;//overwrite u-array.
    }
   
 
    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);

    double gflops = (quda::blas_flops + matSloppy.flops())*1e-9;

    reduceDouble(gflops);

    param.gflops = gflops;

    delete[] sol;

    if (matSloppy.Type() != typeid(DiracStaggeredPC).name() && matSloppy.Type() != typeid(DiracStaggered).name()) 
    {
      delete tmp2_p;
    }

    //clean cH and harmonic matrix?

    delete[] triangH;

    delete[] Cn;
    delete[] Sn;

    delete[] g;

    return r2;
  }

//END of GMRESDR cycle routine.

//Warning: this solver assumes we are working with even-odd preconditioning
  void GmresDR::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
  {
     if (gmres_space_prec != in->Precision()) errorQuda("\nInput/output field precision is incorrect (solver precision: %u spinor precision: %u).\n", gmres_space_prec, in->Precision());

     profile.Start(QUDA_PROFILE_INIT);

     // Check to see that we're not trying to invert on a zero-field source    
     const double b2 = norm2(*in);

     if(b2 == 0){

       profile.Stop(QUDA_PROFILE_INIT);
       printfQuda("Warning: inverting on zero-field source\n");
       *out=*in;

       return;
     }

     cudaColorSpinorField r(*in); //high precision residual    

     ColorSpinorParam csParam(*in);//create spinor parameters

     csParam.create = QUDA_ZERO_FIELD_CREATE;
     // 
     cudaColorSpinorField tmp(*in, csParam);
     //
     cudaColorSpinorField *tmp2_p;

     // tmp only needed for multi-gpu Wilson-like kernels
     if (matSloppy.Type() != typeid(DiracStaggeredPC).name() && matSloppy.Type() != typeid(DiracStaggered).name()) 
     {
       tmp2_p = new cudaColorSpinorField(*in, csParam);
     }
     else
     {
       tmp2_p = &tmp;
     }
     cudaColorSpinorField &tmp2 = *tmp2_p;

     mat(r, *out, tmp, tmp2);
     //
     double r2 = xmyNormCuda(*in, r);//compute residual

     //Allocate Vm array:
     if(gmres_alloc == false)
     {
       printfQuda("\nAllocating resources for the GMRESDR solver...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;
   
       csParam.setPrecision(gmres_space_prec);//GMRESDR internal Krylov subspace precision: may or may not coincide with eigenvector precision!.

       csParam.eigv_dim = param.m+1; // basis dimension (abusive notations!)

       Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors

       checkCudaError();
       printfQuda("\n..done.\n");
       
       gmres_alloc = true;
     }
     //
     Complex *u  = new Complex[ldm];//size is m+1, but set to ldm
     //
     /*********************************The first cycle******************************/

     double beta = sqrt(r2);

     if(beta < 1.0e-15) 
     {
       warningQuda("\nSolution was already found.\n");
       return;
     }

     u[0] = Complex(beta, 0.0);
     //load the first Vm vector:
     axpyCuda(1.0 / beta, r, Vm->Eigenvec(0));

     const double stop = b2*param.tol*param.tol;

     //launch the first Arnoldi cycle:

     double cycler2 = GmresDRCycle(*out, r2, u, stop);
     //
     //Compute full precision residual (not needed for the solo-precision solver.)
     mat(r, *out, tmp, tmp2);
     //
     r2 = xmyNormCuda(*in, r);//compute full precision residual

     //now begin deflation restarts:
     int cycle_idx = 1;
     //
     const int max_cycles = param.deflation_grid;

     args->first_cycle_flag = false;

     while(cylce_idx < param.deflation_grid && && !convergence(r2, stop))
     {
        args->ComputeHarmonicEigenpairs();//also clean cH and harMat
        //
        args->RestartVH(Vm, u);//creates Vm^{new}, H^{new}

        //Update u for the next Arnoldi cycle
        for(int i = 0; i <= args->nev; i++) u[i] = cDotProductCuda(Vm->Eigenvec(i), r);
       
        //Call new GMRESDR cycle
        cycler2 = GmresDRCycle(*out, r2, u, stop);

        //Compute full precision residual (not needed for the solo-precision solver.)
        mat(r, *out, tmp, tmp2);
        //
        r2 = xmyNormCuda(*in, r);//compute residual

        //check cycler2 vs r2 : for mixed precision version.

        cycle_idx += 1;
     }

     delete[] u;

     return;
  }


} // namespace quda
