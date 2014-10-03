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
*/

namespace quda {

   static DeflationParam *defl_param = 0;

   template<typename Float, typename CudaComplex>
   class GmresDRArgs{
     
      private:
      BlasMagmaArgs *gmresdr_magma_args;
      //
      Complex *harVecs;//array of harmonic Ritz vectors
      Complex *harVals;//array of harmonic Ritz values

      int m;
      int nev;
      int ldm;
      
      public:

      GmresDRArgs( ) { };

      GmresDRArgs(int m, int nev);

      ~GmresDRArgs();
      //more implementations here:

   };

   template<typename Float, typename CudaComplex>
   GmresDRArgs<Float, CudaComplex>::GmresDRArgs(int m, int nev): m(m), nev(nev){
    //include pad?
    ldm    = ((m+15)/16)*16;//too naive
       
    //magma initialization:
    const int prec = sizeof(Float);
    gmresdr_magma_args = new BlasMagmaArgs(m, nev, ldm, prec);

    harVecs  = new Complex[ldm*m];//
    harVals  = new Complex[m];//
   
    return;
  }

  template<typename Float, typename CudaComplex>
  GmresDRArgs<Float, CudaComplex>::~GmresDRArgs() {

    delete[] harVecs;
    delete[] harVals;

    delete gmresdr_magma_args;

    return;
  }


  GmresDR::GmresDR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), gmres_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), profile(profile), gmres_alloc(false)
  {
     if(param.nev > MAX_EIGENVEC_WINDOW )
     { 
          warningQuda("\nWarning: the eigenvector window is too big, using default value %d.\n", MAX_EIGENVEC_WINDOW);
          param.nev = MAX_EIGENVEC_WINDOW;
     }

     if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMRESDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//use mixed presicion
  }

  GmresDR::~GmresDR() {

    if(gmres_alloc)   delete Vm;

  }

  void GmresDR::GmresDRFirstCycle(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    if (gmres_space_prec != x.Precision()) errorQuda("\nInput/output field precision is incorrect (solver precision: %u spinor precision: %u).\n", gmres_space_prec, x.Precision());

    profile.Start(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);

    if(b2 == 0){

      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      return;
    }

    cudaColorSpinorField r(b);

    ColorSpinorParam csParam(x);
    //
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // 
    cudaColorSpinorField y(b, csParam);
    //
    cudaColorSpinorField Av(x, csParam);
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

    matSloppy(r, x, tmp, tmp2);
    //
    double r2 = xmyNormCuda(b, r);//compute residual
    //
    zeroCuda(y);
    
    if(gmres_alloc == false){

       printfQuda("\nAllocating resources for the GMRESDR solver...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;

       csParam.setPrecision(x.Precision());//precision for the Arnoldi vectors.

       csParam.eigv_dim = param.m;

       Vm = new cudaColorSpinorField(csParam);

       checkCudaError();
       //
       printfQuda("\n..done.\n");
       
       gmres_alloc = true;
    }

    double beta= norm2(r);

    //load the first vector:
    copyCuda(Vm->Eigenvec(0), r);//convert arrays

    //rescale the vector
    axCuda(1.0 / beta, Vm->Eigenvec(0));

    profile.Stop(QUDA_PROFILE_PREAMBLE);

    profile.Start(QUDA_PROFILE_COMPUTE);

    blas_flops = 0;

    PrintStats("GMRES DR (first cycle)", k, r2, b2);

    double sigma = 0.0;

    const int m   = param.m;
    const int ldh = ((m+16)/16)*16; //ld for m+1 size
    //
    //create array for Givens rotation parameters:
    Complex givens[2][(m+1)]; 

    Complex *H       = new Complex[ldh*(m+1)];//check ldh size..
    Complex *triangH = new Complex[ldh*(m+1)];
    Complex *transpH = new Complex[ldh*(m+1)];

    Complex *g = new Complex[(m+1)]

    while ( !convergence(r2, stop) && k < m) {

      matSloppy(Av, Vm->Eigenvec(k), tmp);
      
      for(int l  = 0; l <= k; l++)
      {
         H[l*ldh+k]=cDotProductCuda(Vm->Eigenvec(l), Av);

         caxpbyCuda(H[l*ldh+k], Vm->Eigenvec(l), -1.0, Av);

         transpH[k*ldh+l] = H[l*ldh+k];
         triangH[l*ldh+k] = H[l*ldh+k];
      } 

      H[(k+1)*ldh+k] = sqrt(norm2(Av));

      triangH[(k+1)*ldh+k] = H[(k+1)*ldh+k];
      transpH[k*ldh+(k+1)] = H[(k+1)*ldh+k];
 
      //perform rotation (principal part):
      for(int l = 0; l < k; l++)
      {
         Complex tmp1 = triangH[l*ldh+k]*givens[1][l] - givens[0][l]*triangH[(l+1)*ldh+k];
         Complex tmp2 = triangH[(l+1)*ldh+k]*givens[1][l] + conj(givens[0][l])*triangH[l*ldh+k];

         triangH[l*ldh+k]     = tmp1;
         triangH[(l+1)*ldh+k] = tmp2;
      } 
      
      beta     = sqrt(norm(triangH[k*ldh+k])+norm(triangH[(k+1)*ldh+k]));

      givens[1][k]     = triangH[(k+1)*m+k].real()/beta;
      givens[0][k]     = triangH[k*m+k]/beta;

      triangH[k*m+k] = beta;//?? 
 
      //update vector g:
      g[k+1] = g[k] * givens[1][k];
      g[k]   = conj(givens[0][k])*g[k];

      //get residual norm:
      r2 = g[k+1].real()*g[k+1].real();

      axCuda(1/H[(k+1)*ldh+k].real(), Av);
      copyCuda(Vm->Eigenvec(k+1), Av);

      k++;

      PrintStats("GMRES DR (first cycle)", k, r2, b2);
    }

    //minimize the norm:
    g[k] /= triangH[k*m+k].real(); 

    caxpyCuda(g[k], Vm->Eigenvec(k), x);

    for(int i = k-1; i >= 0; i++)
    {
      for(int j = i+1; j <= k; j++)
      {
         g[i] -= triangH[i*m + j]*g[j];  
      }
      g[i] /= triangH[k*m+k].real(); 
      caxpyCuda(g[i], Vm->Eigenvec(i), x);
    }

    double beta2 = H[m*ldh+(m-1)].real()*H[m*ldh+(m-1)].real();

    //compute H+beta2 HT^{-1}em * em^{T}
    //reuse g array:
    delete [] g;
    Complex *g = new Complex[m+1];

    g[m-1] = Complex(1.0,0.0); //m-th component

    int *pividx = GmresDRArgs->SolveTransH(g, m, transpH, m, ldh); 

    //just modifies mth column:
    for(int i = 0; i < m; i++)
    {
       H[m*ldh+i] = H[m*ldh+i] + beta2*g[(pibidx[i])];
    }  

    //find nev harmonic Ritz pairs:
    GmresDRArgs->ComputeHarmonicRitz(H);

    //if(convergence(r2, stop)) {};//return

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);

    double gflops = (quda::blas_flops + matSloppy.flops())*1e-9;

    reduceDouble(gflops);

    param.gflops = gflops;

    param.iter += k;

    // compute the true residuals
    matSloppy(r, x, tmp, tmp2);

    param.true_res = sqrt(xmyNormCuda(b, r) / b2);
    PrintSummary("GMRESDR first stage", k, r2, b2);

    // reset the flops counters
    quda::blas_flops = 0;

    matSloppy.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);

    profile.Start(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp) delete tmp2_p;

    profile.Stop(QUDA_PROFILE_FREE);

    delete[] H;
    delete[] triangH;
    delete[] transpH;

    delete[] g;

    return;
  }

//END of eigcg solver.

//Deflation space management:
  void GmresDR::CreateDeflationSpace(cudaColorSpinorField &in, DeflationParam *&dpar)
  {
    printfQuda("\nCreate deflation space...\n");

    if(eigcgSpinor.SiteSubset() != QUDA_PARITY_SITE_SUBSET) errorQuda("\nRitz spinors must be parity spinors\n");//or adjust it

    ColorSpinorParam cudaEigvParam(eigcgSpinor);

    dpar = new DeflationParam(cudaEigvParam, param);

    printfQuda("\n...done.\n");

    //dpar->PrintInfo();

    return;
  }

  void GmresDR::DeleteDeflationSpace(DeflationParam *&dpar)
  {
    if(dpar != 0) 
    {
      delete dpar;
      dpar = 0;
    }

    return;
  }


  void GmresDR::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
  {
     if(defl_param == 0) CreateDeflationSpace(*in, defl_param);

     param.rhs_idx += 1;

     return;
  }


} // namespace quda
