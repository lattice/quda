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
Based on  eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
*/

namespace quda {

   static DeflationParam *defl_param = 0;

   template<typename Float, typename CudaComplex>
   class EigCGArgs{
     
      private:
      BlasMagmaArgs *eigcg_magma_args;
    
      //host Lanczos matrice, and its eigenvalue/vector arrays:
      std::complex<Float> *hTm;//VH A V
      Float  *hTvalm;   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

      //device Lanczos matrix, and its eigenvalue/vector arrays:
      CudaComplex *dTm;     //VH A V
      CudaComplex *dTvecm; //eigenvectors of T[m,  m  ]
      CudaComplex *dTvecm1; //eigenvectors of T[m-1,m-1]

      int m;
      int nev;
      int ldm;
      
      public:

      EigCGArgs(int m, int nev);
      ~EigCGArgs();
      
      //methods for constructing Lanczos matrix:
      void LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0);
      void LoadLanczosOffDiag(int idx, double alpha0, double beta0);
      
      //methods for Rayleigh Ritz procedure: 
      int RestartVm(void* vm, const int cld, const int clen, const int vprec);//complex length

      //methods 
      void FillLanczosDiag(const int _2nev);
      void FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2);
   };

   template<typename Float, typename CudaComplex>
   EigCGArgs<Float, CudaComplex>::EigCGArgs(int m, int nev): m(m), nev(nev){
    //include pad?
    ldm    = ((m+15)/16)*16;//too naive
       
    //magma initialization:
    const int prec = sizeof(Float);
    eigcg_magma_args = new BlasMagmaArgs(m, nev, ldm, prec);

    hTm     = new std::complex<Float>[ldm*m];//VH A V
    hTvalm  = (Float*)malloc(m*sizeof(Float));   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

    //allocate dTm etc. buffers on GPU:
    cudaMalloc(&dTm, ldm*m*sizeof(CudaComplex));//  
    cudaMalloc(&dTvecm, ldm*m*sizeof(CudaComplex));  
    cudaMalloc(&dTvecm1, ldm*m*sizeof(CudaComplex));  

    //set everything to zero:
    cudaMemset(dTm, 0, ldm*m*sizeof(CudaComplex));//?
    cudaMemset(dTvecm, 0, ldm*m*sizeof(CudaComplex));
    cudaMemset(dTvecm1, 0, ldm*m*sizeof(CudaComplex));

    //Error check...
    checkCudaError();

    return;
  }

  template<typename Float, typename CudaComplex>
  EigCGArgs<Float, CudaComplex>::~EigCGArgs() {
    delete[] hTm;

    free(hTvalm);

    cudaFree(dTm);
    cudaFree(dTvecm);
    cudaFree(dTvecm1);

    delete eigcg_magma_args;

    return;
  }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0)
  {
    hTm[idx*ldm+idx] = std::complex<Float>((Float)(1.0/alpha + beta0/alpha0), 0.0);
    return;
  } 

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosOffDiag(int idx, double alpha, double beta)
  {
    hTm[(idx+1)*ldm+idx] = std::complex<Float>((Float)(-sqrt(beta)/alpha), 0.0f);//'U' 
    hTm[idx*ldm+(idx+1)] = hTm[(idx+1)*ldm+idx];//'L'
    return;
  }

  template<typename Float, typename CudaComplex>
  int EigCGArgs<Float, CudaComplex>::RestartVm(void* v, const int cld, const int clen, const int vprec) 
  {
    //Create device version of the Lanczos matrix:
    cudaMemcpy(dTm, hTm, ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);//!

    //Solve m-dimensional eigenproblem:
    cudaMemcpy(dTvecm, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    eigcg_magma_args->MagmaHEEVD((void*)dTvecm, (void*)hTvalm, m);

    //Solve (m-1)-dimensional eigenproblem:
    cudaMemcpy(dTvecm1, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    eigcg_magma_args->MagmaHEEVD((void*)dTvecm1, (void*)hTvalm, m-1);

    //Zero the last row (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
    cudaMemset2D(&dTvecm1[(m-1)], ldm*sizeof(CudaComplex), 0, sizeof(CudaComplex),  (m-1));

    //Attach nev old vectors to nev new vectors (note 2*nev << m):
    cudaMemcpy(&dTvecm[ldm*nev], dTvecm1, ldm*nev*sizeof(CudaComplex), cudaMemcpyDefault);

    //Perform QR-factorization and compute QH*Tm*Q:
    int i = eigcg_magma_args->MagmaORTH_2nev((void*)dTvecm, (void*)dTm);

    //Solve 2nev-dimensional eigenproblem:
    eigcg_magma_args->MagmaHEEVD((void*)dTm, (void*)hTvalm, i);

    //solve zero unused part of the eigenvectors in dTm:
    cudaMemset2D(&(dTm[i]), ldm*sizeof(CudaComplex), 0, (m-i)*sizeof(CudaComplex), i);//check..

    //Restart V:
    eigcg_magma_args->RestartV(v, cld, clen, vprec, (void*)dTvecm, (void*)dTm);

    return i;
  }


  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosDiag(const int _2nev)
 {
    memset(hTm, 0, ldm*m*sizeof(std::complex<Float>));
    for (int i = 0; i < _2nev; i++) hTm[i*ldm+i]= hTvalm[i];//fill-up diagonal

    return;
 }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2)
  {
    if(v->Precision() != u->Precision()) errorQuda("\nIncorrect precision...\n");
    for (int i = 0; i < _2nev; i++){
       std::complex<double> s = cDotProductCuda(*v, u->Eigenvec(i));
       s *= inv_sqrt_r2;
       hTm[_2nev*ldm+i] = std::complex<Float>((Float)s.real(), (Float)s.imag());
       hTm[i*ldm+_2nev] = conj(hTm[_2nev*ldm+i]);
    }
  }

  // set the required parameters for the initCG solver
  void fillInitCGSolveParam(SolverParam &initCGparam) {
    initCGparam.iter   = 0;
    initCGparam.gflops = 0;
    initCGparam.secs   = 0;

    initCGparam.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    initCGparam.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...
  }

  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), search_space_prec(QUDA_INVALID_PRECISION), Vm(0), initCG(0), initCGparam(param), eigcg_alloc(false)
  {
    if((param.rhs_idx < param.deflation_grid) || (param.inv_type == QUDA_EIGCG_INVERTER))
    {
       if(param.nev > MAX_EIGENVEC_WINDOW )
       { 
          warningQuda("\nWarning: the eigenvector window is too big, using default value %d.\n", MAX_EIGENVEC_WINDOW);
          param.nev = MAX_EIGENVEC_WINDOW;
       }

       search_space_prec = param.precision_ritz;
       //
       use_eigcg = true;
       //
       printfQuda("\nIncEigCG will deploy eigCG(m=%d, nev=%d) solver.\n", param.m, param.nev);
    }
    else
    {
       fillInitCGSolveParam(initCGparam);
       //
       if(param.tol_restart < param.tol)//restart was not requested, do normal initCG
          initCG = new CG(mat, matSloppy, initCGparam, profile);
       //  
       use_eigcg = false;
       //
       printfQuda("\nIncEigCG will deploy initCG solver.\n");
    }

    return;
  }

  IncEigCG::~IncEigCG() {

    if(eigcg_alloc) delete Vm;
    if(initCG)      delete initCG;

    return;
  }

  void IncEigCG::EigCG(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    if (param.precision_sloppy == x.Precision()) errorQuda("\nMixedprecision is not supported for the eigCG.\n");

    profile.Start(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    cudaColorSpinorField r(b);

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, csParam);

    mat(r, x, y);
    double r2 = xmyNormCuda(b, r);//compute residual
  
    csParam.setPrecision(param.precision_sloppy);
    cudaColorSpinorField Ap(x, csParam);
    cudaColorSpinorField tmp(x, csParam);

    cudaColorSpinorField *tmp2_p = &tmp;
    // tmp only needed for multi-gpu Wilson-like kernels
    if (mat.Type() != typeid(DiracStaggeredPC).name() && 
	mat.Type() != typeid(DiracStaggered).name()) {
      tmp2_p = new cudaColorSpinorField(x, csParam);
    }
    cudaColorSpinorField &tmp2 = *tmp2_p;

    cudaColorSpinorField *x_sloppy, *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      x_sloppy = &x;
      r_sloppy = &r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x, csParam);
      r_sloppy = new cudaColorSpinorField(r, csParam);
    }

    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;
    cudaColorSpinorField p(rSloppy);

    if(&x != &xSloppy){
      copyCuda(y,x);
      zeroCuda(xSloppy);
    }else{
      zeroCuda(y);
    }
    
    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    
    profile.Stop(QUDA_PROFILE_INIT);
    profile.Start(QUDA_PROFILE_PREAMBLE);

    double r2_old;
    double stop = b2*param.tol*param.tol; // stopping condition of solver

    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double alpha=1.0, beta=0.0;
 
    double pAp;
    int rUpdate = 0;

    int eigvRestart = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    int maxResIncrease = 0; // 0 means we have no tolerance 

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);
    blas_flops = 0;

//eigCG specific code:
    if(eigcg_alloc == false){

       printfQuda("\nAllocating resources for the EigCG solver...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;
       csParam.setPrecision(search_space_prec);//eigCG internal search space precision: must be adjustable.
       csParam.eigv_dim = param.m;

       Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors

       checkCudaError();
       printfQuda("\n..done.\n");
       
       eigcg_alloc = true;
    }

    ColorSpinorParam eigParam(Vm->Eigenvec(0));
    eigParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField  *v0   = new cudaColorSpinorField(Vm->Eigenvec(0), eigParam); //temporary field. 

    cudaColorSpinorField Ap0(Ap);

    //create EigCG objects:
    EigCGArgs<double, cuDoubleComplex> *eigcg_args = new EigCGArgs<double, cuDoubleComplex>(param.m, param.nev); //must be adjustable..
    
    //EigCG additional parameters:
    double alpha0 = 1.0, beta0 = 0.0;

//Begin CG iterations:
    int k=0, l=0;
    
    PrintStats("EigCG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;
    bool relup_flag = false;
    double sigma = 0.0;


    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    k < param.maxiter) {

      double scale = 1.0;

      if(k > 0)
      {
        beta0 = beta;

        if(!relup_flag)
        {
           beta = sigma / r2_old;
           axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
	   if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	     copyCuda(tmp,y);
	     heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
	   }
	   steps_since_reliable++;
        }
       else
       {//after reliable update:
           beta = r2 / r2_old;
	   // explicitly restore the orthogonality of the gradient vector
	   double rp = reDotProductCuda(rSloppy, p) / (r2);
	   axpyCuda(-rp, rSloppy, p);
	   xpayCuda(rSloppy, beta, p);
           scale /= (1.0-rp*beta);
           relup_flag = (l == param.m) ? relup_flag : false;
        }

      }

      //save previous mat-vec result 
      if (l == param.m) copyCuda(Ap0, Ap);

      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp
  
      //construct the Lanczos matrix:
      if(l > 0){
        eigcg_args->LoadLanczosDiag(l-1, alpha, alpha0, beta0);
      }

      //Begin Rayleigh-Ritz procedure:
      if (l == param.m){

         //Restart search space : 
         int cldn = Vm->EigvTotalLength() >> 1; //complex leading dimension
         int clen = Vm->EigvLength()      >> 1; //complex vector length
         //
         int _2nev = eigcg_args->RestartVm(Vm->V(), cldn, clen, Vm->Precision());           

         //Fill-up diagonal elements of the matrix T
         eigcg_args->FillLanczosDiag(_2nev);

         //Compute Ap0 = Ap - beta*Ap0:
         xpayCuda(Ap, -beta, Ap0);//mind precision...

         if(relup_flag){
           axCuda(scale, Ap0);
           relup_flag = false;
         }
           
         copyCuda(*v0, Ap0);//convert arrays here:
         eigcg_args->FillLanczosOffDiag(_2nev, v0, Vm, 1.0 / sqrt(r2));

         eigvRestart++;
         l = _2nev;
      } else{ //no-RR branch:
         if(l > 0){
            eigcg_args->LoadLanczosOffDiag(l-1, alpha, beta);
         }
      }

      //construct Lanczos basis:
      copyCuda(Vm->Eigenvec(l), *r_sloppy);//convert arrays
      //rescale the vector
      scale = 1.0 / sqrt(r2);
      axCuda(scale, Vm->Eigenvec(l));
      //update search space index
      l += 1;

      //end of RR-procedure
      alpha0 = alpha;


      pAp    = reDotProductCuda(p, Ap);
      alpha  = r2 / pAp; 

      // here we are deploying the alternative beta computation 

      r2_old = r2;
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy);
      r2 = real(cg_norm); // (r_new, r_new)
      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k+1-r_k) breaks
      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && delta >= param.tol) updateX = 1;

      if (updateR || updateX) 
      {
	axpyCuda(alpha, p, xSloppy);
	if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
	xpyCuda(x, y); // swap these around?
	mat(r, y, x); // here we can use x as tmp
	r2 = xmyNormCuda(b, r);

	if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
	zeroCuda(xSloppy);

	// break-out check if we have reached the limit of the precision
	static int resIncrease = 0;
	if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
	  warningQuda("EigCG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
	  k++;
	  rUpdate++;
	  if (++resIncrease > maxResIncrease) break; 
	} else {
	  resIncrease = 0;
	}

	rNorm = sqrt(r2);
	maxrr = rNorm;
	maxrx = rNorm;
	r0Norm = rNorm;      
	rUpdate++;

	if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(y,r).z);
        
        relup_flag = true;	
	steps_since_reliable = 0;
      }//end of the reliable update

      k++;

      PrintStats("EigCG", k, r2, b2, heavy_quark_res);
    }

//Free eigcg resources:
    delete eigcg_args;

    if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
    xpyCuda(y, x);

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE){
      printfQuda("EigCG: Reliable updates = %d\n", rUpdate);
      printfQuda("EigCG: Eigenspace restarts = %d\n", eigvRestart);
    }

    // compute the true residuals
    mat(r, x, y);
    param.true_res = sqrt(xmyNormCuda(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    param.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
    param.true_res_hq = 0.0;
#endif      
    PrintSummary("EigCG", k, r2, b2);

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp) delete tmp2_p;

    if (param.precision_sloppy != x.Precision()) {
      delete r_sloppy;
      delete x_sloppy;
    }
//Clean EigCG resources:
    delete v0;

    profile.Stop(QUDA_PROFILE_FREE);

    return;
  }

//END of eigcg solver.

//Deflation space management:
  void IncEigCG::CreateDeflationSpace(cudaColorSpinorField &eigcgSpinor, DeflationParam *&dpar)
  {
    printfQuda("\nCreate deflation space...\n");

    if(eigcgSpinor.SiteSubset() != QUDA_PARITY_SITE_SUBSET) errorQuda("\nRitz spinors must be parity spinors\n");//or adjust it

    dpar = new DeflationParam(param, eigcgSpinor.Volume());

    printfQuda("\n...done.\n");

    //dpar->PrintInfo();

    return;
  }

  void IncEigCG::DeleteDeflationSpace(DeflationParam *&dpar)
  {
    if(dpar != 0) 
    {
      delete dpar;
      dpar = 0;
    }

    return;
  }


  void IncEigCG::ExpandDeflationSpace(DeflationParam *dpar, const int newnevs)
  {
     if(!use_eigcg || (newnevs == 0)) return; //nothing to do

     if(!eigcg_alloc) errorQuda("\nError: cannot expand deflation spase (eigcg ritz vectors were cleaned).\n"); 
     
     printfQuda("\nConstruct projection matrix..\n");

     int added_vecs = 0;

     int new_nevs = newnevs;

     if((new_nevs + dpar->cur_dim) > dpar->tot_dim)
     {
       warningQuda("\nTrying to add too big number of Ritz vectors.\n"); //nothing to do...
       new_nevs = dpar->tot_dim - dpar->cur_dim; 
     }

     //GS orthogonalization
     Complex alpha;

     //alias pointers:
     cpuColorSpinorField *tmpj = 0;
     cpuColorSpinorField *tmpi = 0;

     int l = 0;

     ColorSpinorParam cpuParam(Vm->Eigenvec(0));
     cpuParam.pad       =   0;   //no pad is allowed
     cpuParam.eigv_dim  =   0;
     cpuParam.eigv_id   =  -1;

     cpuParam.create     = QUDA_REFERENCE_FIELD_CREATE;
     cpuParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
     cpuParam.precision  = dpar->ritz_prec;

     for(int i = dpar->cur_dim; i < (dpar->cur_dim + new_nevs); i++)
     {
       size_t offseti = i*dpar->ritz_bytes;
       void   *hptri  = (void*)((char*)dpar->cpuRitzVectors+offseti); 
       cpuParam.v  = hptri;

       tmpi        = new cpuColorSpinorField(cpuParam);

       *tmpi       = Vm->Eigenvec(l);//copy gpu field

       for(int j = 0; j < i; j++)
       {
         size_t offsetj    = j*dpar->ritz_bytes;
         void   *hptrj     = (void*)((char*)dpar->cpuRitzVectors+offsetj); 
         cpuParam.v        = hptrj; 

         tmpj = new cpuColorSpinorField(cpuParam);

         alpha = cDotProductCpu(*tmpj, *tmpi);//<j,i>
         caxpyCpu(-alpha, *tmpj, *tmpi); //i-<j,i>j

         delete tmpj;
         tmpj = 0;
       }
         
       alpha = norm2(*tmpi);

       const double norm2_tol = dpar->ritz_prec == sizeof(double) ? 1e-22 : 1e-16;

       if(alpha.real() > norm2_tol)
       {
         axCpu(1.0 /sqrt(alpha.real()), *tmpi);  
         added_vecs += 1;
       }
       else
       {
          errorQuda("\nCannot orthogonalize %dth vector\n", i);
       }

       Vm->Eigenvec(l) = *tmpi;//use Vm as GPU buffer for futher manipulations...
    
       delete tmpi;
       tmpi = 0;

       l+=1;
    }

    //Create temporary fileds:

    ColorSpinorParam cudaParam(Vm->Eigenvec(0));
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;

    cudaParam.eigv_dim  = 0;
    cudaParam.eigv_id   = -1;

    cudaColorSpinorField *W   = new cudaColorSpinorField(cudaParam); 

    cudaColorSpinorField tmp (*W, cudaParam);

    l = 0;

    void *buff = malloc(dpar->ritz_bytes); 

    cpuParam.v = buff;

    tmpj = new cpuColorSpinorField(cpuParam);
  
    for (int j = dpar->cur_dim; j < (dpar->cur_dim+added_vecs); j++)//
    {

       matDefl(*W, Vm->Eigenvec(l), tmp);

       *tmpj = *W;

       //off-diagonal:
       for (int i = 0; i < j; i++)//row id
       {
          size_t offseti = i*dpar->ritz_bytes;
          void *hptri    = (void*)((char*)dpar->cpuRitzVectors+offseti); 

          cpuParam.v = hptri;

          tmpi = new cpuColorSpinorField(cpuParam);

          alpha  =  cDotProductCpu(*tmpi, *tmpj);
          //
          dpar->proj_matrix[j*dpar->ld+i] = alpha;
          dpar->proj_matrix[i*dpar->ld+j] = conj(alpha);//conj

          delete tmpi;
          tmpi = 0;
       }

       //diagonal:
       alpha  =  cDotProductCuda(Vm->Eigenvec(l), *W);
       //
       dpar->proj_matrix[j*dpar->ld+j] = alpha;

       l += 1;
    }

    dpar->ResetDeflationCurrentDim(added_vecs);

    printfQuda("\n.. done.\n");

    free(buff);
    delete W;

    return;
  }

  void IncEigCG::RelocateRitzVectors(cudaColorSpinorField &spinor, DeflationParam *dpar)
  {
    //clean eigcg CUDA resources
    if(eigcg_alloc)
    {
      delete Vm;
      Vm          = 0;
      eigcg_alloc = false;
    }

    //allocate Ritz vectors on gpu:
    size_t free_bytes, total_bytes;

    cudaMemGetInfo(&free_bytes, &total_bytes);

    printfQuda("\nFree GPU memory: %lu bytes (of total %lu bytes)\n", free_bytes, total_bytes);

    //ok, must take into account initCG resources, just estimate...
    size_t init_cg_bytes  = dpar->ritz_bytes*10;

    free_bytes -= init_cg_bytes;   

    if (free_bytes < param.m*dpar->ritz_bytes) //this won't happen, though
    {
       warningQuda("\nWarning : not enough memory on the device, continue using host resources\n");
       return;
    }

    const int max_cuda_ritz_num = free_bytes /  dpar->ritz_bytes;

    dpar->cuda_ritz_dim = max_cuda_ritz_num > dpar->cpu_ritz_dim ? dpar->cpu_ritz_dim : max_cuda_ritz_num;

    printfQuda("\nAllocating %d cuda Ritz vectors\n", dpar->cuda_ritz_dim);

    //allocate Ritz vectors on the device:
    ColorSpinorParam eigvParam(spinor);

    dpar->AllocateRitzCuda(eigvParam);

    //exctract host eigenvectors:
    eigvParam.pad      =   0;//now pad is allowed
    eigvParam.eigv_dim =   0;    
    eigvParam.eigv_id  =  -1;
    eigvParam.create     = QUDA_REFERENCE_FIELD_CREATE;
    eigvParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    eigvParam.precision  = dpar->ritz_prec;

    cpuColorSpinorField *tmp = 0;
  
    //naive:
    for(int i = 0; i < dpar->cuda_ritz_dim; i++)
    {
       size_t offset    = i*dpar->ritz_bytes;

       void *hptr       = (void*)((char*)dpar->cpuRitzVectors+offset); 

       eigvParam.v      = hptr;

       tmp = new cpuColorSpinorField(eigvParam);

       dpar->cudaRitzVectors->Eigenvec(i) = *tmp;
      
       delete tmp;
       tmp = 0;  
    }

    //update host projection matrix.
    //...not in the current implementation

    //delete host ritz vectors:
    dpar->CleanHostRitzVectors(); //this should be optional.

    return;
  }


  void IncEigCG::DeflateSpinor(cudaColorSpinorField &x, cudaColorSpinorField &b, DeflationParam *dpar)
  {
    zeroCuda(x);

    if(defl_param == 0) errorQuda("\nError: no deflation matrix is available\n");//for now report this as an error.
    //
    if(defl_param->cur_dim == 0) return;//nothing to do

    BlasMagmaArgs *magma_args = new BlasMagmaArgs(sizeof(double));

    Complex  *vec   = new Complex[dpar->ld];

    if(dpar->cuda_ritz_alloc && !use_eigcg) //currently for initCG phase only.
    {
      for(int i = 0; i < dpar->cuda_ritz_dim; i++)  vec[i] = cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(i), b);

      magma_args->SolveProjMatrix((void*)vec, dpar->ld,  dpar->cur_dim, (void*)dpar->proj_matrix, dpar->ld);

      for(int i = 0; i < dpar->cuda_ritz_dim; i++)  caxpyCuda(vec[i], dpar->cudaRitzVectors->Eigenvec(i), x); //a*i+x
    }
    else if(dpar->cpu_ritz_alloc && (dpar->cpu_ritz_dim >= dpar->cur_dim))
    {
      void *hbuff1   = malloc(dpar->ritz_bytes);
      void *hbuff2   = malloc(dpar->ritz_bytes);

      memset(hbuff1, 0, dpar->ritz_bytes);
      memset(hbuff2, 0, dpar->ritz_bytes);

      ColorSpinorParam eigvParam(b); //Eigenvector set is always a parity spinor

      eigvParam.pad      =   0;    
      eigvParam.eigv_dim =   0;    
      eigvParam.eigv_id  =  -1; 

      eigvParam.create     = QUDA_REFERENCE_FIELD_CREATE;
      eigvParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      eigvParam.precision  = dpar->ritz_prec;

      eigvParam.v             = hbuff1;
      cpuColorSpinorField *hb = new cpuColorSpinorField(eigvParam);
      *hb = b;//copy gpu spinor

      eigvParam.v             = hbuff2;
      cpuColorSpinorField *hx = new cpuColorSpinorField(eigvParam);

      //alias pointer:
      cpuColorSpinorField *tmp = 0;

      for(int i = 0; i < dpar->cur_dim; i++)
      {
        size_t offset    = i*dpar->ritz_bytes;
        void *hptr       = (void*)((char*)dpar->cpuRitzVectors+offset); 

        eigvParam.v      = hptr;

        tmp = new cpuColorSpinorField(eigvParam); 
     
        vec[i] = cDotProductCpu(*tmp, *hb);//<i, b>
     
        delete tmp;
        tmp = 0;
      }    

      //lapackF77_zgesv(curnev, H, ldH, (void*)vec);
      magma_args->SolveProjMatrix((void*)vec, dpar->ld,  dpar->cur_dim, (void*)dpar->proj_matrix, dpar->ld);

      for(int i = 0; i < dpar->cur_dim; i++)
      {
        size_t offset    = i*dpar->ritz_bytes;
        void *hptr       = (void*)((char*)dpar->cpuRitzVectors+offset); 

        eigvParam.v      = hptr;

        tmp = new cpuColorSpinorField(eigvParam); 

        caxpyCpu(vec[i], *tmp, *hx); //a*i+x

        delete tmp;
        tmp = 0;
      }
      //copy back to device:
      x = *hx;

      delete hb;
      delete hx;

      free(hbuff1);
      free(hbuff2);
    }
    else //gpu version
    {
      errorQuda("\nError: cannot find Ritz vectors needed for the projection.\n");
    }

    printfQuda("\nDone...\n");

    delete magma_args;

    delete [] vec;

    return;
  }

  //copy EigCG ritz vectors.
  void IncEigCG::SaveRitzVecs(cudaColorSpinorField *out, const int nevs, bool cleanEigCGResources)
  {
     if(out->EigvDim() < nevs) errorQuda("\nNot enough space to copy %d eigenvectors..\n", nevs); 

     if(!eigcg_alloc) 
     {
       warningQuda("\nEigCG resources were cleaned.\n"); 
       return;
     } 


     for(int i = 0; i < nevs; i++) copyCuda(out->Eigenvec(i), Vm->Eigenvec(i));

     if(cleanEigCGResources)
     {
       delete Vm;
       Vm = 0;
       eigcg_alloc = false;
     }
     else//just call zeroCuda..
     {
       zeroCuda(*Vm);
     }

     return;
  }

  void IncEigCG::CleanResources()
  {
    if(eigcg_alloc)
    {
       delete Vm;
       Vm = 0;
       eigcg_alloc = false;
    }
    if(defl_param != 0)
    {
       DeleteDeflationSpace(defl_param);
       defl_param = 0;
    }

    return;
  } 


  static bool first_time_initcg = true; 

  void IncEigCG::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
  {
     if(defl_param == 0) CreateDeflationSpace(*in, defl_param);

     //if this operator applied during the first stage of the incremental eigCG (to construct deflation space):
     //then: call eigCG inverter 
     if(use_eigcg){
        //deflate initial guess:
        DeflateSpinor(*out, *in, defl_param);

        //compute current nev Ritz vectors:
        EigCG(*out, *in);        

        //Construct(extend) projection matrix:
        ExpandDeflationSpace(defl_param, param.nev);
     }
     //second stage here: param.rhs_idx >= param.deflation_grid 
     else{

        if(first_time_initcg)
        {
          RelocateRitzVectors(*out, defl_param);
          first_time_initcg = false;
        }

        DeflateSpinor(*out, *in, defl_param);

        //launch initCG:
        if(initCG)
          (*initCG)(*out, *in);
        else //think about this!
        {
          ColorSpinorParam cudaParam(*out);
          cudaParam.create = QUDA_ZERO_FIELD_CREATE;

          cudaParam.eigv_dim  = 0;
          cudaParam.eigv_id   = -1;

          cudaColorSpinorField *W   = new cudaColorSpinorField(cudaParam); 
          cudaColorSpinorField tmp (*W, cudaParam);

          double tol_swap = initCGparam.tol;

          initCGparam.tol = initCGparam.tol_restart;

          initCG = new CG(mat, matSloppy, initCGparam, profile);

          (*initCG)(*out, *in);           

          delete initCG;

          initCGparam.tol = tol_swap;

          matDefl(*W, *out, tmp);

          xpayCuda(*in, -1, *W); 

          DeflateSpinor(*out, *W, defl_param);

          initCG = new CG(mat, matSloppy, initCGparam, profile);

          (*initCG)(*out, *in);           
        
          delete initCG;
          
        } 

        //copy solver statistics:
        param.iter   = initCGparam.iter;
        //
        param.secs   = initCGparam.secs;
        //
        param.gflops = initCGparam.gflops;
     } 

     param.rhs_idx += 1;

     return;
  }


} // namespace quda
