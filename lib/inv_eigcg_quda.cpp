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

/*
Base eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
//Note: 
1) magma matrices are in coloumn-major format...
2) for coalescing the search space dimension m must be multiple of 16 (use pad or adjust dimension?)
*/

namespace quda {
 
   template<typename Float, typename CudaComplex>
   class EigCGArgs{
     
      private:
      BlasMagmaArgs *eigcg_magma_args;
    
      //host Lanczos matrice, and its eigenvalue/vector arrays:
      std::complex<Float> *hTm;//VH A V
      std::complex<Float> *hTvecm;//eigenvectors of both T[m,  m  ] and T[m-1, m-1] (re-used)
      Float  *hTvalm;   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

      //device Lanczos matrix, and its eigenvalue/vector arrays:
      CudaComplex *dTm;     //VH A V
      CudaComplex *dTvecm0; //eigenvectors of T[m,  m  ]
      CudaComplex *dTvecm1; //eigenvectors of T[m-1,m-1]

      int m;
      int nev;
      int ldTm;

      public:

      EigCGArgs(int m, int nev);
      ~EigCGArgs();
      
      //methods for constructing Lanczos matrix:
      void LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0);
      void LoadLanczosOffDiag(int idx, double alpha, double beta);
      
      //methods for Rayleigh Ritz procedure: 
      int  RunRayleighRitz();
      void RestartVm(CudaComplex* vm, const int complex_eigv_len);

      //methods 
      void FillLanczosDiag(const int _2nev);
      void FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2);
   };

   template<typename Float, typename CudaComplex>
   EigCGArgs<Float, CudaComplex>::EigCGArgs(int m, int nev): m(m), nev(nev){
       
    //magma initialization:
    const int prec = sizeof(Float);
    eigcg_magma_args = new BlasMagmaArgs(m, nev, prec);

    //include pad?
    ldTm    = m;//naive
    hTm     = new std::complex<Float>[ldTm*m];//VH A V
    hTvecm  = new std::complex<Float>[ldTm*m];//eigenvectors of both T[m,  m  ] and T[m-1, m-1] (re-used)
    hTvalm  = new Float[m];   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

    //allocate dTm etc. buffers on GPU:
    cudaMalloc(&dTm, ldTm*m*sizeof(CudaComplex));//  
    cudaMalloc(&dTvecm0, ldTm*m*sizeof(CudaComplex));  
    cudaMalloc(&dTvecm1, ldTm*m*sizeof(CudaComplex));  

    //set everything to zero:
    cudaMemset(dTm, 0, ldTm*m*sizeof(CudaComplex));//?
    cudaMemset(dTvecm0, 0, ldTm*m*sizeof(CudaComplex));
    cudaMemset(dTvecm1, 0, ldTm*m*sizeof(CudaComplex));

    //Error check...
    checkCudaError();
   
    return;
  }

  template<typename Float, typename CudaComplex>
  EigCGArgs<Float, CudaComplex>::~EigCGArgs() {
    delete[] hTm;
    delete[] hTvecm;
    delete[] hTvalm;

    cudaFree(dTm);
    cudaFree(dTvecm0);
    cudaFree(dTvecm1);

    delete eigcg_magma_args;

    checkCudaError();

    return;
  }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0)
  {
    hTm[idx*ldTm+idx] = std::complex<Float>((Float)(1.0/alpha + beta0/alpha0), 0.0);
    return;
  } 

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosOffDiag(int idx, double alpha, double beta)
  {
    hTm[(idx+1)*ldTm+idx] = std::complex<Float>((Float)(-sqrt(beta)/alpha), 0.0f);//'U' 
    hTm[idx*ldTm+(idx+1)] = hTm[(idx+1)*ldTm+idx];//'L'
    return;
  }

  template<typename Float, typename CudaComplex>
  int EigCGArgs<Float, CudaComplex>::RunRayleighRitz() 
  { 
    //Create device version of the Lanczos matrix:
    cudaMemcpy(dTm, hTm, ldTm*m*sizeof(CudaComplex), cudaMemcpyDefault);//!
           
    //run RayleighRitz: 
    int _2nev = eigcg_magma_args->RayleighRitz((void*)dTm, (void*)dTvecm0, (void*)dTvecm1, (void*)hTvecm, (void*)hTvalm);

    return _2nev; 
  }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::RestartVm(CudaComplex* v, const int complex_eigv_len) 
  {
    eigcg_magma_args->Restart_2nev_vectors((void*)v, (void*)dTm, complex_eigv_len);
    return;
  }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosDiag(const int _2nev)
 {
    memset(hTm, 0, ldTm*m*sizeof(std::complex<Float>));
    for (int i = 0; i < _2nev; i++) hTm[i*ldTm+i]= hTvalm[i];//fill-up diagonal

    return;
 }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2)
  {
    if(v->Precision() != u->Precision()) errorQuda("\nIncorrect precision...\n");
    for (int i = 0; i < _2nev; i++){
       std::complex<double> s = cDotProductCuda(*v, u->Eigenvec(i));
       s *= inv_sqrt_r2;
       hTm[_2nev*ldTm+i] = std::complex<Float>((Float)s.real(), (Float)s.imag());
       hTm[i*ldTm+_2nev] = conj(hTm[_2nev*ldTm+i]);
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


  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDefl, ColorSpinorParam *eigvParam, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), initCG(0), initCGparam(param), Vm(0)
  {
    if(param.rhs_idx < param.deflation_grid)
    {
       printfQuda("\nAllocating resources for the EigCG solver...\n");
       if(param.nev >= param.m / 2 ) errorQuda("\nThe eigenvector window is too big! (Or the search space is too small..)\n");
       //Create an eigenvector set:
       eigvParam->create   = QUDA_ZERO_FIELD_CREATE;
       eigvParam->setPrecision(QUDA_SINGLE_PRECISION);//eigCG internal search space is in single precision (currently)
       eigvParam->eigv_dim = param.m;

       Vm = new cudaColorSpinorField(*eigvParam); //search space for Ritz vectors

       //Error check...
       checkCudaError();
       printfQuda("\n..done.\n");
       
       eigcg_alloc = true;
    }
    else
    {
       printfQuda("\nIncEigCG will deploy initCG solver.\n");
       eigcg_alloc = false;
    }
    fillInitCGSolveParam(initCGparam);
    initCG = new CG(mat, matSloppy, initCGparam, profile);

    //Create projection matrix:
    pM = new ProjectionMatrix(param);//Current precision may not be correct...
  }

  IncEigCG::~IncEigCG() {
    if(eigcg_alloc) delete Vm;

    delete initCG;
    delete pM;
  }

  void IncEigCG::EigCG(cudaColorSpinorField &x, cudaColorSpinorField &e, cudaColorSpinorField &b) 
  {
    if(!eigcg_alloc){
       errorQuda("Error: EigCG solver resources were not allocated...\n");
    }

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

//EigCG specific code:
    ColorSpinorParam eigParam(Vm->Eigenvec(0));
    eigParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField  *v0   = new cudaColorSpinorField(Vm->Eigenvec(0), eigParam);  

    cudaColorSpinorField &Ap0 = *tmp2_p;

    //create EigCG objects:
    EigCGArgs<float, cuComplex> *eigcg_args = new EigCGArgs<float, cuComplex>(param.m, param.nev); 
    
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

      if(k > 0){
        beta0 = beta;
        if(!relup_flag){
           beta = sigma / r2_old;
           axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
	   if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	     copyCuda(tmp,y);
	     heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
	   }
	steps_since_reliable++;
        }else{//after reliable update:
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
      if(k > 0){
        eigcg_args->LoadLanczosDiag(l-1, alpha, alpha0, beta0);
      }

      //Begin Rayleigh-Ritz procedure:
      if (l == param.m){
         int _2nev = eigcg_args->RunRayleighRitz();

         //Compute Ritz vectors : V=V(n, m)*dTm(m, l)
         int complex_eigv_len = Vm->EigvLength() / 2;//EigvLength() includes complex
         eigcg_args->RestartVm((cuComplex*)Vm->V(), complex_eigv_len);           

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
         if(k > 0){
            eigcg_args->LoadLanczosOffDiag(l-1, alpha, beta);
         }
      }
      l += 1;
      //construct Lanczos basis:
      copyCuda(Vm->Eigenvec(l-1), *r_sloppy);//convert arrays
      //rescale the vector
      scale = 1.0 / sqrt(r2);
      axCuda(scale, Vm->Eigenvec(l-1));

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

      if (updateR || updateX) {
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

//Copy nev eigvectors:
    Vm->CopyEigenvecSubset(e, param.nev);

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

  void IncEigCG::ConstructProjectionMat(cudaColorSpinorField &u)
  {
     if(pM->curr_dim > u.EigvDim()) errorQuda("\nIncorrect eigenvector window..\n");

     //First compute W=MV
     //Create temporal spinor array:
     ColorSpinorParam cudaParam(u);
     cudaParam.create   = QUDA_ZERO_FIELD_CREATE;
     cudaParam.setPrecision(u.Precision());
     cudaParam.eigv_dim = 0;
     cudaParam.eigv_id  = -1;

     //temporal spinors:
     cudaColorSpinorField *W = new cudaColorSpinorField(cudaParam); 
     cudaColorSpinorField tmp(*W, cudaParam);//temporal array

     //now construct (extend) the projection matrix:
     for(int i = pM->prev_dim ; i < pM->curr_dim; i++)
     {
        matDefl(*W, u.Eigenvec(i), tmp);
        for(int j = 0; j < pM->prev_dim; j++)
        {
           int s             = i * pM->tot_dim + j;
           int conj_s        = j * pM->tot_dim + i;
           pM->hproj[s     ] = cDotProductCuda(u.Eigenvec(j), *W);
           pM->hproj[conj_s] = conj(pM->hproj[s]);
        } 
        for(int j = pM->prev_dim; j < pM->curr_dim; j++)
        {
           int s        = i * pM->tot_dim + j;
           pM->hproj[s] = cDotProductCuda(u.Eigenvec(j), *W);
        } 
     }

     delete W;

     return;
  }

  void IncEigCG::DeflateSpinor(cudaColorSpinorField &in, const cudaColorSpinorField &u)
  {
    if(pM->curr_dim > u.EigvDim()) errorQuda("\nProjection matrix dimension does not match eigenspace dimension.\n");
    if(in.Precision() > u.Precision()) errorQuda("\nPrecisions does not match.\n");
 
    if(pM->prev_dim == 0) return;//nothing to do

    //magma initialization:
    const int prec = in.Precision();
    BlasMagmaArgs *magma_args = new BlasMagmaArgs(prec);//new

    Complex *vec = new Complex[pM->tot_dim];
    memset(vec, 0, pM->tot_dim*sizeof(Complex));

    for(int i = 0; i < pM->prev_dim; i++) vec[i] = cDotProductCuda(in, u.Eigenvec(i));

    //Solve Hx=y:
    magma_args->SolveProjMatrix((void*)vec, pM->tot_dim, pM->prev_dim, (void*)pM->hproj, pM->tot_dim);

    //
    const int complex_len = u.EigvLength() / 2;
    if(in.Precision() == QUDA_DOUBLE_PRECISION)
    {
      magma_args->SpinorMatVec(in.V(), u.V(), (void*)vec, complex_len, pM->prev_dim);
    }
    else if (in.Precision() == QUDA_SINGLE_PRECISION) 
    {
      std::complex<float> *tmp = new std::complex<float>[pM->tot_dim];

      for(int i = 0; i < pM->prev_dim; i++) tmp[i] = std::complex<float>((float)vec[i].real(), (float)vec[i].imag()); 
      magma_args->SpinorMatVec(in.V(), u.V(), (void*)tmp, complex_len, pM->prev_dim); 
   
      delete[] tmp;
    }
    else
    {
       errorQuda("\nUnsupported precision..\n");
    }

    delete[] vec;
    delete magma_args;

    return;
  }

  void IncEigCG::MGS(cudaColorSpinorField &u)
  {
    //Apply MGS algorithm:
    const int w_range = param.nev*(param.rhs_idx+1);
    const int offset  = param.nev*param.rhs_idx;

    if(offset == 0) return; //nothing to do...

    for(int i = offset; i < w_range; i++)
    {
      for(int j = 0; j < i; j++)
      {
        Complex tmp = cDotProductCuda(u.Eigenvec(j), u.Eigenvec(i));
        caxpyCuda(-tmp, u.Eigenvec(j), u.Eigenvec(i));
      }
      //normalize vector:
      double tmp = normCuda(u.Eigenvec(i));//sqrt?
      axCuda(tmp, u.Eigenvec(i));
    }

    return;
  }

  void IncEigCG::LoadProjectionMatrix(const void* in, const int bytes)
  {
    const int cpy_bytes = bytes != 0 ? bytes : pM->prev_dim*pM->tot_dim*sizeof(Complex); 
    if(cpy_bytes == 0) return;//nothing to copy
    pM->LoadProj(in, cpy_bytes);
    return;
  }
  void IncEigCG::SaveProjectionMatrix(void* out)
  {
    const int cpy_bytes = pM->curr_dim*pM->tot_dim*sizeof(Complex); 
    pM->SaveProj(out, cpy_bytes);
    return;
  }

  void IncEigCG::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in, cudaColorSpinorField *u) 
  {
     //if this operator applied during the first stage of the incremental eigCG (to construct deflation space):
     //then: call eigCG inverter 
     if(param.rhs_idx < param.deflation_grid || param.inv_type == QUDA_EIGCG_INVERTER){
        //deflate initial guess:
        DeflateSpinor(*in, *u);

        //compute current nev Ritz vectors:
        EigCG(*out, *u, *in);        

        //orthogonalize new nev vectors against old vectors 
        MGS(*u);

        //Construct(extend) projection matrix:
        ConstructProjectionMat(*u);
     }
     //second stage here: param.rhs_idx >= param.deflation_grid 
     else{
        //deallocate eigCG resources (if they were created)
        if(eigcg_alloc){
          delete Vm;
          eigcg_alloc = false;
        }

        DeflateSpinor(*in, *u);

        //launch initCG:
        (*initCG)(*out, *in);
     } 
//in main routine don't forget: solverParam.updateInvertParam(*param);
     param.rhs_idx++;

     return;
  }


} // namespace quda
