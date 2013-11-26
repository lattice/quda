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
//Warning: magma matrices in coloumn-major format...
WARNING: for coalescing m must be multiple of 16, use pad for other choises
*/

namespace quda {

  ProjectionMatrix::ProjectionMatrix(DiracMatrix &matDefl, SolverParam &param) : matDefl(matDefl), prev_dim(0)
  { 
     if(param.nev == 0 || param.deflation_grid == 0) errorQuda("\nIncorrect deflation space parameters...\n");

     tot_dim      = param.deflation_grid*param.nev;
     curr_dim     = param.nev;
     ld           = ((tot_dim+15) / 16) * tot_dim;

     //precision:
     prec    = param.precision;//always check precision!

     //bytes:
     bytes = ld*tot_dim * (2*prec);//complex matrix

     //allocate complex arrays:
     hproj = malloc(bytes);
     memset(hproj, 0, bytes); 
      
     cudaMalloc(&dproj,bytes);
     cudaMemset(dproj, 0, bytes);
  }

  ProjectionMatrix::~ProjectionMatrix() {
    //free allocated resources:
    free(hproj);
    cudaFree(dproj);
  }

  void ProjectionMatrix::operator()(void *out, cudaColorSpinorField *r, cudaColorSpinorField *u)
  {
     return;
  }

  void ProjectionMatrix::ConstructProj(cudaColorSpinorField &u)
  {
     return;
  }

  void ProjectionMatrix::LoadProj(void *in)
  {
     memcpy(hproj, in, bytes);
     printfQuda("\nCopy %d bytes for the projector matrix..\n", bytes);
     cudaMemcpy(dproj, in, bytes, cudaMemcpyDefault);
     checkCudaError();
     return;
  }

  void ProjectionMatrix::SaveProj(void *out)
  {
     printfQuda("\nCopy %d bytes for the projector matrix..\n", bytes);
     cudaMemcpy(out, dproj, bytes, cudaMemcpyDefault);
     checkCudaError();
     return;
  }

  void ProjectionMatrix::ResetProjMatDim(const int n)
  {
     if(n > tot_dim) errorQuda("\nCannot reset projection matrix dimension.\n");
     prev_dim = curr_dim;
     curr_dim = n;
     return;
  }

  void ProjectionMatrix::PrintInfo()
  {
     printfQuda("\nProjection matrix information:\n");
     printfQuda("Precision %d\n", prec);
     printfQuda("Leading dimension %d\n", ld);
     printfQuda("Total dimension %d\n", tot_dim);
     printfQuda("Current dimension %d\n", curr_dim);
     printfQuda("Bytes: %d\n", bytes);
     printfQuda("Host pointer: %p\n", hproj);
     printfQuda("Device pointer: %p\n\n", dproj);
  }


  // set the required parameters for the initCG solver
  void fillInitCGSolveParam(SolverParam &initCGparam) {
    initCGparam.iter   = 0;
    initCGparam.gflops = 0;
    initCGparam.secs   = 0;

    initCGparam.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    initCGparam.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...
  }


  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, ColorSpinorParam *eigvParam, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), initCG(0), initCGparam(param), Vm(0), hTm(0), hTvecm(0), hTvalm(0), dTm(0), dTvecm0(0), dTvecm1(0)
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

       hTm     = new std::complex<float>[param.m*param.m];//VH A V
       hTvecm  = new std::complex<float>[param.m*param.m];//eigenvectors of both T[m,  m  ] and T[m-1, m-1] (re-used)
       hTvalm  = new float[param.m];   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

       //allocate dT etc. buffers on GPU:
       cudaMalloc(&dTm, param.m*param.m*sizeof(cuComplex));//  
       cudaMalloc(&dTvecm0, param.m*param.m*sizeof(cuComplex));  
       cudaMalloc(&dTvecm1, param.m*param.m*sizeof(cuComplex));  

       //set everything to zero:
       cudaMemset(dTm, 0, param.m*param.m*sizeof(cuComplex));//?
       cudaMemset(dTvecm0, 0, param.m*param.m*sizeof(cuComplex));
       cudaMemset(dTvecm1, 0, param.m*param.m*sizeof(cuComplex));
       //Error check is missing...
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
  }

  IncEigCG::~IncEigCG() {
    if(eigcg_alloc)
    {
      delete Vm;

      delete[] hTm;
      delete[] hTvecm;
      delete[] hTvalm;

      cudaFree(dTm);
      cudaFree(dTvecm0);
      cudaFree(dTvecm1);
    }
    delete initCG;

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

    const int m   = param.m;//include pad , 64-bit aligned
    const int nev = param.nev;

    const int ldTm  = m;

    //magma initialization:

    BlasMagmaArgs *magma_args = new BlasMagmaArgs(m, nev);

    double alpha0 = 1.0, beta0 = 0.0;//EigCG additional parameters

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
           relup_flag = (l == m) ? relup_flag : false;
        }
      }

      //save previous mat-vec result 
      if (l == m) copyCuda(Ap0, Ap);

      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp
  
      //construct the Lanczos matrix:
      if(k > 0){
        hTm[(l-1)*ldTm+(l-1)] = std::complex<float>((float)(1/alpha + beta0/alpha0), 0.0f);
      }

      //Begin Rayleigh-Ritz procedure:
      if (l == m){
         //Create device version of the Lanczos matrix:
         cudaMemcpy(dTm, hTm, ldTm*m*sizeof(cuComplex), cudaMemcpyDefault);
           
         //run main part here: 
         int _2nev = magma_args->RayleighRitz((cuComplex*)dTm, (cuComplex*)dTvecm0, (cuComplex*)dTvecm1, hTvecm, hTvalm);

         //Compute Ritz vectors : V=V(n, m)*dTm(m, l)
         magma_args->Restart_2nev_vectors((cuComplex*)Vm->V(), (cuComplex*)dTm, Vm->EigvLength());//m->ldTm
           
         //Fill-up diagonal elements of the matrix T
         memset(hTm, 0, ldTm*m*sizeof(std::complex<float>));
    	 for (int i = 0; i < _2nev; i++) hTm[i*ldTm+i]= hTvalm[i];//fill-up diagonal

         //Compute Ap0 = Ap - beta*Ap0:
         xpayCuda(Ap, -beta, Ap0);//mind precision...
         if(relup_flag){
           axCuda(scale, Ap0);
           relup_flag = false;
         }
           
         copyCuda(*v0, Ap0);//convert arrays here:
	 for (int i = 0; i < _2nev; i++){
	     std::complex<double> s = cDotProductCuda(*v0, Vm->Eigenvec(i));
	     hTm[_2nev*ldTm+i] = std::complex<float>((float)(s.real()/sqrt(r2)), (float)(s.imag()/sqrt(r2)));
	     hTm[i*ldTm+_2nev] = conj(hTm[_2nev*ldTm+i]);
	 }

         eigvRestart++;
         l = _2nev;
      } else{ //no-RR branch:
         if(k > 0){
            hTm[l*ldTm+(l-1)] = std::complex<float>((float)(sqrt(beta0)/alpha0), 0.0f);//'U' 
            hTm[(l-1)*ldTm+l] = hTm[l*ldTm+(l-1)];//'L'
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
      beta0  = beta;

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

//Shutdown magma:
    delete magma_args;

//Copy nev eigvectors:
    Vm->CopyEigenvecSubset(e, nev);

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

  void IncEigCG::DeflateInitGuess(cudaColorSpinorField &in, const cudaColorSpinorField &u, const ProjectionMatrix &pM)
  {
    //using magma gesv routine:
    return;
  }

  void IncEigCG::OrthRitz(cudaColorSpinorField &u)
  {
    //based on mGS algorithm:
    return;
  }

  void IncEigCG::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in, cudaColorSpinorField *u, ProjectionMatrix *pM) 
  {
     //if this operator applied during the first stage of the incremental eigCG (to construct deflation space):
     //then: call eigCG inverter 
     if((param.rhs_idx == 0) || (param.inv_type == QUDA_EIGCG_INVERTER))
     {
        //compute the first nev Ritz vectors:
        EigCG(*out, *u, *in);

        //Construct projection matrix:
        pM->ConstructProj(*u);

        //finish for this first rhs:
        param.rhs_idx++;
     }
     else if(param.rhs_idx < param.deflation_grid)
     {
        //we are still in the 1st stage
        const int w_range = param.nev*(param.rhs_idx+1);
        const int offset  = param.nev*param.rhs_idx;

        //deflate initial guess:
        DeflateInitGuess(*in, *u, *pM);

        //compute current nev Ritz vectors:
        EigCG(*out, *u, *in);        

        //orthogonalize new nev vectors against old vectors 
        OrthRitz(*u);

        //Construct(extend) projection matrix:
        pM->ConstructProj(*u);

        //finish for this first rhs:
        param.rhs_idx++;
     }
     //second stage here: param.rhs_idx >= param.deflation_grid 
     else
     {
        //deallocate eigCG resources (if they were created)
        if(eigcg_alloc){
          delete Vm;

          delete[] hTm;
          delete[] hTvecm;
          delete[] hTvalm;

          cudaFree(dTm);
          cudaFree(dTvecm0);
          cudaFree(dTvecm1);
          eigcg_alloc = false;
        }

        //deflate initial guess:
        const int range = param.nev*param.deflation_grid;

        DeflateInitGuess(*in, *u, *pM);

        //launch initCG:
        (*initCG)(*out, *in);

        param.rhs_idx++;
     } 
//in main routine don't forget: solverParam.updateInvertParam(*param);

     return;
  }


} // namespace quda
