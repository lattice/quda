#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <face_quda.h>
#include <domain_decomposition.h>
#include <resize_quda.h>
#include <time.h>

// I need to add functionality to the conjugate-gradient solver.
// Should I do this by simple inheritance, or should I use a decorator? 
// Is it even possible to use a decorator?

namespace quda {

  // set the required parameters for the inner solver
  void fillInnerCGInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer) {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.reliable_delta = 1e-20; // no reliable updates within the inner solver

    inner.cuda_prec = outer.cuda_prec_precondition; // preconditioners are uni-precision solvers
    inner.cuda_prec_sloppy = outer.cuda_prec_precondition;

    inner.verbosity = outer.verbosity;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_CG_INVERTER; // used to tell the inner solver it is an inner solver

    if (outer.inv_type == QUDA_CG_INVERTER && outer.cuda_prec_sloppy != outer.cuda_prec_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES; // What does this mean?

  } // copied from inv_gcr_quda.cpp



  PreconCG::PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrec, QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrec), K(NULL), innerProfile("innerSolver")
  {
    Kparam = newQudaInvertParam();

    for(int dir=0; dir<4; ++dir) Kparam.domain_overlap[dir] = invParam.domain_overlap[dir];
    fillInnerCGInvertParam(Kparam, invParam);


    Kparam.dslash_type   = invParam.dslash_type;
    Kparam.inv_type      = invParam.inv_type;
    Kparam.solution_type = invParam.solution_type;
    Kparam.solve_type    = invParam.solve_type;
    Kparam.matpc_type    = invParam.matpc_type;
    Kparam.dirac_order   = invParam.dirac_order;


    Kparam.input_location  = invParam.input_location;
    Kparam.output_location = invParam.output_location;
    Kparam.mass = invParam.mass;
    Kparam.dagger = invParam.dagger;
    Kparam.mass_normalization = invParam.mass_normalization;
    Kparam.preserve_source = invParam.preserve_source;

    Kparam.cpu_prec = invParam.cpu_prec;
    Kparam.cuda_prec = invParam.cuda_prec_precondition;
    Kparam.cuda_prec_sloppy = invParam.cuda_prec_precondition;


    // Can switch off the preconditioning by choosing the number of preconditioner 
    // iterations to be negative
    //if(Kparam.maxiter >= 0) K = new SimpleCG(matPrecon, Kparam, innerProfile);
    if(Kparam.maxiter >= 0) K = new SimpleMR(matPrecon, Kparam, innerProfile);
  }


  PreconCG::~PreconCG() {
    if (K) delete K;
  }

  inline void accumulate_time(double* time_difference, const timeval& start, const timeval& stop)
  {
    long ds = stop.tv_sec - start.tv_sec;
    long dus = stop.tv_usec - start.tv_usec;
    *time_difference += ds + 0.000001*dus;
  }


  void test_time(cudaColorSpinorField* x, cudaColorSpinorField* b, double* time)
  {
    timeval tstart, tstop;
    gettimeofday(&tstart, NULL);

    //    ColorSpinorParam param(*b);
    //    param.create = QUDA_COPY_FIELD_CREATE;
    cudaColorSpinorField r(*b);
    //    cudaColorSpinorField y(*x);

    //  cudaDeviceSynchronize();
    gettimeofday(&tstop, NULL);
    accumulate_time(&(time[2]), tstart, tstop);
    return;
  }

  static void setPreconditionParams(ColorSpinorParam* param,  
      const QudaPrecision& precision,
      const int padding,
      const int* const dims)
  {
    param->create = QUDA_ZERO_FIELD_CREATE;
    param->precision = precision;
    param->nColor = 3;
    param->nDim = 4;
    param->pad = padding;
    param->nSpin = 1;
    param->siteSubset = QUDA_PARITY_SITE_SUBSET;
    param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    param->fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    for(int dir=0; dir<4; ++dir) param->x[dir] = dims[dir];
    param->x[0] /= 2;

    return;
  }


  void PreconCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b)
  {
    profile[QUDA_PROFILE_INIT].Start();

    printfQuda("Calling preconditioned solver\n");  


    timeval precon_start, precon_stop;
    double precon_time = 0.0;
    timeval pcstart, pcstop;
    double pctime = 0.0;
    timeval common_start, common_stop;
    double common_time = 0.0;
    int k=0;
    int rUpdate=0;




    cudaColorSpinorField* minvrPre_ptr;
    cudaColorSpinorField* rPre_ptr;
    cudaColorSpinorField* minvr_ptr;
    cudaColorSpinorField* minvrSloppy_ptr;
    cudaColorSpinorField* p_ptr;


    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    int max_overlap = Kparam.domain_overlap[0];
    for(int dir=1; dir<4; ++dir){
      if(Kparam.domain_overlap[dir] > max_overlap){ 
        max_overlap = Kparam.domain_overlap[dir];
      }
    }


    int X[4]; // smaller sublattice dimensions
    int Y[4]; // extended subdomain dimensions
    X[0] = b.X(0)*2; // assume QUDA_PARITY_SITE_SUBSET
    X[1] = b.X(1);
    X[2] = b.X(2);
    X[3] = b.X(3);
    for(int dir=0; dir<4; ++dir) Y[dir] = X[dir] + 2*Kparam.domain_overlap[dir];

    double simple_time[3] = {0., 0., 0.};

    printfQuda("Y = %d %d %d %d\n", Y[0], Y[1], Y[2], Y[3]);
    fflush(stdout);

    DecompParam dparam;
    initDecompParam(&dparam,X,Y);
    int domain_overlap[4];
    for(int dir=0; dir<4; ++dir) domain_overlap[dir] = invParam.domain_overlap[dir];

    ColorSpinorParam param(b);
    param.nFace  = max_overlap;
    param.create = QUDA_COPY_FIELD_CREATE; 
    cudaColorSpinorField r(b,param);



    Extender extendCuda(r); // function object used to implement overlapping domains
    param.nFace  = b.Nface();
    param.create = QUDA_ZERO_FIELD_CREATE;
    if(K) minvr_ptr = new cudaColorSpinorField(b,param);
    cudaColorSpinorField y(b,param);

    mat(r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    // => r = A*x;
    double r2 = xmyNormCuda(b,r);

    param.precision = invParam.cuda_prec_sloppy;
    cudaColorSpinorField tmpSloppy(x,param);
    param.create = QUDA_COPY_FIELD_CREATE;
    cudaColorSpinorField Ap(x,param);

    cudaColorSpinorField *x_sloppy, *r_sloppy;
    if(x.Precision()  == invParam.cuda_prec_sloppy)
    {
      x_sloppy = &x;
      r_sloppy = &r;
      minvrSloppy_ptr = minvr_ptr;
    }else{
      x_sloppy = new cudaColorSpinorField(x, param);
      param.nFace = max_overlap;
      r_sloppy = new cudaColorSpinorField(r, param);
      param.nFace = b.Nface();
      minvrSloppy_ptr = new cudaColorSpinorField(*minvr_ptr, param);
    }
    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;

    if(&x != &xSloppy){
      copyCuda(y, x); // copy x to y
      zeroCuda(xSloppy);
    }else{
      zeroCuda(x);
    }



    ColorSpinorParam prec_param(x);
    prec_param.create = QUDA_COPY_FIELD_CREATE;
    prec_param.precision = invParam.cuda_prec_precondition;

    if(K){
      setPreconditionParams(&prec_param, 
          invParam.cuda_prec_precondition, 
          rSloppy.Pad(),
          Y);

      rPre_ptr = new cudaColorSpinorField(prec_param);
      // HACK!!!
      int domain_overlap[4];
      for(int dir=0; dir<4; ++dir) domain_overlap[dir] = invParam.domain_overlap[dir];
      gettimeofday(&pcstart, NULL);

      if(max_overlap){
        extendCuda(*rPre_ptr,rSloppy,dparam,domain_overlap);
      }else{
        *rPre_ptr = rSloppy;
      }
      // Create minvrPre_ptr 
      minvrPre_ptr = new cudaColorSpinorField(*rPre_ptr);

      printfQuda("minvrPre_ptr->Precision() = %d\n", minvrPre_ptr->Precision());
      globalReduce = false;
      gettimeofday(&precon_start, NULL); 
      (*K)(*minvrPre_ptr, *rPre_ptr, simple_time);  
      gettimeofday(&precon_stop, NULL);
      accumulate_time(&precon_time, precon_start, precon_stop);
      printfQuda("Time: %lf, %lf\n", simple_time[2], precon_time);
      globalReduce = true;

      if(max_overlap){
        cropCuda(*minvrSloppy_ptr, *minvrPre_ptr, dparam);
      }else{
        *minvrSloppy_ptr = *minvrPre_ptr;
      }

      gettimeofday(&pcstop, NULL);
      accumulate_time(&pctime, pcstart, pcstop);

      p_ptr = new cudaColorSpinorField(*minvrSloppy_ptr);
    }else{
      p_ptr = new cudaColorSpinorField(rSloppy);
    }

    profile[QUDA_PROFILE_INIT].Stop();

    profile[QUDA_PROFILE_PREAMBLE].Start();
    double src_norm = norm2(b);
    double stop = src_norm*invParam.tol*invParam.tol; // stopping condition
    double alpha = 0.0, beta=0.0;
    double pAp;
    double rMinvr  = 0;
    double rMinvr_old = 0.0;
    double r_new_Minvr_old = 0.0;
    double r2_old = 0;
    r2 = norm2(r);
    
    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = invParam.reliable_delta;   
 

    if(K) rMinvr = reDotProductCuda(rSloppy,*minvrSloppy_ptr);

    profile[QUDA_PROFILE_PREAMBLE].Stop();
    profile[QUDA_PROFILE_COMPUTE].Start();

    bool standard_update = true;
    bool reliable_update = false;


    while(r2 > stop && k < invParam.maxiter){

      gettimeofday(&common_start, NULL);
      matSloppy(Ap, *p_ptr, tmpSloppy);
      pAp   = reDotProductCuda(*p_ptr,Ap);


      alpha = (K) ? rMinvr/pAp : r2/pAp;
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy); 
      // r --> r - alpha*A*p
      gettimeofday(&common_stop, NULL);
      accumulate_time(&common_time, common_start, common_stop);
      r2 = real(cg_norm);
      rMinvr_old = rMinvr;

      rNorm = sqrt(r2);

      if(rNorm > maxrx) maxrx = rNorm;
      if(rNorm > maxrr) maxrr = rNorm;
    

      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;


      if( !(updateR || updateX) ){
        r_new_Minvr_old = reDotProductCuda(rSloppy,*minvrSloppy_ptr);

        gettimeofday(&pcstart, NULL);

        if(max_overlap){       
          extendCuda(*rPre_ptr,rSloppy,dparam,domain_overlap);
        }else{
          *rPre_ptr = rSloppy;
        }

        globalReduce = false;
        gettimeofday(&precon_start,NULL);
        (*K)(*minvrPre_ptr, *rPre_ptr, simple_time);
        gettimeofday(&precon_stop,NULL);
        accumulate_time(&precon_time, precon_start, precon_stop);
        globalReduce = true;

        if(max_overlap){
          cropCuda(*minvrSloppy_ptr, *minvrPre_ptr, dparam);
        }else{
          *minvrSloppy_ptr = *minvrPre_ptr;
        }
        gettimeofday(&pcstop, NULL);
        accumulate_time(&pctime, pcstart, pcstop);

        rMinvr = reDotProductCuda(rSloppy,*minvrSloppy_ptr);

        beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; 
        // x = x + alpha*p, p = Minvr + beta*p
        axpyZpbxCuda(alpha, *p_ptr, xSloppy, *minvrSloppy_ptr, beta);
        // update x and p
        // x = x + alpha*p
        // p = Minv_r + beta*p
      } else { // reliable update

        axpyCuda(alpha, *p_ptr, xSloppy); // xSloppy += alpha*p
        copyCuda(x, xSloppy);
        xpyCuda(x, y); // y += x
        // Now compute r 
        mat(r, y, x); // x is just a temporary here
        r2 = xmyNormCuda(b, r);
        copyCuda(rSloppy, r); // copy r to rSloppy
        zeroCuda(xSloppy);

        if(sqrt(r2) > r0Norm && updateX) { 
          // reuse r0Norm for this
          warningQuda("Precon CG: new reliable norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        ++rUpdate;

        if(max_overlap) 
          extendCuda(*rPre_ptr, rSloppy, dparam, domain_overlap); 
        else
          *rPre_ptr = rSloppy;

        globalReduce = false;
        (*K)(*minvrPre_ptr, *rPre_ptr, simple_time);
        globalReduce = true;

        if(max_overlap)
          cropCuda(*minvrSloppy_ptr, *minvrPre_ptr, dparam);
        else
          *minvrSloppy_ptr = *minvrPre_ptr;

        rMinvr = reDotProductCuda(rSloppy,*minvrSloppy_ptr);
        beta = rMinvr/rMinvr_old;        

        xpayCuda(*minvrSloppy_ptr, beta, *p_ptr); // p = minvrSloppy + beta*p
    
      }      

      if(k%100 == 0) printfQuda("r2 = %e\n", r2);

      ++k;
    }
    printfQuda("Number of outer-solver iterations = %d\n",k);


    profile[QUDA_PROFILE_COMPUTE].Stop();

    profile[QUDA_PROFILE_EPILOGUE].Start();

    if(x.Precision() != invParam.cuda_prec_sloppy) copyCuda(x, xSloppy);
    xpyCuda(y, x); // x += y

    // compute the true residual 
    mat(r, x, y);
    double true_res = xmyNormCuda(b, r);
    invParam.true_res = sqrt(true_res / src_norm);
    printfQuda("true_res = %e\n", invParam.true_res);

    profile[QUDA_PROFILE_EPILOGUE].Stop();
    profile[QUDA_PROFILE_FREE].Start();

    // FIX THIS!!
    if(K){ // These are only needed if preconditioning is used
      delete minvrPre_ptr;
      delete rPre_ptr;
      delete minvr_ptr;
      if(x.Precision() != invParam.cuda_prec_sloppy)  delete minvrSloppy_ptr;
    }
    delete p_ptr;


    if(x.Precision() != invParam.cuda_prec_sloppy){
      delete x_sloppy;
      delete r_sloppy;
    }


    profile[QUDA_PROFILE_FREE].Stop();


    printfQuda("PreconCG: Reliable updates = %d\n", rUpdate);

    innerProfile.Print();
#ifdef PRECON_TIME
    printfQuda("SimpleCG matrix time : %lf seconds\n", simple_time[0]);
    printfQuda("SimpleCG other time : %lf seconds\n", simple_time[1]);
    printfQuda("SimpleCG inner total : %lf seconds\n", simple_time[2]);
#endif
    printfQuda("SimpleCG time : %lf seconds\n", precon_time);
    printfQuda("SimpleCG + Copy time : %lf seconds\n", pctime);
    printfQuda("Common time : %lf seconds\n", common_time);
    return;
  }


} // namespace quda
