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
#include <sys/time.h>
#include <face_quda.h>

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
    Solver(invParam, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrec), K(NULL)
  {
    Kparam = newQudaInvertParam();

    for(int dir=0; dir<4; ++dir) invParam.domain_overlap[dir] = 0;
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


    K = new CG(matPrecon, matPrecon, Kparam, profile);
  }


  PreconCG::~PreconCG() {
    if (K) delete K;
  }

  void PreconCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b)
  {
    printfQuda("Calling preconditioned solver\n");
    int k=0;
    int rUpdate;

    cudaColorSpinorField* minvrPre_ptr;
    cudaColorSpinorField* rPre_ptr;
    cudaColorSpinorField* minvr_ptr;
    cudaColorSpinorField* p_ptr;

    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    int max_overlap = invParam.domain_overlap[0];
    for(int dir=1; dir<4; ++dir){
      if(invParam.domain_overlap[dir] > max_overlap){ 
        max_overlap = invParam.domain_overlap[dir];
      }
    }
  
    ColorSpinorParam param(b);
    param.nFace  = max_overlap;
    param.create = QUDA_COPY_FIELD_CREATE; 
    cudaColorSpinorField r(b,param);

    param.nFace  = b.Nface();
    param.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b,param);

    mat(r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    //		cudaColorSpinorField& tmp);
    //
    // => r = A*x;
    double r2 = xmyNormCuda(b,r);
    rUpdate++;

    param.precision = invParam.cuda_prec_sloppy;
    cudaColorSpinorField Ap(x,param);
    cudaColorSpinorField tmp(x,param);


    ColorSpinorParam prec_param(x);
    prec_param.create = QUDA_COPY_FIELD_CREATE;
    prec_param.precision = invParam.cuda_prec_precondition;

    if(K){
      prec_param.create     = QUDA_ZERO_FIELD_CREATE;
      prec_param.precision  = invParam.cuda_prec_precondition;
      prec_param.nColor     = 3;
      prec_param.nDim       = 4;
      prec_param.pad        = 0; // Not sure if this will cause a problem
      prec_param.nSpin      = 1;
      prec_param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      prec_param.siteOrder  = QUDA_EVEN_ODD_SITE_ORDER;
      prec_param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
      prec_param.x[0]       = invParam.domain_overlap[0]/2 + r.X(0); // only works for QUDA_PARITY_SITE_SUBSET
      prec.param.x[1]       = invParam.domain_overlap[1] + r.X(1);   // and even dimensions at the moment.
      prec.param.x[2]       = invParam.domain_overlap[2] + r.X(2);
      prec.param.x[3]       = invParam.domain_overlap[3] + r.X(3);
        
      rPre_ptr = new cudaColorSpinorField(prec_param);
      *rPre_ptr = r;
      minvrPre_ptr = new cudaColorSpinorField(*rPre_ptr);
      minvr_ptr = new cudaColorSpinorField(r);
      K->operator()(*minvrPre_ptr, *rPre_ptr);  
      *minvr_ptr = *minvrPre_ptr;
      p_ptr = new cudaColorSpinorField(*minvr_ptr);
    }else{
      p_ptr = new cudaColorSpinorField(r);
    }


    double src_norm = norm2(b);
    double stop = src_norm*invParam.tol*invParam.tol; // stopping condition
    double alpha = 0.0, beta=0.0;
    double pAp;
    double rMinvr  = reDotProductCuda(r,*minvr_ptr);
    double rMinvr_old = 0.0;
    double r_new_Minvr_old = 0.0;
    double r2_old = 0;
    r2 = normCuda(r);
    printfQuda("r2 = %e\n",r2);


    while(r2 > stop && k < invParam.maxiter){
      mat(Ap, *p_ptr, tmp);
      pAp   = reDotProductCuda(*p_ptr,Ap);
      alpha = (K) ? rMinvr/pAp : r2/pAp;
      printfQuda("alpha = %e\n",alpha);
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, r); 
      // disregard cg_norm
      //axpyCuda(-alpha, Ap, r);
      // r --> r - alpha*A*p

      if(K){
        rMinvr_old = rMinvr;
        r_new_Minvr_old = reDotProductCuda(r,*minvr_ptr);
        *rPre_ptr = r;
        *minvrPre_ptr = *rPre_ptr;
        K->operator()(*minvrPre_ptr, *rPre_ptr);
        *minvr_ptr = *minvrPre_ptr;
        rMinvr = reDotProductCuda(r,*minvr_ptr);


        beta = (rMinvr - r_new_Minvr_old)/rMinvr_old; 
        if(beta < 0){ 
          beta = 0.0; 
        }else{
          if(beta > (rMinvr/rMinvr_old)) beta = rMinvr/rMinvr_old;
        }

        r2 = real(cg_norm);
        // x = x + alpha*p, p = Minvr + beta*p
        axpyZpbxCuda(alpha, *p_ptr, x, *minvr_ptr, beta);
      }else{
        r2_old = r2;
        r2 = real(cg_norm);
        beta = r2/r2_old;
        axpyZpbxCuda(alpha, *p_ptr, x, r, beta);
      }

      printfQuda("r2 = %e\n", r2);
      // update x and p
      // x = x + alpha*p
      // p = Minv_r + beta*p
      ++k;
    }
    printfQuda("Number of iterations = %d\n",k);

    // compute the true residual 
    mat(r, x, y);
    double true_res = xmyNormCuda(b, r);
    invParam.true_res = sqrt(true_res / src_norm);
    printfQuda("true_res = %e\n", invParam.true_res);


    if(K){ // These are only needed if preconditioning is used
      delete minvrPre_ptr;
      delete rPre_ptr;
      delete minvr_ptr;
    }
    delete p_ptr;

    return;
  }


} // namespace quda
