#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include<face_quda.h>

#include <color_spinor_field.h>

// set the required parameters for the inner solver
void fillInnerInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer);

double resNorm(const DiracMatrix &mat, cudaColorSpinorField &b, cudaColorSpinorField &x) {  
  cudaColorSpinorField r(b);
  mat(r, x);
  return xmyNormCuda(b, r);
}


BiCGstab::BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, QudaInvertParam &invParam) :
  Solver(invParam), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), init(false) {

}

BiCGstab::~BiCGstab() {
  if(init) {
    if (wp && wp != pp) delete wp;
    if (zp && zp != pp) delete zp;
    delete yp;
    delete rp;
    delete pp;
    delete vp;
    delete tmpp;
    delete tp;
  }
}

void BiCGstab::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
{
  if (invParam.cuda_prec_sloppy != invParam.prec_precondition)
    errorQuda("BiCGstab does not yet support different sloppy and preconditioner precisions");

  if (!init) {
    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    yp = new cudaColorSpinorField(x, csParam);
    rp = new cudaColorSpinorField(x, csParam); 
    csParam.precision = invParam.cuda_prec_sloppy;
    pp = new cudaColorSpinorField(x, csParam);
    vp = new cudaColorSpinorField(x, csParam);
    tmpp = new cudaColorSpinorField(x, csParam);
    tp = new cudaColorSpinorField(x, csParam);

    // MR preconditioner - we need extra vectors
    if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
      wp = new cudaColorSpinorField(x, csParam);
      zp = new cudaColorSpinorField(x, csParam);
    } else { // dummy assignments
      wp = pp;
      zp = pp;
    }

    init = true;
  }

  cudaColorSpinorField &y = *yp;
  cudaColorSpinorField &r = *rp; 
  cudaColorSpinorField &p = *pp;
  cudaColorSpinorField &v = *vp;
  cudaColorSpinorField &tmp = *tmpp;
  cudaColorSpinorField &t = *tp;

  cudaColorSpinorField &w = *wp;
  cudaColorSpinorField &z = *zp;

  cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;

  if (invParam.cuda_prec_sloppy == x.Precision()) {
    x_sloppy = &x;
    r_sloppy = &r;
    r_0 = &b;
    zeroCuda(*x_sloppy);
    copyCuda(*r_sloppy, b);
  } else {
    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.precision = invParam.cuda_prec_sloppy;
    x_sloppy = new cudaColorSpinorField(x, csParam);
    csParam.create = QUDA_COPY_FIELD_CREATE;
    r_sloppy = new cudaColorSpinorField(b, csParam);
    r_0 = new cudaColorSpinorField(b, csParam);
  }

  // Syntatic sugar
  cudaColorSpinorField &rSloppy = *r_sloppy;
  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &r0 = *r_0;

  QudaInvertParam invert_param_inner = newQudaInvertParam();
  fillInnerInvertParam(invert_param_inner, invParam);

  double b2 = normCuda(b);

  double r2 = b2;
  double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver
  double delta = invParam.reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
  quda::Complex rho(1.0, 0.0);
  quda::Complex rho0 = rho;
  quda::Complex alpha(1.0, 0.0);
  quda::Complex omega(1.0, 0.0);
  quda::Complex beta;

  double3 rho_r2;
  double3 omega_t2;
  
  double rNorm = sqrt(r2);
  //double r0Norm = rNorm;
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invParam.verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);

  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
    quda::blas_flops = 0;    
    stopwatchStart();
  }

  while (r2 > stop && k<invParam.maxiter) {
    
    if (k==0) {
      rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
      copyCuda(p, rSloppy);
    } else {
      if (abs(rho*alpha) == 0.0) beta = 0.0;
      else beta = (rho/rho0) * (alpha/omega);

      cxpaypbzCuda(rSloppy, -beta*omega, v, beta, p);
    }
    
    if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
      errorQuda("Temporary disabled");
      //invertMRCuda(*matPrecon, w, p, &invert_param_inner);
      matSloppy(v, w, tmp);
    } else {
      matSloppy(v, p, tmp);
    }

    if (abs(rho) == 0.0) alpha = 0.0;
    else alpha = rho / cDotProductCuda(r0, v);

    // r -= alpha*v
    caxpyCuda(-alpha, v, rSloppy);

    if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
      errorQuda("Temporary disabled");
      //invertMRCuda(*matPrecon, z, rSloppy, &invert_param_inner);
      matSloppy(t, z, tmp);
    } else {
      matSloppy(t, rSloppy, tmp);
    }
    
    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(t, rSloppy);
    omega = quda::Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);

    if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
      //x += alpha*w + omega*z, r -= omega*t, r2 = (r,r), rho = (r0, r)
      caxpyCuda(alpha, w, xSloppy);
      caxpyCuda(omega, z, xSloppy);
      caxpyCuda(-omega, t, rSloppy);
      rho_r2 = cDotProductNormBCuda(r0, rSloppy);
    } else {
      //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
      rho_r2 = caxpbypzYmbwcDotProductUYNormYCuda(alpha, p, omega, rSloppy, xSloppy, t, r0);
    }

    rho0 = rho;
    rho = quda::Complex(rho_r2.x, rho_r2.y);
    r2 = rho_r2.z;

    if (invParam.verbosity == QUDA_DEBUG_VERBOSE)
      printfQuda("DEBUG: %d iterated residual norm = %e, true residual norm = %e\n",
		 k, norm2(rSloppy), resNorm(matSloppy, b, xSloppy));

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
      xpyCuda(x, y); // swap these around?
      mat(r, y, x);
      r2 = xmyNormCuda(b, r);

      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      //r0Norm = rNorm;      
      rUpdate++;
    }
    
    k++;
    if (invParam.verbosity >= QUDA_VERBOSE) 
      printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);
    
  if (k==invParam.maxiter) warningQuda("Exceeded maximum iterations %d", invParam.maxiter);

  if (invParam.verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
  if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
    invParam.secs += stopwatchReadSeconds();

    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);

    //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
    invParam.gflops += gflops;
    invParam.iter += k;
    
    if (invParam.verbosity >= QUDA_SUMMARIZE) {
      // Calculate the true residual
      mat(r, x);
      double true_res = xmyNormCuda(b, r);
      
      printfQuda("BiCGstab: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		 k, sqrt(r2/b2), sqrt(true_res / b2));    
    }
  }

  if (invParam.cuda_prec_sloppy != x.Precision()) {
    delete r_0;
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}
