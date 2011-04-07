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

#include <sys/time.h>

double timeInterval(struct timeval start, struct timeval end) {
  long ds = end.tv_sec - start.tv_sec;
  long dus = end.tv_usec - start.tv_usec;
  return ds + 0.000001*dus;
}

// set the required parameters for the inner solver
void fillInnerInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer) {
  inner.tol = outer.tol_sloppy;
  inner.maxiter = outer.maxiter_sloppy;
  inner.reliable_delta = 1e-20; // no reliable updates within the inner solver
  
  inner.cuda_prec = outer.cuda_prec_sloppy; // only use sloppy precision on inner solver
  inner.cuda_prec_sloppy = outer.cuda_prec_sloppy;
  
  inner.verbosity = outer.verbosity_sloppy;
  
  inner.iter = 0;
  inner.gflops = 0;
  inner.secs = 0;

  inner.inv_type_sloppy = QUDA_GCR_INVERTER; // used to tell the inner solver it is an inner solver

  if (outer.inv_type == QUDA_GCR_INVERTER && outer.cuda_prec_sloppy != outer.prec_precondition) 
    inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

}

void backSubs(Complex *alpha, Complex **beta, double *gamma, Complex *delta, int n) {
  for (int k=n-1; k>=0;k--) {
    delta[k] = alpha[k];
    for (int j=k+1;j<n; j++) {
      delta[k] -= beta[k][j]*delta[j];
    }
    delta[k] /= gamma[k];
  }
}

void invertGCRCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &pre, 
		   cudaColorSpinorField &x, cudaColorSpinorField &b, QudaInvertParam *invert_param)
{
  typedef std::complex<double> Complex;

  int Nkrylov = invert_param->gcrNkrylov; // size of Krylov space

  ColorSpinorParam param(x);
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField r(x, param); 

  cudaColorSpinorField y(x, param); // high precision accumulator

  // create sloppy fields used for orthogonalization
  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField *p[Nkrylov], *Ap[Nkrylov];
  for (int i=0; i<Nkrylov; i++) {
    p[i] = new cudaColorSpinorField(x, param);
    Ap[i] = new cudaColorSpinorField(x, param);
  }

  cudaColorSpinorField tmp(x, param); //temporary for sloppy mat-vec

  cudaColorSpinorField *x_sloppy, *r_sloppy;
  if (invert_param->cuda_prec_sloppy != invert_param->cuda_prec) {
    param.precision = invert_param->cuda_prec_sloppy;
    x_sloppy = new cudaColorSpinorField(x, param);
    r_sloppy = new cudaColorSpinorField(x, param);
  } else {
    x_sloppy = &x;
    r_sloppy = &r;
  }

  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &rSloppy = *r_sloppy;

  // these low precision fields are used by the inner solver
  bool precMatch = true;
  cudaColorSpinorField *r_pre, *p_pre;
  if (invert_param->prec_precondition != invert_param->cuda_prec_sloppy) {
    param.precision = invert_param->prec_precondition;
    p_pre = new cudaColorSpinorField(x, param);
    r_pre = new cudaColorSpinorField(x, param);
    precMatch = false;
  } else {
    p_pre = NULL;
    r_pre = r_sloppy;
  }
  cudaColorSpinorField &rPre = *r_pre;

  QudaInvertParam invert_param_inner = newQudaInvertParam();
  fillInnerInvertParam(invert_param_inner, *invert_param);

  Complex *alpha = new Complex[Nkrylov];
  Complex **beta = new Complex*[Nkrylov];
  for (int i=0; i<Nkrylov; i++) beta[i] = new Complex[Nkrylov];
  double *gamma = new double[Nkrylov];
  Complex *delta = new Complex[Nkrylov];

  double b2 = normCuda(b);

  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver

  int k = 0;

  // calculate initial residual
  mat(r, x);
  double r2 = xmyNormCuda(b, r);  
  copyCuda(rSloppy, r);

  blas_quda_flops = 0;

  stopwatchStart();

  int total_iter = 0;
  int restart = 0;
  double r2_old = r2;

  if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("GCR: %d total iterations, %d Krylov iterations, r2 = %e\n", total_iter+k, k, r2);

  struct timeval orth0, orth1, pre0, pre1, mat0, mat1, rst0, rst1;
  double orthT = 0, matT = 0, preT = 0, resT = 0;

  while (r2 > stop && total_iter < invert_param->maxiter) {
    
    gettimeofday(&pre0, NULL);

    if (invert_param->inv_type_sloppy != QUDA_INVALID_INVERTER) {
      if (invert_param->tol/(sqrt(r2/b2)) > invert_param->tol_sloppy) // relax stoppng condition
	invert_param_inner.tol = invert_param->tol/sqrt(r2/b2);

      cudaColorSpinorField &pPre = (precMatch ? *p[k] : *p_pre);

      copyCuda(rPre, rSloppy);
      if (invert_param->inv_type_sloppy == QUDA_CG_INVERTER) // inner CG preconditioner
	invertCgCuda(pre, pre, pPre, rPre, &invert_param_inner);
      else if (invert_param->inv_type_sloppy == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
	invertBiCGstabCuda(pre, pre, pre, pPre, rPre, &invert_param_inner);
      else if (invert_param->inv_type_sloppy == QUDA_MR_INVERTER) // inner MR preconditioner
	invertMRCuda(pre, pPre, rPre, &invert_param_inner);
      else
	errorQuda("Unknown inner solver %d", invert_param->inv_type_sloppy);

      copyCuda(*p[k], pPre);
    } else { // no preconditioner
      *p[k] = rSloppy;
    } 

    gettimeofday(&pre1, NULL);


    gettimeofday(&mat0, NULL);
    matSloppy(*Ap[k], *p[k], tmp);
    gettimeofday(&mat1, NULL);

    gettimeofday(&orth0, NULL);
    for (int i=0; i<k; i++) { // 5 (k-1) memory transactions here
      beta[i][k] = cDotProductCuda(*Ap[i], *Ap[k]);
      caxpyCuda(-beta[i][k], *Ap[i], *Ap[k]);
    }
    gettimeofday(&orth1, NULL);
    
    double3 Apr = cDotProductNormACuda(*Ap[k], rSloppy);
    gamma[k] = sqrt(Apr.z); // gamma[k] = Ap[k]
    if (gamma[k] == 0.0) errorQuda("GCR breakdown\n");
    alpha[k] = Complex(Apr.x, Apr.y) / gamma[k]; // alpha = (1/|Ap|) * (Ap, r)

    // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
    r2 = cabxpyAxNormCuda(1.0/gamma[k], -alpha[k], *Ap[k], rSloppy); 

    if (invert_param->verbosity >= QUDA_DEBUG_VERBOSE) 
      printfQuda("GCR: alpha = (%e,%e), x2 = %e\n", real(alpha[k]), imag(alpha[k]), norm2(x));

    k++;
    total_iter++;

    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("GCR: %d total iterations, %d Krylov iterations, r2 = %e\n", total_iter, k, r2);

    gettimeofday(&rst0, NULL);
    // update solution and residual since max Nkrylov reached, converged or reliable update required
    if (k==Nkrylov || r2 < stop || r2/r2_old < invert_param->reliable_delta) { 
      // Update the solution vector
      backSubs(alpha, beta, gamma, delta, k);
      for (int i=0; i<k; i++) caxpyCuda(delta[i], *p[i], xSloppy);

      // recalculate residual in high precision
      copyCuda(x, xSloppy);
      xpyCuda(x, y);

      double r2Sloppy = r2;

      k = 0;
      mat(r, y);
      double r2 = xmyNormCuda(b, r);  

      if (r2 > stop) {
	restart++; // restarting if residual is still too great

	if (invert_param->verbosity >= QUDA_VERBOSE) 
	  printfQuda("\nGCR: restart %d, iterated r2 = %e, true r2 = %e\n", restart, r2Sloppy, r2);
      }

      copyCuda(rSloppy, r);
      zeroCuda(xSloppy);

      if (r2_old < r2) {
	if (invert_param->verbosity >= QUDA_VERBOSE) 
	  printfQuda("GCR: precision limit reached, r2_old = %e < r2 = %e\n", r2_old, r2);
	break;
      }

      r2_old = r2;
    }
    gettimeofday(&rst1, NULL);

    orthT += timeInterval(orth0, orth1);
    matT += timeInterval(mat0, mat1);
    preT += timeInterval(pre0, pre1);
    resT += timeInterval(rst0, rst1);

  }

  copyCuda(x, y);

  if (k>=invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("GCR: number of restarts = %d\n", restart);
  
  invert_param->secs += stopwatchReadSeconds();
  
  double gflops = (blas_quda_flops + mat.flops() + matSloppy.flops() + pre.flops())*1e-9;
  reduceDouble(gflops);

  printfQuda("%f gflops %e Preconditoner = %e, Mat-Vec = %e, orthogonolization %e restart %e\n", 
  	     gflops / stopwatchReadSeconds(), invert_param->secs, preT, matT, orthT, resT);
  invert_param->gflops += gflops;
  invert_param->iter += total_iter;
  
  if (invert_param->verbosity >= QUDA_SUMMARIZE) {
    // Calculate the true residual
    mat(r, x);
    double true_res = xmyNormCuda(b, r);
    
    printfQuda("GCR: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
	       total_iter, sqrt(r2/b2), sqrt(true_res / b2));    
  }

  if (invert_param->cuda_prec_sloppy != invert_param->cuda_prec) {
    delete x_sloppy;
    delete r_sloppy;
  }

  if (invert_param->prec_precondition != invert_param->cuda_prec_sloppy) {
    delete p_pre;
    delete r_pre;
  }

  for (int i=0; i<Nkrylov; i++) {
    delete p[i];
    delete Ap[i];
  }

  delete alpha;
  for (int i=0; i<Nkrylov; i++) delete []beta[i];
  delete []beta;
  delete gamma;
  delete delta;

  return;
}
