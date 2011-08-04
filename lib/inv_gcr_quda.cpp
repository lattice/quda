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

struct timeval orth0, orth1, pre0, pre1, mat0, mat1, rst0, rst1;

double timeInterval(struct timeval start, struct timeval end) {
  long ds = end.tv_sec - start.tv_sec;
  long dus = end.tv_usec - start.tv_usec;
  return ds + 0.000001*dus;
}

// set the required parameters for the inner solver
void fillInnerInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer) {
  inner.tol = outer.tol_precondition;
  inner.maxiter = outer.maxiter_precondition;
  inner.reliable_delta = 1e-20; // no reliable updates within the inner solver
  
  inner.cuda_prec = outer.prec_precondition; // preconditioners are uni-precision solvers
  inner.cuda_prec_sloppy = outer.prec_precondition;
  
  inner.verbosity = outer.verbosity_precondition;
  
  inner.iter = 0;
  inner.gflops = 0;
  inner.secs = 0;

  inner.inv_type_precondition = QUDA_GCR_INVERTER; // used to tell the inner solver it is an inner solver

  if (outer.inv_type == QUDA_GCR_INVERTER && outer.cuda_prec_sloppy != outer.prec_precondition) 
    inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

}

void orthoDir(Complex **beta, cudaColorSpinorField *Ap[], int k) {
  gettimeofday(&orth0, NULL);

  int type = 1;

  switch (type) {
  case 0: // no kernel fusion
    for (int i=0; i<k; i++) { // 5 (k-1) memory transactions here
      beta[i][k] = cDotProductCuda(*Ap[i], *Ap[k]);
      caxpyCuda(-beta[i][k], *Ap[i], *Ap[k]);
    }
    break;
  case 1: // basic kernel fusion
    if (k==0) break;
    beta[0][k] = cDotProductCuda(*Ap[0], *Ap[k]);
    for (int i=0; i<k-1; i++) { // 4 (k-1) memory transactions here
      beta[i+1][k] = caxpyDotzyCuda(-beta[i][k], *Ap[i], *Ap[k], *Ap[i+1]);
    }
    caxpyCuda(-beta[k-1][k], *Ap[k-1], *Ap[k]);
    break;
  case 2: // 
    for (int i=0; i<k-2; i+=3) { // 5 (k-1) memory transactions here
      for (int j=i; j<i+3; j++) beta[j][k] = cDotProductCuda(*Ap[j], *Ap[k]);
      caxpbypczpwCuda(-beta[i][k], *Ap[i], -beta[i+1][k], *Ap[i+1], -beta[i+2][k], *Ap[i+2], *Ap[k]);
    }
    
    if (k%3 != 0) { // need to update the remainder
      if ((k - 3*(k/3)) % 2 == 0) {
	beta[k-2][k] = cDotProductCuda(*Ap[k-2], *Ap[k]);
	beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
	caxpbypzCuda(beta[k-2][k], *Ap[k-2], beta[k-1][k], *Ap[k-1], *Ap[k]);
      } else {
	beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
	caxpyCuda(beta[k-1][k], *Ap[k-1], *Ap[k]);
      }
    }

    break;
  case 3:
    for (int i=0; i<k-1; i+=2) {
      for (int j=i; j<i+2; j++) beta[j][k] = cDotProductCuda(*Ap[j], *Ap[k]);
      caxpbypzCuda(-beta[i][k], *Ap[i], -beta[i+1][k], *Ap[i+1], *Ap[k]);
    }
    
    if (k%2 != 0) { // need to update the remainder
      beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
      caxpyCuda(beta[k-1][k], *Ap[k-1], *Ap[k]);
    }
    break;
  default:
    errorQuda("Orthogonalization type not defined");
  }

  gettimeofday(&orth1, NULL);
}   

void backSubs(const Complex *alpha, Complex** const beta, const double *gamma, Complex *delta, int n) {
  for (int k=n-1; k>=0;k--) {
    delta[k] = alpha[k];
    for (int j=k+1;j<n; j++) {
      delta[k] -= beta[k][j]*delta[j];
    }
    delta[k] /= gamma[k];
  }
}

void updateSolution(cudaColorSpinorField &x, const Complex *alpha, Complex** const beta, 
		    double *gamma, int k, cudaColorSpinorField *p[]) {

  Complex *delta = new Complex[k];

  // Update the solution vector
  backSubs(alpha, beta, gamma, delta, k);
  
  //for (int i=0; i<k; i++) caxpyCuda(delta[i], *p[i], x);
  
  for (int i=0; i<k-2; i+=3) 
    caxpbypczpwCuda(delta[i], *p[i], delta[i+1], *p[i+1], delta[i+2], *p[i+2], x); 
  
  if (k%3 != 0) { // need to update the remainder
    if ((k - 3*(k/3)) % 2 == 0) caxpbypzCuda(delta[k-2], *p[k-2], delta[k-1], *p[k-1], x);
    else caxpyCuda(delta[k-1], *p[k-1], x);
  }

  delete []delta;
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

  double orthT = 0, matT = 0, preT = 0, resT = 0;

  while (r2 > stop && total_iter < invert_param->maxiter) {
    
    gettimeofday(&pre0, NULL);

    if (invert_param->inv_type_precondition != QUDA_INVALID_INVERTER) {
      //if (invert_param->tol/(sqrt(r2/b2)) > invert_param->tol_precondition) // don'trelax stoppng condition
	//invert_param_inner.tol = invert_param->tol/sqrt(r2/b2);

      cudaColorSpinorField &pPre = (precMatch ? *p[k] : *p_pre);

      copyCuda(rPre, rSloppy);
      if (invert_param->inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
	invertCgCuda(pre, pre, pPre, rPre, &invert_param_inner);
      else if (invert_param->inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
	invertBiCGstabCuda(pre, pre, pre, pPre, rPre, &invert_param_inner);
      else if (invert_param->inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
	invertMRCuda(pre, pPre, rPre, &invert_param_inner);
      else
	errorQuda("Unknown inner solver %d", invert_param->inv_type_precondition);

      // relaxation p = omega*p + (1-omega)*r
      if (invert_param->omega!=1.0) axpbyCuda((1.0-invert_param->omega), rPre, invert_param->omega, pPre);

      copyCuda(*p[k], pPre);
    } else { // no preconditioner
      *p[k] = rSloppy;
    } 


    gettimeofday(&pre1, NULL);

    gettimeofday(&mat0, NULL);
    matSloppy(*Ap[k], *p[k], tmp);
    gettimeofday(&mat1, NULL);

    orthoDir(beta, Ap, k);

    double3 Apr = cDotProductNormACuda(*Ap[k], rSloppy);

    gamma[k] = sqrt(Apr.z); // gamma[k] = Ap[k]
    if (gamma[k] == 0.0) errorQuda("GCR breakdown\n");
    alpha[k] = Complex(Apr.x, Apr.y) / gamma[k]; // alpha = (1/|Ap|) * (Ap, r)

    // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
    r2 = cabxpyAxNormCuda(1.0/gamma[k], -alpha[k], *Ap[k], rSloppy); 

    if (invert_param->verbosity >= QUDA_DEBUG_VERBOSE) {
      double x2 = norm2(x);
      double p2 = norm2(*p[k]);
      double Ap2 = norm2(*Ap[k]);
      printfQuda("GCR: alpha = (%e,%e), norm2(x) = %e, norm2(p) = %e, norm2(Ap) = %e\n", 
		 real(alpha[k]), imag(alpha[k]), x2, p2, Ap2);
    }

    k++;
    total_iter++;

    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("GCR: %d total iterations, %d Krylov iterations, r2 = %e\n", total_iter, k, r2);

    gettimeofday(&rst0, NULL);
    // update solution and residual since max Nkrylov reached, converged or reliable update required
    if (k==Nkrylov || r2 < stop || r2/r2_old < invert_param->reliable_delta) { 

      // update the solution vector
      updateSolution(xSloppy, alpha, beta, gamma, k, p);

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

  if (k>=invert_param->maxiter && invert_param->verbosity >= QUDA_SUMMARIZE) 
    warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

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

  return;
}
