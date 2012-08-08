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

void orthoDir(quda::Complex **beta, cudaColorSpinorField *Ap[], int k) {
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

void backSubs(const quda::Complex *alpha, quda::Complex** const beta, const double *gamma, quda::Complex *delta, int n) {
  for (int k=n-1; k>=0;k--) {
    delta[k] = alpha[k];
    for (int j=k+1;j<n; j++) {
      delta[k] -= beta[k][j]*delta[j];
    }
    delta[k] /= gamma[k];
  }
}

void updateSolution(cudaColorSpinorField &x, const quda::Complex *alpha, quda::Complex** const beta, 
		    double *gamma, int k, cudaColorSpinorField *p[]) {

  quda::Complex *delta = new quda::Complex[k];

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

GCR::GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, QudaInvertParam &invParam) :
  Solver(invParam), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0)
{

  Kparam = newQudaInvertParam();
  fillInnerInvertParam(Kparam, invParam);

  if (invParam.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
    K = new CG(matPrecon, matPrecon, Kparam);
  else if (invParam.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
    K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam);
  else if (invParam.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
    K = new MR(matPrecon, Kparam);
  else if (invParam.inv_type_precondition != QUDA_INVALID_INVERTER) // unknown preconditioner
    errorQuda("Unknown inner solver %d", invParam.inv_type_precondition);

}

GCR::~GCR() {
  if (K) delete K;
}

void GCR::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
{
  int Nkrylov = invParam.gcrNkrylov; // size of Krylov space

  ColorSpinorParam param(x);
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField r(x, param); 

  cudaColorSpinorField y(x, param); // high precision accumulator

  // create sloppy fields used for orthogonalization
  param.precision = invParam.cuda_prec_sloppy;
  cudaColorSpinorField **p = new cudaColorSpinorField*[Nkrylov];
  cudaColorSpinorField **Ap = new cudaColorSpinorField*[Nkrylov];
  for (int i=0; i<Nkrylov; i++) {
    p[i] = new cudaColorSpinorField(x, param);
    Ap[i] = new cudaColorSpinorField(x, param);
  }

  cudaColorSpinorField tmp(x, param); //temporary for sloppy mat-vec

  cudaColorSpinorField *x_sloppy, *r_sloppy;
  if (invParam.cuda_prec_sloppy != invParam.cuda_prec) {
    param.precision = invParam.cuda_prec_sloppy;
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
  if (invParam.prec_precondition != invParam.cuda_prec_sloppy || invParam.precondition_cycle > 1) {
    param.precision = invParam.prec_precondition;
    p_pre = new cudaColorSpinorField(x, param);
    r_pre = new cudaColorSpinorField(x, param);
    precMatch = false;
  } else {
    p_pre = NULL;
    r_pre = r_sloppy;
  }
  cudaColorSpinorField &rPre = *r_pre;

  quda::Complex *alpha = new quda::Complex[Nkrylov];
  quda::Complex **beta = new quda::Complex*[Nkrylov];
  for (int i=0; i<Nkrylov; i++) beta[i] = new quda::Complex[Nkrylov];
  double *gamma = new double[Nkrylov];

  double b2 = normCuda(b);

  double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver

  int k = 0;

  // compute parity of the node
  int parity = 0;
  for (int i=0; i<4; i++) parity += commCoords(i);
  parity = parity % 2;

  cudaColorSpinorField rM(rSloppy);
  cudaColorSpinorField xM(rSloppy);

  quda::blas_flops = 0;

  stopwatchStart();

  // calculate initial residual
  mat(r, x);
  double r2 = xmyNormCuda(b, r);  
  copyCuda(rSloppy, r);

  int total_iter = 0;
  int restart = 0;
  double r2_old = r2;

  if (invParam.verbosity >= QUDA_VERBOSE) 
    printfQuda("GCR: %d total iterations, %d Krylov iterations, r2 = %e\n", total_iter+k, k, r2);

  double orthT = 0, matT = 0, preT = 0, resT = 0;

  while (r2 > stop && total_iter < invParam.maxiter) {
    
    gettimeofday(&pre0, NULL);

    for (int m=0; m<invParam.precondition_cycle; m++) {
      if (invParam.inv_type_precondition != QUDA_INVALID_INVERTER) {
	cudaColorSpinorField &pPre = (precMatch ? *p[k] : *p_pre);
	
	if (m==0) { // residual is just source
	  copyCuda(rPre, rSloppy);
	} else { // compute residual
	  copyCuda(rM,rSloppy);
	  axpyCuda(-1.0, *Ap[k], rM);
	  copyCuda(rPre, rM);
	}
	
	if ((parity+m)%2 == 0 || invParam.schwarz_type == QUDA_ADDITIVE_SCHWARZ) (*K)(pPre, rPre);
	else copyCuda(pPre, rPre);
	
	// relaxation p = omega*p + (1-omega)*r
	//if (invParam.omega!=1.0) axpbyCuda((1.0-invParam.omega), rPre, invParam.omega, pPre);
	
	if (m==0) { copyCuda(*p[k], pPre); }
	else { copyCuda(tmp, pPre); xpyCuda(tmp, *p[k]); }
      } else { // no preconditioner
	*p[k] = rSloppy;
      } 
      
      
      gettimeofday(&pre1, NULL);
      preT += timeInterval(pre0, pre1);
      
      gettimeofday(&mat0, NULL);
      matSloppy(*Ap[k], *p[k], tmp);
      gettimeofday(&mat1, NULL);
      matT += timeInterval(mat0, mat1);
    }
    //    exit(0);

    orthoDir(beta, Ap, k);

    double3 Apr = cDotProductNormACuda(*Ap[k], rSloppy);

    gamma[k] = sqrt(Apr.z); // gamma[k] = Ap[k]
    if (gamma[k] == 0.0) errorQuda("GCR breakdown\n");
    alpha[k] = quda::Complex(Apr.x, Apr.y) / gamma[k]; // alpha = (1/|Ap|) * (Ap, r)

    // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
    r2 = cabxpyAxNormCuda(1.0/gamma[k], -alpha[k], *Ap[k], rSloppy); 

    if (invParam.verbosity >= QUDA_DEBUG_VERBOSE) {
      double x2 = norm2(x);
      double p2 = norm2(*p[k]);
      double Ap2 = norm2(*Ap[k]);
      printfQuda("GCR: alpha = (%e,%e), norm2(x) = %e, norm2(p) = %e, norm2(Ap) = %e\n", 
		 real(alpha[k]), imag(alpha[k]), x2, p2, Ap2);
    }

    k++;
    total_iter++;

    if (invParam.verbosity >= QUDA_VERBOSE) 
      printfQuda("GCR: %d total iterations, %d Krylov iterations, r2 = %e\n", total_iter, k, r2);

    gettimeofday(&rst0, NULL);
    // update solution and residual since max Nkrylov reached, converged or reliable update required
    if (k==Nkrylov || r2 < stop || r2/r2_old < invParam.reliable_delta) { 

      // update the solution vector
      updateSolution(xSloppy, alpha, beta, gamma, k, p);

      // recalculate residual in high precision
      copyCuda(x, xSloppy);
      xpyCuda(x, y);

      double r2Sloppy = r2;

      k = 0;
      mat(r, y);
      r2 = xmyNormCuda(b, r);  

      if (r2 > stop) {
	restart++; // restarting if residual is still too great

	if (invParam.verbosity >= QUDA_VERBOSE) 
	  printfQuda("\nGCR: restart %d, iterated r2 = %e, true r2 = %e\n", restart, r2Sloppy, r2);
      }

      copyCuda(rSloppy, r);
      zeroCuda(xSloppy);

      r2_old = r2;
    }
    gettimeofday(&rst1, NULL);

    orthT += timeInterval(orth0, orth1);
    resT += timeInterval(rst0, rst1);

  }

  if (total_iter > 0) copyCuda(x, y);

  if (k>=invParam.maxiter && invParam.verbosity >= QUDA_SUMMARIZE) 
    warningQuda("Exceeded maximum iterations %d", invParam.maxiter);

  if (invParam.verbosity >= QUDA_VERBOSE) printfQuda("GCR: number of restarts = %d\n", restart);
  
  invParam.secs += stopwatchReadSeconds();
  
  double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
  reduceDouble(gflops);

  printfQuda("%f gflops %e Preconditoner = %e, Mat-Vec = %e, orthogonolization %e restart %e\n", 
  	     gflops / stopwatchReadSeconds(), invParam.secs, preT, matT, orthT, resT);
  invParam.gflops += gflops;
  invParam.iter += total_iter;
  
  if (invParam.verbosity >= QUDA_SUMMARIZE) {
    // Calculate the true residual
    mat(r, x);
    double true_res = xmyNormCuda(b, r);
    
    printfQuda("GCR: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
	       total_iter, sqrt(r2/b2), sqrt(true_res / b2));    
  }

  if (invParam.cuda_prec_sloppy != invParam.cuda_prec) {
    delete x_sloppy;
    delete r_sloppy;
  }

  if (invParam.prec_precondition != invParam.cuda_prec_sloppy) {
    delete p_pre;
    delete r_pre;
  }

  for (int i=0; i<Nkrylov; i++) {
    delete p[i];
    delete Ap[i];
  }
  delete[] p;
  delete[] Ap;

  delete alpha;
  for (int i=0; i<Nkrylov; i++) delete []beta[i];
  delete []beta;
  delete gamma;

  return;
}
