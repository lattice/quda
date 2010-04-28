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

void invertCgCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, 
		  cudaColorSpinorField &y, QudaInvertParam *invert_param)
{
  cudaColorSpinorField r(b);

  ColorSpinorParam param;
  param.create = QUDA_ZERO_CREATE;
  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField Ap(x, param);
  cudaColorSpinorField tmp(x, param);

  cudaColorSpinorField *x_sloppy, *r_sloppy;
  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
    zeroCuda(*x_sloppy);
  } else {
    x_sloppy = new cudaColorSpinorField(x, param);
    param.create = QUDA_COPY_CREATE;
    r_sloppy = new cudaColorSpinorField(r, param);
  }

  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &rSloppy = *r_sloppy;
  
  cudaColorSpinorField p(rSloppy);
  zeroCuda(y);

  double b2 = normCuda(b);
  double r2 = b2;
  double r2_old;
  double stop = r2*invert_param->tol*invert_param->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  int k=0;
  int rUpdate = 0;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CG: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    diracSloppy.MdagM(Ap, p);
    //MatVec(Ap, cudaGaugeSloppy, cudaCloverSloppy, cudaCloverInvSloppy, p, invert_param, tmp);
    
    pAp = reDotProductCuda(p, Ap);
    alpha = r2 / pAp;        
    r2_old = r2;
    r2 = axpyNormCuda(-alpha, Ap, rSloppy);

    // reliable update conditions
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
    if (!(updateR || updateX)) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
    } else {
      axpyCuda(alpha, p, xSloppy);
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
      xpyCuda(x, y); // swap these around?
      dirac.MdagM(r, y);
      //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, y, invert_param, x);
      r2 = xmyNormCuda(b, r);
      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;

      beta = r2 / r2_old;
      xpayCuda(rSloppy, beta, p);
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE)
      printfQuda("CG: %d iterations, r2 = %e\n", k, r2);
  }

  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);

  invert_param->secs = stopwatchReadSeconds();

  
  if (k==invert_param->maxiter) 
    warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

  float gflops = (blas_quda_flops + dirac.Flops() + diracSloppy.Flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops = gflops;
  invert_param->iter = k;

  blas_quda_flops = 0;

  //#if 0
  // Calculate the true residual
  dirac.MdagM(r, x);
  //MatVec(r, cudaGaugePrecise, cudaCloverPrecise, cudaCloverInvPrecise, x, y);
  double true_res = xmyNormCuda(b, r);
  printfQuda("Converged after %d iterations, r2 = %e, relative true_r2 = %e\n", 
	     k, r2, true_res / b2);
  //#endif

  if (invert_param->cuda_prec_sloppy != x.Precision()) {
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}



/* 
 *   we assume the lowest mass is in offsets[0]
 */
int 
invertCgCudaMultiMass(Dirac & dirac, Dirac& diracSloppy, cudaColorSpinorField** x, cudaColorSpinorField b,
		      QudaInvertParam *invert_param, 
		      double* offsets, int num_offsets, double* residue_sq)
{
  
  if (num_offsets == 0){
    return 0;
  }

  int finished[num_offsets];
  double shifts[num_offsets];
  double zeta_i[num_offsets], zeta_im1[num_offsets], zeta_ip1[num_offsets];
  double beta_i[num_offsets], beta_im1[num_offsets], alpha[num_offsets];
  int i, j;
  

  int j_low = 0;   
  int num_offsets_now = num_offsets;
  for(i=0;i <num_offsets;i++){
    finished[i]= 0;
    shifts[i] = offsets[i] - offsets[0];
    zeta_im1[i] = zeta_i[i] = 1.0;
    beta_im1[i] = -1.0;
    alpha[i] =0.0;
  }
  
  //double msq_x4 = offsets[0];

  cudaColorSpinorField* r = new cudaColorSpinorField(b);
  
  cudaColorSpinorField *x_sloppy[num_offsets], *r_sloppy;
  
  ColorSpinorParam param;
  param.create = QUDA_ZERO_CREATE;
  param.precision = invert_param->cuda_prec_sloppy;
  
  if (invert_param->cuda_prec_sloppy == x[0]->Precision()) {
    for(i=0;i < num_offsets;i++){
      x_sloppy[i] = x[i];
      zeroCuda(*x_sloppy[i]);
    }
    r_sloppy = r;
  } else {
    for(i=0;i < num_offsets;i++){
      x_sloppy[i] = new cudaColorSpinorField(*x[i], param);
    }
    param.create = QUDA_COPY_CREATE;
    r_sloppy = new cudaColorSpinorField(*r, param);
  }
  

  cudaColorSpinorField* p[num_offsets];  
  for(i=0;i < num_offsets;i++){
    p[i]= new cudaColorSpinorField(*r_sloppy);    
  }
  
  param.create = QUDA_ZERO_CREATE;
  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField* Ap = new cudaColorSpinorField(*r_sloppy, param);
  
  double b2 = 0.0;
  b2 = normCuda(b);
    
  double r2 = b2;
  double r2_old;
  double stop = r2*invert_param->tol*invert_param->tol; // stopping condition of solver
    
  double pAp;
    
  int k=0;
    
  printf("%d iterations, r2 = %e\n", k, r2);
  stopwatchStart();
  while (r2 > stop &&  k < invert_param->maxiter) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    //dslashCuda_st(tmp_sloppy, fatlinkSloppy, longlinkSloppy, p[0], 1 - oddBit, 0);
    //dslashAxpyCuda(Ap, fatlinkSloppy, longlinkSloppy, tmp_sloppy, oddBit, 0, p[0], msq_x4);
    diracSloppy.MdagM(*Ap, *p[0]);
    
    pAp = reDotProductCuda(*p[0], *Ap);
    beta_i[0] = r2 / pAp;        

    zeta_ip1[0] = 1.0;
    for(j=1;j<num_offsets_now;j++) {
      zeta_ip1[j] = zeta_i[j] * zeta_im1[j] * beta_im1[j_low];
      double c1 = beta_i[j_low] * alpha[j_low] * (zeta_im1[j]-zeta_i[j]);
      double c2 = zeta_im1[j] * beta_im1[j_low] * (1.0+shifts[j]*beta_i[j_low]);
      /*THISBLOWSUP
	zeta_ip1[j] /= c1 + c2;
	beta_i[j] = beta_i[j_low] * zeta_ip1[j] / zeta_i[j];
      */
      /*TRYTHIS*/
      if( (c1+c2) != 0.0 )
	zeta_ip1[j] /= (c1 + c2); 
      else {
	zeta_ip1[j] = 0.0;
	finished[j] = 1;
      }
      if( zeta_i[j] != 0.0){
	beta_i[j] = beta_i[j_low] * zeta_ip1[j] / zeta_i[j];
      } else  {
	zeta_ip1[j] = 0.0;
	beta_i[j] = 0.0;
	finished[j] = 1;
	printf("SETTING A ZERO, j=%d, num_offsets_now=%d\n",j,num_offsets_now);
	//if(j==num_offsets_now-1)node0_PRINTF("REDUCING OFFSETS\n");
	if(j==num_offsets_now-1)num_offsets_now--;
	// don't work any more on finished solutions
	// this only works if largest offsets are last, otherwise
	// just wastes time multiplying by zero
      }
    }	
	
    r2_old = r2;

    r2 = axpyNormCuda(-beta_i[j_low], *Ap, *r_sloppy);

	    
    alpha[0] = r2 / r2_old;
	
    for(j=1;j<num_offsets_now;j++){
      /*THISBLOWSUP
	alpha[j] = alpha[j_low] * zeta_ip1[j] * beta_i[j] /
	(zeta_i[j] * beta_i[j_low]);
      */
      /*TRYTHIS*/
      if( zeta_i[j] * beta_i[j_low] != 0.0)
	alpha[j] = alpha[j_low] * zeta_ip1[j] * beta_i[j] /
	  (zeta_i[j] * beta_i[j_low]);
      else {
	alpha[j] = 0.0;
	finished[j] = 1;
      }
    }
	
	
    axpyZpbxCuda(beta_i[0], *p[0], *x_sloppy[0], *r_sloppy, alpha[0]);	
    for(j=1;j < num_offsets_now; j++){
      axpyBzpcxCuda(beta_i[j], *p[j], *x_sloppy[j], zeta_ip1[j], *r_sloppy, alpha[j]);
    }
    
    gettimeofday(&t1, NULL);
	
    for(j=0;j<num_offsets_now;j++){
      beta_im1[j] = beta_i[j];
      zeta_im1[j] = zeta_i[j];
      zeta_i[j] = zeta_ip1[j];
    }
    
    
    k++;
#define TDIFF(t1, t0) (t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec))
    printf("%d iterations, r2 = %e, time=%f\n", k, r2,TDIFF(t1, t0));

  }
    
  if (x[0]->Precision() != x_sloppy[0]->Precision()) {
    for(i=0;i < num_offsets; i++){
      copyCuda(*x[i], *x_sloppy[i]);
    }
  }

  *residue_sq = r2;
  

  invert_param->secs = stopwatchReadSeconds();
    
  if (k==invert_param->maxiter) {
    printf("Exceeded maximum iterations %d\n", invert_param->maxiter);
  }
    
  float gflops = (blas_quda_flops + dirac.Flops() + diracSloppy.Flops())*1e-9;
  invert_param->gflops = gflops;
  invert_param->iter = k;
#if 0
  // Calculate the true residual
  dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x[0],  1-oddBit, 0);
  dslashAxpyCuda(Ap, fatlinkPrecise, longlinkPrecise, tmp, oddBit, 0, x[0], msq_x4);
 
  copyCuda(r, source);
  mxpyCuda(Ap, r);
  double true_res = normCuda(r);
    
  PRINTF("Converged after %d iterations, res = %e, b2=%e, true_res = %e\n", 
	 k, true_res, b2, (true_res / b2));
#endif

  
  delete r;
  for(i=0;i < num_offsets; i++){
    delete p[i];
  }
  delete Ap;
  
  if (invert_param->cuda_prec_sloppy != x[0]->Precision()) {
    for(i=0;i < num_offsets;i++){
      delete x_sloppy[i];
    }
    delete r_sloppy;
  }
  
  
  return k;
}

