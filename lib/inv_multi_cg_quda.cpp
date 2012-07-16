#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <face_quda.h>

//#include <sys/time.h>



/*!
 * Generic Multi Shift Solver 
 * 
 * For staggered, the mass is folded into the dirac operator
 * Otherwise the matrix mass is 'unmodified'. 
 *
 * THe lowest offset is in offsets[0]
 *
 */

MultiShiftCG::MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam) 
  : MultiShiftSolver(invParam), mat(mat), matSloppy(matSloppy) {

}

MultiShiftCG::~MultiShiftCG() {

}

void MultiShiftCG::operator()(cudaColorSpinorField **x, cudaColorSpinorField &b)
{
 
  int num_offset = invParam.num_offset;
  double *offset = invParam.offset;
  double *residue_sq = invParam.tol_offset;
 
  if (num_offset == 0) return;

  int *finished = new int [num_offset];
  double *zeta_i = new double[num_offset];
  double *zeta_im1 = new double[num_offset];
  double *zeta_ip1 = new double[num_offset];
  double *beta_i = new double[num_offset];
  double *beta_im1 = new double[num_offset];
  double *alpha = new double[num_offset];
  int i, j;
  
  int j_low = 0;   
  int num_offset_now = num_offset;
  for (i=0; i<num_offset; i++) {
    finished[i] = 0;
    zeta_im1[i] = zeta_i[i] = 1.0;
    beta_im1[i] = -1.0;
    alpha[i] = 0.0;
  }
  
  //double msq_x4 = offset[0];

  cudaColorSpinorField *r = new cudaColorSpinorField(b);
  
  cudaColorSpinorField **x_sloppy = new cudaColorSpinorField*[num_offset], *r_sloppy;
  
  ColorSpinorParam param;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.precision = invParam.cuda_prec_sloppy;
  
  if (invParam.cuda_prec_sloppy == x[0]->Precision()) {
    for (i=0; i<num_offset; i++){
      x_sloppy[i] = x[i];
      zeroCuda(*x_sloppy[i]);
    }
    r_sloppy = r;
  } else {
    for (i=0; i<num_offset; i++) {
      x_sloppy[i] = new cudaColorSpinorField(*x[i], param);
    }
    param.create = QUDA_COPY_FIELD_CREATE;
    r_sloppy = new cudaColorSpinorField(*r, param);
  }
  
  cudaColorSpinorField **p = new cudaColorSpinorField*[num_offset];  
  for(i=0;i < num_offset;i++){
    p[i]= new cudaColorSpinorField(*r_sloppy);    
  }
  
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.precision = invParam.cuda_prec_sloppy;
  cudaColorSpinorField* Ap = new cudaColorSpinorField(*r_sloppy, param);
  
  cudaColorSpinorField tmp1(*Ap, param);
  cudaColorSpinorField *tmp2_p = &tmp1;
  // tmp only needed for multi-gpu Wilson-like kernels
  if (mat.Type() != typeid(DiracStaggeredPC).name() && 
      mat.Type() != typeid(DiracStaggered).name()) {
    tmp2_p = new cudaColorSpinorField(*Ap, param);
  }
  cudaColorSpinorField &tmp2 = *tmp2_p;

  double b2 = 0.0;
  b2 = normCuda(b);
    
  double r2 = b2;
  double r2_old;


  // Set a lower bound on the tolerance.
  // Not sure if this is the best way to account for finite precision.
  // Perhaps I should enforce a lower bound on "stop" instead.
  
  const double min_tolerance = (param.precision == QUDA_DOUBLE_PRECISION) ? invParam.tol : (param.precision == QUDA_SINGLE_PRECISION) ? 1e-6 : 1e-4;
  const double tolerance = (invParam.tol < min_tolerance) ? min_tolerance : invParam.tol;

  double stop = r2*tolerance*tolerance; // stopping condition of solver
    
  double pAp;
    
  int k = 0;
    
  stopwatchStart();
  while (r2 > stop &&  k < invParam.maxiter) {
    //dslashCuda_st(tmp_sloppy, fatlinkSloppy, longlinkSloppy, p[0], 1 - oddBit, 0);
    //dslashAxpyCuda(Ap, fatlinkSloppy, longlinkSloppy, tmp_sloppy, oddBit, 0, p[0], msq_x4);
    matSloppy(*Ap, *p[0], tmp1, tmp2);
    if (invParam.dslash_type != QUDA_ASQTAD_DSLASH){
      axpyCuda(offset[0], *p[0], *Ap);
    }
    pAp = reDotProductCuda(*p[0], *Ap);
    beta_i[0] = r2 / pAp;        

    zeta_ip1[0] = 1.0;
    for (j=1; j<num_offset_now; j++) {
      zeta_ip1[j] = zeta_i[j] * zeta_im1[j] * beta_im1[j_low];
      double c1 = beta_i[j_low] * alpha[j_low] * (zeta_im1[j]-zeta_i[j]);
      double c2 = zeta_im1[j] * beta_im1[j_low] * (1.0+(offset[j]-offset[0])*beta_i[j_low]);
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
      if( zeta_i[j] != 0.0) {
	beta_i[j] = beta_i[j_low] * zeta_ip1[j] / zeta_i[j];
      } else  {
	zeta_ip1[j] = 0.0;
	beta_i[j] = 0.0;
	finished[j] = 1;
	if (invParam.verbosity >= QUDA_VERBOSE)
	  printfQuda("SETTING A ZERO, j=%d, num_offset_now=%d\n",j,num_offset_now);
	//if(j==num_offset_now-1)node0_PRINTF("REDUCING OFFSET\n");
	if(j==num_offset_now-1) num_offset_now--;
	// don't work any more on finished solutions
	// this only works if largest offsets are last, otherwise
	// just wastes time multiplying by zero
      }
    }	
	
    r2_old = r2;
    r2 = axpyNormCuda(-beta_i[j_low], *Ap, *r_sloppy);

    alpha[0] = r2 / r2_old;
	
    for (j=1; j<num_offset_now; j++) {
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
    for (j=1; j<num_offset_now; j++) {
      axpyBzpcxCuda(beta_i[j], *p[j], *x_sloppy[j], zeta_ip1[j], *r_sloppy, alpha[j]);
    }
    
    for (j=0; j<num_offset_now; j++) {
      beta_im1[j] = beta_i[j];
      zeta_im1[j] = zeta_i[j];
      zeta_i[j] = zeta_ip1[j];
    }

    k++;
    if (invParam.verbosity >= QUDA_VERBOSE){
      printfQuda("Multimass CG: %d iterations, r2 = %e\n", k, r2);
    }
  }
    
  if (x[0]->Precision() != x_sloppy[0]->Precision()) {
    for(i=0;i < num_offset; i++){
      copyCuda(*x[i], *x_sloppy[i]);
    }
  }

  *residue_sq = r2;

  invParam.secs = stopwatchReadSeconds();
     
  if (k==invParam.maxiter) {
    warningQuda("Exceeded maximum iterations %d\n", invParam.maxiter);
  }
    
  double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
  reduceDouble(gflops);

  invParam.gflops = gflops;
  invParam.iter = k;
  
  // Calculate the true residual of the system with the smallest shift
  mat(*r, *x[0]); 
  if (invParam.dslash_type != QUDA_ASQTAD_DSLASH){
    axpyCuda(offset[0],*x[0], *r); // Offset it.
  }
  double true_res = xmyNormCuda(b, *r);
  if (invParam.verbosity >= QUDA_SUMMARIZE){
    printfQuda("MultiShift CG: Converged after %d iterations, r2 = %e, relative true_r2 = %e\n", 
	       k,r2, (true_res / b2));
  }    
  if (invParam.verbosity >= QUDA_VERBOSE){
    printfQuda("MultiShift CG: Converged after %d iterations\n", k);
    printfQuda(" shift=0 resid_rel=%e\n", sqrt(true_res/b2));
    for(int i=1; i < num_offset; i++) { 
      mat(*r, *x[i]); 
      if (invParam.dslash_type != QUDA_ASQTAD_DSLASH){
         axpyCuda(offset[i],*x[i], *r); // Offset it.
      }else{
         axpyCuda(offset[i]-offset[0],*x[i], *r); // Offset it.
      }
      true_res = xmyNormCuda(b, *r);
      printfQuda(" shift=%d resid_rel=%e\n",i, sqrt(true_res/b2));
    }
  }      
  
  if (&tmp2 != &tmp1) delete tmp2_p;

  delete r;
  for(i=0;i < num_offset; i++){
    delete p[i];
  }
  delete p;
  delete Ap;
  
  if (invParam.cuda_prec_sloppy != x[0]->Precision()) {
    for(i=0;i < num_offset;i++){
      delete x_sloppy[i];
    }
    delete r_sloppy;
  }
  delete x_sloppy;
  
  delete []finished;
  delete []zeta_i;
  delete []zeta_im1;
  delete []zeta_ip1;
  delete []beta_i;
  delete []beta_im1;
  delete []alpha;
 
}

