#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <domain_wall_dslash_reference.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

int main(int argc, char **argv)
{
  // set QUDA parameters

  int device = 0; // CUDA device number

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = QUDA_SINGLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
 
  gauge_param.X[0] = 16; 
  gauge_param.X[1] = 16;
  gauge_param.X[2] = 16;
  gauge_param.X[3] = 16;
  inv_param.Ls = 16;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = QUDA_DOMAIN_WALL_DSLASH;
  inv_param.inv_type = QUDA_CG_INVERTER;

  inv_param.mass = 0.01;
  inv_param.m5 = -1.5;
  double kappa5 = 0.5/(5 + inv_param.m5);

  inv_param.tol = 5e-8;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 0.1;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_NORMEQ_PC_SOLVE;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.prec_precondition = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dirac_tune = QUDA_TUNE_YES;
  inv_param.preserve_dirac = QUDA_PRESERVE_DIRAC_YES;

  gauge_param.ga_pad = 0; // 24*24*24;
  inv_param.sp_pad = 0;   // 24*24*24;
  inv_param.cl_pad = 0;   // 24*24*24;

  inv_param.verbosity = QUDA_VERBOSE;

  // Everything between here and the call to initQuda() is application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X, inv_param.Ls);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);

  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  // create a point source at 0
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) *((float*)spinorIn) = 1.0;
  else *((double*)spinorIn) = 1.0;

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // perform the inversion
  invertQuda(spinorOut, spinorIn, &inv_param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printf("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  printf("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) { 
    mat(spinorCheck, gauge, spinorOut, kappa5, 0, inv_param.cpu_prec, 
	gauge_param.cpu_prec, inv_param.mass); 
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.5/kappa5, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {   
    matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, 
	  inv_param.cpu_prec, gauge_param.cpu_prec, inv_param.mass);
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.25/(kappa5*kappa5), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  }

  mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  // finalize the QUDA library
  endQuda();

  return 0;
}
