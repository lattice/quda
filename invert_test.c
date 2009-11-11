#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <quda.h>
#include <util_quda.h>
#include <dslash_reference.h>

int main(int argc, char **argv)
{
  int device = 0;

  void *gauge[4], *clover_inv;

  QudaGaugeParam Gauge_param;
  QudaInvertParam inv_param;

  Gauge_param.X[0] = 24;
  Gauge_param.X[1] = 24;
  Gauge_param.X[2] = 24;
  Gauge_param.X[3] = 64;
  setDims(Gauge_param.X);

  Gauge_param.anisotropy = 1.0;
  Gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  Gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  Gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  Gauge_param.cuda_prec = QUDA_SINGLE_PRECISION;
  Gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
  Gauge_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  Gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  Gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param = &Gauge_param;

  int clover_yes = 0; // 0 for plain Wilson, 1 for clover
  
  if (clover_yes) {
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
  }
  inv_param.inv_type = QUDA_BICGSTAB_INVERTER;

  double mass = -0.94;
  inv_param.kappa = 1.0 / (2.0*(1 + 3/gauge_param->anisotropy + mass));
  inv_param.tol = 5e-7;
  inv_param.maxiter = 10000;
  inv_param.reliable_delta = 1e-1;

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_SINGLE_PRECISION;
  inv_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.sp_pad = 0;

  if (clover_yes) {
    inv_param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec = QUDA_SINGLE_PRECISION;
    inv_param.clover_cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }
  inv_param.verbosity = QUDA_VERBOSE;

  size_t gSize = (Gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  construct_gauge_field(gauge, 1, Gauge_param.cpu_prec);

  if (clover_yes) {
    double norm = 0.2; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorOut = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  int i0 = 0;
  int s0 = 0;
  int c0 = 0;
  construct_spinor_field(spinorIn, 1, i0, s0, c0, inv_param.cpu_prec);

  double time0 = -((double)clock()); // Start the timer

  initQuda(device);
  loadGaugeQuda((void*)gauge, &Gauge_param);
  if (clover_yes) loadCloverQuda(NULL, clover_inv, &inv_param);

  invertQuda(spinorOut, spinorIn, &inv_param);

  time0 += clock(); // stop the timer
  time0 /= CLOCKS_PER_SEC;

  printf("Cuda Space Required:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, Gauge_param.gaugeGiB);
  if (clover_yes) printf("   Clover: %f GiB\n", inv_param.cloverGiB);
  printf("\nDone: %i iter / %g secs = %g gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, Gauge_param.cpu_prec);
  if  (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
    ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

  mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  endQuda();

  return 0;
}
