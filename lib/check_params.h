// check_params.h

// This file defines functions to either initialize, check, or print
// the QUDA gauge and inverter parameters.  It gets included in
// interface_quda.cpp, after either INIT_PARAM, CHECK_PARAM, or
// PRINT_PARAM is defined.
//
// If you're reading this file because it was mentioned in a "QUDA
// error" message, it probably means that you forgot to set one of the
// gauge or inverter parameters in your application before calling
// loadGaugeQuda() or invertQuda().

#include <float.h>
#define INVALID_INT QUDA_INVALID_ENUM
#define INVALID_DOUBLE DBL_MIN

// define macro to carry out the appropriate action for a given parameter

#if defined INIT_PARAM
#define P(x, val) ret.x = val
#elif defined CHECK_PARAM
#define P(x, val) if (param->x == val) errorQuda("Parameter " #x " undefined")
#elif defined PRINT_PARAM
#define P(x, val) \
  printfQuda((val == INVALID_DOUBLE) ? #x " = %g\n" : #x " = %d\n", param->x)
#else
#error INIT_PARAM, CHECK_PARAM, and PRINT_PARAM all undefined in check_params.h
#endif


// define the appropriate function for GaugeParam

#if defined INIT_PARAM
QudaGaugeParam newQudaGaugeParam(void) {
  QudaGaugeParam ret;
#elif defined CHECK_PARAM
static void checkGaugeParam(QudaGaugeParam *param) {
#else
void printQudaGaugeParam(QudaGaugeParam *param) {
  printfQuda("QUDA Gauge Parameters:\n");
#endif

  P(X[0], INVALID_INT);
  P(X[1], INVALID_INT);
  P(X[2], INVALID_INT);
  P(X[3], INVALID_INT);
  P(anisotropy, INVALID_DOUBLE);
  P(type, QUDA_INVALID_GAUGE);
  P(gauge_order, QUDA_INVALID_GAUGE_ORDER);
  P(t_boundary, QUDA_INVALID_T_BOUNDARY);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(reconstruct, QUDA_RECONSTRUCT_INVALID);
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(reconstruct_sloppy, QUDA_RECONSTRUCT_INVALID);
  P(gauge_fix, QUDA_GAUGE_FIXED_INVALID);
  P(ga_pad, INVALID_INT);

#ifdef PRINT_PARAM
  P(packed_size, INVALID_INT);
  P(gaugeGiB, INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}


// define the appropriate function for InvertParam

#if defined INIT_PARAM
QudaInvertParam newQudaInvertParam(void) {
  QudaInvertParam ret;
#elif defined CHECK_PARAM
static void checkInvertParam(QudaInvertParam *param) {
#else
void printQudaInvertParam(QudaInvertParam *param) {
  printfQuda("QUDA Inverter Parameters:\n");
#endif

  P(dslash_type, QUDA_INVALID_DSLASH);
  P(inv_type, QUDA_INVALID_INVERTER);

#if defined INIT_PARAM
  P(in_parity, QUDA_INVALID_PARITY);
  P(mass, INVALID_DOUBLE);
  P(kappa, INVALID_DOUBLE);
#else
  if (param->dslash_type == QUDA_STAGGERED_DSLASH) {
    P(in_parity, QUDA_INVALID_PARITY);
    P(mass, INVALID_DOUBLE);
  } else {
    P(kappa, INVALID_DOUBLE);
  }
#endif

  P(tol, INVALID_DOUBLE);
  P(maxiter, INVALID_INT);
  P(reliable_delta, INVALID_DOUBLE);
  P(solution_type, QUDA_INVALID_SOLUTION);
  P(solver_type, QUDA_INVALID_SOLUTION);
  P(matpc_type, QUDA_MATPC_INVALID);
  P(mass_normalization, QUDA_INVALID_NORMALIZATION);
  P(preserve_source, QUDA_PRESERVE_SOURCE_INVALID);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(dirac_order, QUDA_INVALID_DIRAC_ORDER);
  P(sp_pad, INVALID_INT);

#ifndef INIT_PARAM
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
#endif
    P(clover_cpu_prec, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec_sloppy, QUDA_INVALID_PRECISION);
    P(clover_order, QUDA_INVALID_CLOVER_ORDER);
    P(cl_pad, INVALID_INT);
#ifndef INIT_PARAM
  }
#endif

  P(verbosity, QUDA_INVALID_VERBOSITY);

#ifdef PRINT_PARAM
  P(iter, INVALID_INT);
  P(spinorGiB, INVALID_DOUBLE);
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH)
    P(cloverGiB, INVALID_DOUBLE);
  P(gflops, INVALID_DOUBLE);
  P(secs, INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}


// clean up

#undef INVALID_INT
#undef INVALID_DOUBLE
#undef P
