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

#if defined INIT_PARAM
  P(anisotropy, INVALID_DOUBLE);
  P(tadpole_coeff, INVALID_DOUBLE);
#else
  if (param->type == QUDA_WILSON_LINKS) {
    P(anisotropy, INVALID_DOUBLE);
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS ||
	     param->type == QUDA_ASQTAD_LONG_LINKS) {
    P(tadpole_coeff, INVALID_DOUBLE);
  }
#endif

  P(type, QUDA_INVALID_LINKS);
  P(gauge_order, QUDA_INVALID_GAUGE_ORDER);
  P(t_boundary, QUDA_INVALID_T_BOUNDARY);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(reconstruct, QUDA_RECONSTRUCT_INVALID);
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(reconstruct_sloppy, QUDA_RECONSTRUCT_INVALID);
  P(gauge_fix, QUDA_GAUGE_FIXED_INVALID);
  P(ga_pad, INVALID_INT);
  
#if defined INIT_PARAM
  P(packed_size, 0);
  P(gaugeGiB, 0.0);
#elif defined PRINT_PARAM
  P(packed_size, INVALID_INT);
  P(gaugeGiB, INVALID_DOUBLE);
#endif

#if defined INIT_PARAM
  P(flag, 0);
#else
  P(flag, INVALID_INT);
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
  P(mass, INVALID_DOUBLE);
  P(kappa, INVALID_DOUBLE);
  P(m5, INVALID_DOUBLE);
  P(Ls, INVALID_INT);
  P(mu, INVALID_DOUBLE);
  P(twist_flavor, QUDA_TWIST_INVALID);
#else
  // asqtad and domain wall use mass parameterization
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    P(mass, INVALID_DOUBLE);
  } else { // Wilson and clover use kappa parameterization
    P(kappa, INVALID_DOUBLE);
  }
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    P(m5, INVALID_DOUBLE);
    P(Ls, INVALID_INT);
  }
  if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    P(mu, INVALID_DOUBLE);
    P(twist_flavor, QUDA_TWIST_INVALID);
  }
#endif

  P(tol, INVALID_DOUBLE);
  P(maxiter, INVALID_INT);
  P(reliable_delta, INVALID_DOUBLE);
  P(solution_type, QUDA_INVALID_SOLUTION);
  P(solve_type, QUDA_INVALID_SOLVE);
  P(matpc_type, QUDA_MATPC_INVALID);
  P(dagger, QUDA_DAG_INVALID);
  P(mass_normalization, QUDA_INVALID_NORMALIZATION);
  P(preserve_source, QUDA_PRESERVE_SOURCE_INVALID);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(gamma_basis, QUDA_INVALID_GAMMA_BASIS);
  P(dirac_order, QUDA_INVALID_DIRAC_ORDER);
  P(sp_pad, INVALID_INT);

  P(dirac_tune, QUDA_TUNE_INVALID);
  P(preserve_dirac, QUDA_PRESERVE_DIRAC_INVALID);

#if defined INIT_PARAM
  P(gcrNkrylov, INVALID_INT);
#else
  if (param->inv_type == QUDA_GCR_INVERTER) {
    P(gcrNkrylov, INVALID_INT);
  }
#endif

  // domain decomposition parameters
  //P(inv_type_sloppy, QUDA_INVALID_INVERTER); // disable since invalid means no preconditioner
#if defined INIT_PARAM
  P(inv_type_precondition, QUDA_INVALID_INVERTER);
  P(tol_precondition, INVALID_DOUBLE);
  P(maxiter_precondition, INVALID_INT);
  P(verbosity_precondition, QUDA_INVALID_VERBOSITY);
  P(prec_precondition, QUDA_INVALID_PRECISION);
#else
  if (param->inv_type_precondition == QUDA_BICGSTAB_INVERTER || 
      param->inv_type_precondition == QUDA_CG_INVERTER || 
      param->inv_type_precondition == QUDA_MR_INVERTER) {
    P(tol_precondition, INVALID_DOUBLE);
    P(maxiter_precondition, INVALID_INT);
    P(verbosity_precondition, QUDA_INVALID_VERBOSITY);
    P(prec_precondition, QUDA_INVALID_PRECISION);
  }
#endif



  
#ifdef INIT_PARAM
  P(use_init_guess, QUDA_USE_INIT_GUESS_NO); //set the default to no
  P(omega, 1.0); // set default to no relaxation
#else
  P(use_init_guess, QUDA_USE_INIT_GUESS_INVALID);
  P(omega, INVALID_DOUBLE);
#endif

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
  //p(ghostDim[0],0);
  //p(ghostDim[1],0);
  //p(ghostDim[2],0);
  //p(ghostDim[3],0);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}


// clean up

#undef INVALID_INT
#undef INVALID_DOUBLE
#undef P
