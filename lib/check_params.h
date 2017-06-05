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
#define P(x, val)							\
  { if (val == INVALID_DOUBLE) printfQuda(#x " = %g\n", (double)param->x); \
    else printfQuda(#x " = %d\n", (int)param->x); }
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

#if defined INIT_PARAM
  P(location, QUDA_CPU_FIELD_LOCATION);
#else
  P(location, QUDA_INVALID_FIELD_LOCATION);
#endif

  for (int i=0; i<4; i++) P(X[i], INVALID_INT);

#if defined INIT_PARAM
  P(anisotropy, INVALID_DOUBLE);
  P(tadpole_coeff, INVALID_DOUBLE);
  P(scale, INVALID_DOUBLE);
#else
  if (param->type == QUDA_WILSON_LINKS) {
    P(anisotropy, INVALID_DOUBLE);
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS ||
	     param->type == QUDA_ASQTAD_LONG_LINKS) {
    P(tadpole_coeff, INVALID_DOUBLE);
    //P(scale, INVALID_DOUBLE);
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
#if defined INIT_PARAM
  P(cuda_prec_precondition, QUDA_INVALID_PRECISION);
  P(reconstruct_precondition, QUDA_RECONSTRUCT_INVALID);
#else
  if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
    param->cuda_prec_precondition = param->cuda_prec_sloppy;
  if (param->reconstruct_precondition == QUDA_RECONSTRUCT_INVALID)
    param->reconstruct_precondition = param->reconstruct_sloppy;
#endif

  P(gauge_fix, QUDA_GAUGE_FIXED_INVALID);

  P(ga_pad, INVALID_INT);
  
#if defined INIT_PARAM
  P(gaugeGiB, 0.0);
#else
  P(gaugeGiB, INVALID_DOUBLE);
#endif


#if defined INIT_PARAM
  P(staggered_phase_type, QUDA_STAGGERED_PHASE_NO);
  P(staggered_phase_applied, 0);
  P(i_mu, 0.0);
  P(overlap, 0);
#else
  P(staggered_phase_type, QUDA_STAGGERED_PHASE_INVALID);
  P(staggered_phase_applied, INVALID_INT);
  P(i_mu, INVALID_DOUBLE);
  P(overlap, INVALID_INT);
#endif

#if defined INIT_PARAM
  P(overwrite_mom, 0);
  P(use_resident_gauge, 0);
  P(use_resident_mom, 0);
  P(make_resident_gauge, 0);
  P(make_resident_mom, 0);
  P(return_result_gauge, 1);
  P(return_result_mom, 1);
#else
  P(overwrite_mom, INVALID_INT);
  P(use_resident_gauge, INVALID_INT);
  P(use_resident_mom, INVALID_INT);
  P(make_resident_gauge, INVALID_INT);
  P(make_resident_mom, INVALID_INT);
  P(return_result_gauge, INVALID_INT);
  P(return_result_mom, INVALID_INT);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

// define the appropriate function for EigParam

#if defined INIT_PARAM
QudaEigParam newQudaEigParam(void) {
  QudaEigParam ret;
#elif defined CHECK_PARAM
static void checkEigParam(QudaEigParam *param) {
#else
void printQudaEigParam(QudaEigParam *param) {
  printfQuda("QUDA Eig Parameters:\n");
#endif

#if defined INIT_PARAM
  P(RitzMat_lanczos, QUDA_INVALID_SOLUTION);
  P(RitzMat_Convcheck, QUDA_INVALID_SOLUTION);
  P(eig_type, QUDA_INVALID_TYPE);
  P(NPoly, 0);
  P(Stp_residual, 0.0);
  P(nk, 0);
  P(np, 0);
  P(f_size, 0);
  P(eigen_shift, 0.0);
#else
  P(NPoly, INVALID_INT);
  P(Stp_residual, INVALID_DOUBLE);
  P(nk, INVALID_INT);
  P(np, INVALID_INT);
  P(f_size, INVALID_INT);
  P(eigen_shift, INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}
// define the appropriate function for InvertParam

#if defined INIT_PARAM
QudaInvertParam newQudaInvertParam(void) {
  QudaInvertParam ret;
  QudaInvertParam *param=&ret;
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
  if (param->dslash_type == QUDA_STAGGERED_DSLASH || 
      param->dslash_type == QUDA_ASQTAD_DSLASH || 
      param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    P(mass, INVALID_DOUBLE);
  } else { // Wilson and clover use kappa parameterization
    P(kappa, INVALID_DOUBLE);
  }
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      param->dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    P(m5, INVALID_DOUBLE);
    P(Ls, INVALID_INT);
  }
  if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    P(mu, INVALID_DOUBLE);
    P(twist_flavor, QUDA_TWIST_INVALID);
  }
#endif

  P(tol, INVALID_DOUBLE);

#ifdef INIT_PARAM
  P(residual_type, QUDA_L2_RELATIVE_RESIDUAL);
#else
  P(residual_type, QUDA_INVALID_RESIDUAL);
#endif

  if (param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
    P(tol_hq, INVALID_DOUBLE);
  }

  P(maxiter, INVALID_INT);
  P(reliable_delta, INVALID_DOUBLE);
#ifdef INIT_PARAM 
  P(use_sloppy_partial_accumulator, 0); /**< Default is to use a high-precision accumulator (not yet supported in all solvers) */
  P(solution_accumulator_pipeline, 1); /**< Default is solution accumulator depth of 1 */
  P(max_res_increase, 1); /**< Default is to allow one consecutive residual increase */
  P(max_res_increase_total, 10); /**< Default is to allow ten residual increase */
  P(heavy_quark_check, 10); /**< Default is to update heavy quark residual after 10 iterations */
 #else
  P(use_sloppy_partial_accumulator, INVALID_INT);
  P(solution_accumulator_pipeline, INVALID_INT);
  P(max_res_increase, INVALID_INT);
  P(max_res_increase_total, INVALID_INT);
  P(heavy_quark_check, INVALID_INT);
#endif

#ifndef CHECK_PARAM
  P(pipeline, 0); /** Whether to use a pipelined solver */
  P(num_offset, 0); /**< Number of offsets in the multi-shift solver */
  P(num_src, 1); /**< Number of offsets in the multi-shift solver */
  P(overlap, 0); /**< width of domain overlaps */
#endif

#ifdef INIT_PARAM
  P(compute_action, 0);
  P(compute_true_res, 1);
#else
  P(compute_action, INVALID_INT);
  P(compute_true_res, INVALID_INT);
#endif

  if (param->num_offset > 0) {

    for (int i=0; i<param->num_offset; i++) {
      P(offset[i], INVALID_DOUBLE);
      P(tol_offset[i], INVALID_DOUBLE);     
      if (param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL)
	P(tol_hq_offset[i], INVALID_DOUBLE);
#ifndef CHECK_PARAM
      P(true_res_offset[i], INVALID_DOUBLE); 
      P(iter_res_offset[i], INVALID_DOUBLE);
#endif
      if (param->compute_action) P(residue[i], INVALID_DOUBLE);
    }
#ifndef CHECK_PARAM
    P(action[0], INVALID_DOUBLE);
    P(action[1], INVALID_DOUBLE);
#endif
  }

  P(solution_type, QUDA_INVALID_SOLUTION);
  P(solve_type, QUDA_INVALID_SOLVE);
  P(matpc_type, QUDA_MATPC_INVALID);
  P(dagger, QUDA_DAG_INVALID);
  P(mass_normalization, QUDA_INVALID_NORMALIZATION);
#ifndef CHECK_PARAM
  P(solver_normalization, QUDA_DEFAULT_NORMALIZATION);
#endif
  P(preserve_source, QUDA_PRESERVE_SOURCE_INVALID);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);

  // leave the default behviour to cpu pointers
#if defined INIT_PARAM
  P(input_location, QUDA_CPU_FIELD_LOCATION);
  P(output_location, QUDA_CPU_FIELD_LOCATION);
  P(clover_location, QUDA_CPU_FIELD_LOCATION);
#else
  P(input_location, QUDA_INVALID_FIELD_LOCATION);
  P(output_location, QUDA_INVALID_FIELD_LOCATION);
  P(clover_location, QUDA_INVALID_FIELD_LOCATION);
#endif

#if defined INIT_PARAM
  P(cuda_prec_precondition, QUDA_INVALID_PRECISION);
#else
  if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
    param->cuda_prec_precondition = param->cuda_prec_sloppy;
#endif

  P(gamma_basis, QUDA_INVALID_GAMMA_BASIS);
  P(dirac_order, QUDA_INVALID_DIRAC_ORDER);
  P(sp_pad, INVALID_INT);

#if defined INIT_PARAM
  P(Nsteps, INVALID_INT);
#else
  if(param->inv_type == QUDA_MPCG_INVERTER || param->inv_type == QUDA_MPBICGSTAB_INVERTER){
    P(Nsteps, INVALID_INT);
  }
#endif

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
  P(preconditioner, 0);
  P(tol_precondition, INVALID_DOUBLE);
  P(maxiter_precondition, INVALID_INT);
  P(verbosity_precondition, QUDA_INVALID_VERBOSITY);
  P(schwarz_type, QUDA_ADDITIVE_SCHWARZ); // defaults match previous interface behaviour
  P(precondition_cycle, 1);               // defaults match previous interface behaviour
#else
  if (param->inv_type_precondition == QUDA_BICGSTAB_INVERTER || 
      param->inv_type_precondition == QUDA_CG_INVERTER || 
      param->inv_type_precondition == QUDA_MR_INVERTER) {
    P(tol_precondition, INVALID_DOUBLE);
    P(maxiter_precondition, INVALID_INT);
    P(verbosity_precondition, QUDA_INVALID_VERBOSITY);
    P(schwarz_type, QUDA_INVALID_SCHWARZ);
    P(precondition_cycle, 0);              
  }
#endif



  
#ifdef INIT_PARAM
  P(use_init_guess, QUDA_USE_INIT_GUESS_NO); //set the default to no
  //P(compute_null_vector, QUDA_COMPUTE_NULL_VECTOR_NO); //set the default to no
  P(omega, 1.0); // set default to no relaxation
#else
  P(use_init_guess, QUDA_USE_INIT_GUESS_INVALID);
  //P(compute_null_vector, QUDA_COMPUTE_NULL_VECTOR_INVALID);
  P(omega, INVALID_DOUBLE);
#endif

#ifndef INIT_PARAM
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
#endif
    P(clover_cpu_prec, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec_sloppy, QUDA_INVALID_PRECISION);
#if defined INIT_PARAM
    P(clover_cuda_prec_precondition, QUDA_INVALID_PRECISION);
    P(compute_clover_trlog, 0);
    P(compute_clover, 0);
    P(compute_clover_inverse, 0);
    P(return_clover, 0);
    P(return_clover_inverse, 0);
    P(clover_rho, 0.0);
#else
    if (param->clover_cuda_prec_precondition == QUDA_INVALID_PRECISION)
      param->clover_cuda_prec_precondition = param->clover_cuda_prec_sloppy;
    P(compute_clover_trlog, QUDA_INVALID_PRECISION);
    P(compute_clover, QUDA_INVALID_PRECISION);
    P(compute_clover_inverse, QUDA_INVALID_PRECISION);
    P(return_clover, QUDA_INVALID_PRECISION);
    P(return_clover_inverse, QUDA_INVALID_PRECISION);
    P(clover_rho, INVALID_DOUBLE);
#endif
    P(clover_order, QUDA_INVALID_CLOVER_ORDER);
    P(cl_pad, INVALID_INT);

    P(clover_coeff, INVALID_DOUBLE);
#ifndef INIT_PARAM
  }
#endif

  P(verbosity, QUDA_INVALID_VERBOSITY);

#ifdef INIT_PARAM
  P(iter, 0);
  P(spinorGiB, 0.0);
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH)
    P(cloverGiB, 0.0);
  P(gflops, 0.0);
  P(secs, 0.0);
#elif defined(PRINT_PARAM)
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


#if defined INIT_PARAM
  P(cuda_prec_ritz, QUDA_INVALID_PRECISION);
  P(nev, 0);
  P(max_search_dim, 0);
  P(rhs_idx, 0);
  P(deflation_grid, 0);

  P(use_reduced_vector_set, true);
  P(use_cg_updates, false);
  P(cg_iterref_tol, 5e-2);
  P(eigcg_max_restarts, 2);
  P(max_restart_num, 3);
  P(inc_tol, 1e-2);
  P(eigenval_tol, 1e-1);
#else
  //P(cuda_prec_ritz, QUDA_INVALID_PRECISION);
  P(nev, INVALID_INT);
  P(max_search_dim, INVALID_INT);
  P(rhs_idx, INVALID_INT);
  P(deflation_grid, INVALID_INT);
  P(cg_iterref_tol, INVALID_DOUBLE);
  P(eigcg_max_restarts, INVALID_INT);
  P(max_restart_num, INVALID_INT);
  P(inc_tol, INVALID_DOUBLE);
  P(eigenval_tol, INVALID_DOUBLE);
#endif

#if defined INIT_PARAM
  P(use_resident_solution, 0);
  P(make_resident_solution, 0);
#else
  P(use_resident_solution, INVALID_INT);
  P(make_resident_solution, INVALID_INT);
#endif


#if defined INIT_PARAM
  P(use_resident_chrono, 0);
  P(make_resident_chrono, 0);
  P(max_chrono_dim, 0);
  P(chrono_index, 0);
#else
  P(use_resident_chrono, INVALID_INT);
  P(make_resident_chrono, INVALID_INT);
  P(max_chrono_dim, INVALID_INT);
  P(chrono_index, INVALID_INT);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}


#if defined INIT_PARAM
 QudaMultigridParam newQudaMultigridParam(void) {
   QudaMultigridParam ret;
#elif defined CHECK_PARAM
   static void checkMultigridParam(QudaMultigridParam *param) {
#else
void printQudaMultigridParam(QudaMultigridParam *param) {
  printfQuda("QUDA Multigrid Parameters:\n");
#endif

#ifdef INIT_PARAM
  // do nothing
#elif defined CHECK_PARAM
  checkInvertParam(param->invert_param);
#else
  printQudaInvertParam(param->invert_param);
#endif

  P(n_level, INVALID_INT);

#ifdef INIT_PARAM
  int n_level = QUDA_MAX_MG_LEVEL;
#else
  int n_level = param->n_level;
#endif

  for (int i=0; i<n_level; i++) {
#ifdef INIT_PARAM
    P(verbosity[i], QUDA_SILENT);
#else
    P(verbosity[i], QUDA_INVALID_VERBOSITY);
#endif
#ifdef INIT_PARAM
    P(setup_inv_type[i], QUDA_BICGSTAB_INVERTER);
#else
    P(setup_inv_type[i], QUDA_INVALID_INVERTER);
#endif
#ifdef INIT_PARAM
    P(setup_tol[i], 5e-6);
#else
    P(setup_tol[i], INVALID_DOUBLE);
#endif
    P(smoother[i], QUDA_INVALID_INVERTER);
    P(smoother_solve_type[i], QUDA_INVALID_SOLVE);

    // these parameters are not set for the bottom grid
    if (i<n_level-1) {
      for (int j=0; j<4; j++) P(geo_block_size[i][j], INVALID_INT);
      P(spin_block_size[i], INVALID_INT);
      P(n_vec[i], INVALID_INT);
      P(cycle_type[i], QUDA_MG_CYCLE_INVALID);
      P(nu_pre[i], INVALID_INT);
      P(nu_post[i], INVALID_INT);
      P(coarse_grid_solution_type[i], QUDA_INVALID_SOLUTION);
    }

#ifdef INIT_PARAM
    P(mu_factor[i], 1);
#else
    P(mu_factor[i], INVALID_DOUBLE);
#endif
    P(smoother_tol[i], INVALID_DOUBLE);
#ifdef INIT_PARAM
    P(global_reduction[i], QUDA_BOOLEAN_YES);
#else
    P(global_reduction[i], QUDA_BOOLEAN_INVALID);
#endif

    P(omega[i], INVALID_DOUBLE);
    P(location[i], QUDA_INVALID_FIELD_LOCATION);
  }

  P(compute_null_vector, QUDA_COMPUTE_NULL_VECTOR_INVALID);
  P(generate_all_levels, QUDA_BOOLEAN_INVALID);

#ifdef CHECK_PARAM
  // if only doing top-level null-space generation, check that n_vec
  // is equal on all levels
  if (param->generate_all_levels == QUDA_BOOLEAN_NO && param->compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
    for (int i=1; i<n_level-1; i++)
      if (param->n_vec[0] != param->n_vec[i])
	errorQuda("n_vec %d != %d must be equal on all levels if generate_all_levels == false",
		  param->n_vec[0], param->n_vec[i]);
  }
#endif

  P(run_verify, QUDA_BOOLEAN_INVALID);

#ifdef INIT_PARAM
  P(gflops, 0.0);
  P(secs, 0.0);
#elif defined(PRINT_PARAM)
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
