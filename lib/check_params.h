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

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaGaugeParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
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
  P(scale, 1.0);
#else
  if (param->type == QUDA_WILSON_LINKS) {
    P(anisotropy, INVALID_DOUBLE);
  } else if (param->type == QUDA_ASQTAD_LONG_LINKS) {
    P(tadpole_coeff, INVALID_DOUBLE);
    P(scale, INVALID_DOUBLE);
  }
#endif

  P(type, QUDA_INVALID_LINKS);
  P(gauge_order, QUDA_INVALID_GAUGE_ORDER);
  P(t_boundary, QUDA_INVALID_T_BOUNDARY);
  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);
  P(reconstruct, QUDA_RECONSTRUCT_INVALID);

#ifndef CHECK_PARAM
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(reconstruct_sloppy, QUDA_RECONSTRUCT_INVALID);
  P(cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION);
  P(reconstruct_refinement_sloppy, QUDA_RECONSTRUCT_INVALID);
  P(cuda_prec_precondition, QUDA_INVALID_PRECISION);
  P(reconstruct_precondition, QUDA_RECONSTRUCT_INVALID);
  P(cuda_prec_eigensolver, QUDA_INVALID_PRECISION);
  P(reconstruct_eigensolver, QUDA_RECONSTRUCT_INVALID);
#else
  if (param->cuda_prec_sloppy == QUDA_INVALID_PRECISION)
    param->cuda_prec_sloppy = param->cuda_prec;
  if (param->cuda_prec_eigensolver == QUDA_INVALID_PRECISION) param->cuda_prec_eigensolver = param->cuda_prec;
  if (param->reconstruct_sloppy == QUDA_RECONSTRUCT_INVALID)
    param->reconstruct_sloppy = param->reconstruct;
  if (param->reconstruct_eigensolver == QUDA_RECONSTRUCT_INVALID)
    param->reconstruct_eigensolver = param->reconstruct_sloppy;
  if (param->cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
    param->cuda_prec_refinement_sloppy = param->cuda_prec_sloppy;
  if (param->reconstruct_refinement_sloppy == QUDA_RECONSTRUCT_INVALID)
    param->reconstruct_refinement_sloppy = param->reconstruct_sloppy;
  if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
    param->cuda_prec_precondition = param->cuda_prec_sloppy;
  if (param->reconstruct_precondition == QUDA_RECONSTRUCT_INVALID)
    param->reconstruct_precondition = param->reconstruct_sloppy;
#endif

  P(gauge_fix, QUDA_GAUGE_FIXED_INVALID);
  P(ga_pad, INVALID_INT);

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
  P(gauge_offset, 0);
  P(mom_offset, 0);
  P(site_size, 0);
#else
  P(overwrite_mom, INVALID_INT);
  P(use_resident_gauge, INVALID_INT);
  P(use_resident_mom, INVALID_INT);
  P(make_resident_gauge, INVALID_INT);
  P(make_resident_mom, INVALID_INT);
  P(return_result_gauge, INVALID_INT);
  P(return_result_mom, INVALID_INT);
  P(gauge_offset, (size_t)INVALID_INT);
  P(mom_offset, (size_t)INVALID_INT);
  P(site_size, (size_t)INVALID_INT);
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

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaEigParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#if defined INIT_PARAM
  P(use_eigen_qr, QUDA_BOOLEAN_FALSE);
  P(use_poly_acc, QUDA_BOOLEAN_FALSE);
  P(poly_deg, 0);
  P(a_min, 0.0);
  P(a_max, 0.0);
  P(preserve_deflation, QUDA_BOOLEAN_FALSE);
  P(preserve_deflation_space, 0);
  P(preserve_evals, QUDA_BOOLEAN_TRUE);
  P(use_dagger, QUDA_BOOLEAN_FALSE);
  P(use_norm_op, QUDA_BOOLEAN_FALSE);
  P(compute_svd, QUDA_BOOLEAN_FALSE);
  P(require_convergence, QUDA_BOOLEAN_TRUE);
  P(spectrum, QUDA_SPECTRUM_LR_EIG);
  P(n_ev, 0);
  P(n_kr, 0);
  P(n_conv, 0);
  P(n_ev_deflate, -1);
  P(batched_rotate, 0);
  P(tol, 0.0);
  P(qr_tol, 0.0);
  P(check_interval, 0);
  P(max_restarts, 0);
  P(max_ortho_attempts, 10);
  P(arpack_check, QUDA_BOOLEAN_FALSE);
  P(nk, 0);
  P(np, 0);
  P(eig_type, QUDA_EIG_TR_LANCZOS);
  P(extlib_type, QUDA_EIGEN_EXTLIB);
  P(mem_type_ritz, QUDA_MEMORY_DEVICE);
  P(ortho_block_size, 0);
#else
  P(use_eigen_qr, QUDA_BOOLEAN_INVALID);
  P(use_poly_acc, QUDA_BOOLEAN_INVALID);
  P(poly_deg, INVALID_INT);
  P(a_min, INVALID_DOUBLE);
  P(a_max, INVALID_DOUBLE);
  P(preserve_deflation, QUDA_BOOLEAN_INVALID);
  P(preserve_evals, QUDA_BOOLEAN_INVALID);
  P(use_dagger, QUDA_BOOLEAN_INVALID);
  P(use_norm_op, QUDA_BOOLEAN_INVALID);
  P(compute_svd, QUDA_BOOLEAN_INVALID);
  P(require_convergence, QUDA_BOOLEAN_INVALID);
  P(n_ev, INVALID_INT);
  P(n_kr, INVALID_INT);
  P(n_conv, INVALID_INT);
  P(n_ev_deflate, INVALID_INT);
  P(batched_rotate, INVALID_INT);
  P(tol, INVALID_DOUBLE);
  P(qr_tol, INVALID_DOUBLE);
  P(check_interval, INVALID_INT);
  P(max_restarts, INVALID_INT);
  P(max_ortho_attempts, INVALID_INT);
  P(arpack_check, QUDA_BOOLEAN_INVALID);
  P(nk, INVALID_INT);
  P(np, INVALID_INT);
  P(eig_type, QUDA_EIG_INVALID);
  P(extlib_type, QUDA_EXTLIB_INVALID);
  P(mem_type_ritz, QUDA_MEMORY_INVALID);
  P(ortho_block_size, INVALID_INT);
#endif

  // only need to enfore block size checking if doing a block eigen solve
#ifdef CHECK_PARAM
  if (param->eig_type == QUDA_EIG_BLK_TR_LANCZOS)
#endif
    P(block_size, INVALID_INT);

#if defined INIT_PARAM
  P(location, QUDA_CUDA_FIELD_LOCATION);
#else
  P(location, QUDA_INVALID_FIELD_LOCATION);
#endif

#if defined INIT_PARAM
  P(save_prec, QUDA_DOUBLE_PRECISION);
#else
  P(save_prec, QUDA_INVALID_PRECISION);
#endif

#if defined INIT_PARAM
  P(io_parity_inflate, QUDA_BOOLEAN_FALSE);
#else
  P(io_parity_inflate, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

// define the appropriate function for clover subset from InvertParam
#if defined INIT_PARAM
void newQudaCloverParam(QudaInvertParam *param)
{
  QudaInvertParam &ret = *param;
#elif defined CHECK_PARAM
static void checkCloverParam(QudaInvertParam *param)
{
#else
void printQudaCloverParam(QudaInvertParam *param)
{
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaInvertParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#if defined INIT_PARAM
  P(clover_location, QUDA_CPU_FIELD_LOCATION);
#else
  P(clover_location, QUDA_INVALID_FIELD_LOCATION);
#endif

#ifndef INIT_PARAM
  if (param->dslash_type == QUDA_CLOVER_WILSON_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || param->dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
#endif
    P(clover_cpu_prec, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec, QUDA_INVALID_PRECISION);

#ifndef CHECK_PARAM
    P(clover_cuda_prec_sloppy, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec_precondition, QUDA_INVALID_PRECISION);
    P(clover_cuda_prec_eigensolver, QUDA_INVALID_PRECISION);
#else
  if (param->clover_cuda_prec_sloppy == QUDA_INVALID_PRECISION)
    param->clover_cuda_prec_sloppy = param->clover_cuda_prec;
  if (param->clover_cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
    param->clover_cuda_prec_refinement_sloppy = param->clover_cuda_prec_sloppy;
  if (param->clover_cuda_prec_precondition == QUDA_INVALID_PRECISION)
    param->clover_cuda_prec_precondition = param->clover_cuda_prec_sloppy;
  if (param->clover_cuda_prec_eigensolver == QUDA_INVALID_PRECISION)
    param->clover_cuda_prec_eigensolver = param->clover_cuda_prec_sloppy;
#endif

#ifdef INIT_PARAM
    P(compute_clover_trlog, 1);
    P(compute_clover, 0);
    P(compute_clover_inverse, 0);
    P(return_clover, 0);
    P(return_clover_inverse, 0);
    P(clover_rho, 0.0);
    P(clover_coeff, 0.0);
    P(clover_csw, 0.0);
#else
  P(compute_clover_trlog, QUDA_INVALID_PRECISION);
  P(compute_clover, QUDA_INVALID_PRECISION);
  P(compute_clover_inverse, QUDA_INVALID_PRECISION);
  P(return_clover, QUDA_INVALID_PRECISION);
  P(return_clover_inverse, QUDA_INVALID_PRECISION);
  P(clover_rho, INVALID_DOUBLE);
  P(clover_coeff, INVALID_DOUBLE);
  P(clover_csw, INVALID_DOUBLE);
#endif
    P(clover_order, QUDA_INVALID_CLOVER_ORDER);

#ifndef INIT_PARAM
  }
#endif
}

// define the appropriate function for InvertParam

#if defined INIT_PARAM
QudaInvertParam newQudaInvertParam(void)
{
  QudaInvertParam ret;
  QudaInvertParam *param=&ret;
#elif defined CHECK_PARAM
static void checkInvertParam(QudaInvertParam *param, void *out_ptr=nullptr, void *in_ptr=nullptr) {
#else
void printQudaInvertParam(QudaInvertParam *param) {
  printfQuda("QUDA Inverter Parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaInvertParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

  P(dslash_type, QUDA_INVALID_DSLASH);
  P(inv_type, QUDA_INVALID_INVERTER);

#if defined INIT_PARAM
  P(mass, INVALID_DOUBLE);
  P(kappa, INVALID_DOUBLE);
  P(m5, INVALID_DOUBLE);
  P(Ls, INVALID_INT);
  P(mu, INVALID_DOUBLE);
  P(epsilon, INVALID_DOUBLE);
  P(tm_rho, 0.0);
  P(twist_flavor, QUDA_TWIST_INVALID);
  P(laplace3D, INVALID_INT);
#else
  // asqtad and domain wall use mass parameterization
  if (param->dslash_type == QUDA_STAGGERED_DSLASH || param->dslash_type == QUDA_ASQTAD_DSLASH
      || param->dslash_type == QUDA_DOMAIN_WALL_DSLASH || param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
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
  if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH || param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    P(mu, INVALID_DOUBLE);
    P(twist_flavor, QUDA_TWIST_INVALID);
  }
  if (param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) { P(tm_rho, INVALID_DOUBLE); }
  if (param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) { P(epsilon, INVALID_DOUBLE); }
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
#ifndef CHECK_PARAM
  P(reliable_delta_refinement, INVALID_DOUBLE);
#else
  if (param->reliable_delta_refinement == INVALID_DOUBLE) param->reliable_delta_refinement = param->reliable_delta;
#endif

#ifdef INIT_PARAM
  P(use_alternative_reliable, 0); /**< Default is to not use alternative relative updates, e.g., use delta to determine reliable trigger */
  P(use_sloppy_partial_accumulator, 0); /**< Default is to use a high-precision accumulator (not yet supported in all solvers) */
  P(solution_accumulator_pipeline, 1); /**< Default is solution accumulator depth of 1 */
  P(max_res_increase, 1); /**< Default is to allow one consecutive residual increase */
  P(max_res_increase_total, 10); /**< Default is to allow ten residual increase */
  P(max_hq_res_increase, 1);     /**< Default is to allow one consecutive heavy-quark residual increase */
  P(max_hq_res_restart_total, 10); /**< Default is to allow ten heavy-quark restarts */
  P(heavy_quark_check, 10); /**< Default is to update heavy quark residual after 10 iterations */
 #else
  P(use_alternative_reliable, INVALID_INT);
  P(use_sloppy_partial_accumulator, INVALID_INT);
  P(solution_accumulator_pipeline, INVALID_INT);
  P(max_res_increase, INVALID_INT);
  P(max_res_increase_total, INVALID_INT);
  P(max_hq_res_increase, INVALID_INT);
  P(max_hq_res_restart_total, INVALID_INT);
  P(heavy_quark_check, INVALID_INT);
#endif

#ifndef CHECK_PARAM
  P(pipeline, 0); /** Whether to use a pipelined solver */
  P(num_offset, 0); /**< Number of offsets in the multi-shift solver */
  P(num_src, 1); /**< Number of offsets in the multi-shift solver */
  P(overlap, 0); /**< width of domain overlaps */
#endif

#ifdef INIT_PARAM
  for (int d = 0; d < 4; d++) { P(split_grid[d], 1); } /**< Grid of sub-partitions */
  P(num_src_per_sub_partition, 1);                     /**< Number of sources per sub-partitions */
#else
  for (int d = 0; d < 4; d++) { P(split_grid[d], INVALID_INT); } /**< Grid of sub-partitions */
  P(num_src_per_sub_partition, INVALID_INT);                     /**< Number of sources per sub-partitions */
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

#ifndef CHECK_PARAM
  P(preserve_source, QUDA_PRESERVE_SOURCE_INVALID); // deprecated
#endif

  P(cpu_prec, QUDA_INVALID_PRECISION);
  P(cuda_prec, QUDA_INVALID_PRECISION);

#ifndef CHECK_PARAM
  P(cuda_prec_sloppy, QUDA_INVALID_PRECISION);
  P(cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION);
  P(cuda_prec_precondition, QUDA_INVALID_PRECISION);
  P(cuda_prec_eigensolver, QUDA_INVALID_PRECISION);
#else
  if (param->cuda_prec_sloppy == QUDA_INVALID_PRECISION)
    param->cuda_prec_sloppy = param->cuda_prec;
  if (param->cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
    param->cuda_prec_refinement_sloppy = param->cuda_prec_sloppy;
  if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
    param->cuda_prec_precondition = param->cuda_prec_sloppy;
  if (param->cuda_prec_eigensolver == QUDA_INVALID_PRECISION) param->cuda_prec_eigensolver = param->cuda_prec_sloppy;
#endif

  // leave the default behaviour to cpu pointers
#if defined INIT_PARAM
  P(input_location, QUDA_CPU_FIELD_LOCATION);
  P(output_location, QUDA_CPU_FIELD_LOCATION);
  P(clover_location, QUDA_CPU_FIELD_LOCATION);
#else
  P(input_location, QUDA_INVALID_FIELD_LOCATION);
  P(output_location, QUDA_INVALID_FIELD_LOCATION);
  P(clover_location, QUDA_INVALID_FIELD_LOCATION);
#endif

#ifdef CHECK_PARAM
  if (in_ptr && quda::get_pointer_location(in_ptr) != param->input_location) {
    warningQuda("input_location=%d, however supplied pointer is location=%d", param->input_location, quda::get_pointer_location(in_ptr));
    param->input_location = quda::get_pointer_location(in_ptr);
  }

  if (out_ptr && quda::get_pointer_location(out_ptr) != param->output_location) {
    warningQuda("output_location=%d, however supplied pointer is location=%d", param->output_location, quda::get_pointer_location(out_ptr));
    param->output_location = quda::get_pointer_location(out_ptr);
  }
#endif

  P(gamma_basis, QUDA_INVALID_GAMMA_BASIS);
  P(dirac_order, QUDA_INVALID_DIRAC_ORDER);

#if defined INIT_PARAM
  P(Nsteps, INVALID_INT);
#endif

#if defined INIT_PARAM
  P(gcrNkrylov, INVALID_INT);
#else
  if (param->inv_type == QUDA_GCR_INVERTER || param->inv_type == QUDA_BICGSTABL_INVERTER
      || quda::is_ca_solver(param->inv_type)) {
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
  P(schwarz_type, QUDA_INVALID_SCHWARZ);
  P(accelerator_type_precondition, QUDA_INVALID_ACCELERATOR);
  P(precondition_cycle, 1);               // defaults match previous interface behaviour
#else
  if (param->inv_type_precondition == QUDA_BICGSTAB_INVERTER || param->inv_type_precondition == QUDA_CG_INVERTER
      || param->inv_type_precondition == QUDA_CA_CG_INVERTER || param->inv_type_precondition == QUDA_MR_INVERTER) {
    P(tol_precondition, INVALID_DOUBLE);
    P(maxiter_precondition, INVALID_INT);
    P(verbosity_precondition, QUDA_INVALID_VERBOSITY);
    P(precondition_cycle, 0);
  }
#endif

#ifndef INIT_PARAM
  if (param->accelerator_type_precondition == QUDA_MADWF_ACCELERATOR) {
#endif
    P(madwf_diagonal_suppressor, INVALID_DOUBLE);
    P(madwf_ls, INVALID_INT);
    P(madwf_null_miniter, INVALID_INT);
    P(madwf_null_tol, INVALID_DOUBLE);
    P(madwf_train_maxiter, INVALID_INT);
#ifndef INIT_PARAM
  }
#endif

#ifdef INIT_PARAM
  P(madwf_param_infile[0], '\0');
  P(madwf_param_outfile[0], '\0');
#endif

#ifdef INIT_PARAM
  P(madwf_param_load, QUDA_BOOLEAN_FALSE);
  P(madwf_param_save, QUDA_BOOLEAN_FALSE);
#else
  P(madwf_param_load, QUDA_BOOLEAN_INVALID);
  P(madwf_param_save, QUDA_BOOLEAN_INVALID);
#endif

#if defined(INIT_PARAM)
  P(eig_param, 0);
#endif

#ifdef INIT_PARAM
  P(use_init_guess, QUDA_USE_INIT_GUESS_NO); //set the default to no
  P(omega, 1.0); // set default to no relaxation
#else
  P(use_init_guess, QUDA_USE_INIT_GUESS_INVALID);
  P(omega, INVALID_DOUBLE);
#endif

#if defined(INIT_PARAM)
  newQudaCloverParam(param);
#elif defined(CHECK_PARAM)
  checkCloverParam(param);
#else
  printQudaCloverParam(param);
#endif

#ifdef INIT_PARAM
  P(ca_basis, QUDA_POWER_BASIS);
  P(ca_lambda_min, 0.0);
  P(ca_lambda_max, -1.0);
#else
  if (quda::is_ca_solver(param->inv_type)) {
    P(ca_basis, QUDA_INVALID_BASIS);
    if (param->ca_basis == QUDA_CHEBYSHEV_BASIS) {
      P(ca_lambda_min, INVALID_DOUBLE);
      P(ca_lambda_max, INVALID_DOUBLE);
    }
  }
#endif

#ifdef INIT_PARAM
  P(ca_basis_precondition, QUDA_POWER_BASIS);
  P(ca_lambda_min_precondition, 0.0);
  P(ca_lambda_max_precondition, -1.0);
#else
  if (quda::is_ca_solver(param->inv_type)) {
    P(ca_basis_precondition, QUDA_INVALID_BASIS);
    if (param->ca_basis_precondition == QUDA_CHEBYSHEV_BASIS) {
      P(ca_lambda_min_precondition, INVALID_DOUBLE);
      P(ca_lambda_max_precondition, INVALID_DOUBLE);
    }
  }
#endif

  P(verbosity, QUDA_INVALID_VERBOSITY);

#ifdef INIT_PARAM
  P(iter, 0);
  P(gflops, 0.0);
  P(secs, 0.0);
#elif defined(PRINT_PARAM)
  P(iter, INVALID_INT);
  P(gflops, INVALID_DOUBLE);
  P(secs, INVALID_DOUBLE);
#endif


#if defined INIT_PARAM
  P(cuda_prec_ritz, QUDA_SINGLE_PRECISION);
  P(n_ev, 8);
  P(max_search_dim, 64);
  P(rhs_idx, 0);
  P(deflation_grid, 1);

  P(eigcg_max_restarts, 4);
  P(max_restart_num, 3);
  P(tol_restart,5e-5);
  P(inc_tol, 1e-2);
  P(eigenval_tol, 1e-1);
#else
  P(cuda_prec_ritz, QUDA_INVALID_PRECISION);
  P(n_ev, INVALID_INT);
  P(max_search_dim, INVALID_INT);
  P(rhs_idx, INVALID_INT);
  P(deflation_grid, INVALID_INT);
  P(eigcg_max_restarts, INVALID_INT);
  P(max_restart_num, INVALID_INT);
  P(tol_restart,INVALID_DOUBLE);
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
  P(chrono_use_resident, 0);
  P(chrono_make_resident, 0);
  P(chrono_replace_last, 0);
  P(chrono_max_dim, 0);
  P(chrono_index, 0);
#else
  P(chrono_use_resident, INVALID_INT);
  P(chrono_make_resident, INVALID_INT);
  P(chrono_replace_last, INVALID_INT);
  P(chrono_max_dim, INVALID_INT);
  P(chrono_index, INVALID_INT);
#endif

#if !defined CHECK_PARAM
  P(chrono_precision, QUDA_INVALID_PRECISION);
#else
  // default the chrono precision to using outer precision
  if (param->chrono_precision == QUDA_INVALID_PRECISION) param->chrono_precision = param->cuda_prec;
#endif

#if defined INIT_PARAM
  P(extlib_type, QUDA_EIGEN_EXTLIB);
#else
  P(extlib_type, QUDA_EXTLIB_INVALID);
#endif

#if defined INIT_PARAM
  P(native_blas_lapack, QUDA_BOOLEAN_TRUE);
#else
  P(native_blas_lapack, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
#ifdef NVSHMEM_COMMS
  P(use_mobius_fused_kernel, QUDA_BOOLEAN_FALSE);
#else
  P(use_mobius_fused_kernel, QUDA_BOOLEAN_TRUE);
#endif
#else
  P(use_mobius_fused_kernel, QUDA_BOOLEAN_INVALID);
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

#if defined CHECK_PARAM
   if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
     errorQuda("Unexpected QudaMultigridParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
     P(struct_size, (size_t)INVALID_INT);
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

#ifdef INIT_PARAM
  P(setup_type, QUDA_NULL_VECTOR_SETUP);
#else
  P(setup_type, QUDA_INVALID_SETUP_TYPE);
#endif

#ifdef INIT_PARAM
  P(pre_orthonormalize, QUDA_BOOLEAN_FALSE);
#else
  P(pre_orthonormalize, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  P(post_orthonormalize, QUDA_BOOLEAN_TRUE);
#else
  P(post_orthonormalize, QUDA_BOOLEAN_INVALID);
#endif

  for (int i=0; i<n_level; i++) {
#ifdef INIT_PARAM
    P(verbosity[i], QUDA_SILENT);
#else
    P(verbosity[i], QUDA_INVALID_VERBOSITY);
#endif
#ifdef INIT_PARAM
#ifdef QUDA_MMA_AVAILABLE
    P(setup_use_mma[i], QUDA_BOOLEAN_TRUE);
#else
    P(setup_use_mma[i], QUDA_BOOLEAN_FALSE);
#endif
    P(dslash_use_mma[i], QUDA_BOOLEAN_FALSE);
#else
    P(setup_use_mma[i], QUDA_BOOLEAN_INVALID);
    P(dslash_use_mma[i], QUDA_BOOLEAN_INVALID);
#endif
#ifdef INIT_PARAM
    P(setup_inv_type[i], QUDA_BICGSTAB_INVERTER);
#else
    P(setup_inv_type[i], QUDA_INVALID_INVERTER);
#endif
#ifdef INIT_PARAM
    P(num_setup_iter[i], 1);
#else
    P(num_setup_iter[i], INVALID_INT);
#endif
#ifdef INIT_PARAM
    P(use_eig_solver[i], QUDA_BOOLEAN_FALSE);
#else
    P(use_eig_solver[i], QUDA_BOOLEAN_INVALID);
#endif
#ifdef INIT_PARAM
    P(setup_tol[i], 5e-6);
    P(setup_maxiter[i], 500);
    P(setup_maxiter_refresh[i], 0);
#else
    P(setup_tol[i], INVALID_DOUBLE);
    P(setup_maxiter[i], INVALID_INT);
    P(setup_maxiter_refresh[i], INVALID_INT);
#endif

#ifdef INIT_PARAM
    P(setup_ca_basis[i], QUDA_POWER_BASIS);
    P(setup_ca_basis_size[i], 4);
    P(setup_ca_lambda_min[i], 0.0);
    P(setup_ca_lambda_max[i], -1.0);
#else
    P(setup_ca_basis[i], QUDA_INVALID_BASIS);
    P(setup_ca_basis_size[i], INVALID_INT);
    P(setup_ca_lambda_min[i], INVALID_DOUBLE);
    P(setup_ca_lambda_max[i], INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
    P(n_block_ortho[i], 1);
    P(block_ortho_two_pass[i], QUDA_BOOLEAN_TRUE);
#else
    P(n_block_ortho[i], INVALID_INT);
    P(block_ortho_two_pass[i], QUDA_BOOLEAN_INVALID);
#endif

    P(coarse_solver[i], QUDA_INVALID_INVERTER);
    P(coarse_solver_maxiter[i], INVALID_INT);
    P(smoother[i], QUDA_INVALID_INVERTER);
    P(smoother_solve_type[i], QUDA_INVALID_SOLVE);

#ifdef INIT_PARAM
    P(coarse_solver_ca_basis[i], QUDA_POWER_BASIS);
    P(coarse_solver_ca_basis_size[i], 4);
    P(coarse_solver_ca_lambda_min[i], 0.0);
    P(coarse_solver_ca_lambda_max[i], -1.0);
#else
    P(coarse_solver_ca_basis[i], QUDA_INVALID_BASIS);
    P(coarse_solver_ca_basis_size[i], INVALID_INT);
    P(coarse_solver_ca_lambda_min[i], INVALID_DOUBLE);
    P(coarse_solver_ca_lambda_max[i], INVALID_DOUBLE);
#endif

#ifndef CHECK_PARAM
    P(smoother_halo_precision[i], QUDA_INVALID_PRECISION);
    P(smoother_schwarz_type[i], QUDA_INVALID_SCHWARZ);
    P(smoother_schwarz_cycle[i], 1);
#else
    P(smoother_schwarz_cycle[i], INVALID_INT);
#endif

    // these parameters are not set for the bottom grid
    if (i<n_level-1) {
      for (int j=0; j<4; j++) P(geo_block_size[i][j], INVALID_INT);
      P(spin_block_size[i], INVALID_INT);
#ifdef INIT_PARAM
      P(precision_null[i], QUDA_SINGLE_PRECISION);
#else
      P(precision_null[i], INVALID_INT);
#endif
      P(cycle_type[i], QUDA_MG_CYCLE_INVALID);
      P(nu_pre[i], INVALID_INT);
      P(nu_post[i], INVALID_INT);
      P(coarse_grid_solution_type[i], QUDA_INVALID_SOLUTION);
    }

#ifdef INIT_PARAM
    P(smoother_solver_ca_basis[i], QUDA_POWER_BASIS);
    P(smoother_solver_ca_lambda_min[i], 0.0);
    P(smoother_solver_ca_lambda_max[i], -1.0);
#else
    P(smoother_solver_ca_basis[i], QUDA_INVALID_BASIS);
    P(smoother_solver_ca_lambda_min[i], INVALID_DOUBLE);
    P(smoother_solver_ca_lambda_max[i], INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
    if (i<QUDA_MAX_MG_LEVEL) {
          P(n_vec[i], INVALID_INT);
    }
#else
    if (i<n_level-1) {
      P(n_vec[i], INVALID_INT);
    }
#endif

#ifdef INIT_PARAM
    P(transfer_type[i], QUDA_TRANSFER_AGGREGATE);
#else
    P(transfer_type[i], QUDA_TRANSFER_INVALID);
#endif

#ifdef INIT_PARAM
    P(mu_factor[i], 1);
#else
    P(mu_factor[i], INVALID_DOUBLE);
#endif
    P(coarse_solver_tol[i], INVALID_DOUBLE);
    P(smoother_tol[i], INVALID_DOUBLE);
#ifdef INIT_PARAM
    P(global_reduction[i], QUDA_BOOLEAN_TRUE);
#else
    P(global_reduction[i], QUDA_BOOLEAN_INVALID);
#endif

    P(omega[i], INVALID_DOUBLE);
    P(location[i], QUDA_INVALID_FIELD_LOCATION);

#ifdef INIT_PARAM
    P(setup_location[i], QUDA_CUDA_FIELD_LOCATION);
#else
    P(setup_location[i], QUDA_INVALID_FIELD_LOCATION);
#endif
  }

#ifdef INIT_PARAM
  P(setup_minimize_memory, QUDA_BOOLEAN_FALSE);
#else
  P(setup_minimize_memory, QUDA_BOOLEAN_INVALID);
#endif

  P(compute_null_vector, QUDA_COMPUTE_NULL_VECTOR_INVALID);
  P(generate_all_levels, QUDA_BOOLEAN_INVALID);

#ifdef CHECK_PARAM
  // if only doing top-level null-space generation, check that n_vec
  // is equal on all levels
  if (param->generate_all_levels == QUDA_BOOLEAN_FALSE && param->compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
    for (int i=1; i<n_level-1; i++)
      if (param->n_vec[0] != param->n_vec[i])
	errorQuda("n_vec %d != %d must be equal on all levels if generate_all_levels == false",
		  param->n_vec[0], param->n_vec[i]);
  }
#endif

  P(run_verify, QUDA_BOOLEAN_INVALID);

#ifdef INIT_PARAM
  P(run_low_mode_check, QUDA_BOOLEAN_FALSE);
  P(run_oblique_proj_check, QUDA_BOOLEAN_FALSE);
  P(coarse_guess, QUDA_BOOLEAN_FALSE);
  P(preserve_deflation, QUDA_BOOLEAN_FALSE);
#else
  P(run_low_mode_check, QUDA_BOOLEAN_INVALID);
  P(run_oblique_proj_check, QUDA_BOOLEAN_INVALID);
  P(coarse_guess, QUDA_BOOLEAN_INVALID);
  P(preserve_deflation, QUDA_BOOLEAN_INVALID);
#endif

  for (int i = 0; i < n_level - 1; i++) {
#ifdef INIT_PARAM
    P(vec_load[i], QUDA_BOOLEAN_FALSE);
    P(vec_store[i], QUDA_BOOLEAN_FALSE);
#else
    P(vec_load[i], QUDA_BOOLEAN_INVALID);
    P(vec_store[i], QUDA_BOOLEAN_INVALID);
#endif
  }

#ifdef INIT_PARAM
  P(gflops, 0.0);
  P(secs, 0.0);
#elif defined(PRINT_PARAM)
  P(gflops, INVALID_DOUBLE);
  P(secs, INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
  P(allow_truncation, QUDA_BOOLEAN_FALSE);
#else
  P(allow_truncation, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  P(staggered_kd_dagger_approximation, QUDA_BOOLEAN_FALSE);
#else
  P(staggered_kd_dagger_approximation, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  P(thin_update_only, QUDA_BOOLEAN_FALSE);
#else
  P(thin_update_only, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
QudaGaugeObservableParam newQudaGaugeObservableParam(void)
{
  QudaGaugeObservableParam ret;
#elif defined CHECK_PARAM
static void checkGaugeObservableParam(QudaGaugeObservableParam *param)
{
#else
void printQudaGaugeObservableParam(QudaGaugeObservableParam *param)
{
  printfQuda("QUDA Gauge-Observable Parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaGaugeObervableParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(su_project, QUDA_BOOLEAN_FALSE);
  P(compute_plaquette, QUDA_BOOLEAN_FALSE);
  P(compute_polyakov_loop, QUDA_BOOLEAN_FALSE);
  P(compute_gauge_loop_trace, QUDA_BOOLEAN_FALSE);
  P(traces, nullptr);
  P(input_path_buff, nullptr);
  P(path_length, nullptr);
  P(loop_coeff, nullptr);
  P(num_paths, INVALID_INT);
  P(max_length, INVALID_INT);
  P(factor, INVALID_DOUBLE);
  P(compute_qcharge, QUDA_BOOLEAN_FALSE);
  P(compute_qcharge_density, QUDA_BOOLEAN_FALSE);
  P(qcharge_density, nullptr);
  P(remove_staggered_phase, QUDA_BOOLEAN_FALSE);
#else
  P(su_project, QUDA_BOOLEAN_INVALID);
  P(compute_plaquette, QUDA_BOOLEAN_INVALID);
  P(compute_polyakov_loop, QUDA_BOOLEAN_INVALID);
  P(compute_gauge_loop_trace, QUDA_BOOLEAN_INVALID);
  if (param->compute_gauge_loop_trace == QUDA_BOOLEAN_TRUE) {
    P(num_paths, INVALID_INT);
    P(max_length, INVALID_INT);
    P(factor, INVALID_DOUBLE);
  }
  P(compute_qcharge, QUDA_BOOLEAN_INVALID);
  P(compute_qcharge_density, QUDA_BOOLEAN_INVALID);
  P(remove_staggered_phase, QUDA_BOOLEAN_INVALID);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
QudaGaugeSmearParam newQudaGaugeSmearParam(void)
{
  QudaGaugeSmearParam ret;
#elif defined CHECK_PARAM
static void checkGaugeSmearParam(QudaGaugeSmearParam *param)
{
#else
void printQudaGaugeSmearParam(QudaGaugeSmearParam *param)
{
  printfQuda("QUDA Gauge Smear Parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaGaugeSmearParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

  P(smear_type, QUDA_GAUGE_SMEAR_INVALID);

#ifdef INIT_PARAM
  P(n_steps, 0);
  P(meas_interval, 0);
  P(alpha, 0.0);
  P(rho, 0.0);
  P(epsilon, 0.0);
#else
  P(n_steps, (unsigned int)INVALID_INT);
  P(meas_interval, (unsigned int)INVALID_INT);
  P(alpha, INVALID_DOUBLE);
  P(rho, INVALID_DOUBLE);
  P(epsilon, INVALID_DOUBLE);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
QudaBLASParam newQudaBLASParam(void)
{
  QudaBLASParam ret;
#elif defined CHECK_PARAM
static void checkBLASParam(QudaBLASParam *param)
{
#else
void printQudaBLASParam(QudaBLASParam *param)
{
  printfQuda("QUDA blas parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaBLASParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(trans_a, QUDA_BLAS_OP_N);
  P(trans_b, QUDA_BLAS_OP_N);
  P(m, INVALID_INT);
  P(n, INVALID_INT);
  P(k, INVALID_INT);
  P(lda, INVALID_INT);
  P(ldb, INVALID_INT);
  P(ldc, INVALID_INT);
  P(a_offset, 0);
  P(b_offset, 0);
  P(c_offset, 0);
  P(a_stride, 1);
  P(b_stride, 1);
  P(c_stride, 1);
  P(batch_count, 1);
  P(data_type, QUDA_BLAS_DATATYPE_S);
  P(data_order, QUDA_BLAS_DATAORDER_ROW);
  P(blas_type, QUDA_BLAS_INVALID);
  P(inv_mat_size, INVALID_INT);
#else
  P(trans_a, QUDA_BLAS_OP_INVALID);
  P(trans_b, QUDA_BLAS_OP_INVALID);
  P(m, INVALID_INT);
  P(n, INVALID_INT);
  P(k, INVALID_INT);
  P(lda, INVALID_INT);
  P(ldb, INVALID_INT);
  P(ldc, INVALID_INT);
  P(a_offset, INVALID_INT);
  P(b_offset, INVALID_INT);
  P(c_offset, INVALID_INT);
  P(a_stride, INVALID_INT);
  P(b_stride, INVALID_INT);
  P(c_stride, INVALID_INT);
  P(batch_count, INVALID_INT);
  P(data_type, QUDA_BLAS_DATATYPE_INVALID);
  P(data_order, QUDA_BLAS_DATAORDER_INVALID);
  P(blas_type, QUDA_BLAS_INVALID);
  P(inv_mat_size, INVALID_INT);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

// clean up

#undef INVALID_INT
#undef INVALID_DOUBLE
#undef P
