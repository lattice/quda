// This file defines functions to either initialize, check, or print
// the QUDA gauge and inverter parameters.  It gets included in
// interface_quda.cpp, after either INIT_PARAM, CHECK_PARAM, or
// PRINT_PARAM is defined.
//
// If you're reading this file because it was mentioned in a "QUDA
// error" message, it probably means that you forgot to set one of the
// gauge or inverter parameters in your application before calling
// loadGaugeQuda() or invertQuda().

#include "check_params.h"

QudaGaugeParam newQudaGaugeParam(void) {
  QudaGaugeParam param;
  paramType type = INIT;
  parseQudaGaugeParam(&param, type);
  return param;
}

void checkGaugeParam(QudaGaugeParam *param) 
{
  paramType type = CHECK; 
  parseQudaGaugeParam(param, type);
}

void printQudaGaugeParam(QudaGaugeParam *param) 
{
  printfQuda("QUDA Gauge Parameters:\n");
  paramType type = PRINT; 
  parseQudaGaugeParam(param, type);
}


void parseQudaGaugeParam(QudaGaugeParam *param, paramType type) 
{
  if(type == INIT) {
    P(param, location, QUDA_CPU_FIELD_LOCATION, type);
  } else {
    P(param, location, QUDA_INVALID_FIELD_LOCATION, type);
  }

  for (int i=0; i<4; i++) P(param, X[i], INVALID_INT, type);
  
  if(type == INIT) {
    P(param, anisotropy, INVALID_DOUBLE, type);
    P(param, tadpole_coeff, INVALID_DOUBLE, type);
    P(param, scale, 1.0, type);
  } else {
    if (param->type == QUDA_WILSON_LINKS) {
      P(param, anisotropy, INVALID_DOUBLE, type);
    } else if (param->type == QUDA_ASQTAD_LONG_LINKS) {
      P(param, tadpole_coeff, INVALID_DOUBLE, type);
      P(param, scale, INVALID_DOUBLE, type);
    }
  }
    
  P(param, type, QUDA_INVALID_LINKS, type);
  P(param, gauge_order, QUDA_INVALID_GAUGE_ORDER, type);
  P(param, t_boundary, QUDA_INVALID_T_BOUNDARY, type);
  P(param, cpu_prec, QUDA_INVALID_PRECISION, type);
  P(param, cuda_prec, QUDA_INVALID_PRECISION, type);
  P(param, reconstruct, QUDA_RECONSTRUCT_INVALID, type);

  if(type != CHECK) {
    P(param, cuda_prec_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, reconstruct_sloppy, QUDA_RECONSTRUCT_INVALID, type);
    P(param, cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, reconstruct_refinement_sloppy, QUDA_RECONSTRUCT_INVALID, type);
    P(param, cuda_prec_precondition, QUDA_INVALID_PRECISION, type);
    P(param, reconstruct_precondition, QUDA_RECONSTRUCT_INVALID, type);
  } else {
    if (param->cuda_prec_sloppy == QUDA_INVALID_PRECISION)
      param->cuda_prec_sloppy = param->cuda_prec;
    if (param->reconstruct_sloppy == QUDA_RECONSTRUCT_INVALID)
      param->reconstruct_sloppy = param->reconstruct;
    if (param->cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
      param->cuda_prec_refinement_sloppy = param->cuda_prec_sloppy;
    if (param->reconstruct_refinement_sloppy == QUDA_RECONSTRUCT_INVALID)
      param->reconstruct_refinement_sloppy = param->reconstruct_sloppy;
    if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
      param->cuda_prec_precondition = param->cuda_prec_sloppy;
    if (param->reconstruct_precondition == QUDA_RECONSTRUCT_INVALID)
      param->reconstruct_precondition = param->reconstruct_sloppy;
  }

  P(param, gauge_fix, QUDA_GAUGE_FIXED_INVALID, type);
  P(param, ga_pad, INVALID_INT, type);

  if(type == INIT) {
    P(param, staggered_phase_type, QUDA_STAGGERED_PHASE_NO, type);
    P(param, staggered_phase_applied, 0, type);
    P(param, i_mu, 0.0, type);
    P(param, overlap, 0, type);
  } else {
    P(param, staggered_phase_type, QUDA_STAGGERED_PHASE_INVALID, type);
    P(param, staggered_phase_applied, INVALID_INT, type);
    P(param, i_mu, INVALID_DOUBLE, type);
    P(param, overlap, INVALID_INT, type);
  }

  if(type == INIT) { 
    P(param, overwrite_mom, 0, type);
    P(param, use_resident_gauge, 0, type);
    P(param, use_resident_mom, 0, type);
    P(param, make_resident_gauge, 0, type);
    P(param, make_resident_mom, 0, type);
    P(param, return_result_gauge, 1, type);
    P(param, return_result_mom, 1, type);
    P(param, gauge_offset, 0, type);
    P(param, mom_offset, 0, type);
    P(param, site_size, 0, type);
  } else {
    P(param, overwrite_mom, INVALID_INT, type);
    P(param, use_resident_gauge, INVALID_INT, type);
    P(param, use_resident_mom, INVALID_INT, type);
    P(param, make_resident_gauge, INVALID_INT, type);
    P(param, make_resident_mom, INVALID_INT, type);
    P(param, return_result_gauge, INVALID_INT, type);
    P(param, return_result_mom, INVALID_INT, type);
    P(param, gauge_offset, (size_t)INVALID_INT, type);
    P(param, mom_offset, (size_t)INVALID_INT, type);
    P(param, site_size, (size_t)INVALID_INT, type);
  }
}

// define the appropriate function for EigParam




QudaEigParam newQudaEigParam(void) 
{
  QudaEigParam param;
  paramType type = INIT;
  parseQudaEigParam(&param, type);
  return param;
}

void checkEigParam(QudaEigParam *param) 
{
  paramType type = CHECK; 
  parseQudaEigParam(param, type);
}

void printQudaEigParam(QudaEigParam *param) 
{
  printfQuda("QUDA Eig Parameters:\n");
  paramType type = PRINT; 
  parseQudaEigParam(param, type);
}

void parseQudaEigParam(QudaEigParam *param, paramType type) 
{
  if(type == INIT) {
    P(param, use_poly_acc, QUDA_BOOLEAN_FALSE, type);
    P(param, poly_deg, 0, type);
    P(param, a_min, 0.0, type);
    P(param, a_max, 0.0, type);
    P(param, preserve_deflation, QUDA_BOOLEAN_FALSE, type);
    //P(param, preserve_deflation_space, 0, type);
    P(param, preserve_evals, QUDA_BOOLEAN_TRUE, type);
    P(param, use_dagger, QUDA_BOOLEAN_FALSE, type);
    P(param, use_norm_op, QUDA_BOOLEAN_FALSE, type);
    P(param, compute_svd, QUDA_BOOLEAN_FALSE, type);
    P(param, require_convergence, QUDA_BOOLEAN_TRUE, type);
    P(param, spectrum, QUDA_SPECTRUM_LR_EIG, type);
    P(param, n_ev, 0, type);
    P(param, n_kr, 0, type);
    P(param, n_conv, 0, type);
    P(param, n_ev_deflate, -1, type);
    P(param, batched_rotate, 0, type);
    P(param, tol, 0.0, type);
    P(param, check_interval, 0, type);
    P(param, max_restarts, 0, type);
    P(param, arpack_check, QUDA_BOOLEAN_FALSE, type);
    P(param, nk, 0, type);
    P(param, np, 0, type);
    P(param, eig_type, QUDA_EIG_TR_LANCZOS, type);
    P(param, extlib_type, QUDA_EIGEN_EXTLIB, type);
    P(param, mem_type_ritz, QUDA_MEMORY_DEVICE, type);
    P(param, location, QUDA_CUDA_FIELD_LOCATION, type);
    P(param, io_parity_inflate, QUDA_BOOLEAN_FALSE, type);
  } else {
    P(param, use_poly_acc, QUDA_BOOLEAN_INVALID, type);
    P(param, poly_deg, INVALID_INT, type);
    P(param, a_min, INVALID_DOUBLE, type);
    P(param, a_max, INVALID_DOUBLE, type);
    P(param, preserve_deflation, QUDA_BOOLEAN_INVALID, type);
    P(param, preserve_evals, QUDA_BOOLEAN_INVALID, type);
    P(param, use_dagger, QUDA_BOOLEAN_INVALID, type);
    P(param, use_norm_op, QUDA_BOOLEAN_INVALID, type);
    P(param, compute_svd, QUDA_BOOLEAN_INVALID, type);
    P(param, require_convergence, QUDA_BOOLEAN_INVALID, type);
    P(param, n_ev, INVALID_INT, type);
    P(param, n_kr, INVALID_INT, type);
    P(param, n_conv, INVALID_INT, type);
    P(param, n_ev_deflate, INVALID_INT, type);
    P(param, batched_rotate, INVALID_INT, type);
    P(param, tol, INVALID_DOUBLE, type);
    P(param, check_interval, INVALID_INT, type);
    P(param, max_restarts, INVALID_INT, type);
    P(param, arpack_check, QUDA_BOOLEAN_INVALID, type);
    P(param, nk, INVALID_INT, type);
    P(param, np, INVALID_INT, type);
    P(param, eig_type, QUDA_EIG_INVALID, type);
    P(param, extlib_type, QUDA_EXTLIB_INVALID, type);
    P(param, mem_type_ritz, QUDA_MEMORY_INVALID, type);
    P(param, location, QUDA_INVALID_FIELD_LOCATION, type);
    P(param, io_parity_inflate, QUDA_BOOLEAN_INVALID, type);
  }
    
  // only need to enfore block size checking if doing a block eigen solve
  if(type == CHECK && param->eig_type == QUDA_EIG_BLK_TR_LANCZOS)
    P(param, block_size, INVALID_INT, type);
}


// define the appropriate function for clover subset from InvertParam


void newQudaCloverParam(QudaInvertParam *param) {
  paramType type = INIT;
  parseQudaCloverParam(param, type);
}

void checkCloverParam(QudaInvertParam *param) 
{
  paramType type = CHECK; 
  parseQudaCloverParam(param, type);
}

void printQudaCloverParam(QudaInvertParam *param) 
{
  printfQuda("QUDA Clover Parameters:\n");
  paramType type = PRINT; 
  parseQudaCloverParam(param, type);
}


void parseQudaCloverParam(QudaInvertParam *param, paramType type) 
{
  if(type == INIT) {
    P(param, clover_location, QUDA_CPU_FIELD_LOCATION, type);
  } else {
    P(param, clover_location, QUDA_INVALID_FIELD_LOCATION, type);
  }

  P(param, clover_cpu_prec, QUDA_INVALID_PRECISION, type);
  P(param, clover_cuda_prec, QUDA_INVALID_PRECISION, type);


  if(type != CHECK) {
    P(param, clover_cuda_prec_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, clover_cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, clover_cuda_prec_precondition, QUDA_INVALID_PRECISION, type);
  } else {
    if (param->clover_cuda_prec_sloppy == QUDA_INVALID_PRECISION)
      param->clover_cuda_prec_sloppy = param->clover_cuda_prec;
    if (param->clover_cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
      param->clover_cuda_prec_refinement_sloppy = param->clover_cuda_prec_sloppy;
    if (param->clover_cuda_prec_precondition == QUDA_INVALID_PRECISION)
      param->clover_cuda_prec_precondition = param->clover_cuda_prec_sloppy;
  }
    
  if(type == INIT) {
    P(param, compute_clover_trlog, 0, type);
    P(param, compute_clover, 0, type);
    P(param, compute_clover_inverse, 0, type);
    P(param, return_clover, 0, type);
    P(param, return_clover_inverse, 0, type);
    P(param, clover_rho, 0.0, type);
  } else {
    P(param, compute_clover_trlog, QUDA_INVALID_PRECISION, type);
    P(param, compute_clover, QUDA_INVALID_PRECISION, type);
    P(param, compute_clover_inverse, QUDA_INVALID_PRECISION, type);
    P(param, return_clover, QUDA_INVALID_PRECISION, type);
    P(param, return_clover_inverse, QUDA_INVALID_PRECISION, type);
    P(param, clover_rho, INVALID_DOUBLE, type);
  }

  P(param, clover_order, QUDA_INVALID_CLOVER_ORDER, type);
  P(param, cl_pad, INVALID_INT, type);  
  P(param, clover_coeff, INVALID_DOUBLE, type);
}

// define the appropriate function for InvertParam



QudaInvertParam newQudaInvertParam(void)
{
  QudaInvertParam ret;
  QudaInvertParam *param=&ret;
  paramType type = INIT;
  parseQudaInvertParam(param, type);
  return ret;
}

void checkInvertParam(QudaInvertParam *param, void *out_ptr, void *in_ptr) 
{
  paramType type = CHECK;
  if (in_ptr && quda::get_pointer_location(in_ptr) != param->input_location) {
    warningQuda("input_location=%d, however supplied pointer is location=%d", param->input_location, quda::get_pointer_location(in_ptr));
    param->input_location = quda::get_pointer_location(in_ptr);
  }
  
  if (out_ptr && quda::get_pointer_location(out_ptr) != param->output_location) {
    warningQuda("output_location=%d, however supplied pointer is location=%d", param->output_location, quda::get_pointer_location(out_ptr));
    param->output_location = quda::get_pointer_location(out_ptr);
  }  
  parseQudaInvertParam(param, type);
}

void printQudaInvertParam(QudaInvertParam *param) 
{
  printfQuda("QUDA Inverter Parameters:\n");
  paramType type = PRINT;
  parseQudaInvertParam(param, type); 
}

void parseQudaInvertParam(QudaInvertParam *param, paramType type) {
  
  P(param, dslash_type, QUDA_INVALID_DSLASH, type);
  P(param, inv_type, QUDA_INVALID_INVERTER, type);
  
  if(type == INIT) {
    P(param, mass, INVALID_DOUBLE, type);
    P(param, kappa, INVALID_DOUBLE, type);
    P(param, m5, INVALID_DOUBLE, type);
    P(param, Ls, INVALID_INT, type);
    P(param, mu, INVALID_DOUBLE, type);
    P(param, twist_flavor, QUDA_TWIST_INVALID, type);
    P(param, laplace3D, INVALID_INT, type);
    P(param, residual_type, QUDA_L2_RELATIVE_RESIDUAL, type);
    P(param, use_alternative_reliable, 0, type); /**< Default is to not use alternative relative updates, e.g., use delta to determine reliable trigger */
    P(param, use_sloppy_partial_accumulator, 0, type); /**< Default is to use a high-precision accumulator (not yet supported in all solvers) */
    P(param, solution_accumulator_pipeline, 1, type); /**< Default is solution accumulator depth of 1 */
    P(param, max_res_increase, 1, type); /**< Default is to allow one consecutive residual increase */
    P(param, max_res_increase_total, 10, type); /**< Default is to allow ten residual increase */
    P(param, max_hq_res_increase, 1, type);     /**< Default is to allow one consecutive heavy-quark residual increase */
    P(param, max_hq_res_restart_total, 10, type); /**< Default is to allow ten heavy-quark restarts */
    P(param, heavy_quark_check, 10, type); /**< Default is to update heavy quark residual after 10 iterations */
    P(param, compute_action, 0, type);
    P(param, compute_true_res, 1, type);
  } else {
    // asqtad and domain wall use mass parameterization
    if (param->dslash_type == QUDA_STAGGERED_DSLASH || param->dslash_type == QUDA_ASQTAD_DSLASH
	|| param->dslash_type == QUDA_DOMAIN_WALL_DSLASH || param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
	|| param->dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      P(param, mass, INVALID_DOUBLE, type);
    } else { // Wilson and clover use kappa parameterization
      P(param, kappa, INVALID_DOUBLE, type);
    }
    if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
	param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
	param->dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
      P(param, m5, INVALID_DOUBLE, type);
      P(param, Ls, INVALID_INT, type);
    }
    if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      P(param, mu, INVALID_DOUBLE, type);
      P(param, twist_flavor, QUDA_TWIST_INVALID, type);
    }
    P(param, residual_type, QUDA_INVALID_RESIDUAL, type);
    P(param, use_alternative_reliable, INVALID_INT, type);
    P(param, use_sloppy_partial_accumulator, INVALID_INT, type);
    P(param, solution_accumulator_pipeline, INVALID_INT, type);
    P(param, max_res_increase, INVALID_INT, type);
    P(param, max_res_increase_total, INVALID_INT, type);
    P(param, max_hq_res_increase, INVALID_INT, type);
    P(param, max_hq_res_restart_total, INVALID_INT, type);
    P(param, heavy_quark_check, INVALID_INT, type);    
    P(param, compute_action, INVALID_INT, type);
    P(param, compute_true_res, INVALID_INT, type);
  }
  
  
  P(param, tol, INVALID_DOUBLE, type);
  if (param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
    P(param, tol_hq, INVALID_DOUBLE, type);
  }
  P(param, maxiter, INVALID_INT, type);
  P(param, reliable_delta, INVALID_DOUBLE, type);

  if(type != CHECK) {
    P(param, reliable_delta_refinement, INVALID_DOUBLE, type);
    P(param, pipeline, 0, type); /** Whether to use a pipelined solver */
    P(param, num_offset, 0, type); /**< Number of offsets in the multi-shift solver */
    P(param, num_src, 1, type); /**< Number of offsets in the multi-shift solver */
    P(param, overlap, 0, type); /**< width of domain overlaps */    
  } else {
    if (param->reliable_delta_refinement == INVALID_DOUBLE) param->reliable_delta_refinement = param->reliable_delta;
  }
  
  if (param->num_offset > 0) {
    
    for (int i=0; i<param->num_offset; i++) {
      P(param, offset[i], INVALID_DOUBLE, type);
      P(param, tol_offset[i], INVALID_DOUBLE, type);
      if (param->residual_type & QUDA_HEAVY_QUARK_RESIDUAL)
	P(param, tol_hq_offset[i], INVALID_DOUBLE, type);
      if(type != CHECK) {
	P(param, true_res_offset[i], INVALID_DOUBLE, type);
	P(param, iter_res_offset[i], INVALID_DOUBLE, type);
      }
      if (param->compute_action) P(param, residue[i], INVALID_DOUBLE, type);
    }
    if(type != CHECK) {
      P(param, action[0], INVALID_DOUBLE, type);
      P(param, action[1], INVALID_DOUBLE, type);
    }
  }

  P(param, solution_type, QUDA_INVALID_SOLUTION, type);
  P(param, solve_type, QUDA_INVALID_SOLVE, type);
  P(param, matpc_type, QUDA_MATPC_INVALID, type);
  P(param, dagger, QUDA_DAG_INVALID, type);
  P(param, mass_normalization, QUDA_INVALID_NORMALIZATION, type);

  if(type != CHECK) P(param, solver_normalization, QUDA_DEFAULT_NORMALIZATION, type);
  P(param, preserve_source, QUDA_PRESERVE_SOURCE_INVALID, type);
  P(param, cpu_prec, QUDA_INVALID_PRECISION, type);
  P(param, cuda_prec, QUDA_INVALID_PRECISION, type);
  
  if(type != CHECK) { 
    P(param, cuda_prec_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, cuda_prec_refinement_sloppy, QUDA_INVALID_PRECISION, type);
    P(param, cuda_prec_precondition, QUDA_INVALID_PRECISION, type);
  } else {
    if (param->cuda_prec_sloppy == QUDA_INVALID_PRECISION)
      param->cuda_prec_sloppy = param->cuda_prec;
    if (param->cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION)
      param->cuda_prec_refinement_sloppy = param->cuda_prec_sloppy;
    if (param->cuda_prec_precondition == QUDA_INVALID_PRECISION)
      param->cuda_prec_precondition = param->cuda_prec_sloppy;
  }
  
  // leave the default behaviour to cpu pointers
  if(type == INIT) {
    P(param, input_location, QUDA_CPU_FIELD_LOCATION, type);
    P(param, output_location, QUDA_CPU_FIELD_LOCATION, type);
    P(param, clover_location, QUDA_CPU_FIELD_LOCATION, type);
  } else {
    P(param, input_location, QUDA_INVALID_FIELD_LOCATION, type);
    P(param, output_location, QUDA_INVALID_FIELD_LOCATION, type);
    P(param, clover_location, QUDA_INVALID_FIELD_LOCATION, type);
  }
  
  
  P(param, gamma_basis, QUDA_INVALID_GAMMA_BASIS, type);
  P(param, dirac_order, QUDA_INVALID_DIRAC_ORDER, type);
  P(param, sp_pad, INVALID_INT, type);
  
  if(type == INIT) { 
    P(param, Nsteps, INVALID_INT, type);
  } else {
    if(param->inv_type == QUDA_MPCG_INVERTER || param->inv_type == QUDA_MPBICGSTAB_INVERTER){
      P(param, Nsteps, INVALID_INT, type);
    }
  }

  if(type == INIT) { 
    P(param, gcrNkrylov, INVALID_INT, type);
  } else {
    if (param->inv_type == QUDA_GCR_INVERTER ||
      param->inv_type == QUDA_CA_GCR_INVERTER ||
      param->inv_type == QUDA_CA_CG_INVERTER ||
	param->inv_type == QUDA_CA_CGNE_INVERTER ||
	param->inv_type == QUDA_CA_CGNR_INVERTER) {
      P(param, gcrNkrylov, INVALID_INT, type);
    }
  }

  // domain decomposition parameters
  //P(param, inv_type_sloppy, QUDA_INVALID_INVERTER, type); // disable since invalid means no preconditioner
  if(type == INIT) { 
    P(param, inv_type_precondition, QUDA_INVALID_INVERTER, type);
    //P(param, preconditioner, 0, type);
    P(param, tol_precondition, INVALID_DOUBLE, type);
    P(param, maxiter_precondition, INVALID_INT, type);
    P(param, verbosity_precondition, QUDA_INVALID_VERBOSITY, type);
    P(param, schwarz_type, QUDA_INVALID_SCHWARZ, type);
    P(param, precondition_cycle, 1, type);               // defaults match previous interface behaviour
  } else {
    if (param->inv_type_precondition == QUDA_BICGSTAB_INVERTER || param->inv_type_precondition == QUDA_CG_INVERTER
	|| param->inv_type_precondition == QUDA_MR_INVERTER) {
      P(param, tol_precondition, INVALID_DOUBLE, type);
      P(param, maxiter_precondition, INVALID_INT, type);
      P(param, verbosity_precondition, QUDA_INVALID_VERBOSITY, type);
      P(param, precondition_cycle, 0, type);
    }
  }

  if(type == INIT) { 
    //P(param, eig_param, 0, type);
  } else if(type == CHECK) {
    if (param->eig_param && param->inv_type_precondition != QUDA_INVALID_INVERTER) {
      errorQuda("At present cannot combine deflation with Schwarz preconditioner");
    }
  }
  
  if(type == INIT) { 
    P(param, use_init_guess, QUDA_USE_INIT_GUESS_NO, type); //set the default to no
    P(param, omega, 1.0, type); // set default to no relaxation
  } else {
    P(param, use_init_guess, QUDA_USE_INIT_GUESS_INVALID, type);
    P(param, omega, INVALID_DOUBLE, type);
  }

  if(type == INIT) { 
    newQudaCloverParam(param);
  } else if (type == CHECK) {
    checkCloverParam(param);
  } else {
    printQudaCloverParam(param);
  }

  if(type == INIT) { 
    P(param, ca_basis, QUDA_POWER_BASIS, type);
    P(param, ca_lambda_min, 0.0, type);
    P(param, ca_lambda_max, -1.0, type);
  } else {
    if (param->inv_type == QUDA_CA_CG_INVERTER ||
	param->inv_type == QUDA_CA_CGNE_INVERTER ||
	param->inv_type == QUDA_CA_CGNR_INVERTER) {
      P(param, ca_basis, QUDA_INVALID_BASIS, type);
      if (param->ca_basis == QUDA_CHEBYSHEV_BASIS) {
	P(param, ca_lambda_min, INVALID_DOUBLE, type);
	P(param, ca_lambda_max, INVALID_DOUBLE, type);
      }
    }
  }

  P(param, verbosity, QUDA_INVALID_VERBOSITY, type);

  if(type == INIT) {
    P(param, iter, 0, type);
    P(param, gflops, 0.0, type);
    P(param, secs, 0.0, type);
  } else if (type == PRINT) { 
    P(param, iter, INVALID_INT, type);
    P(param, gflops, INVALID_DOUBLE, type);
    P(param, secs, INVALID_DOUBLE, type);
  }

  if(type == INIT) { 
    P(param, cuda_prec_ritz, QUDA_SINGLE_PRECISION, type);
    P(param, n_ev, 8, type);
    P(param, max_search_dim, 64, type);
    P(param, rhs_idx, 0, type);
    P(param, deflation_grid, 1, type);
    
    P(param, eigcg_max_restarts, 4, type);
    P(param, max_restart_num, 3, type);
    P(param, tol_restart,5e-5, type);
    P(param, inc_tol, 1e-2, type);
    P(param, eigenval_tol, 1e-1, type);
  } else {
    P(param, cuda_prec_ritz, QUDA_INVALID_PRECISION, type);
    P(param, n_ev, INVALID_INT, type);
    P(param, max_search_dim, INVALID_INT, type);
    P(param, rhs_idx, INVALID_INT, type);
    P(param, deflation_grid, INVALID_INT, type);
    P(param, eigcg_max_restarts, INVALID_INT, type);
    P(param, max_restart_num, INVALID_INT, type);
    P(param, tol_restart,INVALID_DOUBLE, type);
    P(param, inc_tol, INVALID_DOUBLE, type);
    P(param, eigenval_tol, INVALID_DOUBLE, type);
  }
  
  if(type == INIT) { 
    P(param, use_resident_solution, 0, type);
    P(param, make_resident_solution, 0, type);
  } else {
    P(param, use_resident_solution, INVALID_INT, type);
    P(param, make_resident_solution, INVALID_INT, type);
  }

  if(type == INIT) { 
    P(param, chrono_use_resident, 0, type);
    P(param, chrono_make_resident, 0, type);
    P(param, chrono_replace_last, 0, type);
    P(param, chrono_max_dim, 0, type);
    P(param, chrono_index, 0, type);
  } else {
    P(param, chrono_use_resident, INVALID_INT, type);
    P(param, chrono_make_resident, INVALID_INT, type);
    P(param, chrono_replace_last, INVALID_INT, type);
    P(param, chrono_max_dim, INVALID_INT, type);
    P(param, chrono_index, INVALID_INT, type);
  }


  if(type != CHECK) {
    P(param, chrono_precision, QUDA_INVALID_PRECISION, type);
  } else {
    // default the chrono precision to using outer precision
    if (param->chrono_precision == QUDA_INVALID_PRECISION) param->chrono_precision = param->cuda_prec;
  }

  if(type == INIT) { 
    P(param, extlib_type, QUDA_EIGEN_EXTLIB, type);
  } else { 
    P(param, extlib_type, QUDA_EXTLIB_INVALID, type);
  }
}




QudaMultigridParam newQudaMultigridParam(void) {
  QudaMultigridParam param;
  paramType type = INIT;
  parseQudaMultigridParam(&param, type);
  return param;
}

void checkMultigridParam(QudaMultigridParam *param) {
  paramType type = CHECK;
  parseQudaMultigridParam(param, type);
}

void printQudaMultigridParam(QudaMultigridParam *param) {
  printfQuda("QUDA Multigrid Parameters:\n");
  paramType type = PRINT;
  parseQudaMultigridParam(param, type);
}

void parseQudaMultigridParam(QudaMultigridParam *param, paramType type)
{
  if(type == INIT) {}
  else if (type == CHECK) checkInvertParam(param->invert_param);
  else printQudaInvertParam(param->invert_param);
  
  P(param, n_level, INVALID_INT, type);

  int n_level = 0;
  if(type == INIT) {
    n_level = QUDA_MAX_MG_LEVEL;
    P(param, setup_type, QUDA_NULL_VECTOR_SETUP, type);
    P(param, pre_orthonormalize, QUDA_BOOLEAN_FALSE, type);
    P(param, post_orthonormalize, QUDA_BOOLEAN_TRUE, type);
  } else {
    n_level = param->n_level;
    P(param, setup_type, QUDA_INVALID_SETUP_TYPE, type);
    P(param, pre_orthonormalize, QUDA_BOOLEAN_INVALID, type);
    P(param, post_orthonormalize, QUDA_BOOLEAN_INVALID, type);
  }


  for (int i=0; i<n_level; i++) {
    if(type == INIT) { 
      P(param, verbosity[i], QUDA_SILENT, type);
      P(param, setup_inv_type[i], QUDA_BICGSTAB_INVERTER, type);
      P(param, num_setup_iter[i], 1, type);
      P(param, use_eig_solver[i], QUDA_BOOLEAN_FALSE, type);
      P(param, setup_tol[i], 5e-6, type);
      P(param, setup_maxiter[i], 500, type);
      P(param, setup_maxiter_refresh[i], 0, type);
      P(param, setup_ca_basis[i], QUDA_POWER_BASIS, type);
      P(param, setup_ca_basis_size[i], 4, type);
      P(param, setup_ca_lambda_min[i], 0.0, type);
      P(param, setup_ca_lambda_max[i], -1.0, type);
      P(param, n_block_ortho[i], 1, type);
      P(param, coarse_solver_ca_basis[i], QUDA_POWER_BASIS, type);
      P(param, coarse_solver_ca_basis_size[i], 4, type);
      P(param, coarse_solver_ca_lambda_min[i], 0.0, type);
      P(param, coarse_solver_ca_lambda_max[i], -1.0, type);
      
    } else {
      P(param, verbosity[i], QUDA_INVALID_VERBOSITY, type);
      P(param, setup_inv_type[i], QUDA_INVALID_INVERTER, type);
      P(param, num_setup_iter[i], INVALID_INT, type);
      P(param, use_eig_solver[i], QUDA_BOOLEAN_INVALID, type);
      P(param, setup_tol[i], INVALID_DOUBLE, type);
      P(param, setup_maxiter[i], INVALID_INT, type);
      P(param, setup_maxiter_refresh[i], INVALID_INT, type);
      P(param, setup_ca_basis[i], QUDA_INVALID_BASIS, type);
      P(param, setup_ca_basis_size[i], INVALID_INT, type);
      P(param, setup_ca_lambda_min[i], INVALID_DOUBLE, type);
      P(param, setup_ca_lambda_max[i], INVALID_DOUBLE, type);
      P(param, n_block_ortho[i], INVALID_INT, type);
      P(param, coarse_solver_ca_basis[i], QUDA_INVALID_BASIS, type);
      P(param, coarse_solver_ca_basis_size[i], INVALID_INT, type);
      P(param, coarse_solver_ca_lambda_min[i], INVALID_DOUBLE, type);
      P(param, coarse_solver_ca_lambda_max[i], INVALID_DOUBLE, type);      
    }

    P(param, coarse_solver[i], QUDA_INVALID_INVERTER, type);
    P(param, coarse_solver_maxiter[i], INVALID_INT, type);
    P(param, smoother[i], QUDA_INVALID_INVERTER, type);
    P(param, smoother_solve_type[i], QUDA_INVALID_SOLVE, type);

    if(type != CHECK) {
      P(param, smoother_halo_precision[i], QUDA_INVALID_PRECISION, type);
      P(param, smoother_schwarz_type[i], QUDA_INVALID_SCHWARZ, type);
      P(param, smoother_schwarz_cycle[i], 1, type);
    } else {
      P(param, smoother_schwarz_cycle[i], INVALID_INT, type);
    }

    // these parameters are not set for the bottom grid
    if (i<n_level-1) {
      for (int j=0; j<4; j++) P(param, geo_block_size[i][j], INVALID_INT, type);
      P(param, spin_block_size[i], INVALID_INT, type);

      if(type == INIT) {
	P(param, precision_null[i], QUDA_SINGLE_PRECISION, type);
      } else {
	//P(param, precision_null[i], INVALID_INT, type);
      }
      P(param, cycle_type[i], QUDA_MG_CYCLE_INVALID, type);
      P(param, nu_pre[i], INVALID_INT, type);
      P(param, nu_post[i], INVALID_INT, type);
      P(param, coarse_grid_solution_type[i], QUDA_INVALID_SOLUTION, type);
    }

    if(type == INIT) {
      if (i<QUDA_MAX_MG_LEVEL) {
	P(param, n_vec[i], INVALID_INT, type);
      }
    } else { 
      if (i<n_level-1) {
	P(param, n_vec[i], INVALID_INT, type);
      }
    }
      
    if(type == INIT) {
      P(param, mu_factor[i], 1, type);
      P(param, global_reduction[i], QUDA_BOOLEAN_TRUE, type);
    } else {
      P(param, mu_factor[i], INVALID_DOUBLE, type);
      P(param, global_reduction[i], QUDA_BOOLEAN_INVALID, type);
    }

    P(param, coarse_solver_tol[i], INVALID_DOUBLE, type);
    P(param, smoother_tol[i], INVALID_DOUBLE, type);
    P(param, omega[i], INVALID_DOUBLE, type);
    P(param, location[i], QUDA_INVALID_FIELD_LOCATION, type);

    if(type == INIT) { 
      P(param, setup_location[i], QUDA_CUDA_FIELD_LOCATION, type);
    } else {
      P(param, setup_location[i], QUDA_INVALID_FIELD_LOCATION, type);
    }
  }

  if(type == INIT) { 
    P(param, setup_minimize_memory, QUDA_BOOLEAN_FALSE, type);
  } else {
    P(param, setup_minimize_memory, QUDA_BOOLEAN_INVALID, type);
  }

  P(param, compute_null_vector, QUDA_COMPUTE_NULL_VECTOR_INVALID, type);
  P(param, generate_all_levels, QUDA_BOOLEAN_INVALID, type);

  if(type == CHECK) {
    // if only doing top-level null-space generation, check that n_vec
    // is equal on all levels
    if (param->generate_all_levels == QUDA_BOOLEAN_FALSE && param->compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
      for (int i=1; i<n_level-1; i++)
	if (param->n_vec[0] != param->n_vec[i])
	  errorQuda("n_vec %d != %d must be equal on all levels if generate_all_levels == false",
		    param->n_vec[0], param->n_vec[i]);
    }
  }

  P(param, run_verify, QUDA_BOOLEAN_INVALID, type);

  if(type == INIT) {   
    P(param, run_low_mode_check, QUDA_BOOLEAN_FALSE, type);
    P(param, run_oblique_proj_check, QUDA_BOOLEAN_FALSE, type);
    P(param, coarse_guess, QUDA_BOOLEAN_FALSE, type);
    P(param, preserve_deflation, QUDA_BOOLEAN_FALSE, type);
  } else {
    P(param, run_low_mode_check, QUDA_BOOLEAN_INVALID, type);
    P(param, run_oblique_proj_check, QUDA_BOOLEAN_INVALID, type);
    P(param, coarse_guess, QUDA_BOOLEAN_INVALID, type);
    P(param, preserve_deflation, QUDA_BOOLEAN_INVALID, type);
  }

  for (int i = 0; i < n_level - 1; i++) {
    if(type == INIT) {
      P(param, vec_load[i], QUDA_BOOLEAN_FALSE, type);
      P(param, vec_store[i], QUDA_BOOLEAN_FALSE, type);
    } else {
      P(param, vec_load[i], QUDA_BOOLEAN_INVALID, type);
      P(param, vec_store[i], QUDA_BOOLEAN_INVALID, type);
    }
  }

  if(type == INIT) {
    P(param, gflops, 0.0, type);
    P(param, secs, 0.0, type);
  } else if(type == PRINT) {
    P(param, gflops, INVALID_DOUBLE, type);
    P(param, secs, INVALID_DOUBLE, type);
  }

  if(type == INIT) {
    P(param, is_staggered, QUDA_BOOLEAN_FALSE, type);
  } else {
    P(param, is_staggered, QUDA_BOOLEAN_INVALID, type);
  }
}



QudaGaugeObservableParam newQudaGaugeObservableParam(void)
{
  QudaGaugeObservableParam param;
  paramType type = INIT;
  parseQudaGaugeObservableParam(&param, type);
  return param;
}
void checkGaugeObservableParam(QudaGaugeObservableParam *param)
{
  paramType type = CHECK;
  parseQudaGaugeObservableParam(param, type);
}

void printQudaGaugeObservableParam(QudaGaugeObservableParam *param)
{
  printfQuda("QUDA Gauge-Observable Parameters:\n");
  paramType type = PRINT;
  parseQudaGaugeObservableParam(param, type);
}
 
void parseQudaGaugeObservableParam(QudaGaugeObservableParam *param, paramType type)
{
  if(type == INIT) { 
    P(param, su_project, QUDA_BOOLEAN_FALSE, type);
    P(param, compute_plaquette, QUDA_BOOLEAN_FALSE, type);
    P(param, compute_qcharge, QUDA_BOOLEAN_FALSE, type);
    P(param, compute_qcharge_density, QUDA_BOOLEAN_FALSE, type);
    //P(param, qcharge_density, nullptr, type);
  } else {
    P(param, su_project, QUDA_BOOLEAN_INVALID, type);
    P(param, compute_plaquette, QUDA_BOOLEAN_INVALID, type);
    P(param, compute_qcharge, QUDA_BOOLEAN_INVALID, type);
    P(param, compute_qcharge_density, QUDA_BOOLEAN_INVALID, type);
  }
}
 


 
QudaCublasParam newQudaCublasParam(void)
{
  QudaCublasParam param;
  paramType type = INIT;
  parseQudaCublasParam(&param, type);
  return param;
}
void checkCublasParam(QudaCublasParam *param)
{
  paramType type = CHECK;
  parseQudaCublasParam(param, type);
}

void printQudaCublasParam(QudaCublasParam *param)
{
  printfQuda("QUDA Gauge-Observable Parameters:\n");
  paramType type = PRINT;
  parseQudaCublasParam(param, type);
}
 
void parseQudaCublasParam(QudaCublasParam *param, paramType type)
{
  if(type == INIT) {
    P(param, trans_a, QUDA_CUBLAS_OP_N, type);
    P(param, trans_b, QUDA_CUBLAS_OP_N, type);
    P(param, m, 0, type);
    P(param, n, 0, type);
    P(param, k, 0, type);
    P(param, lda, 0, type);
    P(param, ldb, 0, type);
    P(param, ldc, 0, type);
    P(param, a_offset, 0, type);
    P(param, b_offset, 0, type);
    P(param, c_offset, 0, type);
    P(param, batch_count, 1, type);
    P(param, data_type, QUDA_CUBLAS_DATATYPE_S, type);
    P(param, data_order, QUDA_CUBLAS_DATAORDER_ROW, type);
  } else {
    P(param, trans_a, QUDA_CUBLAS_OP_INVALID, type);
    P(param, trans_b, QUDA_CUBLAS_OP_INVALID, type);
    P(param, m, INVALID_INT, type);
    P(param, n, INVALID_INT, type);
    P(param, k, INVALID_INT, type);
    P(param, lda, INVALID_INT, type);
    P(param, ldb, INVALID_INT, type);
    P(param, ldc, INVALID_INT, type);
    P(param, a_offset, INVALID_INT, type);
    P(param, b_offset, INVALID_INT, type);
    P(param, c_offset, INVALID_INT, type);
    P(param, batch_count, INVALID_INT, type);
    P(param, data_type, QUDA_CUBLAS_DATATYPE_INVALID, type);
    P(param, data_order, QUDA_CUBLAS_DATAORDER_INVALID, type);
  }
}

// clean up

#undef INVALID_INT
#undef INVALID_DOUBLE
#undef P
