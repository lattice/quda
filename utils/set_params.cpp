#include <algorithm>
#include <command_line_params.h>
#include <host_utils.h>
#include "misc.h"

void setGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.type = QUDA_SU3_LINKS;

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.cuda_prec_precondition = cuda_prec;

  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.reconstruct_precondition = link_recon;
  gauge_param.reconstruct_refinement_sloppy = link_recon;

  gauge_param.anisotropy = 1.0;
  gauge_param.tadpole_coeff = 1.0;

  gauge_param.ga_pad = 0;
  gauge_param.mom_ga_pad = 0;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
}

void setWilsonGaugeParam(QudaGaugeParam &gauge_param)
{
  setGaugeParam(gauge_param);
  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = fermion_t_boundary;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_precondition = link_recon_precondition;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;

  int pad_size = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  pad_size = std::max({x_face_size, y_face_size, z_face_size, t_face_size});
#endif
  gauge_param.ga_pad = pad_size;
}

void setStaggeredGaugeParam(QudaGaugeParam &gauge_param)
{
  setGaugeParam(gauge_param);

  gauge_param.cuda_prec_sloppy = prec_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  gauge_param.cuda_prec_precondition = prec_precondition;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_precondition = link_recon;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gauge_param.tadpole_coeff = 1.0;

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.scale = -1.0 / 24.0;
    if (eps_naik != 0) { gauge_param.scale *= (1.0 + eps_naik); }
  } else {
    gauge_param.scale = 1.0;
  }

  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.t_boundary = fermion_t_boundary;
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gauge_param.type = QUDA_WILSON_LINKS;

  int pad_size = 0;
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  pad_size = std::max({x_face_size, y_face_size, z_face_size, t_face_size});
#endif
  gauge_param.ga_pad = pad_size;
}

void setInvertParam(QudaInvertParam &inv_param)
{
  // Set dslash type
  inv_param.dslash_type = dslash_type;

  // Use kappa or mass normalisation
  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.kappa = 1.0 / (8 + mass);
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1.0 + 3.0 / anisotropy);
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.mass = 1.0 / kappa - 8.0;
  }
  printfQuda("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);

  // Use 3D or 4D laplace
  inv_param.laplace3D = laplace3D;

  // Some fermion specific parameters
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
             || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    inv_param.m5 = m5;
    kappa5 = 0.5 / (5 + inv_param.m5);
    inv_param.Ls = Lsdim;
    for (int k = 0; k < Lsdim; k++) { // for mobius only
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = b5;
      inv_param.c_5[k] = c5;
    }
    inv_param.eofa_pm = eofa_pm;
    inv_param.eofa_shift = eofa_shift;
    inv_param.mq1 = eofa_mq1;
    inv_param.mq2 = eofa_mq2;
    inv_param.mq3 = eofa_mq3;
  } else {
    inv_param.Ls = 1;
  }

  // Set clover specific parameters
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  // General parameter setup
  inv_param.inv_type = inv_type;
  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  inv_param.pipeline = pipeline;
  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart;
  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    inv_param.residual_type;

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // Offsets used only by multi-shift solver
  // These should be set in the application code. We set the them here by way of
  // example
  inv_param.num_offset = multishift;
  for (int i = 0; i < inv_param.num_offset; i++) inv_param.offset[i] = 0.06 + i * i * 0.1;
  // these can be set individually
  for (int i = 0; i < inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;

  inv_param.schwarz_type = precon_schwarz_type;
  inv_param.precondition_cycle = precon_schwarz_cycle;
  inv_param.tol_precondition = tol_precondition;
  inv_param.maxiter_precondition = maxiter_precondition;
  inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;
  // inv_param.cuda_prec_ritz = cuda_prec_ritz;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.verbosity = verbosity;

  inv_param.extlib_type = solver_ext_lib;
}

// Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param)
{
  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  if ((eig_type == QUDA_EIG_TR_LANCZOS || eig_type == QUDA_EIG_IR_LANCZOS)
      && !(eig_spectrum == QUDA_SPECTRUM_LR_EIG || eig_spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to Lanczos type solver");
  }

  // The solver will exit when nConv extremal eigenpairs have converged
  if (eig_nConv < 0) {
    eig_param.nConv = eig_nEv;
    eig_nConv = eig_nEv;
  } else {
    eig_param.nConv = eig_nConv;
  }

  eig_param.block_size = eig_param.eig_type == QUDA_EIG_TR_LANCZOS ? 1 : eig_block_size;
  eig_param.nEv = eig_nEv;
  eig_param.nKr = eig_nKr;
  eig_param.tol = eig_tol;
  eig_param.batched_rotate = eig_batched_rotate;
  eig_param.require_convergence = eig_require_convergence ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  eig_param.cuda_prec_ritz = cuda_prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.use_dagger = eig_use_dagger ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  if (eig_compute_svd) {
    eig_param.use_dagger = QUDA_BOOLEAN_FALSE;
    eig_param.use_norm_op = QUDA_BOOLEAN_TRUE;
  }

  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
  eig_param.io_parity_inflate = eig_io_parity_inflate ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
}

void setMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1 + 3 / anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i = 0; i < mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    for (int j = 4; j < QUDA_MAX_DIM; j++) mg_param.geo_block_size[i][j] = 1;
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];
    mg_param.setup_maxiter_refresh[i] = setup_maxiter_refresh[i];

    // Basis to use for CA-CGN(E/R) setup
    mg_param.setup_ca_basis[i] = setup_ca_basis[i];

    // Basis size for CACG setup
    mg_param.setup_ca_basis_size[i] = setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = setup_ca_lambda_max[i];

    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i];          // default to 24 vectors if not set
    mg_param.n_block_ortho[i] = n_block_ortho[i];             // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null;                   // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    // Basis to use for CA-CGN(E/R) coarse solver
    mg_param.coarse_solver_ca_basis[i] = coarse_solver_ca_basis[i];

    // Basis size for CACG coarse solver/
    mg_param.coarse_solver_ca_basis_size[i] = coarse_solver_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = smoother_solve_type[i];

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = mg_schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (mg_schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = mg_schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0],
                  solve_type);

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }

    } else {

      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", coarse_solve_type[i]);
      }
    }

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_FALSE;

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // Is NOT a staggered solve
  mg_param.is_staggered = QUDA_BOOLEAN_FALSE;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = reliable_delta;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;
}

void setMultigridInvertParam(QudaInvertParam &inv_param)
{
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1 + 3 / anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = mg_verbosity[0];

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.pipeline = pipeline;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // Offsets used only by multi-shift solver
  // should be set in application
  inv_param.num_offset = multishift;
  for (int i = 0; i < inv_param.num_offset; i++) inv_param.offset[i] = 0.06 + i * i * 0.1;
  // these can be set individually
  for (int i = 0; i < inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;

  // domain decomposition preconditioner is disabled when using MG
  inv_param.schwarz_type = QUDA_INVALID_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

// Parameters defining the eigensolver
void setMultigridEigParam(QudaEigParam &mg_eig_param, int level)
{
  mg_eig_param.eig_type = mg_eig_type[level];
  mg_eig_param.spectrum = mg_eig_spectrum[level];
  if ((mg_eig_type[level] == QUDA_EIG_TR_LANCZOS || mg_eig_type[level] == QUDA_EIG_IR_LANCZOS)
      && !(mg_eig_spectrum[level] == QUDA_SPECTRUM_LR_EIG || mg_eig_spectrum[level] == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.block_size = mg_eig_param.eig_type == QUDA_EIG_TR_LANCZOS ? 1 : mg_eig_block_size[level];
  mg_eig_param.nEv = mg_eig_nEv[level];
  mg_eig_param.nKr = mg_eig_nKr[level];
  mg_eig_param.nConv = nvec[level];
  mg_eig_param.batched_rotate = mg_eig_batched_rotate[level];
  mg_eig_param.require_convergence = mg_eig_require_convergence[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.tol = mg_eig_tol[level];
  mg_eig_param.check_interval = mg_eig_check_interval[level];
  mg_eig_param.max_restarts = mg_eig_max_restarts[level];
  mg_eig_param.cuda_prec_ritz = cuda_prec;

  mg_eig_param.compute_svd = QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_norm_op = mg_eig_use_normop[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_dagger = mg_eig_use_dagger[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.use_poly_acc = mg_eig_use_poly_acc[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.poly_deg = mg_eig_poly_deg[level];
  mg_eig_param.a_min = mg_eig_amin[level];
  mg_eig_param.a_max = mg_eig_amax[level];

  // set file i/o parameters
  // Give empty strings, Multigrid will handle IO.
  strcpy(mg_eig_param.vec_infile, "");
  strcpy(mg_eig_param.vec_outfile, "");
  mg_eig_param.io_parity_inflate = QUDA_BOOLEAN_FALSE;
  strcpy(mg_eig_param.QUDA_logfile, eig_QUDA_logfile);
}

void setContractInvertParam(QudaInvertParam &inv_param)
{
  inv_param.Ls = 1;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;

  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  // Quda performs contractions in Degrand-Rossi gamma basis,
  // but the user may suppy vectors in any supported order.
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
}

void setStaggeredMGInvertParam(QudaInvertParam &inv_param)
{
  // Solver params
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.mass = mass;

  // outer solver parameters
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = tol;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-4;
  inv_param.pipeline = pipeline;

  inv_param.Ls = 1;

  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  /* ESW HACK: comment this out to do a non-MG solve. */
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  // inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.verbosity_precondition = QUDA_SUMMARIZE; // ESW HACK
  inv_param.cuda_prec_precondition = cuda_prec_precondition;

  // Specify Krylov sub-size for GCR, BICGSTAB(L)
  inv_param.gcrNkrylov = gcrNkrylov;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dslash_type = dslash_type;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  // these can be set individually
  for (int i = 0; i < inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }

  // domain decomposition preconditioner is disabled when using MG
  inv_param.schwarz_type = QUDA_INVALID_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

void setStaggeredInvertParam(QudaInvertParam &inv_param)
{
  // Solver params
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.mass = mass;
  inv_param.kappa = kappa = 1.0 / (8.0 + mass); // for Laplace operator
  inv_param.laplace3D = laplace3D;              // for Laplace operator

  // outer solver parameters
  inv_param.inv_type = inv_type;
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = false;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.pipeline = pipeline;

  inv_param.Ls = 1; // Nsrc

  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    inv_param.residual_type;
  inv_param.heavy_quark_check = (inv_param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 5 : 0);

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.Nsteps = 2;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
  inv_param.tol_precondition = tol_precondition;
  inv_param.maxiter_precondition = maxiter_precondition;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.cuda_prec_precondition = prec_precondition;

  // Specify Krylov sub-size for GCR, BICGSTAB(L), basis size for CA-CG, CA-GCR
  inv_param.gcrNkrylov = gcrNkrylov;

  // Specify basis for CA-CG, lambda min/max for Chebyshev basis
  //   lambda_max < lambda_max . use power iters to generate
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;

  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dslash_type = dslash_type;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.sp_pad = 0;
}

void setStaggeredMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4.0 + inv_param.mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.is_staggered = QUDA_BOOLEAN_TRUE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i = 0; i < mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];

    // Basis to use for CA-CGN(E/R) setup
    mg_param.setup_ca_basis[i] = setup_ca_basis[i];

    // Basis size for CACG setup
    mg_param.setup_ca_basis_size[i] = setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = setup_ca_lambda_max[i];

    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = (i == 0) ? 24 : nvec[i] == 0 ? 96 : nvec[i]; // default to 96 vectors if not set
    mg_param.n_block_ortho[i] = n_block_ortho[i];                    // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null;                          // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec;        // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    // Basis to use for CA-CGN(E/R) coarse solver
    mg_param.coarse_solver_ca_basis[i] = coarse_solver_ca_basis[i];

    // Basis size for CACG coarse solver/
    mg_param.coarse_solver_ca_basis_size[i] = coarse_solver_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = smoother_solve_type[i];

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = mg_schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (mg_schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = mg_schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %s and outer solve type %s",
                  get_solve_str(coarse_solve_type[0]), get_solve_str(solve_type));

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %s\n", get_solve_str(solve_type));
      }

    } else {

      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %s\n", get_solve_str(coarse_solve_type[i]));
      }
    }

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
    nu_pre[i] = 2;
    nu_post[i] = 2;
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_FALSE;

  // coarsening the spin on the first restriction is undefined for staggered fields.
  mg_param.spin_block_size[0] = 0;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = reliable_delta;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;
}

void setDeflatedInvertParam(QudaInvertParam &inv_param)
{
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1 + 3 / anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  // inv_param.solution_type = QUDA_MATPC_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  if (inv_type != QUDA_EIGCG_INVERTER && inv_type != QUDA_INC_EIGCG_INVERTER && inv_type != QUDA_GMRESDR_INVERTER)
    errorQuda("Requested solver %s is not a deflated solver type", get_solver_str(inv_type));

  //! For deflated solvers only:
  inv_param.inv_type = inv_type;
  inv_param.tol = tol;
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.rhs_idx = 0;

  inv_param.nev = nev;
  inv_param.max_search_dim = max_search_dim;
  inv_param.deflation_grid = deflation_grid;
  inv_param.tol_restart = tol_restart;
  inv_param.eigcg_max_restarts = eigcg_max_restarts;
  inv_param.max_restart_num = max_restart_num;
  inv_param.inc_tol = inc_tol;
  inv_param.eigenval_tol = eigenval_tol;

  if (inv_param.inv_type == QUDA_EIGCG_INVERTER || inv_param.inv_type == QUDA_INC_EIGCG_INVERTER) {
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  } else if (inv_param.inv_type == QUDA_GMRESDR_INVERTER) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param.tol_restart = 0.0; // restart is not requested...
  }

  inv_param.cuda_prec_ritz = cuda_prec_ritz;
  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;

  inv_param.inv_type_precondition = precon_type;
  inv_param.gcrNkrylov = 6;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // Offsets used only by multi-shift solver
  // should be set in application
  inv_param.num_offset = multishift;
  for (int i = 0; i < inv_param.num_offset; i++) inv_param.offset[i] = 0.06 + i * i * 0.1;
  // these can be set individually
  for (int i = 0; i < inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-1;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = precon_schwarz_type;
  inv_param.precondition_cycle = precon_schwarz_cycle;
  inv_param.tol_precondition = tol_precondition;
  inv_param.maxiter_precondition = maxiter_precondition;
  inv_param.omega = 1.0;

  inv_param.extlib_type = solver_ext_lib;
}

void setDeflationParam(QudaEigParam &df_param)
{
  df_param.import_vectors = QUDA_BOOLEAN_FALSE;
  df_param.run_verify = QUDA_BOOLEAN_FALSE;

  df_param.nk = df_param.invert_param->nev;
  df_param.np = df_param.invert_param->nev * df_param.invert_param->deflation_grid;
  df_param.extlib_type = deflation_ext_lib;

  df_param.cuda_prec_ritz = prec_ritz;
  df_param.location = location_ritz;
  df_param.mem_type_ritz = mem_type_ritz;

  // set file i/o parameters
  strcpy(df_param.vec_infile, eig_vec_infile);
  strcpy(df_param.vec_outfile, eig_vec_outfile);
  df_param.io_parity_inflate = eig_io_parity_inflate ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
}

void setQudaStaggeredInvTestParams()
{
  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (test_type != 0) { errorQuda("Test type %d is not supported for the Laplace operator.\n", test_type); }

    solve_type = QUDA_DIRECT_SOLVE;
    solution_type = QUDA_MAT_SOLUTION;
    matpc_type = QUDA_MATPC_EVEN_EVEN; // doesn't matter

  } else {

    if (test_type == 0 && (inv_type == QUDA_CG_INVERTER || inv_type == QUDA_PCG_INVERTER)
        && solve_type != QUDA_NORMOP_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
      warningQuda("The full spinor staggered operator (test 0) can't be inverted with (P)CG. Switching to BiCGstab.\n");
      inv_type = QUDA_BICGSTAB_INVERTER;
    }

    if (solve_type == QUDA_INVALID_SOLVE) {
      if (test_type == 0) {
        solve_type = QUDA_DIRECT_SOLVE;
      } else {
        solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }

    if (test_type == 1 || test_type == 3 || test_type == 5) {
      matpc_type = QUDA_MATPC_EVEN_EVEN;
    } else if (test_type == 2 || test_type == 4 || test_type == 6) {
      matpc_type = QUDA_MATPC_ODD_ODD;
    } else if (test_type == 0) {
      matpc_type = QUDA_MATPC_EVEN_EVEN; // it doesn't matter
    }

    if (test_type == 0 || test_type == 1 || test_type == 2) {
      solution_type = QUDA_MAT_SOLUTION;
    } else {
      solution_type = QUDA_MATPC_SOLUTION;
    }
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) { prec_sloppy = prec; }

  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) { prec_refinement_sloppy = prec_sloppy; }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) { link_recon_sloppy = link_recon; }

  if (inv_type != QUDA_CG_INVERTER && (test_type == 5 || test_type == 6)) {
    errorQuda("Preconditioning is currently not supported in multi-shift solver solvers");
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }
}

void setQudaStaggeredEigTestParams()
{
  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    // LAPLACE operator path, only DIRECT solves feasible.
    if (test_type != 0) { errorQuda("Test type %d is not supported for the Laplace operator.\n", test_type); }
    solve_type = QUDA_DIRECT_SOLVE;
    solution_type = QUDA_MAT_SOLUTION;
  } else {
    // STAGGERED operator path
    if (solve_type == QUDA_INVALID_SOLVE) {
      if (test_type == 0) {
        solve_type = QUDA_DIRECT_SOLVE;
      } else {
        solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }
    // If test type is not 3, it is 4 or 0. If 0, the matpc type is irrelevant
    if (test_type == 3)
      matpc_type = QUDA_MATPC_EVEN_EVEN;
    else
      matpc_type = QUDA_MATPC_ODD_ODD;

    if (test_type == 0) {
      solution_type = QUDA_MAT_SOLUTION;
    } else {
      solution_type = QUDA_MATPC_SOLUTION;
    }
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }
}
