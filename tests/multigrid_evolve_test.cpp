#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <qio_field.h>
#include <color_spinor_field.h>

#include <gauge_field.h>
#include <gauge_tools.h>
#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) {
    printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
    printfQuda(" - level %d number of pre-smoother applications %d\n", i+1, nu_pre[i]);
    printfQuda(" - level %d number of post-smoother applications %d\n", i+1, nu_post[i]);
  }

  printfQuda("Outer solver paramers\n");
  printfQuda(" - pipeline = %d\n", pipeline);

  printfQuda("Eigensolver parameters\n");
  for (int i = 0; i < mg_levels; i++) {
    if (low_mode_check || mg_eig[i]) {
      printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
      printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
      printfQuda(" - level %d number of eigenvectors requested nConv %d\n", i + 1, nvec[i]);
      printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_nEv[i]);
      printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_nKr[i]);
      printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
      printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
      printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1, mg_eig_use_dagger[i] ? "true" : "false",
                 mg_eig_use_normop[i] ? "true" : "false");
      if (mg_eig_use_poly_acc[i]) {
        printfQuda(" - level %d Chebyshev polynomial degree %d\n", i + 1, mg_eig_poly_deg[i]);
        printfQuda(" - level %d Chebyshev polynomial minumum %e\n", i + 1, mg_eig_amin[i]);
        printfQuda(" - level %d Chebyshev polynomial maximum %e\n", i + 1, mg_eig_amax[i]);
      }
      printfQuda("\n");
    }
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

// Parameters defining the eigensolver
void setEigParam(QudaEigParam &mg_eig_param, int level)
{
  mg_eig_param.eig_type = mg_eig_type[level];
  mg_eig_param.spectrum = mg_eig_spectrum[level];
  if ((mg_eig_type[level] == QUDA_EIG_TR_LANCZOS || mg_eig_type[level] == QUDA_EIG_IR_LANCZOS)
      && !(mg_eig_spectrum[level] == QUDA_SPECTRUM_LR_EIG || mg_eig_spectrum[level] == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.nEv = mg_eig_nEv[level];
  mg_eig_param.nKr = mg_eig_nKr[level];
  mg_eig_param.nConv = nvec[level];
  mg_eig_param.batched_rotate = mg_eig_batched_rotate[level];
  mg_eig_param.require_convergence = mg_eig_require_convergence[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  mg_eig_param.tol = mg_eig_tol[level];
  mg_eig_param.check_interval = mg_eig_check_interval[level];
  mg_eig_param.max_restarts = mg_eig_max_restarts[level];
  mg_eig_param.cuda_prec_ritz = cuda_prec;

  mg_eig_param.compute_svd = QUDA_BOOLEAN_NO;
  mg_eig_param.use_norm_op = mg_eig_use_normop[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_eig_param.use_dagger = mg_eig_use_dagger[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  mg_eig_param.use_poly_acc = mg_eig_use_poly_acc[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_eig_param.poly_deg = mg_eig_poly_deg[level];
  mg_eig_param.a_min = mg_eig_amin[level];
  mg_eig_param.a_max = mg_eig_amax[level];

  // set file i/o parameters
  // Give empty strings, Multigrid will handle IO.
  strcpy(mg_eig_param.vec_infile, "");
  strcpy(mg_eig_param.vec_outfile, "");

  strcpy(mg_eig_param.QUDA_logfile, eig_QUDA_logfile);
}

void setMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param;

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
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
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
  } else {
    inv_param.mu = 0.0;
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    for (int j = 4; j < QUDA_MAX_DIM; j++) mg_param.geo_block_size[i][j] = 1;
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
    mg_param.spin_block_size[i] = 1;
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

    mg_param.setup_maxiter_refresh[i] = setup_maxiter_refresh[i];
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.n_block_ortho[i] = n_block_ortho[i];    // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
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
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

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
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0], solve_type);

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
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_NO;

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_YES;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_YES;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_param.preserve_deflation = QUDA_BOOLEAN_NO;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = mg_verbosity[0];
}

void setInvertParam(QudaInvertParam &inv_param) {

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
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
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
  } else {
    inv_param.mu = 0.0;
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

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

 void setReunitarizationConsts(){
   using namespace quda;
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error,
                               reunit_allow_svd, reunit_svd_only,
                               svd_rel_error, svd_abs_error);

  }

void CallUnitarizeLinks(quda::cudaGaugeField *cudaInGauge){
   using namespace quda;
   int *num_failures_dev = (int*)device_malloc(sizeof(int));
   int num_failures;
   cudaMemset(num_failures_dev, 0, sizeof(int));
   unitarizeLinks(*cudaInGauge, num_failures_dev);

   cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
   if(num_failures>0) errorQuda("Error in the unitarization\n");
   device_free(num_failures_dev);
  }

int main(int argc, char **argv)
{
  // We give here the default values to some of the array
  solve_type = QUDA_DIRECT_PC_SOLVE;
  for (int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SUMMARIZE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    setup_maxiter_refresh[i] = 100;
    mu_factor[i] = 1.;
    coarse_solve_type[i] = QUDA_INVALID_SOLVE;
    smoother_solve_type[i] = QUDA_INVALID_SOLVE;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_MR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 10;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
    mg_eig_nEv[i] = nvec[i];
    mg_eig_nKr[i] = 3 * nvec[i];
    mg_eig_require_convergence[i] = QUDA_BOOLEAN_YES;
    mg_eig_type[i] = QUDA_EIG_TR_LANCZOS;
    mg_eig_spectrum[i] = QUDA_SPECTRUM_SR_EIG;
    mg_eig_check_interval[i] = 5;
    mg_eig_max_restarts[i] = 100;
    mg_eig_use_normop[i] = QUDA_BOOLEAN_NO;
    mg_eig_use_dagger[i] = QUDA_BOOLEAN_NO;
    mg_eig_use_poly_acc[i] = QUDA_BOOLEAN_YES;
    mg_eig_poly_deg[i] = 100;
    mg_eig_amin[i] = 1.0;
    mg_eig_amax[i] = 5.0;

    setup_ca_basis[i] = QUDA_POWER_BASIS;
    setup_ca_basis_size[i] = 4;
    setup_ca_lambda_min[i] = 0.0;
    setup_ca_lambda_max[i] = -1.0; // use power iterations

    coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    coarse_solver_ca_basis_size[i] = 4;
    coarse_solver_ca_lambda_min[i] = 0.0;
    coarse_solver_ca_lambda_max[i] = -1.0;

    strcpy(mg_vec_infile[i], "");
    strcpy(mg_vec_outfile[i], "");
  }
  reliable_delta = 1e-4;

  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  QudaMultigridParam mg_param = newQudaMultigridParam();
  // Set sub structures
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];
  for (int i = 0; i < mg_levels; i++) {
    if (mg_eig[i]) {
      mg_eig_param[i] = newQudaEigParam();
      setEigParam(mg_eig_param[i], i);
      mg_param.eig_param[i] = &mg_eig_param[i];
    } else {
      mg_param.eig_param[i] = nullptr;
    }
  }

  // Set MG
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);

  display_test_info();

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      // unit SU(3) field
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      // random SU(3) field
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = inv_param.clover_cpu_prec;
    clover = malloc(V*cloverSiteSize*cSize);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    if (!compute_clover) construct_clover_field(clover, norm, diag, inv_param.clover_cpu_prec);

    inv_param.compute_clover = compute_clover;
    if (compute_clover) inv_param.return_clover = 1;
    inv_param.compute_clover_inverse = 1;
    inv_param.return_clover_inverse = 1;
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorOut = malloc(V * spinorSiteSize * sSize * inv_param.Ls);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  {
    using namespace quda;
    GaugeFieldParam gParam(0, gauge_param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = gauge_param.type;
    gParam.reconstruct = gauge_param.reconstruct;
    gParam.order       = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    cudaGaugeField *gauge = new cudaGaugeField(gParam);

    int pad = 0;
    int y[4];
    int R[4] = {0,0,0,0};
    for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
    for(int dir=0; dir<4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
    GaugeFieldParam gParamEx(y, prec, link_recon,
			     pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = gParam.order;
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = gParam.t_boundary;
    gParamEx.nFace = 1;
    for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
    cudaGaugeField *gaugeEx = new cudaGaugeField(gParamEx);
    // CURAND random generator initialization
    RNG *randstates = new RNG(*gauge, 1234);
    randstates->Init();

    int nsteps = 10;
    int nhbsteps = 1;
    int novrsteps = 1;
    bool coldstart = false;
    double beta_value = 6.2;

    if(link_recon != QUDA_RECONSTRUCT_8 && coldstart) InitGaugeField( *gaugeEx);
    else{
      InitGaugeField( *gaugeEx, *randstates );
    }
    // Reunitarization setup
    setReunitarizationConsts();
    plaquette(*gaugeEx);

    Monte(*gaugeEx, *randstates, beta_value, 100 * nhbsteps, 100 * novrsteps);

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    // load the gauge field from gauge
    gauge_param.gauge_order = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    double3 plaq = plaquette(*gaugeEx);
    double charge = qChargeQuda();

    // Demonstrate MG evolution on an evolving gauge field
    //----------------------------------------------------
    printfQuda("\n======================================================\n");
    printfQuda("Running MG gauge evolution test at constant quark mass\n");
    printfQuda("======================================================\n");
    printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", 0, plaq.x, charge,
               inv_param.mass, inv_param.kappa, inv_param.mu);

    // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
    if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE)
      inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
      loadCloverQuda(clover, clover_inv, &inv_param);

    inv_param.solve_type = solve_type; // restore actual solve_type we want to do

    // create a point source at 0 (in each subvolume...  FIXME)
    memset(spinorIn, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorCheck, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)spinorIn)[i] = rand() / (float)RAND_MAX;
    } else {
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = rand() / (double)RAND_MAX;
    }

    // do reference BiCGStab solve
    QudaInvertParam inv_param2 = inv_param;
    inv_param2.inv_type = QUDA_BICGSTABL_INVERTER;
    inv_param2.gcrNkrylov = 4;
    inv_param2.inv_type_precondition = QUDA_INVALID_INVERTER;
    inv_param2.reliable_delta = 0.1;
    inv_param2.maxiter = 10000;
    inv_param2.chrono_use_resident = true;
    inv_param2.chrono_make_resident = true;
    inv_param2.chrono_index = 0 ;
    inv_param2.chrono_max_dim = 7;
    inv_param2.chrono_precision = inv_param2.cuda_prec_sloppy; // use sloppy precision for chrono basis
    inv_param2.use_init_guess = QUDA_USE_INIT_GUESS_YES;

    invertQuda(spinorOut, spinorIn, &inv_param2);

    // setup the multigrid solver
    void *mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
    invertQuda(spinorOut, spinorIn, &inv_param);

    for (int step = 1; step < nsteps; ++step) {
      freeGaugeQuda();
      Monte( *gaugeEx, *randstates, beta_value, nhbsteps, novrsteps);

      //Reunitarize gauge links...
      CallUnitarizeLinks(gaugeEx);

      // copy into regular field
      copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

      loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
      plaq = plaquette(*gaugeEx);
      charge = qChargeQuda();
      printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", step, plaq.x,
                 charge, inv_param.mass, inv_param.kappa, inv_param.mu);

      // reference BiCGStab for comparison
      invertQuda(spinorOut, spinorIn, &inv_param2);

      updateMultigridQuda(mg_preconditioner, &mg_param); // update the multigrid operator for new gauge and clover fields
      invertQuda(spinorOut, spinorIn, &inv_param);

      if (inv_param.iter == inv_param.maxiter) {
        char vec_outfile[QUDA_MAX_MG_LEVEL][256];
        for (int i=0; i<mg_param.n_level; i++) {
          strcpy(vec_outfile[i], mg_param.vec_outfile[i]);
          sprintf(mg_param.vec_outfile[i], "dump_step_evolve_%d", step);
        }
        warningQuda("Solver failed to converge within max iteration count - dumping null vectors to %s",
                    mg_param.vec_outfile[0]);

        dumpMultigridQuda(mg_preconditioner, &mg_param);
        for (int i=0; i<mg_param.n_level; i++) {
          strcpy(mg_param.vec_outfile[i], vec_outfile[i]); // restore output file name
        }
      }
    }

    // free the multigrid solver
    destroyMultigridQuda(mg_preconditioner);

    // Demonstrate MG evolution on a fixed gauge field and different masses
    //---------------------------------------------------------------------
    // setup the multigrid solver
    printfQuda("\n====================================================\n");
    printfQuda("Running MG mass scaling test at constant gauge field\n");
    printfQuda("====================================================\n");

    invertQuda(spinorOut, spinorIn, &inv_param2);

    mg_param.preserve_deflation = mg_eig_preserve_deflation ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
    for (int i = 0; i < mg_param.n_level; i++) mg_param.setup_maxiter_refresh[i] = 0;

    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
    invertQuda(spinorOut, spinorIn, &inv_param);

    freeGaugeQuda();

    // Reunitarize gauge links...
    CallUnitarizeLinks(gaugeEx);

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    for (int step = 1; step < nsteps; ++step) {

      plaq = plaquette(*gaugeEx);
      charge = qChargeQuda();

      // Increment the mass/kappa and mu values to emulate heavy/light flavour updates
      if (kappa == -1.0) {
        inv_param.mass = mass + 0.01 * step;
        inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass + 0.01 * step));
        mg_param.invert_param->mass = inv_param.mass;
        mg_param.invert_param->kappa = inv_param.kappa;
        inv_param2.mass = mass + 0.01 * step;
        inv_param2.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass + 0.01 * step));
      } else {
        inv_param.kappa = kappa - 0.001 * step;
        inv_param.mass = 0.5 / (kappa - 0.001 * step) - (1 + 3 / anisotropy);
        mg_param.invert_param->mass = inv_param.mass;
        mg_param.invert_param->kappa = inv_param.kappa;
        inv_param2.kappa = kappa - 0.001 * step;
        inv_param2.mass = 0.5 / (kappa - 0.001 * step) - (1 + 3 / anisotropy);
      }
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        // Multiply by -1.0 to emulate twist switch
        inv_param.mu = -1.0 * mu + 0.01 * step;
        inv_param2.mu = -1.0 * mu + 0.01 * step;
        mg_param.invert_param->mu = inv_param.mu;
      }

      printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", step, plaq.x,
                 charge, inv_param.mass, inv_param.kappa, inv_param.mu);

      // reference BiCGStab for comparison
      invertQuda(spinorOut, spinorIn, &inv_param2);

      updateMultigridQuda(mg_preconditioner, &mg_param); // update the multigrid operator for new mass and mu values
      invertQuda(spinorOut, spinorIn, &inv_param);

      if (inv_param.iter == inv_param.maxiter) {
        char vec_outfile[QUDA_MAX_MG_LEVEL][256];
        for (int i = 0; i < mg_param.n_level; i++) {
          strcpy(vec_outfile[i], mg_param.vec_outfile[i]);
          sprintf(mg_param.vec_outfile[i], "dump_step_shift_%d", step);
        }
        warningQuda("Solver failed to converge within max iteration count - dumping null vectors to %s",
                    mg_param.vec_outfile[0]);

        dumpMultigridQuda(mg_preconditioner, &mg_param);
        for (int i = 0; i < mg_param.n_level; i++) {
          strcpy(mg_param.vec_outfile[i], vec_outfile[i]); // restore output file name
        }
      }
    }

    // free the multigrid solver
    destroyMultigridQuda(mg_preconditioner);

    delete gauge;
    delete gaugeEx;
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();
 
    randstates->Release();
    delete randstates;
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  //printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
  //inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, 0.0);

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  free(spinorIn);
  free(spinorCheck);
  free(spinorOut);

  return 0;
}
