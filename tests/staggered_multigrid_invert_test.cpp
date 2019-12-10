#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_reference.h>
#include "misc.h"
#include <gauge_field.h>
#include <covdev_reference.h>
#include <unitarization_links.h>
#include <random_quda.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <blas_quda.h>

// Various CPU fields lifted from
// staggered_invert_test.cpp

#define mySpinorSiteSize 6

static int n_naiks = 1;


#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif

cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;


QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

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
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return ;
}

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_sloppy = prec_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  gauge_param.cuda_prec_precondition = prec_precondition;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gauge_param.anisotropy = 1.0;

  // For asqtad:
  //gauge_param.tadpole_coeff = tadpole_coeff;
  //gauge_param.scale = dslash_type != QUDA_ASQTAD_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gauge_param.tadpole_coeff = 1.0;
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.scale = -1.0/24.0;
    if (eps_naik != 0) {
      gauge_param.scale *= (1.0+eps_naik);
    }
  } else {
    gauge_param.scale = 1.0;
  }
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  gauge_param.ga_pad = 0;

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

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;//this will be used to setup SolverParam parent in MGParam class

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

  mg_param.is_staggered = QUDA_BOOLEAN_YES;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
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
    mg_param.n_block_ortho[i] = n_block_ortho[i]; // number of times to Gram-Schmidt
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
    nu_pre[i] = 2;
    nu_post[i] = 2;
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_NO;

  // coarsening the spin on the first restriction is undefined for staggered fields.
  mg_param.spin_block_size[0] = 0;

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

void setInvertParam(QudaInvertParam &inv_param) {

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

  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  /* ESW HACK: comment this out to do a non-MG solve. */
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  //inv_param.verbosity_precondition = mg_verbosity[0];
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
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

int main(int argc, char **argv)
{
  // We give here the default values to some of the array
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SUMMARIZE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    mu_factor[i] = 1.;
    coarse_solve_type[i] = QUDA_INVALID_SOLVE;
    smoother_solve_type[i] = QUDA_INVALID_SOLVE;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_GCR_INVERTER; 
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
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

  // Give the dslash type a reasonable default.
  dslash_type = QUDA_STAGGERED_DSLASH;

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
  for(int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }


  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  // Need to add support for LAPLACE
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // ESW HACK: needs to be addressed
  /*if (solve_type == QUDA_DIRECT_PC_SOLVE || coarse_solve_type[0] == QUDA_DIRECT_PC_SOLVE || smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE) {
    printfQuda("staggered_multigtid_invert_test doesn't support preconditioned outer solve yet.\n");
    exit(0);
  }*/

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];

  // Since the top level is just a unitary rotation, we manually
  // set mg_eig[0] to false (and throw a warning if a user set it to true)
  if (mg_eig[0]) {
    printfQuda("Warning: Cannot specify near-null vectors for top level.\n");
    mg_eig[0] = false;
  }
  for(int i=0; i<mg_levels; i++) {
    mg_eig_param[i] = newQudaEigParam();
    setEigParam(mg_eig_param[i], i);
  }

  setGaugeParam(gauge_param);
  setInvertParam(inv_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();

  mg_param.invert_param = &mg_inv_param;
  for(int i=0; i<mg_levels; i++) {
    mg_param.eig_param[i] = &mg_eig_param[i];
  }

  setMultigridParam(mg_param);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.


  setDims(gauge_param.X);
  dw_setDims(gauge_param.X,1); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);


  /* Taken from staggered_invert_test to load gauge fields */
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* milc_fatlink = nullptr;
  void* milc_longlink = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  milc_fatlink = malloc(4*V*gaugeSiteSize*gSize);
  milc_longlink = malloc(4*V*gaugeSiteSize*gSize);

  // for load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  // load a field WITHOUT PHASES
  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_fatlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gauge_param.cpu_prec,&gauge_param,compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
    //createSiteLinkCPU(inlink, gauge_param.cpu_prec, 0); // 0 for no phases
  }

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);

  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you 
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      memset(qdp_longlink[dir],0,V*gaugeSiteSize*gSize);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {
      computeFatLongGPU(qdp_fatlink, qdp_longlink, qdp_inlink, gauge_param, gSize, n_naiks, eps_naik);
    } else { //

      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      }
    }

    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);

    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  }

  // Alright, we've created all the void** links.
  // Create the void* pointers
  reorderQDPtoMILC(milc_fatlink,qdp_fatlink,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink,qdp_longlink,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);


  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  csParam.x[4] = 1;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;  
  in = new cpuColorSpinorField(csParam);  
  out = new cpuColorSpinorField(csParam);  
  ref = new cpuColorSpinorField(csParam);  
  tmp = new cpuColorSpinorField(csParam);  

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField* cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField* cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif
  
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct = gauge_param.reconstruct_sloppy = gauge_param.reconstruct_refinement_sloppy
      = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_precondition = link_recon_precondition;
    loadGaugeQuda(milc_longlink, &gauge_param);
  }

  /* end stuff stolen from staggered_invert_test */

  //if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.solve_type = solve_type; // restore actual solve_type we want to do

  /* ESW HACK: comment this out to do a non-MG solve. */

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;

  // Test: create a dummy invert param just to make sure
  // we're setting up gauge fields and such correctly.

  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  for (int k = 0; k < Nsrc; k++) {

    construct_spinor_source(in->V(), 1, 3, inv_param.cpu_prec, csParam.x, *rng);
    invertQuda(out->V(), in->V(), &inv_param);

    time[k] = inv_param.secs;
    gflops[k] = inv_param.gflops/inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs);

  }
  
  rng->Release();
  delete rng;

  double nrm2=0;
  double src2=0;

  int len = 0;
  if (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    len = V;
  } else {
    len = Vh;
  }

  // Check solution
  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

    //In QUDA, the full staggered operator has the sign convention
    //{{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
    //have the minus sign. Passing in QUDA_DAG_YES solves this
    //discrepancy
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Even()), qdp_fatlink, qdp_longlink, ghost_fatlink,
        ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Odd()), QUDA_EVEN_PARITY, QUDA_DAG_YES,
        inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Odd()), qdp_fatlink, qdp_longlink, ghost_fatlink,
        ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Even()), QUDA_ODD_PARITY, QUDA_DAG_YES,
        inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    //if (dslash_type == QUDA_LAPLACE_DSLASH) {
    //  xpay(out->V(), kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    //  ax(0.5/kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    //} else {
      axpy(2*mass, out->V(), ref->V(), ref->Length(), gauge_param.cpu_prec);
    //}

    // Reference debugging code: print the first component
    // of the even and odd partities within a solution vector.
    /*
    printfQuda("\nLength: %lu\n", ref->Length());

    // for verification
    printfQuda("\n\nEven:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Even().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Even().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Even().V()))[0]);

    printfQuda("\n\nOdd:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Odd().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Odd().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Odd().V()))[0]);
    printfQuda("\n\n");
    */

    mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);

  } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {

    matdagmat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, out, mass, 0, inv_param.cpu_prec,
                gauge_param.cpu_prec, tmp, QUDA_EVEN_PARITY, dslash_type);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      printfQuda("%f %f\n", ((float*)in->V())[12], ((float*)ref->V())[12]);
    } else {
      printfQuda("%f %f\n", ((double*)in->V())[12], ((double*)ref->V())[12]);
    }

    mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
  }

  double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
  double l2r = sqrt(nrm2/src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
      inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

  // Compute timings
  auto mean_time = 0.0;
  auto mean_time2 = 0.0;
  auto mean_gflops = 0.0;
  auto mean_gflops2 = 0.0;
  for (int i=0; i<Nsrc; i++) {
    mean_time += time[i];
    mean_time2 += time[i] * time[i];
    mean_gflops += gflops[i];
    mean_gflops2 += gflops[i] * gflops[i];
  }

  mean_time /= Nsrc;
  mean_time2 /= Nsrc;
  auto stddev_time = Nsrc > 1 ? sqrt((Nsrc / ((double)Nsrc - 1.0)) * (mean_time2 - mean_time * mean_time)) : std::numeric_limits<double>::infinity();
  mean_gflops /= Nsrc;
  mean_gflops2 /= Nsrc;
  auto stddev_gflops = Nsrc > 1 ? sqrt((Nsrc / ((double)Nsrc - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) : std::numeric_limits<double>::infinity();
  printfQuda("%d solves, with mean solve time %g (stddev = %g), mean GFLOPS %g (stddev = %g)\n", Nsrc, mean_time,
             stddev_time, mean_gflops, stddev_gflops);

  delete []time;
  delete []gflops;

  // Clean up gauge fields, at least
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) { free(qdp_inlink[dir]); qdp_inlink[dir] = nullptr; }
    if (qdp_fatlink[dir] != nullptr) { free(qdp_fatlink[dir]); qdp_fatlink[dir] = nullptr; }
    if (qdp_longlink[dir] != nullptr) { free(qdp_longlink[dir]); qdp_longlink[dir] = nullptr; }
  }
  if (milc_fatlink != nullptr) { free(milc_fatlink); milc_fatlink = nullptr; }
  if (milc_longlink != nullptr) { free(milc_longlink); milc_longlink = nullptr; }

  if (cpuFat != nullptr) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong != nullptr) { delete cpuLong; cpuLong = nullptr; }

  delete in;
  delete out;
  delete ref;
  delete tmp;

  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
