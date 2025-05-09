#include <string.h>
#include <vector>
#include <fstream>
#include <array>
#include "quda.h"
#include "quda_internal.h"
#include "milc_interface_internal.hpp"
#include "invert_quda.h"

namespace quda
{

  bool mgInputStruct::update(std::vector<std::string> &input_line)
  {
    int error_code = 0; // no error
                        // 1 = wrong number of arguments

    if (parse_2_args(error_code, input_line, "mg_levels", [&](const char *input) { mg_levels = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "verify_results",
                            [&](const char *input) { verify_results = input[0] == 't' ? true : false; })) {
    } else if (parse_2_args(error_code, input_line, "preconditioner_precision",
                            [&](const char *input) { preconditioner_precision = getQudaPrecision(input); })) {
    } else if (parse_2_args(error_code, input_line, "optimized_kd",
                            [&](const char *input) { optimized_kd = getQudaTransferType(input); })) {
    } else if (parse_2_args(error_code, input_line, "use_mma", [&](const char *input) {
                 if (input[0] == 't') {
                   for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
                     setup_use_mma[i] = true;
                     dslash_use_mma[i] = true;
                     // transfer_use_mma[i] = true; // FIXME, stick with default for now
                     // collapse_mrhs[i] = true; // FIXME, stick with default for now
                   }
                 } else {
                   for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
                     setup_use_mma[i] = false;
                     dslash_use_mma[i] = false;
                     // transfer_use_mma[i] = false; // FIXME, stick with default for now
                     // collapse_mrhs[i] = false; // FIXME, stick with default for now
                   }
                 }
               })) {
    } else if (parse_2_args(error_code, input_line, "allow_truncation",
                            [&](const char *input) { allow_truncation = input[0] == 't' ? true : false; })) {
    } else if (parse_2_args(error_code, input_line, "dagger_approximation",
                            [&](const char *input) { dagger_approximation = input[0] == 't' ? true : false; })) {
    } else if (parse_2_args(error_code, input_line, "block_solver_batch_size",
                            [&](const char *input) { block_solver_batch_size = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "mg_verbosity",
                            [&](int level, const char *input) { mg_verbosity[level] = getQudaVerbosity(input); })) {
    }
    /* begin setup */
    else if (parse_3_args(error_code, input_line, "nvec",
                          [&](int level, const char *input) { nvec[level] = atoi(input); })) {
    } else if (parse_3_geo_args(error_code, input_line, "geo_block_size", [&](int level, std::array<const char *, 4> vals) {
                 for (int d = 0; d < 4; d++) geo_block_size[level][d] = atoi(vals[d]);
               })) {
    } else if (parse_3_args(error_code, input_line, "setup_inv",
                            [&](int level, const char *input) { setup_inv[level] = getQudaInverterType(input); })) {
    } else if (parse_3_args(error_code, input_line, "setup_tol",
                            [&](int level, const char *input) { setup_tol[level] = atof(input); })) {
    } else if (parse_3_args(error_code, input_line, "setup_maxiter",
                            [&](int level, const char *input) { setup_maxiter[level] = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "setup_ca_basis_size",
                            [&](int level, const char *input) { setup_ca_basis_size[level] = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "mg_vec_infile",
                            [&](int level, const char *input) { strcpy(mg_vec_infile[level], input); })) {
    } else if (parse_3_args(error_code, input_line, "mg_vec_outfile",
                            [&](int level, const char *input) { strcpy(mg_vec_outfile[level], input); })) {
    } else if (parse_3_args(error_code, input_line, "mg_vec_partfile", [&](int level, const char *input) {
                 mg_vec_partfile[level] = input[0] == 't' ? true : false;
               })) {
    }
    /* begin solvers */
    else if (parse_3_args(error_code, input_line, "coarse_solve_type",
                          [&](int level, const char *input) { coarse_solve_type[level] = getQudaSolveType(input); })) {
    } else if (parse_3_args(error_code, input_line, "coarse_solver",
                            [&](int level, const char *input) { coarse_solver[level] = getQudaInverterType(input); })) {
    } else if (parse_3_args(error_code, input_line, "coarse_solver_tol",
                            [&](int level, const char *input) { coarse_solver_tol[level] = atof(input); })) {
    } else if (parse_3_args(error_code, input_line, "coarse_solver_maxiter",
                            [&](int level, const char *input) { coarse_solver_maxiter[level] = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "coarse_solver_ca_basis_size",
                            [&](int level, const char *input) { coarse_solver_ca_basis_size[level] = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "smoother_type",
                            [&](int level, const char *input) { smoother_type[level] = getQudaInverterType(input); })) {
    } else if (parse_3_args(error_code, input_line, "nu_pre",
                            [&](int level, const char *input) { nu_pre[level] = atoi(input); })) {
    } else if (parse_3_args(error_code, input_line, "nu_post",
                            [&](int level, const char *input) { nu_post[level] = atoi(input); })) {
    }
    /* Begin deflation */
    else if (parse_2_args(error_code, input_line, "deflate_n_ev", [&](const char *input) { deflate_n_ev = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_n_kr",
                            [&](const char *input) { deflate_n_kr = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_max_restarts",
                            [&](const char *input) { deflate_max_restarts = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_tol", [&](const char *input) { deflate_tol = atof(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_block_size",
                            [&](const char *input) { deflate_block_size = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_use_poly_acc",
                            [&](const char *input) { deflate_use_poly_acc = input[0] == 't' ? true : false; })) {
    } else if (parse_2_args(error_code, input_line, "deflate_a_min",
                            [&](const char *input) { deflate_a_min = atof(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_poly_deg",
                            [&](const char *input) { deflate_poly_deg = atoi(input); })) {
    } else if (parse_2_args(error_code, input_line, "deflate_vec_partfile",
                            [&](const char *input) { deflate_vec_partfile = input[0] == 't' ? true : false; })) {
    } else {
      printf("Invalid option %s\n", input_line[0].c_str());
      return false;
    }

    if (error_code == 1) {
      // intentionally printf b/c we're only running this on rank zero
      printf("Input option %s has an invalid number of arguments\n", input_line[0].c_str());
      return false;
    }

    return true;
  }

  void milcSetMultigridEigParam(QudaEigParam &mg_eig_param, const mgInputStruct &input_struct, int level)
  {
    mg_eig_param.eig_type
      = (input_struct.deflate_block_size > 1) ? QUDA_EIG_BLK_TR_LANCZOS : QUDA_EIG_TR_LANCZOS; // mg_eig_type[level];
    mg_eig_param.spectrum = QUDA_SPECTRUM_SR_EIG; // mg_eig_spectrum[level];
    if ((mg_eig_param.eig_type == QUDA_EIG_TR_LANCZOS || mg_eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS)
        && !(mg_eig_param.spectrum == QUDA_SPECTRUM_LR_EIG || mg_eig_param.spectrum == QUDA_SPECTRUM_SR_EIG)) {
      errorQuda("Only a real spectrum type (LR or SR) can be passed to a Lanczos type solver");
    }

    mg_eig_param.n_ev = input_struct.deflate_n_ev; // mg_eig_n_ev[level];
    mg_eig_param.n_kr = input_struct.deflate_n_kr; // mg_eig_n_kr[level];
    mg_eig_param.n_conv = input_struct.nvec[level];
    mg_eig_param.n_ev_deflate = -1; // deflate everything that converged
    mg_eig_param.compute_evals_batch_size
      = (input_struct.nvec[level] % 16 == 0) ? 16 : 1; // compute the eigenvalues in appropriate batches
    mg_eig_param.block_size
      = (mg_eig_param.eig_type == QUDA_EIG_TR_LANCZOS || mg_eig_param.eig_type == QUDA_EIG_IR_ARNOLDI) ?
      1 :
      input_struct.deflate_block_size; // mg_eig_block_size[level];
    mg_eig_param.batched_rotate = 0;   // mg_eig_batched_rotate[level];
    mg_eig_param.require_convergence
      = QUDA_BOOLEAN_TRUE; // mg_eig_require_convergence[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    mg_eig_param.tol = input_struct.deflate_tol;                   // mg_eig_tol[level];
    mg_eig_param.check_interval = 10;                              // mg_eig_check_interval[level];
    mg_eig_param.max_restarts = input_struct.deflate_max_restarts; // mg_eig_max_restarts[level];

    mg_eig_param.compute_svd = QUDA_BOOLEAN_FALSE;
    mg_eig_param.use_norm_op = QUDA_BOOLEAN_TRUE; // mg_eig_use_normop[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_eig_param.use_dagger = QUDA_BOOLEAN_FALSE; // mg_eig_use_dagger[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    mg_eig_param.use_poly_acc = input_struct.deflate_use_poly_acc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_eig_param.poly_deg = input_struct.deflate_poly_deg; // mg_eig_poly_deg[level];
    mg_eig_param.a_min = input_struct.deflate_a_min;       // mg_eig_amin[level];
    mg_eig_param.a_max = 0.0;                              // compute estimate // mg_eig_amax[level];

    // set file i/o parameters
    // Give empty strings, Multigrid will handle IO.
    strcpy(mg_eig_param.vec_infile, "");
    strcpy(mg_eig_param.vec_outfile, "");
    mg_eig_param.io_parity_inflate = QUDA_BOOLEAN_FALSE; // do not inflate coarse vectors
    mg_eig_param.save_prec = QUDA_SINGLE_PRECISION;      // cannot save in fixed point
    mg_eig_param.partfile = QUDA_BOOLEAN_FALSE;          // ignored, multigrid parameters take precedence

    strcpy(mg_eig_param.QUDA_logfile, "" /*eig_QUDA_logfile*/);
  }

  void milcSetMultigridParam(milcMultigridPack *mg_pack, QudaPrecision host_precision, QudaPrecision device_precision,
                             QudaPrecision device_precision_sloppy, double mass, const char *const mg_param_file)
  {
    static const QudaVerbosity verbosity = getVerbosity();
    QudaMultigridParam &mg_param = mg_pack->mg_param;

    auto &input_struct = mg_pack->input_struct;

    // Load input struct on rank 0
    if (comm_rank() == 0) {
      std::ifstream input_file(mg_param_file, std::ios_base::in);

      if (!input_file.is_open()) { errorQuda("MILC interface MG input file %s does not exist!", mg_param_file); }

      // enter parameter loop
      char buffer[1024];
      std::vector<std::string> elements;
      while (!input_file.eof()) {

        elements.clear();

        // get line
        input_file.getline(buffer, 1024);

        // split on spaces, tabs
        char *pch = strtok(buffer, " \t");
        while (pch != nullptr) {
          elements.emplace_back(std::string(pch));
          pch = strtok(nullptr, " \t");
        }

        // skip empty lines, comments
        if (elements.size() == 0 || elements[0][0] == '#') continue;

        // debug: print back out
        if (verbosity == QUDA_VERBOSE) {
          for (auto elem : elements) { printf("%s ", elem.c_str()); }
          printf("\n");
        }

        input_struct.update(elements);
      }
    }

    comm_barrier();
    comm_broadcast((void *)&input_struct, sizeof(mgInputStruct));

    auto mg_levels = input_struct.mg_levels;

    // Prepare eigenvector params
    for (int i = 0; i < mg_levels; i++) {
      mg_pack->mg_eig_param[i] = newQudaEigParam();
      milcSetMultigridEigParam(mg_pack->mg_eig_param[i], input_struct, i);
    }

    mg_pack->mg_inv_param = newQudaInvertParam();
    mg_pack->mg_param = newQudaMultigridParam();
    mg_pack->last_mass = mass;

    mg_pack->mg_param.invert_param = &mg_pack->mg_inv_param;
    for (int i = 0; i < mg_levels; i++) { mg_pack->mg_param.eig_param[i] = &mg_pack->mg_eig_param[i]; }

    QudaInvertParam &inv_param
      = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

    inv_param.Ls = 1;

    inv_param.cpu_prec = host_precision;
    inv_param.cuda_prec = device_precision;
    inv_param.cuda_prec_sloppy = device_precision_sloppy;
    inv_param.cuda_prec_precondition = input_struct.preconditioner_precision;
    inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    inv_param.dirac_order = QUDA_DIRAC_ORDER;

    inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
    inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

    inv_param.dslash_type = QUDA_ASQTAD_DSLASH; // dslash_type;

    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (4.0 + inv_param.mass));

    inv_param.dagger = QUDA_DAG_NO;
    inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

    // this gets ignored
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN; // matpc_type;

    // req'd for staggered/hisq
    inv_param.solution_type = QUDA_MAT_SOLUTION;

    auto solve_type = QUDA_DIRECT_SOLVE;
    inv_param.solve_type = solve_type;

    // whether or not we allow dropping a long link when an aggregation size is smaller than 3
    mg_param.allow_truncation = input_struct.allow_truncation ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // whether or not we use the dagger approximation
    mg_param.staggered_kd_dagger_approximation
      = input_struct.dagger_approximation ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    mg_param.invert_param = &inv_param;
    mg_param.n_level = mg_levels; // set from file

    for (int i = 0; i < mg_param.n_level; i++) {

      if (i == 0) {
        for (int j = 0; j < 4; j++) {
          mg_param.geo_block_size[i][j]
            = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? 2 : 1; // Kahler-Dirac blocking
        }
      } else {
        for (int j = 0; j < 4; j++) { mg_param.geo_block_size[i][j] = input_struct.geo_block_size[i][j]; }
      }

      // mg_param.use_eig_solver[i] = QUDA_BOOLEAN_FALSE; //mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      if (i == mg_param.n_level - 1 && input_struct.nvec[i] > 0) {
        mg_param.use_eig_solver[i] = QUDA_BOOLEAN_TRUE;
      } else {
        mg_param.use_eig_solver[i] = QUDA_BOOLEAN_FALSE;
      }

      mg_param.verbosity[i] = input_struct.mg_verbosity[i];
      mg_param.setup_use_mma[i] = input_struct.setup_use_mma[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      mg_param.dslash_use_mma[i] = input_struct.dslash_use_mma[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      mg_param.transfer_use_mma[i] = input_struct.transfer_use_mma[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      mg_param.collapse_mrhs[i] = input_struct.collapse_mrhs[i];
      mg_param.setup_inv_type[i] = input_struct.setup_inv[i];
      mg_param.num_setup_iter[i] = 1; // num_setup_iter[i];
      mg_param.setup_tol[i] = input_struct.setup_tol[i];
      mg_param.setup_maxiter[i] = input_struct.setup_maxiter[i];

      // Basis to use for CA solver setup --- heuristic for CA-GCR is empirical
      if (is_ca_solver(input_struct.setup_inv[i])) {
        if (input_struct.setup_inv[i] == QUDA_CA_GCR_INVERTER && input_struct.setup_ca_basis_size[i] <= 8)
          mg_param.setup_ca_basis[i] = QUDA_POWER_BASIS;
        else
          mg_param.setup_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // setup_ca_basis[i];
      } else {
        mg_param.setup_ca_basis[i] = QUDA_POWER_BASIS; // setup_ca_basis[i];
      }

      // Basis size for CA solver setup
      mg_param.setup_ca_basis_size[i] = input_struct.setup_ca_basis_size[i];

      // Minimum and maximum eigenvalue for Chebyshev CA basis setup
      mg_param.setup_ca_lambda_min[i] = 0.0;  // setup_ca_lambda_min[i];
      mg_param.setup_ca_lambda_max[i] = -1.0; // use power iterations // setup_ca_lambda_max[i];

      mg_param.spin_block_size[i] = 1;
      // change this to refresh fields when mass or links change
      mg_param.setup_maxiter_refresh[i] = 0; // setup_maxiter_refresh[i];
      mg_param.n_vec[i]
        = (i == 0) ? ((input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? 24 : 3) : input_struct.nvec[i];
      mg_param.n_vec_batch[i] = (i == 0) ? 1 : (mg_param.n_vec[i] % 16 == 0 ? 16 : 1);
      mg_param.n_block_ortho[i] = 2; // n_block_ortho[i];                          // number of times to Gram-Schmidt
      mg_param.precision_null[i] = input_struct.preconditioner_precision; // precision to store the null-space basis
      mg_param.smoother_halo_precision[i]
        = input_struct.preconditioner_precision; // precision of the halo exchange in the smoother
      mg_param.nu_pre[i] = input_struct.nu_pre[i];
      mg_param.nu_post[i] = input_struct.nu_post[i];
      mg_param.mu_factor[i] = 1.; // mu_factor[i];

      mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

      // top level: coarse vs optimized KD, otherwise standard aggregation.
      if (i == 0) {
        mg_param.transfer_type[i] = input_struct.optimized_kd;
      } else {
        mg_param.transfer_type[i] = QUDA_TRANSFER_AGGREGATE;
      }

      // set the coarse solver wrappers including bottom solver
      mg_param.coarse_solver[i] = input_struct.coarse_solver[i];
      mg_param.coarse_solver_tol[i] = input_struct.coarse_solver_tol[i];
      mg_param.coarse_solver_maxiter[i] = input_struct.coarse_solver_maxiter[i];

      // Basis size for CA coarse solvers
      if (input_struct.coarse_solver_ca_basis_size[i] > input_struct.coarse_solver_maxiter[i]) {
        mg_param.coarse_solver_ca_basis_size[i] = input_struct.coarse_solver_maxiter[i];
      } else {
        mg_param.coarse_solver_ca_basis_size[i] = input_struct.coarse_solver_ca_basis_size[i];
      }

      // Basis to use for CA basis coarse solvers --- heuristic for CA-GCR is empirical
      if (is_ca_solver(input_struct.coarse_solver[i])) {
        if (input_struct.coarse_solver[i] == QUDA_CA_GCR_INVERTER && mg_param.coarse_solver_ca_basis_size[i] <= 8)
          mg_param.coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
        else
          mg_param.coarse_solver_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // coarse_solver_ca_basis[i];
      } else {
        mg_param.coarse_solver_ca_basis[i] = QUDA_POWER_BASIS; // coarse_solver_ca_basis[i];
      }

      // Minimum and maximum eigenvalue for Chebyshev CA basis
      mg_param.coarse_solver_ca_lambda_min[i] = 0.0;  // coarse_solver_ca_lambda_min[i];
      mg_param.coarse_solver_ca_lambda_max[i] = -1.0; // use power iterations // coarse_solver_ca_lambda_max[i];

      mg_param.smoother[i] = input_struct.smoother_type[i];

      // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
      mg_param.smoother_tol[i] = 1e-10; // smoother_tol[i];

      // Basis to use for CA basis smoothers --- heuristic for CA-GCR is empirical
      if (is_ca_solver(input_struct.smoother_type[i])) {
        if (input_struct.smoother_type[i] == QUDA_CA_GCR_INVERTER && mg_param.nu_pre[i] <= 8 && mg_param.nu_post[i] <= 8)
          mg_param.smoother_solver_ca_basis[i] = QUDA_POWER_BASIS;
        else
          mg_param.smoother_solver_ca_basis[i] = QUDA_CHEBYSHEV_BASIS; // smoother_solver_ca_basis[i];
      } else {
        mg_param.smoother_solver_ca_basis[i] = QUDA_POWER_BASIS; // smoother_solver_ca_basis[i];
      }

      // Minimum and maximum eigenvalue for Chebyshev CA basis smoothers
      mg_param.smoother_solver_ca_lambda_min[i] = 0.0;  // smoother_solver_ca_lambda_min[i];
      mg_param.smoother_solver_ca_lambda_max[i] = -1.0; // smoother_solver_ca_lambda_max[i];

      // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
      // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
      // from test routines: // smoother_solve_type[i];
      switch (i) {
      case 0: mg_param.smoother_solve_type[0] = QUDA_DIRECT_SOLVE; break;
      case 1:
        mg_param.smoother_solve_type[1]
          = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? QUDA_DIRECT_PC_SOLVE : QUDA_DIRECT_SOLVE;
        break;
      default: mg_param.smoother_solve_type[i] = input_struct.coarse_solve_type[i]; break;
      }

      // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
      mg_param.smoother_schwarz_type[i] = QUDA_INVALID_SCHWARZ; // schwarz_type[i];

      // if using Schwarz preconditioning then use local reductions only
      mg_param.global_reduction[i]
        = QUDA_BOOLEAN_TRUE; // (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

      // set number of Schwarz cycles to apply
      mg_param.smoother_schwarz_cycle[i] = 1; // schwarz_cycle[i];

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

        // Always this for now
        if (solve_type == QUDA_DIRECT_SOLVE) {
          mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
        } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
          mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
        } else {
          errorQuda("Unexpected solve_type = %d\n", solve_type);
        }

      } else if (i == 1) {

        // Always this for now.
        mg_param.coarse_grid_solution_type[i]
          = (input_struct.optimized_kd == QUDA_TRANSFER_COARSE_KD) ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;
      } else {

        if (input_struct.coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
          mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
        } else if (input_struct.coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
          mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
        } else {
          errorQuda("unexpected solve type = %d\n", input_struct.coarse_solve_type[i]);
        }
      }

      mg_param.omega[i] = 0.85; // ignored // omega; // over/under relaxation factor

      mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;       //  solver_location[i];
      mg_param.setup_location[i] = QUDA_CUDA_FIELD_LOCATION; // setup_location[i];
    }

    // coarsening the spin on the first restriction is undefined for staggered fields.
    mg_param.spin_block_size[0] = 0;
    if (input_struct.optimized_kd == QUDA_TRANSFER_OPTIMIZED_KD
        || input_struct.optimized_kd == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
      mg_param.spin_block_size[1] = 0;

    mg_param.setup_type = QUDA_NULL_VECTOR_SETUP;     // setup_type;
    mg_param.pre_orthonormalize = QUDA_BOOLEAN_FALSE; // pre_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.post_orthonormalize = QUDA_BOOLEAN_TRUE; // post_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    mg_param.compute_null_vector
      = QUDA_COMPUTE_NULL_VECTOR_YES; // generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;

    mg_param.generate_all_levels = QUDA_BOOLEAN_TRUE; // generate_all_levels ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    mg_param.run_verify = input_struct.verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.run_low_mode_check = QUDA_BOOLEAN_FALSE;     // low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.run_oblique_proj_check = QUDA_BOOLEAN_FALSE; // oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.preserve_deflation = QUDA_BOOLEAN_TRUE;      // FIXME, controversial, should update if mass changes?

    // set file i/o parameters
    for (int i = 0; i < mg_param.n_level; i++) {
      strcpy(mg_param.vec_infile[i], input_struct.mg_vec_infile[i]);
      strcpy(mg_param.vec_outfile[i], input_struct.mg_vec_outfile[i]);
      if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
      if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
      if (i != mg_param.n_level - 1)
        mg_param.mg_vec_partfile[i] = input_struct.mg_vec_partfile[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      else
        mg_param.mg_vec_partfile[i] = input_struct.deflate_vec_partfile ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }

    mg_param.coarse_guess = QUDA_BOOLEAN_FALSE; // mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // these need to tbe set for now but are actually ignored by the MG setup
    // needed to make it pass the initialization test
    inv_param.inv_type = QUDA_GCR_INVERTER;
    inv_param.tol = 1e-10;
    inv_param.maxiter = 1000;
    inv_param.reliable_delta = 1e-6; // reliable_delta;
    inv_param.gcrNkrylov = 10;

    inv_param.verbosity = verbosity;

    inv_param.verbosity = input_struct.mg_verbosity[0];

    // We need to pass this back to the fat/long links for the outer-most level.
    mg_pack->preconditioner_precision = input_struct.preconditioner_precision;
  }

}; // namespace quda
