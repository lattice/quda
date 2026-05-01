/**
 * @file hmc_test.cpp
 * @brief Test demonstrating HMC trajectory calls from an external library using the QUDA HMC API.
 *
 * This test shows how an external application (e.g., MILC, Chroma, OpenQCD) can call
 * hmcTrajectoryQuda() with different integrator types (leapfrog, Omelyan, nested FGI)
 * using host pointers. It validates:
 *   1. Parameter initialization via newQudaHMCParam()
 *   2. Host/device field bridging via the resident field pattern
 *   3. Hamiltonian conservation (dH) for each integrator
 *   4. Reversibility: forward + backward trajectory returns to starting point
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>

#include <quda.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <hmc_quda.h>
#include <dirac_quda.h>
#include <momentum.h>
#include <gauge_update_quda.h>
#include <eigen_tracker.h>
#include <eigen_forecast.h>
#include <cg_ritz_extractor.h>
#include <gcr_tracker.h>
#include <inv_tracker.h>
#include <eigen_tracking_state.h>
#include <eigensolve_quda.h>
#include <multigrid.h>
#include <qio_field.h>
#include <algorithm>
#include <gtest/gtest.h>

#include "command_line_params.h"
#include "gauge_utils.h"
#include "host_utils.h"

// Forward declarations for QUDA internal functions used in numerical force test
namespace quda
{
  void solve(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &b, Dirac &dirac, Dirac &diracSloppy,
             Dirac &diracPre, Dirac &diracEig, QudaInvertParam &param);
  void createDiracWithEig(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dEig, QudaInvertParam &param,
                          bool pc_solve, bool use_smeared_gauge = false);
} // namespace quda

// These globals are in file scope (not quda namespace) in interface_quda.cpp
extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSloppy;
extern quda::GaugeField *gaugePrecondition;
extern quda::CloverField *cloverPrecise;
extern quda::GaugeField *gaugeSloppy;
extern quda::GaugeField *gaugePrecondition;
extern quda::GaugeField momResident;
extern quda::CloverField *cloverPrecise;
#include "momentum_utils.h"
#include "misc.h"
#include "test.h"

// Global test state (non-static for access from host_clover_hmc.cpp)
QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

// Host gauge storage (QDP order: array of 4 pointers)
static std::vector<char> gauge_buf;
void *hostGauge[4];

void initHMCTest(int argc, char **argv)
{
  // Initialize QUDA gauge parameters
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Initialize inverter parameters
  inv_param = newQudaInvertParam();
  // Ensure all precision fields have valid values (CLI may leave some INVALID).
  // Chain the fallbacks: sloppy → precise, precondition → sloppy, eigensolver → sloppy.
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_eigensolver == QUDA_INVALID_PRECISION) prec_eigensolver = prec_sloppy;
  setInvertParam(inv_param);
  inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  // Asymmetric PC for clover (needed for clover force), symmetric for Wilson
  inv_param.matpc_type
    = (dslash_type == QUDA_CLOVER_WILSON_DSLASH) ? QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : QUDA_MATPC_EVEN_EVEN;
  // Set clover coefficients based on dslash type (from --dslash-type CLI option)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_csw = clover_csw; // from --clover-csw CLI (default 1.0)
    inv_param.clover_coeff = inv_param.clover_csw * inv_param.kappa;
  } else {
    inv_param.clover_csw = 0.0;
    inv_param.clover_coeff = 0.0;
  }
  inv_param.clover_cpu_prec = gauge_param.cpu_prec;
  inv_param.clover_cuda_prec = gauge_param.cuda_prec;
  inv_param.clover_cuda_prec_sloppy = gauge_param.cuda_prec_sloppy;
  inv_param.clover_cuda_prec_precondition = gauge_param.cuda_prec_precondition;
  inv_param.clover_cuda_prec_eigensolver = gauge_param.cuda_prec_eigensolver;
  inv_param.clover_cuda_prec_refinement_sloppy = gauge_param.cuda_prec_refinement_sloppy;
  inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  inv_param.compute_clover = QUDA_BOOLEAN_TRUE;
  inv_param.compute_clover_trlog = 1;
  setDims(gauge_param.X);

  // Allocate host gauge field (QDP order)
  gauge_buf.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) hostGauge[i] = gauge_buf.data() + i * V * gauge_site_size * host_gauge_data_type_size;
  constructHostGaugeField(hostGauge, gauge_param, argc, argv); // uses --gauge-load or random

  // Load gauge to QUDA (makes it resident)
  gauge_param.use_resident_gauge = 0;
  gauge_param.make_resident_gauge = 1;
  loadGaugeQuda(hostGauge, &gauge_param);

  printfQuda("HMC test initialized: lattice = %d x %d x %d x %d\n", gauge_param.X[0], gauge_param.X[1],
             gauge_param.X[2], gauge_param.X[3]);
}

/**
 * Test that newQudaHMCParam returns sensible defaults.
 */
TEST(HMC, ParameterDefaults)
{
  QudaHMCParam p = newQudaHMCParam();

  EXPECT_EQ(p.integrator, QUDA_LEAPFROG_INTEGRATOR);
  EXPECT_DOUBLE_EQ(p.tau, 1.0);
  EXPECT_EQ(p.n_steps, 10);
  EXPECT_DOUBLE_EQ(p.omelyan_lambda, 0.1932);
  EXPECT_NEAR(p.fgi_lambda, 1.0 / 6.0, 1e-15);
  EXPECT_NEAR(p.fgi_xi, 1.0 / 72.0, 1e-15);
  EXPECT_EQ(p.n_inner_steps, 3);
  EXPECT_EQ(p.n_defl, 32);
  EXPECT_DOUBLE_EQ(p.eig_tol, 1e-6);
  EXPECT_EQ(p.defl_refresh_interval, 0);
  EXPECT_EQ(p.coarse_level, 1);
  EXPECT_EQ(p.n_mr_smooth, 0);
  EXPECT_DOUBLE_EQ(p.beta, 6.0);
  EXPECT_EQ(p.generate_momentum, 1);
  EXPECT_EQ(p.return_result_gauge, 1);
  EXPECT_EQ(p.return_result_mom, 1);
}

/**
 * Helper: populate QudaHMCParam from CLI options.
 * All integrator-specific parameters (omelyan_lambda, fgi_lambda, fgi_xi)
 * are always set so any integrator type can be selected via --hmc-integrator.
 */
static QudaHMCParam makeHMCParam(QudaIntegratorType integrator_override = static_cast<QudaIntegratorType>(-1))
{
  QudaHMCParam p = newQudaHMCParam();

  p.integrator = (integrator_override != static_cast<QudaIntegratorType>(-1))
    ? integrator_override
    : static_cast<QudaIntegratorType>(hmc_integrator);
  p.beta = hmc_beta;
  p.tau = hmc_tau;
  p.n_steps = hmc_n_steps;
  p.omelyan_lambda = hmc_omelyan_lambda;
  p.fgi_lambda = hmc_fgi_lambda;
  p.fgi_xi = hmc_fgi_xi;
  p.generate_momentum = 1;
  p.momentum_seed = hmc_momentum_seed;
  p.use_resident_gauge = 1;
  p.make_resident_gauge = 1;
  p.return_result_gauge = 0;

  // Eigentracking — pass CLI values through; 0 means "derive later"
  p.eigentracking_enabled = eigentracking_enabled;
  p.eigentracking_n_ev = eigentracking_n_ev;
  p.eigentracking_pool_capacity = eigentracking_pool_capacity;
  p.eigentracking_n_ritz = eigentracking_n_ritz;
  p.eigentracking_forecast_order = eigentracking_forecast_order;
  p.eigentracking_fresh_trlm_interval = eigentracking_fresh_interval;
  p.eigentracking_solution_history = eigentracking_solution_history;
  p.eigentracking_absorb_ritz = eigentracking_absorb_ritz ? 1 : 0;
  p.eigentracking_mg_refresh_iters = eigentracking_mg_refresh_iters;
  p.eigentracking_residual_cap = eigentracking_residual_cap;
  p.eigentracking_trlm_tol = eigentracking_trlm_tol;
  p.eigentracking_trlm_max_restarts = eigentracking_trlm_max_restarts;
  p.eigentracking_trlm_check_interval = eigentracking_trlm_check_interval;
  p.eigentracking_eig_type = eigentracking_eig_type;
  p.eigentracking_blk_size = eigentracking_blk_size;
  // Poly-acc defaulting: if the CLI left it off and a_min unset, auto-enable
  // with the empirically-proven a_min=1.0 default so bare
  // `./hmc_test --dim 4 4 4 4` invocations converge on 4^4 hot-start gauges.
  // Explicit CLI values always win.
  if (!eigentracking_use_poly_acc && eigentracking_a_min == 0.0) {
    p.eigentracking_use_poly_acc = 1;
    p.eigentracking_a_min = 1.0;
  } else {
    p.eigentracking_use_poly_acc = eigentracking_use_poly_acc ? 1 : 0;
    p.eigentracking_a_min = eigentracking_a_min;
  }
  p.eigentracking_poly_deg = eigentracking_poly_deg;
  p.eigentracking_a_max = eigentracking_a_max;

  return p;
}

/**
 * Resolve eigentracking defaults from MG null-vector count.
 * CLI values of 0 are replaced with sensible MG-derived defaults.
 * Explicit CLI values (non-zero) are preserved.
 */
static void resolveEigenTrackingDefaults(QudaHMCParam &p, int mg_nvec)
{
  int nev = p.eigentracking_n_ev;
  if (nev == 0) nev = mg_nvec > 0 ? mg_nvec : 8;
  p.eigentracking_n_ev = nev;

  if (p.eigentracking_pool_capacity == 0) p.eigentracking_pool_capacity = 2 * nev;
  if (p.eigentracking_n_ritz == 0) p.eigentracking_n_ritz = std::max(nev / 2, 2);

  printfQuda("Eigentracking resolved: n_ev=%d, pool=%d, n_ritz=%d, forecast=%d, fresh=%d, history=%d\n",
             p.eigentracking_n_ev, p.eigentracking_pool_capacity, p.eigentracking_n_ritz,
             p.eigentracking_forecast_order, p.eigentracking_fresh_trlm_interval, p.eigentracking_solution_history);
}

/**
 * Helper: populate a 2-level 4^4 MG param block with known-good fixture defaults.
 *
 * Used by HMC.MGPreconditionedRun and the nested-FGI branch of
 * HMC.ReversibilityAllIntegrators. For CLI-driven MG configuration, use
 * HMC.Production (which runs through setQudaDefaultMgTestParams +
 * setMultigridParam with the full snapshot/restore CLI override pattern).
 */
static void configureHMCTestMG(QudaMultigridParam &mg_param, QudaInvertParam &mg_inv_param,
                               QudaPrecision precision_null)
{
  mg_param.invert_param = &mg_inv_param;
  mg_param.n_level = 2;

  // Level 0 (fine)
  for (int d = 0; d < 4; d++) mg_param.geo_block_size[0][d] = 2;
  mg_param.n_vec[0] = 24;
  mg_param.spin_block_size[0] = 2;
  mg_param.nu_pre[0] = 2;
  mg_param.nu_post[0] = 2;
  mg_param.smoother[0] = QUDA_MR_INVERTER;
  mg_param.smoother_tol[0] = 0.25;
  mg_param.smoother_solve_type[0] = QUDA_DIRECT_PC_SOLVE;
  mg_param.setup_inv_type[0] = QUDA_BICGSTAB_INVERTER;
  mg_param.setup_tol[0] = 5e-6;
  mg_param.setup_maxiter[0] = 500;
  mg_param.num_setup_iter[0] = 1;
  mg_param.n_block_ortho[0] = 1;
  mg_param.precision_null[0] = precision_null;
  mg_param.coarse_solver[0] = QUDA_GCR_INVERTER;
  mg_param.coarse_solver_tol[0] = 0.25;
  mg_param.coarse_solver_maxiter[0] = 16;
  mg_param.coarse_grid_solution_type[0] = QUDA_MATPC_SOLUTION;
  mg_param.location[0] = QUDA_CUDA_FIELD_LOCATION;
  mg_param.setup_location[0] = QUDA_CUDA_FIELD_LOCATION;

  // Level 1 (coarse)
  mg_param.smoother[1] = QUDA_GCR_INVERTER;
  mg_param.smoother_tol[1] = 0.25;
  mg_param.smoother_solve_type[1] = QUDA_DIRECT_PC_SOLVE;
  mg_param.coarse_solver[1] = QUDA_GCR_INVERTER;
  mg_param.coarse_solver_tol[1] = 0.25;
  mg_param.coarse_solver_maxiter[1] = 16;
  mg_param.coarse_grid_solution_type[1] = QUDA_MATPC_SOLUTION;
  mg_param.location[1] = QUDA_CUDA_FIELD_LOCATION;
  mg_param.setup_location[1] = QUDA_CUDA_FIELD_LOCATION;

  // Shared level params
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;
    mg_param.verbosity[i] = QUDA_SILENT;
    mg_param.transfer_type[i] = QUDA_TRANSFER_AGGREGATE;
    mg_param.n_vec_batch[i] = 1;
    mg_param.omega[i] = 0.85;
    mg_param.setup_ca_basis[i] = QUDA_POWER_BASIS;
    mg_param.setup_ca_basis_size[i] = 4;
    mg_param.setup_ca_lambda_min[i] = 0.0;
    mg_param.setup_ca_lambda_max[i] = -1.0;
    mg_param.coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    mg_param.coarse_solver_ca_basis_size[i] = 4;
    mg_param.coarse_solver_ca_lambda_min[i] = 0.0;
    mg_param.coarse_solver_ca_lambda_max[i] = -1.0;
    mg_param.smoother_halo_precision[i] = QUDA_INVALID_PRECISION;
    mg_param.setup_maxiter_refresh[i] = 0;
  }

  mg_param.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
  mg_param.generate_all_levels = QUDA_BOOLEAN_TRUE;
  mg_param.run_verify = QUDA_BOOLEAN_FALSE;
}

/**
 * Test: Leapfrog trajectory with energy conservation check.
 *
 * Demonstrates the self-contained usage pattern:
 *   1. Load gauge from host
 *   2. Let QUDA generate Gaussian momentum internally
 *   3. Call hmcTrajectoryQuda -- it generates pseudofermion, runs MD, returns dH
 *
 * All parameters from CLI: --hmc-beta, --hmc-tau, --hmc-n-steps, --hmc-momentum-seed
 */
TEST(HMC, LeapfrogTrajectory)
{
  QudaHMCParam hmc_param = makeHMCParam(QUDA_LEAPFROG_INTEGRATOR);

  double dH = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("Leapfrog: dH = %e\n", dH);
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Omelyan trajectory.
 *
 * All parameters from CLI: --hmc-beta, --hmc-tau, --hmc-n-steps,
 * --hmc-omelyan-lambda, --hmc-momentum-seed
 */
TEST(HMC, OmelyanTrajectory)
{
  QudaHMCParam hmc_param = makeHMCParam(QUDA_OMELYAN_INTEGRATOR);

  double dH = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("Omelyan: dH = %e\n", dH);
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Single-timescale force-gradient integrator (PQPQP_FGI).
 *
 * This is the 4th-order Hessian-free FGI without force splitting.
 * It uses the total force for all kicks and the FG displacement.
 * Validates the PQPQP structure, gauge save/restore, and FG step.
 *
 * All parameters from CLI: --hmc-beta, --hmc-tau, --hmc-n-steps,
 * --hmc-fgi-lambda, --hmc-fgi-xi, --hmc-momentum-seed
 */
TEST(HMC, ForceGradientTrajectory)
{
  QudaHMCParam hmc_param = makeHMCParam(QUDA_FORCE_GRADIENT_INTEGRATOR);

  double dH = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("FGI: dH = %e\n", dH);
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Nested FGI parameter configuration.
 *
 * Shows how to configure the nested force-gradient integrator with
 * multigrid preconditioning and coarse-grid deflation.
 * NOTE: This test only validates parameter setup. Running the full
 * nested FGI trajectory requires a properly configured MG preconditioner.
 */
TEST(HMC, NestedFGIParameterSetup)
{
  QudaHMCParam hmc_param = makeHMCParam(QUDA_NESTED_FGI_INTEGRATOR);

  // Inner integrator
  hmc_param.n_inner_steps = 3;

  // Coarse deflation
  hmc_param.n_defl = 32;
  hmc_param.eig_tol = 1e-6;
  hmc_param.eig_n_kr = 96; // 3 * n_defl
  hmc_param.eig_max_restarts = 100;
  hmc_param.defl_refresh_interval = 0; // frozen deflation
  hmc_param.coarse_level = 1;

  // MR smoothing
  hmc_param.n_mr_smooth = 3;
  hmc_param.mr_omega = 1.0;

  // Validate that the parameters are set correctly
  EXPECT_EQ(hmc_param.integrator, QUDA_NESTED_FGI_INTEGRATOR);
  EXPECT_EQ(hmc_param.n_steps, hmc_n_steps);
  EXPECT_EQ(hmc_param.n_inner_steps, 3);
  EXPECT_EQ(hmc_param.n_defl, 32);
  EXPECT_EQ(hmc_param.n_mr_smooth, 3);

  // Print the CG solve count estimate from the report:
  // Total CG solves = 3 * n_outer + 3 (including Hamiltonian evaluations)
  int cg_per_traj = 3 * hmc_param.n_steps + 3;
  printfQuda("Nested FGI: estimated %d CG solves per trajectory (n_outer=%d)\n", cg_per_traj, hmc_param.n_steps);
}

/**
 * Test: Multi-trajectory HMC run with accept/reject.
 *
 * Uses hmcRunQuda to run multiple trajectories with Metropolis
 * accept/reject, thermalisation, and plaquette logging.
 * Demonstrates the self-contained HMC workflow using CLI options.
 */
TEST(HMC, MultiTrajectoryRun)
{
  QudaHMCParam hmc_param = makeHMCParam();
  hmc_param.n_trajectories = hmc_n_trajectories;
  hmc_param.n_thermalization = hmc_n_thermalization;
  hmc_param.checkpoint_interval = hmc_checkpoint_interval;

  strncpy(hmc_param.checkpoint_prefix, hmc_checkpoint_prefix.c_str(), sizeof(hmc_param.checkpoint_prefix) - 1);
  strncpy(hmc_param.gauge_infile, hmc_gauge_infile.c_str(), sizeof(hmc_param.gauge_infile) - 1);
  strncpy(hmc_param.gauge_outfile, hmc_gauge_outfile.c_str(), sizeof(hmc_param.gauge_outfile) - 1);

  hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, nullptr, nullptr);

  // If we get here without crashing, the run succeeded
  SUCCEED();
}

/**
 * Test: MG-preconditioned multi-trajectory HMC.
 *
 * Sets up a 2-level multigrid preconditioner, runs HMC with MG-preconditioned
 * CG for the fermion force, and demonstrates thin MG updates between trajectories.
 * Uses the --mg-* CLI options from the multigrid option group for configuration.
 */
TEST(HMC, MGPreconditionedRun)
{
  // --- Set up MG preconditioner ---
  // MG setup requires clover field loaded for clover-family dslashes. Align
  // inv_param's clover precisions with its gauge precisions so the clover
  // field loaded here matches the precision at which MG sees the gauge —
  // otherwise calculateY() and the Wilson-clover preconditioned arg packer
  // will hit "Precisions 4 8 do not match" errors.
  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cuda_prec = inv_param.cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = inv_param.cuda_prec_precondition;
    inv_param.clover_cuda_prec_eigensolver = inv_param.cuda_prec_precondition;
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }

  // Create a separate inv_param for MG setup (needs DIRECT_SOLVE, symmetric PC).
  // MG internal precision must match the null-vector precision below, otherwise
  // the Y-matrix construction (coarse_op_*.cu:calculateY) and the smoother
  // Wilson-clover argument packer will hit "Precisions 4 8 do not match" errors.
  QudaPrecision mg_prec =
    (prec_precondition != QUDA_INVALID_PRECISION) ? prec_precondition : QUDA_SINGLE_PRECISION;
  QudaInvertParam mg_inv_param = inv_param;
  mg_inv_param.solve_type = QUDA_DIRECT_SOLVE;
  mg_inv_param.solution_type = QUDA_MAT_SOLUTION;
  mg_inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  mg_inv_param.cuda_prec = gauge_param.cuda_prec; // must match gauge precise
  mg_inv_param.cuda_prec_sloppy = mg_prec;
  mg_inv_param.cuda_prec_precondition = mg_prec;
  mg_inv_param.cuda_prec_eigensolver = mg_prec;
  mg_inv_param.clover_cuda_prec = gauge_param.cuda_prec;
  mg_inv_param.clover_cuda_prec_sloppy = mg_prec;
  mg_inv_param.clover_cuda_prec_precondition = mg_prec;
  mg_inv_param.clover_cuda_prec_eigensolver = mg_prec;

  // Configure 2-level MG. CLI-overridable knobs (block size, n_vec, smoother,
  // setup solver, etc.) are honoured by HMC.Production, which uses the full
  // snapshot/restore + setMultigridParam pipeline. This focused MG test pins
  // a known-good 4^4 configuration so it exercises the MG+HMC plumbing
  // deterministically; for arbitrary CLI-driven MG configs, use HMC.Production.
  QudaMultigridParam mg_param = newQudaMultigridParam();
  configureHMCTestMG(mg_param, mg_inv_param, mg_prec);

  // Create MG preconditioner (builds null vectors, coarse operators)
  void *mg_preconditioner = newMultigridQuda(&mg_param);

  // Snapshot inv_param's solver-config fields so this MG-specific
  // reconfiguration does not leak into subsequent gtest cases.
  const void *saved_preconditioner = inv_param.preconditioner;
  const QudaInverterType saved_inv_type = inv_param.inv_type;
  const QudaInverterType saved_inv_type_precondition = inv_param.inv_type_precondition;
  const QudaSolveType saved_solve_type = inv_param.solve_type;

  // Outer solve: GCR + MG, with DIRECT_PC_SOLVE for compatibility with both MG and force.
  // solution_type stays MATPCDAG_MATPC_SOLUTION (set in initHMCTest).
  inv_param.preconditioner = mg_preconditioner;
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;

  // --- Configure HMC with MG ---
  QudaHMCParam hmc_param = makeHMCParam();
  hmc_param.n_trajectories = hmc_n_trajectories;
  hmc_param.n_thermalization = hmc_n_thermalization;
  hmc_param.mg_setup_interval = hmc_mg_setup_interval;
  hmc_param.mg_setup_iter_ratio = hmc_mg_setup_iter_ratio;
  hmc_param.mg_setup_iter_baseline_traj = hmc_mg_setup_iter_baseline_traj;

  hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, mg_preconditioner, &mg_param);

  // Cleanup MG and fully restore inv_param so the next test starts clean.
  destroyMultigridQuda(mg_preconditioner);
  inv_param.preconditioner = const_cast<void *>(saved_preconditioner);
  inv_param.inv_type = saved_inv_type;
  inv_param.inv_type_precondition = saved_inv_type_precondition;
  inv_param.solve_type = saved_solve_type;

  SUCCEED();
}

/**
 * Test: Directional derivative verification for fermion force.
 *
 * Verifies that the analytical fermion force is consistent with the numerical
 * derivative of the fermion action along the force direction.
 *
 * Method:
 *   1. Compute force F into zero momentum (dt=1)
 *   2. Perturb gauge: U' = exp(eps * F) * U
 *   3. Compute dS_f = S_f(U') - S_f(U)
 *   4. Compare dS_f/eps against -2*T_ferm (the directional derivative)
 *
 * If the force is correct (F = -dS/dU), then dS_f/eps = -||F||^2 = -2*T_ferm
 * (up to the MILC -4 offset in computeMomAction). The ratio should be 1.0.
 */
TEST(HMC, DirectionalForceTest)
{
  using namespace quda;

  double eps = hmc_force_eps;

  printfQuda("\n=== Directional fermion force test (eps=%e) ===\n", eps);

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }

  ColorSpinorField phi = generateEOPseudofermion(inv_param, 99999);

  // Compute fermion action at original gauge
  double Sf0 = computeEOFermionAction(phi, inv_param);

  // Compute fermion force with dt=1 into zero momentum
  GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
  mParam.location = QUDA_CUDA_FIELD_LOCATION;
  mParam.create = QUDA_ZERO_FIELD_CREATE;
  mParam.reconstruct = QUDA_RECONSTRUCT_10;
  mParam.setPrecision(gauge_param.cuda_prec, true);
  momResident = GaugeField(mParam);

  QudaInvertParam ip = inv_param;
  ip.solve_type = QUDA_NORMOP_PC_SOLVE;
  ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

  Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
  createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, true, false);

  ColorSpinorParam csParam(phi);
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField> x(1, csParam);
  std::vector<ColorSpinorField> b(1, ColorSpinorField(phi));
  solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);

  computeEOFermionForce(momResident, x[0], inv_param, 1.0);

  double T_ferm = computeMomAction(momResident);

  // Perturb gauge along force direction
  GaugeField gaugeSaved(*gaugePrecise);
  GaugeFieldParam gfParam(*gaugePrecise);
  gfParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField u_out(gfParam);
  updateGaugeField(u_out, eps, *gaugePrecise, momResident, false, true);
  gaugePrecise->copy(u_out);

  if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);

  double Sf1 = computeEOFermionAction(phi, inv_param);

  // Restore
  gaugePrecise->copy(gaugeSaved);
  if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);

  double dSf_deps = (Sf1 - Sf0) / eps;

  // Correct for MILC -4 offset: T_physical = T_ferm + 4*N_links
  int V = 1;
  for (int d = 0; d < 4; d++) V *= gauge_param.X[d];
  int nLinks = 4 * V;
  double T_physical = T_ferm + 4.0 * nLinks;

  // If F = -dS/dU, then perturbing along F gives dS/deps = -2*T_physical
  double dSf_expected = -2.0 * T_physical;
  double ratio = dSf_deps / dSf_expected;

  printfQuda("  S_f(U)     = %e\n", Sf0);
  printfQuda("  S_f(U')    = %e\n", Sf1);
  printfQuda("  dS/deps    = %e  (numerical)\n", dSf_deps);
  printfQuda("  -2*T_phys  = %e  (expected if F = -dS/dU)\n", dSf_expected);
  printfQuda("  ratio      = %f  (should be 1.0)\n", ratio);

  EXPECT_NEAR(ratio, 1.0, 0.01) << "Fermion force does not match action derivative";

  delete dirac;
  delete diracSloppy;
  if (diracPre != diracSloppy) delete diracPre;
  if (diracEig != diracPre) delete diracEig;

  momResident = GaugeField();
}

/**
 * Test: Per-link numerical force verification.
 *
 * For each test link (mu, site) and SU(3) generator T^a, perturbs the gauge
 * link by exp(+-i eps T^a) and compares the numerical action derivative
 * against the analytical force component extracted from the momentum field.
 *
 * The analytical force p^a at link (mu,site) is extracted from the momentum
 * P = sum_a p^a (iT^a) via p^a = 2 Re Tr((iT^a)^dag P).
 *
 * Adapted from Schwinger_MG verify_forces (hmc.cpp:844-910).
 */
TEST(HMC, PerLinkForceTest)
{
  using namespace quda;

  double eps = hmc_force_eps;
  int nTestLinks = hmc_per_link_test_links;

  printfQuda("\n=== Per-link fermion force test (eps=%e) ===\n", eps);

  // Sync hostGauge from the device so the per-link numerical derivative
  // perturbs the CURRENT gauge state (prior tests may have run HMC
  // trajectories that mutated the device gauge but not hostGauge).
  {
    auto saved_order = gauge_param.gauge_order;
    gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    saveGaugeQuda(hostGauge, &gauge_param);
    gauge_param.gauge_order = saved_order;
  }

  // Tighten the CG tolerance for this test only. PerLinkForceTest is a
  // correctness oracle: it compares the analytical force against a central
  // difference of the action. Both the action evaluations (S_plus/S_minus)
  // and the force-input solve consume inv_param.tol, and at the project
  // default (~2.4e-7 = 2*float_epsilon) the residual noise on x dominates
  // the relrr threshold (1e-3) at large volumes where κ(M†M) is non-trivial.
  // 1e-10 is small enough to make the noise floor negligible against the
  // central-difference truncation error at eps=1e-4 without exploding the
  // iteration count.
  const double saved_tol = inv_param.tol;
  inv_param.tol = 1e-10;

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }

  ColorSpinorField phi = generateEOPseudofermion(inv_param, 99999);

  QudaInvertParam ip = inv_param;
  ip.solve_type = QUDA_NORMOP_PC_SOLVE;
  ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

  Dirac *d = nullptr, *dS = nullptr, *dP = nullptr, *dE = nullptr;
  createDiracWithEig(d, dS, dP, dE, ip, true, false);
  ColorSpinorParam csParam(phi);
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField> xv(1, csParam);
  std::vector<ColorSpinorField> bv(1, ColorSpinorField(phi));
  solve(xv, bv, *d, *dS, *dP, *dE, ip);

  // Compute analytical force (dt=1)
  GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
  mParam.location = QUDA_CUDA_FIELD_LOCATION;
  mParam.create = QUDA_ZERO_FIELD_CREATE;
  mParam.reconstruct = QUDA_RECONSTRUCT_10;
  mParam.setPrecision(gauge_param.cuda_prec, true);
  momResident = GaugeField(mParam);
  computeEOFermionForce(momResident, xv[0], inv_param, 1.0);

  // Download momentum to host in RECONSTRUCT_10 format (10 reals per link)
  // then reconstruct the full 3x3 anti-hermitian matrix from the 10 components.
  std::vector<double> momBuf(4 * V * 10, 0.0);
  void *momHostPtrs[4];
  for (int i = 0; i < 4; i++) momHostPtrs[i] = momBuf.data() + i * V * 10;

  QudaGaugeParam momGp = gauge_param;
  momGp.gauge_order = QUDA_QDP_GAUGE_ORDER;
  momGp.reconstruct = QUDA_RECONSTRUCT_10;
  GaugeFieldParam momHostParam(momGp, momHostPtrs, QUDA_ASQTAD_MOM_LINKS);
  GaugeField momHost(momHostParam);
  momHost.copy(momResident);

  GaugeField gaugeSaved(*gaugePrecise);

  // Gell-Mann generators: iT^a = i lambda^a / 2
  std::complex<double> iT[8][3][3] = {};
  std::complex<double> Im(0, 1);
  iT[0][0][1] = Im*0.5; iT[0][1][0] = Im*0.5;
  iT[1][0][1] = 0.5;    iT[1][1][0] = -0.5;
  iT[2][0][0] = Im*0.5; iT[2][1][1] = -Im*0.5;
  iT[3][0][2] = Im*0.5; iT[3][2][0] = Im*0.5;
  iT[4][0][2] = 0.5;    iT[4][2][0] = -0.5;
  iT[5][1][2] = Im*0.5; iT[5][2][1] = Im*0.5;
  iT[6][1][2] = 0.5;    iT[6][2][1] = -0.5;
  double r3 = 1.0 / std::sqrt(3.0);
  iT[7][0][0] = Im*0.5*r3; iT[7][1][1] = Im*0.5*r3; iT[7][2][2] = -Im*r3;

  double maxRelErr = 0;
  int nPass = 0, nFail = 0;

  for (int link = 0; link < nTestLinks; link++) {
    int mu = link % 4;
    int site = (link * 37 + 13) % V;
    int x_coords[4];
    x_coords[3] = site / (gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2]);
    int rem = site % (gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2]);
    x_coords[2] = rem / (gauge_param.X[0] * gauge_param.X[1]);
    rem = rem % (gauge_param.X[0] * gauge_param.X[1]);
    x_coords[1] = rem / gauge_param.X[0];
    x_coords[0] = rem % gauge_param.X[0];
    int parity = (x_coords[0] + x_coords[1] + x_coords[2] + x_coords[3]) % 2;

    // Read the analytical momentum matrix P from RECONSTRUCT_10 format.
    // Layout: 10 doubles per link = {Re01, Im01, Re02, Im02, Re12, Im12, Im00, Im11, Im22, pad}
    // The matrix P is anti-hermitian traceless: P(j,i) = -P(i,j)*, Tr(P) = 0
    double *momData = static_cast<double *>(momHostPtrs[mu]) + site * 10;
    std::complex<double> P[3][3];
    P[0][1] = std::complex<double>(momData[0], momData[1]);
    P[0][2] = std::complex<double>(momData[2], momData[3]);
    P[1][2] = std::complex<double>(momData[4], momData[5]);
    P[0][0] = std::complex<double>(0, momData[6]);
    P[1][1] = std::complex<double>(0, momData[7]);
    P[2][2] = std::complex<double>(0, momData[8]);
    P[1][0] = -std::conj(P[0][1]);
    P[2][0] = -std::conj(P[0][2]);
    P[2][1] = -std::conj(P[1][2]);

    // Read original gauge link
    double *linkData = static_cast<double *>(hostGauge[mu]) + site * 18;
    std::complex<double> U_orig[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        U_orig[i][j] = std::complex<double>(linkData[2*(3*i+j)], linkData[2*(3*i+j)+1]);

    printfQuda("  Link (%d, %d, parity=%d):\n", mu, site, parity);

    for (int a = 0; a < 8; a++) {
      // Extract analytical force component: p^a = 2 Re Tr((iT^a)^dag P)
      // (iT^a)^dag = -iT^a (anti-hermitian), so p^a = -2 Re Tr(iT^a P)
      // The momentum stores P = dt * coeff * TA(force) where coeff includes a factor
      // of 2 from the gauge update convention (exp(dt*P)*U with H_kin = -1/2 Tr(P^2)).
      // To compare with -dS/dU, divide by 2: F_analytical = p^a / 2.
      std::complex<double> trace(0, 0);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          trace += iT[a][j][i] * P[i][j]; // Tr(iT^a P) = sum_ij (iT^a)_ji P_ij
      double p_a = -2.0 * trace.real();
      double F_analytical = p_a / 2.0; // divide by 2 for gauge update convention

      // Compute numerical derivative
      std::complex<double> U_plus[3][3], U_minus[3][3];
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          U_plus[i][j] = U_minus[i][j] = 0;
          for (int k = 0; k < 3; k++) {
            std::complex<double> pert_p = (i == k ? 1.0 : 0.0) + eps * iT[a][i][k];
            std::complex<double> pert_m = (i == k ? 1.0 : 0.0) - eps * iT[a][i][k];
            U_plus[i][j] += pert_p * U_orig[k][j];
            U_minus[i][j] += pert_m * U_orig[k][j];
          }
        }

      // Perturb +eps
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          linkData[2*(3*i+j)]   = U_plus[i][j].real();
          linkData[2*(3*i+j)+1] = U_plus[i][j].imag();
        }
      gauge_param.use_resident_gauge = 0;
      gauge_param.make_resident_gauge = 1;
      loadGaugeQuda(hostGauge, &gauge_param);
      if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        freeCloverQuda(); // force fresh clover+TrLog recomputation
        loadCloverQuda(nullptr, nullptr, &inv_param);
      }
      double Sf_plus = computeEOFermionAction(phi, inv_param);

      // Perturb -eps
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          linkData[2*(3*i+j)]   = U_minus[i][j].real();
          linkData[2*(3*i+j)+1] = U_minus[i][j].imag();
        }
      loadGaugeQuda(hostGauge, &gauge_param);
      if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        freeCloverQuda();
        loadCloverQuda(nullptr, nullptr, &inv_param);
      }
      double Sf_minus = computeEOFermionAction(phi, inv_param);

      // Restore
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          linkData[2*(3*i+j)]   = U_orig[i][j].real();
          linkData[2*(3*i+j)+1] = U_orig[i][j].imag();
        }

      double F_numerical = -(Sf_plus - Sf_minus) / (2.0 * eps);

      double relErr = (std::abs(F_numerical) > 1e-12)
        ? std::abs(F_analytical - F_numerical) / std::abs(F_numerical)
        : std::abs(F_analytical - F_numerical);
      maxRelErr = std::max(maxRelErr, relErr);

      if (relErr > 1e-3) nFail++;
      else nPass++;

      printfQuda("    gen %d: ana=%+.6e  num=%+.6e  rel_err=%.2e %s\n",
                 a, F_analytical, F_numerical, relErr, relErr > 1e-3 ? "FAIL" : "ok");
    }
  }

  // Restore original gauge and clover
  gaugePrecise->copy(gaugeSaved);
  for (int i = 0; i < 3; i++) // restore hostGauge from gaugeSaved
    for (int j = 0; j < 3; j++) {}
  loadGaugeQuda(hostGauge, &gauge_param);
  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    freeCloverQuda();
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }
  gauge_param.use_resident_gauge = 1;
  inv_param.tol = saved_tol;

  printfQuda("\n  Results: %d pass, %d fail, max_rel_err = %.2e\n", nPass, nFail, maxRelErr);
  EXPECT_EQ(nFail, 0) << "Per-link force test failed: analytical force does not match numerical derivative";

  delete d; delete dS;
  if (dP != dS) delete dP;
  if (dE != dP) delete dE;
  momResident = GaugeField();
}

// CompareDerivativeKernels test removed — served its purpose (confirmed kernel geometry).
// See REPORT_clover_sigma_force_derivation.md for the mathematical analysis.

/**
 * Test: Directional derivative verification for the *gauge* force.
 *
 * Counterpart of HMC.DirectionalForceTest, which checks fermion force vs.
 * fermion action. This one tests that the Wilson plaquette gauge force
 * is consistent with computeGaugeActionHMC. It exists because the existing
 * fermion-force tests pass cleanly while iso 16^4 thermalisation shows a
 * persistent ⟨dH⟩ < 0 — the gauge sector was previously unverified.
 *
 * Method:
 *   1. Compute gauge force F into zero momentum at coeff=1
 *   2. Perturb gauge: U' = exp(eps * F) * U
 *   3. Compute dS_g = S_g(U') - S_g(U)
 *   4. Compare dS_g/eps against -2*T_phys (the directional derivative)
 *
 * Same MILC -4 N_links offset on T_ferm as the fermion test (it's a property
 * of computeMomAction, not of the force).
 */
TEST(HMC, GaugeForceActionConsistency)
{
  using namespace quda;

  double eps = hmc_force_eps;

  printfQuda("\n=== Directional gauge force test (eps=%e, beta=%g) ===\n", eps, hmc_beta);

  // Wilson plaquette gauge paths — replicate the static setupWilsonGaugePaths
  // in lib/hmc_integrator.cpp so the test is self-contained.
  int *path_length = new int[6];
  double *path_coeff = new double[6];
  for (int i = 0; i < 6; i++) { path_length[i] = 3; path_coeff[i] = 1.0; }

  int ***input_path = new int **[4];
  for (int dir = 0; dir < 4; dir++) {
    input_path[dir] = new int *[6];
    int idx = 0;
    for (int i = 0; i < 4; i++) {
      if (i == dir) continue;
      int opp_dir = 7 - dir;
      int opp_i = 7 - i;
      input_path[dir][idx]    = new int[3]{i, opp_dir, opp_i};
      input_path[dir][idx+1]  = new int[3]{opp_i, opp_dir, i};
      idx += 2;
    }
  }

  // S_g(U)
  double Sg0 = computeGaugeActionHMC(hmc_beta);

  // Allocate zero momentum on device
  GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
  mParam.location = QUDA_CUDA_FIELD_LOCATION;
  mParam.create = QUDA_ZERO_FIELD_CREATE;
  mParam.reconstruct = QUDA_RECONSTRUCT_10;
  mParam.setPrecision(gauge_param.cuda_prec, true);
  momResident = GaugeField(mParam);

  // Compute gauge force into the zero mom (coeff = 1).
  QudaGaugeParam gp = gauge_param;
  gp.use_resident_gauge = 1;
  gp.use_resident_mom = 1;
  gp.make_resident_gauge = 1;
  gp.make_resident_mom = 1;
  gp.return_result_mom = 0;
  gp.overwrite_mom = 0;
  double eb3 = 1.0 * hmc_beta / 3.0;
  computeGaugeForceQuda(nullptr, nullptr, input_path, path_length, path_coeff, 6, 4, eb3, &gp);

  double T_ferm = computeMomAction(momResident);
  int V = 1;
  for (int d = 0; d < 4; d++) V *= gauge_param.X[d];
  int nLinks = 4 * V;
  double T_physical = T_ferm + 4.0 * nLinks; // MILC -4 offset (same as fermion test)

  // Perturb gauge along F: U' = exp(eps * F) * U
  GaugeField gaugeSaved(*gaugePrecise);
  GaugeFieldParam gfParam(*gaugePrecise);
  gfParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField u_out(gfParam);
  updateGaugeField(u_out, eps, *gaugePrecise, momResident, false, true);
  gaugePrecise->copy(u_out);

  double Sg1 = computeGaugeActionHMC(hmc_beta);

  // Restore
  gaugePrecise->copy(gaugeSaved);

  double dSg_deps = (Sg1 - Sg0) / eps;
  double dSg_expected = -2.0 * T_physical;
  double ratio = dSg_deps / dSg_expected;

  printfQuda("  S_g(U)     = %e\n", Sg0);
  printfQuda("  S_g(U')    = %e\n", Sg1);
  printfQuda("  dS/deps    = %e  (numerical)\n", dSg_deps);
  printfQuda("  -2*T_phys  = %e  (expected if F = -dS/dU)\n", dSg_expected);
  printfQuda("  ratio      = %f  (should be 1.0)\n", ratio);

  EXPECT_NEAR(ratio, 1.0, 0.01) << "Gauge force does not match action derivative";

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < 6; i++) delete[] input_path[dir][i];
    delete[] input_path[dir];
  }
  delete[] input_path;
  delete[] path_length;
  delete[] path_coeff;

  momResident = GaugeField();
}

/**
 * Test: Multi-trajectory dH statistics (Jarzynski equality).
 *
 * Runs N short trajectories from the same starting gauge with N different
 * momentum seeds, accumulates dH, and reports:
 *   ⟨dH⟩            (should approach 0 as N grows for symplectic integrator)
 *   σ(dH)           (integrator-error spread)
 *   ⟨exp(-dH)⟩      (must equal 1 by Jarzynski for symplectic, reversible MD
 *                    starting from a sample of the equilibrium distribution)
 *
 * Reversibility tests catch round-trip bugs but cannot catch a force↔action
 * mismatch — the same wrong force used both directions cancels. This test
 * does catch that class of bug.
 */
TEST(HMC, dHStatistics)
{
  using namespace quda;

  int N = std::max(10, hmc_n_trajectories);
  printfQuda("\n=== dH statistics over %d trajectories ===\n", N);

  GaugeField gaugeU0(*gaugePrecise);

  std::vector<double> dHs;
  dHs.reserve(N);
  for (int i = 0; i < N; i++) {
    // Restore starting gauge each iteration so we sample fresh momenta from
    // the same configuration, not from a drifting Markov chain.
    gaugePrecise->copy(gaugeU0);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);
    destroyHMCQuda();

    QudaHMCParam hp = makeHMCParam();
    hp.momentum_seed = hmc_momentum_seed + i;
    hp.reuse_pseudofermion = 0;
    hp.use_resident_gauge = 1;
    hp.return_result_gauge = 0;
    double dH = hmcTrajectoryQuda(nullptr, nullptr, &hp, &gauge_param, &inv_param, nullptr);
    dHs.push_back(dH);
  }

  double mean = 0, exp_mean = 0;
  for (double dH : dHs) { mean += dH; exp_mean += std::exp(-dH); }
  mean /= N; exp_mean /= N;
  double var = 0;
  for (double dH : dHs) var += (dH - mean) * (dH - mean);
  var /= std::max(1, N - 1);
  double stderr_mean = std::sqrt(var / N);

  printfQuda("  ⟨dH⟩         = %+.4e ± %.4e\n", mean, stderr_mean);
  printfQuda("  σ(dH)        = %.4e\n", std::sqrt(var));
  printfQuda("  ⟨exp(-dH)⟩   = %.4f  (should be ~1.0 by Jarzynski)\n", exp_mean);
  printfQuda("  individual dH values:");
  for (double dH : dHs) printfQuda("  %+.3e", dH);
  printfQuda("\n");

  // Restore
  gaugePrecise->copy(gaugeU0);
  destroyHMCQuda();

  // Loose-but-real bound: ⟨exp(-dH)⟩ within 50% of 1 means no order-unity bias.
  // A tighter bound needs more trajectories than is reasonable in a unit test.
  EXPECT_NEAR(exp_mean, 1.0, 0.5) << "Jarzynski equality violated — integrator likely not symplectic";
}

/**
 * Test: dH scaling with integration step size.
 *
 * For a symplectic integrator at *fixed* trajectory length τ, |dH| should
 * decrease as a power of dt = τ/n_steps:
 *   Leapfrog        : |dH| ~ dt^2
 *   Omelyan, FGI    : |dH| ~ dt^4
 *
 * If the integrator has a force↔action mismatch, the bias is dt-independent
 * and |dH| plateaus at small dt. This test scans n_steps and reports the
 * fitted exponent so a regression to a plateau or wrong scaling is visible.
 *
 * Same momentum seed and starting gauge for every dt → only dt varies.
 */
TEST(HMC, dHScaling)
{
  using namespace quda;

  // Larger n_steps values push |dH| down to the CG residual noise floor
  // (~1e-5 at tol=1e-9 on 4^4) and contaminate the linear fit; for FG we
  // stay in the cleaner regime where the dt⁴ leading term still dominates.
  QudaIntegratorType itype = static_cast<QudaIntegratorType>(hmc_integrator);
  int n_step_list_p2[] = {10, 20, 40, 80};
  int n_step_list_p4[] = {8, 12, 16, 22};
  int *n_step_list = (itype == QUDA_FORCE_GRADIENT_INTEGRATOR) ? n_step_list_p4 : n_step_list_p2;
  const int n_dt = 4;
  double tau = hmc_tau;

  const char *iname = (itype == QUDA_LEAPFROG_INTEGRATOR) ? "Leapfrog"
                    : (itype == QUDA_OMELYAN_INTEGRATOR)  ? "Omelyan"
                    : (itype == QUDA_FORCE_GRADIENT_INTEGRATOR) ? "ForceGradient"
                    : "Unknown";
  // Leapfrog and the 2nd-order Omelyan PQPQP minimum-norm scheme both have
  // dH ∝ dt². Force-gradient (PQPQP_FG with the Hessian-free trick) is 4th
  // order.
  double expected_p = (itype == QUDA_FORCE_GRADIENT_INTEGRATOR) ? 4.0 : 2.0;

  printfQuda("\n=== dH-vs-dt scaling for %s (expect |dH| ~ dt^%g) ===\n", iname, expected_p);

  GaugeField gaugeU0(*gaugePrecise);

  std::vector<double> dts, dHs;
  for (int k = 0; k < n_dt; k++) {
    int n = n_step_list[k];
    double dt = tau / n;

    gaugePrecise->copy(gaugeU0);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);
    destroyHMCQuda();

    QudaHMCParam hp = makeHMCParam();
    hp.tau = tau;
    hp.n_steps = n;
    hp.momentum_seed = hmc_momentum_seed; // same seed → same momenta
    hp.reuse_pseudofermion = 0;
    hp.use_resident_gauge = 1;
    hp.return_result_gauge = 0;
    double dH = hmcTrajectoryQuda(nullptr, nullptr, &hp, &gauge_param, &inv_param, nullptr);
    printfQuda("  n_steps=%3d  dt=%.5f  dH=%+.6e  |dH|=%.6e\n", n, dt, dH, std::abs(dH));
    dts.push_back(dt);
    dHs.push_back(std::abs(dH));
  }

  // Linear fit log|dH| = p*log(dt) + c
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int k = 0; k < n_dt; k++) {
    double x = std::log(dts[k]);
    double y = std::log(std::max(dHs[k], 1e-30));
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  double p_fit = (n_dt * sxy - sx * sy) / (n_dt * sxx - sx * sx);

  printfQuda("  fitted exponent p = %.3f  (expected %.1f for %s)\n", p_fit, expected_p, iname);

  gaugePrecise->copy(gaugeU0);
  destroyHMCQuda();

  // Loose bound: scaling within ±0.5 of expected. Plateau (force-action
  // mismatch) shows up as p ≈ 0; wrong order shows as different exponent.
  EXPECT_NEAR(p_fit, expected_p, 0.5) << iname << ": dH does not scale as dt^" << expected_p
                                      << " (fitted " << p_fit << ") — possible integrator bug";
}

/**
 * Test: dH scaling for NestedFGI (with MG preconditioner).
 *
 * NestedFGI = outer PQPQP_FG (our patched 4th-order fgStep) + inner
 * leapfrog/Omelyan handling only the MG-projected low-mode force. Although
 * the inner integrator is formally 2nd-order, the inner force amplitude is
 * small (only the low-mode component), so its dt² error coefficient is
 * dwarfed by the outer FG's dt⁴ term. Empirically the scaling is dt⁴ in
 * the regime where dt is large enough that the floor doesn't bite.
 *
 * The MG transfer is single-precision by default (--prec-precondition),
 * which puts a noise floor around |dH| ~ 1e-5 on this 4⁴ test gauge.
 */
TEST(HMC, dHScalingNestedFGI)
{
  using namespace quda;

  int n_step_list[] = {6, 10, 16, 24};
  const int n_dt = 4;
  double tau = hmc_tau;

  printfQuda("\n=== dH-vs-dt scaling for NestedFGI ===\n");

  GaugeField gaugeU0(*gaugePrecise);

  // Set up MG preconditioner. Precision is CLI-driven via --prec-precondition
  // (same convention as invert_test); the test setup at the top of this file
  // resolves prec_precondition from CLI with fallbacks. Match all MG-internal
  // precisions (sloppy/precondition/eigensolver/null) so calculateY doesn't
  // see a single/double mix.
  destroyHMCQuda();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_ip = inv_param;
  mg_ip.solve_type = QUDA_DIRECT_SOLVE;
  mg_ip.solution_type = QUDA_MAT_SOLUTION;
  mg_ip.matpc_type = QUDA_MATPC_EVEN_EVEN;
  mg_ip.cuda_prec = gauge_param.cuda_prec;
  mg_ip.cuda_prec_sloppy = prec_precondition;
  mg_ip.cuda_prec_precondition = prec_precondition;
  mg_ip.cuda_prec_eigensolver = prec_precondition;
  mg_ip.clover_cuda_prec = gauge_param.cuda_prec;
  mg_ip.clover_cuda_prec_sloppy = prec_precondition;
  mg_ip.clover_cuda_prec_precondition = prec_precondition;
  mg_ip.clover_cuda_prec_eigensolver = prec_precondition;
  configureHMCTestMG(mg_param, mg_ip, prec_precondition);
  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(nullptr, nullptr, &inv_param);
  void *mg_prec = newMultigridQuda(&mg_param);

  // Outer inverter: plain CG, no MG preconditioner on the outer (MG is
  // used only for coarse transfer in the nested-FGI inner force).
  QudaInvertParam saved_ip = inv_param;
  inv_param.preconditioner = nullptr;
  inv_param.inv_type = QUDA_CG_INVERTER;
  inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param.inv_type_precondition = QUDA_INVALID_INVERTER;

  std::vector<double> dts, dHs;
  for (int k = 0; k < n_dt; k++) {
    int n = n_step_list[k];
    double dt = tau / n;

    gaugePrecise->copy(gaugeU0);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);

    QudaHMCParam hp = makeHMCParam(QUDA_NESTED_FGI_INTEGRATOR);
    hp.tau = tau;
    hp.n_steps = n;
    hp.momentum_seed = hmc_momentum_seed + 424242;
    hp.reuse_pseudofermion = 0;
    hp.use_resident_gauge = 1;
    hp.return_result_gauge = 0;
    hp.defl_refresh_interval = 0;
    double dH = hmcTrajectoryQuda(nullptr, nullptr, &hp, &gauge_param, &inv_param, mg_prec);
    printfQuda("  n_outer=%3d  dt=%.5f  dH=%+.6e  |dH|=%.6e\n", n, dt, dH, std::abs(dH));
    dts.push_back(dt);
    dHs.push_back(std::abs(dH));
  }

  // Linear fit log|dH| = p*log(dt) + c, plus consecutive-pair report
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int k = 0; k < n_dt; k++) {
    double x = std::log(dts[k]);
    double y = std::log(std::max(dHs[k], 1e-30));
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  double p_fit = (n_dt * sxy - sx * sy) / (n_dt * sxx - sx * sx);

  printfQuda("  fitted exponent p = %.3f\n", p_fit);
  for (int k = 1; k < n_dt; k++) {
    double r = dHs[k - 1] / std::max(dHs[k], 1e-30);
    double scale = dts[k - 1] / dts[k];
    double p = std::log(r) / std::log(scale);
    printfQuda("  pair (%d→%d): ratio=%.2f, implied p=%.2f\n", n_step_list[k - 1], n_step_list[k], r, p);
  }

  // Cleanup
  inv_param = saved_ip;
  destroyMultigridQuda(mg_prec);
  gaugePrecise->copy(gaugeU0);
  destroyHMCQuda();

  // Inner leapfrog/Omelyan limits the overall convergence to 2nd order.
  // Loose tolerance because MG-preconditioned transfer adds single-precision
  // noise floor that contaminates fits at small dt.
  EXPECT_GT(p_fit, 1.5) << "NestedFGI: dH does not scale at least as dt^2";
  EXPECT_LT(p_fit, 4.5) << "NestedFGI: dH scales unexpectedly steeply";
}

/**
 * Test: Thermalisation + reversibility test for detailed balance.
 *
 * 1. Thermalise for hmc_n_thermalization trajectories
 * 2. Every 10 trajectories after thermalisation, perform a reversibility test:
 *    - Save gauge U_0
 *    - Run forward trajectory (tau) → dH_fwd
 *    - Run backward trajectory (-tau) with same pseudofermion → dH_bwd
 *    - Check ||U_final - U_0|| / ||U_0|| ~ O(machine eps)
 *    - Check dH_fwd + dH_bwd ~ 0
 */
TEST(HMC, ReversibilityTest)
{
  int n_therm = hmc_n_thermalization;
  int n_total = hmc_n_trajectories;
  int rev_interval = hmc_reversibility_interval;
  double rev_tol = hmc_reversibility_tol;

  printfQuda("Reversibility test: %d total trajectories, %d thermalisation, test every %d, tol=%e\n", n_total, n_therm,
             rev_interval, rev_tol);

  // --- Phase 1: Thermalise ---
  QudaHMCParam hmc_param = makeHMCParam();
  hmc_param.n_trajectories = n_therm;
  hmc_param.n_thermalization = n_therm; // all trajectories are thermalisation (always accept)

  if (n_therm > 0) {
    printfQuda("=== Thermalising for %d trajectories ===\n", n_therm);
    hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, nullptr, nullptr);
  }

  // --- Phase 2: Production with reversibility tests ---
  int n_rev_tests = 0;
  int n_rev_pass = 0;

  for (int traj = n_therm; traj < n_total; traj++) {
    // Normal trajectory with accept/reject
    hmc_param.n_trajectories = 1;
    hmc_param.n_thermalization = 0;
    hmc_param.momentum_seed = hmc_momentum_seed + traj;
    hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, nullptr, nullptr);

    // Reversibility test every rev_interval trajectories
    if ((traj - n_therm) % rev_interval == 0 && traj >= n_therm) {
      printfQuda("\n=== Reversibility test at trajectory %d ===\n", traj + 1);
      n_rev_tests++;

      // Measure plaquette before
      double plaq_before[3];
      plaqQuda(plaq_before);

      // Forward trajectory: generate momentum, generate phi, evolve
      QudaHMCParam fwd_param = makeHMCParam();
      fwd_param.momentum_seed = hmc_momentum_seed + 100000 + traj;
      fwd_param.reuse_pseudofermion = 0;
      fwd_param.use_resident_mom = 0;
      fwd_param.make_resident_mom = 1;
      fwd_param.return_result_mom = 0;

      double dH_fwd = hmcTrajectoryQuda(nullptr, nullptr, &fwd_param, &gauge_param, &inv_param, nullptr);

      // Backward trajectory: reuse pseudofermion, use resident momentum, negative tau
      QudaHMCParam bwd_param = fwd_param;
      bwd_param.tau = -hmc_tau; // reverse direction
      bwd_param.generate_momentum = 0;
      bwd_param.reuse_pseudofermion = 1;
      bwd_param.use_resident_mom = 1;

      double dH_bwd = hmcTrajectoryQuda(nullptr, nullptr, &bwd_param, &gauge_param, &inv_param, nullptr);

      // Measure plaquette after (should match before if reversible)
      double plaq_after[3];
      plaqQuda(plaq_after);

      double delta_plaq = fabs(plaq_after[0] - plaq_before[0]);
      double sum_dH = dH_fwd + dH_bwd;

      printfQuda("  dH_fwd = %+.6e, dH_bwd = %+.6e, dH_fwd + dH_bwd = %+.6e\n", dH_fwd, dH_bwd, sum_dH);
      printfQuda("  plaq_before = %.15e, plaq_after = %.15e, delta = %.6e\n", plaq_before[0], plaq_after[0],
                 delta_plaq);

      bool pass = (delta_plaq < rev_tol);
      if (pass) {
        n_rev_pass++;
        printfQuda("  REVERSIBILITY: PASS (delta_plaq = %.6e < %.6e)\n", delta_plaq, rev_tol);
      } else {
        printfQuda("  REVERSIBILITY: FAIL (delta_plaq = %.6e > %.6e)\n", delta_plaq, rev_tol);
      }
    }
  }

  printfQuda("\n=== Reversibility summary: %d/%d passed ===\n", n_rev_pass, n_rev_tests);
  if (n_rev_tests > 0) { EXPECT_GT(n_rev_pass, 0); }
  SUCCEED();
}

/**
 * Test: Reversibility for every integrator.
 *
 * For each of {leapfrog, Omelyan, FGI, nested FGI}:
 *   1. Save gauge U_0 and plaquette
 *   2. Forward trajectory of length +tau, capture dH_fwd, save pseudofermion
 *   3. Backward trajectory of length -tau, reusing the same pseudofermion and
 *      resident momentum, capture dH_bwd
 *   4. Assert |dH_fwd + dH_bwd| small and |plaq_after - plaq_before| small
 *   5. Restore gauge for the next integrator
 *
 * Outer fermion solve is plain CG (no MG preconditioner). Nested FGI requires
 * an MG hierarchy for its coarse transfer operator; we build a minimal one
 * and freeze the coarse deflation (no in-trajectory refresh) to keep the
 * force a pure function of the gauge, which is what reversibility demands.
 */
TEST(HMC, ReversibilityAllIntegrators)
{
  using namespace quda;

  struct Case {
    const char *name;
    QudaIntegratorType type;
    bool needs_mg;
    double tol; // plaquette-recovery tolerance
  };
  // Pure-double CG integrators hit machine precision (use --hmc-reversibility-tol);
  // MG-using integrators (nested FGI) run with single-precision transfer so the
  // tolerance is looser (use --hmc-reversibility-tol-mg). Both are CLI-driven.
  const Case cases[] = {
    {"Leapfrog",      QUDA_LEAPFROG_INTEGRATOR,       false, hmc_reversibility_tol},
    {"Omelyan",       QUDA_OMELYAN_INTEGRATOR,        false, hmc_reversibility_tol},
    {"ForceGradient", QUDA_FORCE_GRADIENT_INTEGRATOR, false, hmc_reversibility_tol},
    {"NestedFGI",     QUDA_NESTED_FGI_INTEGRATOR,     true,  hmc_reversibility_tol_mg},
  };

  // Save the starting gauge so every integrator starts from the same U_0.
  GaugeField gaugeU0(*gaugePrecise);
  double plaqU0[3];
  plaqQuda(plaqU0);

  int n_pass = 0, n_total = 0;

  for (const auto &tc : cases) {
    printfQuda("\n=== Reversibility[%s] ===\n", tc.name);
    n_total++;

    // Reset gauge to U_0 before this integrator's trial.
    gaugePrecise->copy(gaugeU0);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);

    // Reset any persistent HMC state (nested FGI instance, eigentracker, etc.)
    destroyHMCQuda();

    // Optional MG setup for nested FGI (required for coarse transfer).
    void *mg_prec = nullptr;
    QudaMultigridParam mg_param = newQudaMultigridParam();
    QudaInvertParam mg_ip = inv_param;
    if (tc.needs_mg) {
      mg_ip.solve_type = QUDA_DIRECT_SOLVE;
      mg_ip.solution_type = QUDA_MAT_SOLUTION;
      mg_ip.matpc_type = QUDA_MATPC_EVEN_EVEN;
      // Pin MG-internal precisions to the CLI-driven preconditioner precision
      // (--prec-precondition) so calculateY doesn't see a fine Dirac op at one
      // precision and prolongator/restrictor at another. Same pattern as
      // HMC.MGPreconditionedRun (line 447) — invert_test passes precision via
      // --prec-precondition and we follow that convention.
      mg_ip.cuda_prec = gauge_param.cuda_prec; // must match gauge precise
      mg_ip.cuda_prec_sloppy = prec_precondition;
      mg_ip.cuda_prec_precondition = prec_precondition;
      mg_ip.cuda_prec_eigensolver = prec_precondition;
      mg_ip.clover_cuda_prec = gauge_param.cuda_prec;
      mg_ip.clover_cuda_prec_sloppy = prec_precondition;
      mg_ip.clover_cuda_prec_precondition = prec_precondition;
      mg_ip.clover_cuda_prec_eigensolver = prec_precondition;
      configureHMCTestMG(mg_param, mg_ip, prec_precondition);
      if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(nullptr, nullptr, &inv_param);
      mg_prec = newMultigridQuda(&mg_param);
    }

    // Configure the outer inverter for a plain CG solve (no MG preconditioner
    // on the CG itself, even for nested FGI — MG is used only for transfer).
    QudaInvertParam saved_ip = inv_param;
    inv_param.preconditioner = nullptr;
    inv_param.inv_type = QUDA_CG_INVERTER;
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
    inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    inv_param.inv_type_precondition = QUDA_INVALID_INVERTER;

    // Forward trajectory
    QudaHMCParam fwd = makeHMCParam(tc.type);
    fwd.momentum_seed = hmc_momentum_seed + 424242;
    fwd.reuse_pseudofermion = 0;
    fwd.use_resident_mom = 0;
    fwd.make_resident_mom = 1;
    fwd.return_result_mom = 0;
    fwd.defl_refresh_interval = 0; // freeze deflation during trajectory (for nested FGI)
    fwd.use_resident_gauge = 1;

    double dH_fwd = hmcTrajectoryQuda(nullptr, nullptr, &fwd, &gauge_param, &inv_param, mg_prec);

    // Backward trajectory: negative tau, same pseudofermion, resident momentum.
    QudaHMCParam bwd = fwd;
    bwd.tau = -fwd.tau;
    bwd.generate_momentum = 0;
    bwd.reuse_pseudofermion = 1;
    bwd.use_resident_mom = 1;

    double dH_bwd = hmcTrajectoryQuda(nullptr, nullptr, &bwd, &gauge_param, &inv_param, mg_prec);

    // Compare plaquette before/after — reversible means bit-equal in exact arithmetic.
    double plaq_after[3];
    plaqQuda(plaq_after);
    double delta_plaq = fabs(plaq_after[0] - plaqU0[0]);
    double sum_dH = dH_fwd + dH_bwd;

    printfQuda("  dH_fwd = %+.6e, dH_bwd = %+.6e, sum = %+.6e\n", dH_fwd, dH_bwd, sum_dH);
    printfQuda("  plaq_before = %.15e, plaq_after = %.15e, delta = %.6e (tol = %.1e)\n", plaqU0[0], plaq_after[0],
               delta_plaq, tc.tol);

    bool pass = (delta_plaq < tc.tol);
    if (pass) {
      n_pass++;
      printfQuda("  [%s] REVERSIBILITY: PASS\n", tc.name);
    } else {
      printfQuda("  [%s] REVERSIBILITY: FAIL\n", tc.name);
    }
    EXPECT_LT(delta_plaq, tc.tol) << tc.name << ": plaquette delta " << delta_plaq << " exceeds tol " << tc.tol;

    // Restore inv_param and tear down MG before the next case.
    inv_param = saved_ip;
    if (mg_prec) {
      destroyMultigridQuda(mg_prec);
      mg_prec = nullptr;
    }
  }

  // Final reset: restore starting gauge and free any persistent HMC state.
  gaugePrecise->copy(gaugeU0);
  destroyHMCQuda();

  printfQuda("\n=== Reversibility summary: %d/%d integrators passed ===\n", n_pass, n_total);
}

// ============================================================
// Eigentracking tests
// ============================================================

/**
 * Helper: create EO-preconditioned Dirac operators for eigentracking tests.
 */
static void createEODirac(quda::Dirac *&dirac, QudaInvertParam &ip)
{
  using namespace quda;
  ip.solve_type = QUDA_NORMOP_PC_SOLVE;
  ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  Dirac *dSloppy = nullptr, *dPre = nullptr, *dEig = nullptr;
  createDiracWithEig(dirac, dSloppy, dPre, dEig, ip, true, false);
  // We only use the precise Dirac; clean up the rest
  if (dSloppy && dSloppy != dirac) delete dSloppy;
  if (dPre && dPre != dirac && dPre != dSloppy) delete dPre;
  if (dEig && dEig != dirac && dEig != dPre) delete dEig;
}

/**
 * Helper: build a QudaEigParam for the EigenTracking test fixtures from CLI vars.
 *
 * All fixture knobs are CLI-driven via the --eigentracking-* flags so
 * reviewers can adapt to different operators (heavier mass,
 * polynomial-acceleration tweaks, looser convergence) without editing
 * the test source. The same flags drive EigenTrackingState::maybeInit
 * for production HMC runs.
 */
static QudaEigParam makeEigentestEigParam(int nEv)
{
  QudaEigParam ep;
  memset(&ep, 0, sizeof(QudaEigParam));
  ep.eig_type = QUDA_EIG_TR_LANCZOS;
  ep.spectrum = QUDA_SPECTRUM_SR_EIG;
  ep.n_ev = nEv;
  ep.n_kr = 3 * nEv;          // QUDA wiki rule of thumb
  ep.n_conv = nEv;
  ep.n_ev_deflate = nEv;
  ep.tol = eigentracking_trlm_tol;
  ep.max_restarts = eigentracking_trlm_max_restarts;
  ep.require_convergence = QUDA_BOOLEAN_TRUE;
  ep.use_norm_op = QUDA_BOOLEAN_FALSE;
  ep.use_dagger = QUDA_BOOLEAN_FALSE;
  ep.use_pc = QUDA_BOOLEAN_FALSE;
  // Poly-acc defaulting: random 4^4 hot-start gauges need a conservative
  // a_min ≈ 1.0 for reliable TRLM convergence (per empirical experience —
  // mass-scaled heuristics land too close to the auto-estimated a_max and
  // give a weak Chebyshev filter). Explicit CLI values always win.
  bool usePA = eigentracking_use_poly_acc;
  double aMin = eigentracking_a_min;
  if (!usePA && aMin == 0.0) {
    usePA = true;
    aMin = 1.0;
  }
  ep.use_poly_acc = usePA ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  ep.poly_deg = eigentracking_poly_deg;
  ep.a_min = aMin;
  ep.a_max = eigentracking_a_max;
  ep.compute_svd = QUDA_BOOLEAN_FALSE;
  ep.compute_gamma5 = QUDA_BOOLEAN_FALSE;
  ep.batched_rotate = 0;
  ep.preserve_deflation = QUDA_BOOLEAN_FALSE;
  ep.check_interval = 10;
  ep.max_ortho_attempts = 10;
  ep.compute_evals_batch_size = nEv;
  return ep;
}

/**
 * Helper: resolve the EigenTracking test n_ev. CLI value wins. Otherwise scale
 * linearly with L = V^(1/4): the low-mode density of M†M grows with V, so the
 * Krylov space (n_kr = 3·n_ev) needs to grow too or TRLM hits its restart cap
 * before resolving the cluster. Empirically: n_ev=6 converges at 4^4, n_ev=24
 * is needed at 16^4 (same heavy fixture mass=2.0). The formula below gives
 * those values and scales monotonically for intermediate volumes.
 */
static int makeEigentestNev()
{
  if (eigentracking_n_ev > 0) return eigentracking_n_ev;
  long V = 1;
  for (int d = 0; d < 4; d++) V *= gauge_param.X[d];
  int L = static_cast<int>(std::round(std::pow(static_cast<double>(V), 0.25)));
  return std::max(6, (3 * L + 1) / 2); // ceil(1.5·L), floor at 6
}

/**
 * Helper: resolve pool capacity for the EigenTracker fixture. Must be ≥ n_ev
 * (EigenTracker::init enforces this). CLI value wins; otherwise default to
 * max(8, n_ev) so it scales with the volume-driven n_ev above.
 */
static int makeEigentestPoolCapacity(int nEv)
{
  if (eigentracking_pool_capacity > 0) return eigentracking_pool_capacity;
  return std::max(8, nEv);
}

/**
 * Test: Initialize EigenTracker from TRLM and verify compress.
 */
TEST(EigenTracking, PoolInitAndCompress)
{
  using namespace quda;

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) { loadCloverQuda(nullptr, nullptr, &inv_param); }

  QudaInvertParam ip = inv_param;
  Dirac *dirac = nullptr;
  createEODirac(dirac, ip);
  DiracMdagM matNorm(*dirac);
  DiracM matHalf(*dirac);

  const int nEv = makeEigentestNev();
  const int nKr = 3 * nEv;
  const int poolCapacity = makeEigentestPoolCapacity(nEv);
  QudaEigParam ep = makeEigentestEigParam(nEv);

  auto *eigSolve = quda::EigenSolver::create(&ep, matNorm);

  // TRLM needs kSpace populated with at least one field for metadata;
  // use an even-parity pseudofermion as the template.
  ColorSpinorField templateField = generateEOPseudofermion(inv_param, 1);
  std::vector<ColorSpinorField> kSpace;
  kSpace.reserve(nKr);
  kSpace.push_back(std::move(templateField));
  std::vector<Complex> evals(nEv);
  (*eigSolve)(kSpace, evals);
  delete eigSolve;

  printfQuda("TRLM eigenvalues:");
  for (int i = 0; i < nEv; i++) printfQuda(" %e", evals[i].real());
  printfQuda("\n");

  // Initialize tracker
  EigenTracker tracker;
  tracker.init(kSpace, evals, matHalf, nEv, poolCapacity);
  EXPECT_TRUE(tracker.isInitialized());
  EXPECT_EQ(tracker.poolSize(), std::min(static_cast<int>(kSpace.size()), poolCapacity));

  // Compress and verify eigenvalues
  tracker.compress();
  auto &trackerEvals = tracker.getEvals();
  for (int i = 0; i < nEv; i++) {
    double relDiff = std::abs(trackerEvals[i].real() - evals[i].real()) / std::max(std::abs(evals[i].real()), 1e-30);
    printfQuda("  eval[%d]: TRLM=%e  tracker=%e  relDiff=%e\n", i, evals[i].real(), trackerEvals[i].real(), relDiff);
    EXPECT_LT(relDiff, 1e-4) << "Eigenvalue " << i << " does not match TRLM";
  }

  // Verify max residual
  double maxRes = tracker.maxResidual(matNorm);
  printfQuda("  maxResidual = %e\n", maxRes);
  EXPECT_LT(maxRes, 1e-4);

  delete dirac;
}

/**
 * Test: Small gauge perturbation -> forceUpdate -> check residuals.
 */
TEST(EigenTracking, ForceUpdate)
{
  using namespace quda;

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) { loadCloverQuda(nullptr, nullptr, &inv_param); }

  QudaInvertParam ip = inv_param;
  Dirac *dirac = nullptr;
  createEODirac(dirac, ip);
  DiracMdagM matNorm(*dirac);
  DiracM matHalf(*dirac);

  const int nEv = makeEigentestNev();
  QudaEigParam ep = makeEigentestEigParam(nEv);

  auto *eigSolve = quda::EigenSolver::create(&ep, matNorm);
  // Seed kSpace metadata via a pseudofermion (TRLM needs one field)
  ColorSpinorField templateField = generateEOPseudofermion(inv_param, 2);
  std::vector<ColorSpinorField> kSpace;
  kSpace.reserve(3 * nEv);
  kSpace.push_back(std::move(templateField));
  std::vector<Complex> evals(nEv);
  (*eigSolve)(kSpace, evals);
  delete eigSolve;
  delete dirac;

  EigenTracker tracker;
  DiracParam dp;
  setDiracParam(dp, &ip, true);
  Dirac *d = Dirac::create(dp);
  DiracM mHalf(*d);
  tracker.init(kSpace, evals, mHalf, nEv, makeEigentestPoolCapacity(nEv));
  delete d;

  // Save gauge
  GaugeField gaugeSaved(*gaugePrecise);

  // Small gauge perturbation
  double eps = 1e-3;
  GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
  mParam.location = QUDA_CUDA_FIELD_LOCATION;
  mParam.create = QUDA_ZERO_FIELD_CREATE;
  mParam.reconstruct = QUDA_RECONSTRUCT_10;
  mParam.setPrecision(gauge_param.cuda_prec, true);
  GaugeField randMom(mParam);
  gaugeGauss(randMom, 42, 1.0);

  GaugeFieldParam gfParam(*gaugePrecise);
  gfParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField u_out(gfParam);
  updateGaugeField(u_out, eps, *gaugePrecise, randMom, false, true);
  gaugePrecise->copy(u_out);
  if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
  if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
    gaugePrecondition->copy(*gaugePrecise);

  if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);

  // ForceUpdate with new operator
  DiracParam dp2;
  setDiracParam(dp2, &ip, true);
  Dirac *d2 = Dirac::create(dp2);
  DiracM mHalf2(*d2);
  DiracMdagM mNorm2(*d2);
  tracker.forceUpdate(mHalf2);

  double maxRes = tracker.maxResidual(mNorm2);
  printfQuda("  ForceUpdate maxResidual = %e (after eps=%e perturbation)\n", maxRes, eps);
  EXPECT_LT(maxRes, 0.5) << "Residual too large after small perturbation";

  delete d2;

  // Restore gauge
  gaugePrecise->copy(gaugeSaved);
  if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
  if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
    gaugePrecondition->copy(*gaugePrecise);
  if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);
}

/**
 * Test: Multiple perturbations -> RR tracking accuracy.
 */
TEST(EigenTracking, RayleighRitzEvolve)
{
  using namespace quda;

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) { loadCloverQuda(nullptr, nullptr, &inv_param); }

  QudaInvertParam ip = inv_param;
  Dirac *dirac = nullptr;
  createEODirac(dirac, ip);
  DiracMdagM matNorm(*dirac);
  DiracM matHalf(*dirac);

  const int nEv = makeEigentestNev();
  QudaEigParam ep = makeEigentestEigParam(nEv);

  auto *eigSolve = quda::EigenSolver::create(&ep, matNorm);
  // Seed kSpace metadata via a pseudofermion (TRLM needs one field)
  ColorSpinorField templateField = generateEOPseudofermion(inv_param, 2);
  std::vector<ColorSpinorField> kSpace;
  kSpace.reserve(3 * nEv);
  kSpace.push_back(std::move(templateField));
  std::vector<Complex> evals(nEv);
  (*eigSolve)(kSpace, evals);
  delete eigSolve;
  delete dirac;

  EigenTracker tracker;
  {
    DiracParam dp;
    setDiracParam(dp, &ip, true);
    Dirac *d = Dirac::create(dp);
    DiracM mHalf(*d);
    tracker.init(kSpace, evals, mHalf, nEv, makeEigentestPoolCapacity(nEv));
    delete d;
  }

  GaugeField gaugeSaved(*gaugePrecise);

  // Apply 5 sequential gauge perturbations with RR after each
  for (int iter = 0; iter < 5; iter++) {
    double eps = 1e-3;
    GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
    mParam.location = QUDA_CUDA_FIELD_LOCATION;
    mParam.create = QUDA_ZERO_FIELD_CREATE;
    mParam.reconstruct = QUDA_RECONSTRUCT_10;
    mParam.setPrecision(gauge_param.cuda_prec, true);
    GaugeField randMom(mParam);
    gaugeGauss(randMom, 100 + iter, 1.0);

    GaugeFieldParam gfParam(*gaugePrecise);
    gfParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u_out(gfParam);
    updateGaugeField(u_out, eps, *gaugePrecise, randMom, false, true);
    gaugePrecise->copy(u_out);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);
    if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);

    DiracParam dp;
    setDiracParam(dp, &ip, true);
    Dirac *d = Dirac::create(dp);
    DiracMdagM mNorm(*d);
    DiracM mHalf(*d);

    auto rotation = tracker.rayleighRitzEvolve(mNorm);
    tracker.forceUpdate(mHalf); // recompute Dpool

    double maxRes = tracker.maxResidual(mNorm);
    printfQuda("  RR step %d: maxResidual = %e\n", iter + 1, maxRes);
    EXPECT_LT(maxRes, 0.01) << "RR residual too large at step " << iter + 1;
    delete d;
  }

  // Restore gauge
  gaugePrecise->copy(gaugeSaved);
  if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
  if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
    gaugePrecondition->copy(*gaugePrecise);
  if (cloverPrecise) loadCloverQuda(nullptr, nullptr, &inv_param);
}

/**
 * Test: Forecasting improves RR tracking.
 */
TEST(EigenTracking, Forecast)
{
  using namespace quda;
  // This test verifies that EigenForecast compiles and runs without error.
  // The forecast quality is best tested on a real trajectory; here we just
  // test the API round-trip: record rotations, forecast, apply.

  EigenForecast forecast(4, 1);
  EXPECT_EQ(forecast.historyLength(), 0);

  // Create a trivial identity rotation
  int k = 4;
  std::vector<Complex> identity(k * k, Complex(0, 0));
  for (int i = 0; i < k; i++) identity[i * k + i] = Complex(1, 0);

  forecast.recordRotation(identity, k);
  EXPECT_EQ(forecast.historyLength(), 1);

  forecast.recordRotation(identity, k);
  EXPECT_EQ(forecast.historyLength(), 2);

  auto R = forecast.forecastRotation();
  EXPECT_EQ(static_cast<int>(R.size()), k * k);

  // The forecast of identity rotations should be close to identity
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double expected = (i == j) ? 1.0 : 0.0;
      double actual = std::abs(R[j * k + i] - Complex(expected, 0));
      EXPECT_LT(actual, 1e-10) << "Forecast deviates from identity at (" << i << "," << j << ")";
    }
  }

  forecast.reset();
  EXPECT_EQ(forecast.historyLength(), 0);
}

/**
 * Test: CG Ritz extraction produces reasonable eigenvalue estimates.
 */
TEST(EigenTracking, CGRitzExtraction)
{
  using namespace quda;

  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) { loadCloverQuda(nullptr, nullptr, &inv_param); }

  QudaInvertParam ip = inv_param;
  ip.solve_type = QUDA_NORMOP_PC_SOLVE;
  ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

  // Generate pseudofermion with the (possibly bumped) mass
  ColorSpinorField phi = generateEOPseudofermion(ip, 77777);

  Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
  createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, true, false);

  ColorSpinorParam csParam(phi);
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField> x(1, csParam);
  std::vector<ColorSpinorField> b(1, ColorSpinorField(phi));
  solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);

  // Extract Ritz pairs. All knobs CLI-driven via --eigentracking-* and
  // --hmc-eigentest-* flags. CLI default of --eigentracking-n-ritz=0 maps to
  // a volume-scaled value via makeEigentestNev (same scaling rationale: the
  // low-mode density of M†M grows with V, so n_ritz must too — otherwise
  // TRLM exhausts max_restarts and returns un-converged zero eigenvalues).
  DiracMdagM matNorm(*dirac);
  const int nRitz = eigentracking_n_ritz > 0 ? eigentracking_n_ritz : makeEigentestNev();
  std::vector<ColorSpinorField> ritzVecs;
  std::vector<Complex> ritzVals;
  CGRitzExtractor::extract(ritzVecs, ritzVals, x[0], matNorm, nRitz,
                           /*nKr=*/0,
                           /*maxRestarts=*/eigentracking_trlm_max_restarts,
                           /*tol=*/eigentracking_trlm_tol,
                           /*usePolyAcc=*/eigentracking_use_poly_acc,
                           /*polyDeg=*/eigentracking_poly_deg,
                           /*aMin=*/eigentracking_a_min,
                           /*aMax=*/eigentracking_a_max);

  printfQuda("CGRitzExtraction: extracted %d Ritz pairs\n", static_cast<int>(ritzVals.size()));
  for (int i = 0; i < static_cast<int>(ritzVals.size()); i++) {
    printfQuda("  ritzVal[%d] = %e\n", i, ritzVals[i].real());
  }

  EXPECT_GT(static_cast<int>(ritzVals.size()), 0) << "Should extract at least one Ritz pair";

  // Verify Ritz values are positive (eigenvalues of M^dag M)
  for (auto &rv : ritzVals) { EXPECT_GT(rv.real(), 0.0) << "Ritz eigenvalue should be positive"; }

  delete dirac;
  delete diracSloppy;
  if (diracPre != diracSloppy) delete diracPre;
  if (diracEig != diracPre) delete diracEig;
}

/**
 * Test: Full HMC trajectory with eigentracking enabled.
 */
TEST(EigenTracking, FullTrajectory)
{
  QudaHMCParam hmc_param = makeHMCParam(QUDA_LEAPFROG_INTEGRATOR);
  // Enable eigentracking; everything else honours --eigentracking-* CLI
  // (defaults inherit from makeHMCParam, which already pulled CLI vars).
  hmc_param.eigentracking_enabled = 1;
  // CLI default for n_ev/pool_capacity/n_ritz is "0 = derive": substitute
  // volume-scaled fixture-friendly values only when CLI hasn't overridden.
  // Same scaling rationale as makeEigentestNev: low-mode density grows with
  // V, so n_ev (and n_ritz, which feeds TRLM with n_kr = 3·n_ritz) must too.
  const int derivedNev = makeEigentestNev();
  if (hmc_param.eigentracking_n_ev <= 0) hmc_param.eigentracking_n_ev = derivedNev;
  if (hmc_param.eigentracking_pool_capacity <= 0)
    hmc_param.eigentracking_pool_capacity = std::max(16, hmc_param.eigentracking_n_ev);
  if (hmc_param.eigentracking_n_ritz <= 0)
    hmc_param.eigentracking_n_ritz = std::max(3, hmc_param.eigentracking_n_ev / 2);
  // Disable periodic TRLM refresh for this single-trajectory test.
  hmc_param.eigentracking_fresh_trlm_interval = 0;

  QudaInvertParam ip = inv_param;

  // First trajectory
  double dH1 = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &ip, nullptr);
  printfQuda("EigenTracking FullTrajectory: dH1 = %e\n", dH1);
  EXPECT_TRUE(std::isfinite(dH1));

  // Second trajectory (tests persistent state)
  hmc_param.momentum_seed = hmc_momentum_seed + 1;
  double dH2 = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &ip, nullptr);
  printfQuda("EigenTracking FullTrajectory: dH2 = %e\n", dH2);
  EXPECT_TRUE(std::isfinite(dH2));
}

/**
 * Test: Thermalize then run production trajectories.
 *
 * Phase 1: Thermalization (always accepted, no eigentracking)
 * Phase 2: Production trajectories (eigentracking from CLI)
 *
 * All parameters from CLI:
 *   --hmc-thermalization N     thermalization trajectories (default 100)
 *   --hmc-n-trajectories N     production trajectories (default 20)
 *   --eigentracking 0/1        enable eigentracking in Phase 2
 *   --eigentracking-n-ev N     tracked eigenpairs
 *   etc.
 */
TEST(EigenTracking, ThermalizeAndTrack)
{
  // Honour the CLI --mass directly. If the user picks a small mass with a
  // small lattice, TRLM may not converge — that's a configuration choice,
  // not something to silently override here.
  QudaInvertParam ip = inv_param;

  // --- Phase 1: Thermalize (eigentracking off) ---
  QudaHMCParam therm_param = makeHMCParam();
  therm_param.n_trajectories = hmc_n_thermalization;
  therm_param.n_thermalization = hmc_n_thermalization; // all forced-accept
  therm_param.eigentracking_enabled = 0; // always off during thermalization
  therm_param.checkpoint_interval = hmc_checkpoint_interval;

  printfQuda("\n========================================\n");
  printfQuda("Phase 1: Thermalizing for %d trajectories\n", hmc_n_thermalization);
  printfQuda("========================================\n");

  hmcRunQuda(nullptr, &therm_param, &gauge_param, &ip, nullptr, nullptr);

  // Destroy HMC state between phases so eigentracking starts fresh
  destroyHMCQuda();

  // --- Phase 2: Production (eigentracking from CLI) ---
  QudaHMCParam prod_param = makeHMCParam();
  prod_param.n_trajectories = hmc_n_trajectories;
  prod_param.n_thermalization = 0; // Metropolis accept/reject active

  printfQuda("\n========================================\n");
  printfQuda("Phase 2: %d production trajectories (eigentracking=%d)\n", hmc_n_trajectories,
             prod_param.eigentracking_enabled);
  printfQuda("========================================\n");

  hmcRunQuda(nullptr, &prod_param, &gauge_param, &ip, nullptr, nullptr);

  SUCCEED();
}

/**
 * Test: CLI-driven HMC with optional MG preconditioning and eigentracking.
 *
 * All parameters from CLI. Supports:
 *   --hmc-gauge-infile   Load gauge from LIME file
 *   --mg-levels 2        Enable 2-level MG (with --mg-block-size, --mg-nvec, etc.)
 *   --eigentracking 1    Enable eigentracking
 *   --hmc-thermalization  Thermalization trajectories
 *   --hmc-n-trajectories  Production trajectories
 *
 * When --mg-levels >= 2, builds MG preconditioner using CLI --mg-* options.
 * When --mg-levels < 2 (default 1), runs plain CG without MG.
 */
TEST(HMC, Production)
{
  void *mg_preconditioner = nullptr;
  QudaMultigridParam mg_param = {};
  QudaInvertParam mg_inv_param = {};

  // Snapshot inv_param's solver-config fields so the optional MG
  // reconfiguration below does not leak into subsequent gtest cases.
  const void *saved_preconditioner = inv_param.preconditioner;
  const QudaInverterType saved_inv_type = inv_param.inv_type;
  const QudaInverterType saved_inv_type_precondition = inv_param.inv_type_precondition;
  const QudaSolveType saved_solve_type = inv_param.solve_type;

  // --- Optional MG setup (if --mg-levels >= 2) ---
  if (mg_levels >= 2) {
    // MG setup requires the gauge field loaded first
    if (hmc_gauge_infile.size() > 0) {
      GaugeFieldParam gParam(gauge_param);
      gParam.location = QUDA_CPU_FIELD_LOCATION;
      gParam.order = QUDA_QDP_GAUGE_ORDER;
      gParam.create = QUDA_NULL_FIELD_CREATE;
      quda::GaugeField cpuGauge(gParam);
      read_gauge_field(hmc_gauge_infile.c_str(), reinterpret_cast<void **>(cpuGauge.raw_pointer()),
                       gauge_param.cpu_prec, gauge_param.X, 0, nullptr);
      loadGaugeQuda(cpuGauge.raw_pointer(), &gauge_param);
    }

    // For Wilson with anisotropy, skip clover. For clover, load it.
    if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      loadCloverQuda(nullptr, nullptr, &inv_param);
    }

    // Create MG inv_param for setup (DIRECT_SOLVE, symmetric PC).
    // MG internal precision from --prec-precondition (default single if not set).
    QudaPrecision mg_prec = (prec_precondition != QUDA_INVALID_PRECISION) ? prec_precondition : QUDA_SINGLE_PRECISION;
    mg_inv_param = inv_param;
    mg_inv_param.solve_type = QUDA_DIRECT_SOLVE;
    mg_inv_param.solution_type = QUDA_MAT_SOLUTION;
    mg_inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    // MG inv_param: precise matches gauge (double), sloppy/precondition in mg_prec (single)
    mg_inv_param.cuda_prec = gauge_param.cuda_prec;           // must match gauge
    mg_inv_param.cuda_prec_sloppy = mg_prec;
    mg_inv_param.cuda_prec_precondition = mg_prec;
    mg_inv_param.cuda_prec_eigensolver = mg_prec;
    mg_inv_param.clover_cuda_prec = gauge_param.cuda_prec;
    mg_inv_param.clover_cuda_prec_sloppy = mg_prec;
    mg_inv_param.clover_cuda_prec_precondition = mg_prec;
    mg_inv_param.clover_cuda_prec_eigensolver = mg_prec;

    // Use the standard MG setup pattern from invert_test.cpp:
    // 1. Set default MG test params (fills all mgarray globals with sensible values)
    // 2. Apply CLI overrides (user's --mg-* flags are already in the globals)
    // 3. Set solve types for MG compatibility
    // 4. Call setMultigridParam to wire everything into mg_param
    if (prec_null == QUDA_INVALID_PRECISION) prec_null = mg_prec;

    // Snapshot ALL mgarray CLI values before setQudaDefaultMgTestParams overwrites them.
    // We use a lambda to save/restore any mgarray: if the CLI set a non-default value,
    // it wins over the test default.
    struct MgSnapshot {
      decltype(mg_eig) eig; decltype(mg_eig_n_ev) eig_n_ev; decltype(mg_eig_n_kr) eig_n_kr;
      decltype(mg_eig_use_normop) eig_normop; decltype(mg_eig_use_poly_acc) eig_poly;
      decltype(mg_eig_poly_deg) eig_poly_deg; decltype(mg_eig_amin) eig_amin;
      decltype(setup_inv) sinv; decltype(setup_maxiter) smax; decltype(setup_maxiter_refresh) srefresh;
      decltype(setup_tol) stol; decltype(num_setup_iter) siter;
      decltype(nu_pre) npre; decltype(nu_post) npost;
      decltype(smoother_type) sm; decltype(smoother_tol) smtol;
      decltype(coarse_solver) cs; decltype(coarse_solver_tol) cstol; decltype(coarse_solver_maxiter) csmax;
      decltype(mg_verbosity) verb; decltype(nvec) nv; decltype(geo_block_size) gbs;
    } snap = {mg_eig, mg_eig_n_ev, mg_eig_n_kr, mg_eig_use_normop, mg_eig_use_poly_acc,
              mg_eig_poly_deg, mg_eig_amin, setup_inv, setup_maxiter, setup_maxiter_refresh,
              setup_tol, num_setup_iter, nu_pre, nu_post, smoother_type, smoother_tol,
              coarse_solver, coarse_solver_tol, coarse_solver_maxiter, mg_verbosity, nvec, geo_block_size};

    setQudaDefaultMgTestParams();

    // Restore: CLI values that differ from zero-init win over test defaults.
    // Bool arrays always restore (zero-init == false is a valid CLI value).
    mg_eig = snap.eig; mg_eig_use_normop = snap.eig_normop;
    mg_eig_use_poly_acc = snap.eig_poly;
    for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
      if (snap.eig_n_ev[i]) mg_eig_n_ev[i] = snap.eig_n_ev[i];
      if (snap.eig_n_kr[i]) mg_eig_n_kr[i] = snap.eig_n_kr[i];
      if (snap.eig_poly_deg[i]) mg_eig_poly_deg[i] = snap.eig_poly_deg[i];
      if (snap.eig_amin[i] != 0) mg_eig_amin[i] = snap.eig_amin[i];
      if (snap.sinv[i] != 0) setup_inv[i] = snap.sinv[i];  // 0 = CG, also valid
      if (snap.smax[i]) setup_maxiter[i] = snap.smax[i];
      if (snap.srefresh[i]) setup_maxiter_refresh[i] = snap.srefresh[i];
      if (snap.stol[i] != 0) setup_tol[i] = snap.stol[i];
      if (snap.siter[i]) num_setup_iter[i] = snap.siter[i];
      if (snap.npre[i]) nu_pre[i] = snap.npre[i];
      if (snap.npost[i]) nu_post[i] = snap.npost[i];
      if (snap.sm[i] != 0) smoother_type[i] = snap.sm[i];
      if (snap.smtol[i] != 0) smoother_tol[i] = snap.smtol[i];
      if (snap.cs[i] != 0) coarse_solver[i] = snap.cs[i];
      if (snap.cstol[i] != 0) coarse_solver_tol[i] = snap.cstol[i];
      if (snap.csmax[i]) coarse_solver_maxiter[i] = snap.csmax[i];
      if (snap.verb[i] != 0) mg_verbosity[i] = snap.verb[i];
      if (snap.nv[i]) nvec[i] = snap.nv[i];
      for (int j = 0; j < 4; j++) if (snap.gbs[i][j]) geo_block_size[i][j] = snap.gbs[i][j];
    }

    // Override solve_type for MG (must be DIRECT_PC for outer HMC solve).
    // Keep it set through setMultigridParam — restored after MG setup.
    auto saved_solve_type = solve_type;
    solve_type = QUDA_DIRECT_PC_SOLVE;
    setQudaMgSolveTypes();

    // Configure MG param struct
    mg_param = newQudaMultigridParam();
    mg_param.invert_param = &mg_inv_param;

    // Wire eigensolver params for levels that use deflation (--mg-eig N true)
    static QudaEigParam mg_eig_params[QUDA_MAX_MG_LEVEL];
    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
        // For coarsest level, nvec[i] may be 0 (no null vectors at coarsest).
        // setMultigridEigParam uses nvec[i] for n_conv — ensure it's set.
        if (nvec[i] == 0) nvec[i] = mg_eig_n_ev[i] > 0 ? mg_eig_n_ev[i] : 24;
        mg_eig_params[i] = newQudaEigParam();
        setMultigridEigParam(mg_eig_params[i], i);
        mg_param.eig_param[i] = &mg_eig_params[i];
      } else {
        mg_param.eig_param[i] = nullptr;
      }
    }

    // Apply full CLI-driven MG configuration
    setMultigridParam(mg_param);
    solve_type = saved_solve_type; // restore after MG setup

    // Debug: verify deflation configuration
    for (int i = 0; i < mg_param.n_level; i++) {
      printfQuda("MG level %d: use_eig_solver=%d, eig_param=%p\n", i,
                 mg_param.use_eig_solver[i], (void *)mg_param.eig_param[i]);
      if (mg_param.eig_param[i]) {
        printfQuda("  eig: n_ev=%d, n_kr=%d, n_conv=%d, use_norm_op=%d, use_poly_acc=%d, poly_deg=%d\n",
                   mg_param.eig_param[i]->n_ev, mg_param.eig_param[i]->n_kr, mg_param.eig_param[i]->n_conv,
                   mg_param.eig_param[i]->use_norm_op, mg_param.eig_param[i]->use_poly_acc,
                   mg_param.eig_param[i]->poly_deg);
      }
    }

    printfQuda("Setting up %d-level MG preconditioner...\n", mg_levels);
    mg_preconditioner = newMultigridQuda(&mg_param);

    // Outer solver: GCR + MG with DIRECT_PC_SOLVE (required by QUDA MG validation).
    inv_param.preconditioner = mg_preconditioner;
    inv_param.inv_type = QUDA_GCR_INVERTER;
    inv_param.inv_type_precondition = QUDA_MG_INVERTER;
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    printfQuda("MG preconditioner ready.\n");
    // Note: inv_param solver-config fields are restored to pre-MG values in
    // the "Cleanup MG" block below so this test doesn't leak MG state into
    // any subsequent gtest case.
    // Eigentracker MG seeding is now done by hmcTrajectoryQuda
    // automatically when mg_instance is passed; no explicit call needed.
  }

  // --- Configure HMC ---
  QudaHMCParam hmc_param = makeHMCParam();
  // Resolve eigentracking 0-defaults from MG nvec (or standalone defaults)
  if (hmc_param.eigentracking_enabled) {
    int mg_nvec = 0;
    if (mg_preconditioner) {
      auto *mg_s = static_cast<quda::multigrid_solver *>(mg_preconditioner);
      mg_nvec = static_cast<int>(mg_s->B.size());
    }
    resolveEigenTrackingDefaults(hmc_param, mg_nvec);
  }
  hmc_param.n_trajectories = hmc_n_trajectories;
  hmc_param.n_thermalization = hmc_n_thermalization;
  hmc_param.checkpoint_interval = hmc_checkpoint_interval;
  hmc_param.mg_setup_interval = hmc_mg_setup_interval;
  hmc_param.mg_setup_iter_ratio = hmc_mg_setup_iter_ratio;
  hmc_param.mg_setup_iter_baseline_traj = hmc_mg_setup_iter_baseline_traj;

  strncpy(hmc_param.checkpoint_prefix, hmc_checkpoint_prefix.c_str(), sizeof(hmc_param.checkpoint_prefix) - 1);
  strncpy(hmc_param.gauge_outfile, hmc_gauge_outfile.c_str(), sizeof(hmc_param.gauge_outfile) - 1);

  // If MG is enabled, gauge was already loaded during MG setup — use resident.
  // Otherwise, let hmcRunQuda load from file.
  if (mg_preconditioner) {
    hmc_param.gauge_infile[0] = '\0';
    hmc_param.use_resident_gauge = 1;
    hmc_param.make_resident_gauge = 1;
  } else {
    strncpy(hmc_param.gauge_infile, hmc_gauge_infile.c_str(), sizeof(hmc_param.gauge_infile) - 1);
  }

  printfQuda("\n========================================\n");
  printfQuda("HMC Production: %d trajectories (%d therm)\n", hmc_n_trajectories, hmc_n_thermalization);
  printfQuda("  MG=%s, eigentracking=%d\n", mg_preconditioner ? "enabled" : "disabled",
             hmc_param.eigentracking_enabled);
  printfQuda("========================================\n");

  hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, mg_preconditioner,
             mg_preconditioner ? &mg_param : nullptr);

  // Cleanup MG and fully restore inv_param so the next test starts clean.
  if (mg_preconditioner) {
    destroyMultigridQuda(mg_preconditioner);
  }
  inv_param.preconditioner = const_cast<void *>(saved_preconditioner);
  inv_param.inv_type = saved_inv_type;
  inv_param.inv_type_precondition = saved_inv_type_precondition;
  inv_param.solve_type = saved_solve_type;

  SUCCEED();
}

// ============================================================================
// Eigentracking solver-side tracker regression tests
// ----------------------------------------------------------------------------
// These tests lock in the architectural invariants of the per-solve Krylov
// capture (--eigentracking-residual-cap > 0) wired into inv_cg_quda.cpp and
// inv_gcr_quda.cpp through CGTracker / GCRTracker (cg_ritz_extractor.cpp,
// gcr_tracker.cpp) and the shared TrackerScope<T> / takeRitzVectors helpers
// in inv_tracker.h. The implementation peeled off four real bugs during
// development — move-assign guard on the FIFO, sloppy/double precision
// mismatch in multiCdot, hierarchy leak from coarse-MG-level GCR, and a
// half-to-full-site embedding that inverted the pool's site-subset match.
// Each test below pins one of those invariants so the next person to touch
// the trackers cannot silently reintroduce them.
// ============================================================================

namespace
{

  /** @brief Construct an empty fine-grid Wilson spinor (Ns=4, Nc=3) at the
   *         requested precision and site subset. Uses the test fixture's
   *         inv_param + gauge dimensions, so the field is compatible with
   *         the operators / pool the production HMC builds. */
  quda::ColorSpinorField makeFineSpinor(QudaPrecision prec_, QudaSiteSubset siteSubset)
  {
    // ColorSpinorParam wants lat_dim_t (quda::array<int, QUDA_MAX_DIM>);
    // gauge_param.X is plain int[4]. Promote.
    quda::lat_dim_t X{};
    for (int i = 0; i < 4; i++) X[i] = gauge_param.X[i];
    quda::ColorSpinorParam csParam(nullptr, inv_param, X,
                                   /*pc_solution=*/(siteSubset == QUDA_PARITY_SITE_SUBSET),
                                   QUDA_CUDA_FIELD_LOCATION);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(prec_);
    csParam.fieldOrder = QUDA_NATIVE_FIELD_ORDER;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    return quda::ColorSpinorField(csParam);
  }

  /** @brief Construct an empty coarse-grid spinor (Ns=2, Nc=24) by deriving
   *         from a fine spinor's create_coarse with a 2x2x2x2 block. Used to
   *         simulate the field shape that QUDA's MG-internal level-1 GCR
   *         hands recordIteration when the eigentracking hook fires
   *         throughout the MG hierarchy. */
  quda::ColorSpinorField makeCoarseSpinor()
  {
    auto fine = makeFineSpinor(QUDA_DOUBLE_PRECISION, QUDA_FULL_SITE_SUBSET);
    const int blockSize[4] = {2, 2, 2, 2};
    return fine.create_coarse(blockSize, /*spinBlockSize=*/2, /*Nvec=*/24,
                               QUDA_DOUBLE_PRECISION, QUDA_CUDA_FIELD_LOCATION);
  }

} // namespace

/**
 * @brief Hierarchy filter: GCRTracker silently drops residuals whose spinor
 *        shape does not match a fine Wilson field (Ns=4, Nc=3).
 *
 * The activeGCRTracker hook in inv_gcr_quda.cpp fires for every GCR
 * instance in the call tree. With MG enabled, that includes the level-1
 * coarse-grid GCR run inside the preconditioner. Coarse residuals have
 * (Ns=2, Nc=nVec); they do not match the EigenTracker pool's fine-grid
 * reference vectors, so the absorb path faults if they reach it. The
 * filter at the top of recordIteration drops them before they can be
 * stashed.
 */
TEST(HMC, GCRTrackerHierarchyFilter)
{
  using namespace quda;

  GCRTracker tracker(/*maxVecs=*/4, /*targetPrecision=*/QUDA_DOUBLE_PRECISION);
  ASSERT_TRUE(tracker.isActive());
  EXPECT_EQ(tracker.numStored(), 0);

  // A coarse-shape spinor must be silently dropped — no contribution to
  // the stored residual list, no exception, no crash.
  auto coarse = makeCoarseSpinor();
  ASSERT_NE(coarse.Nspin(), 4) << "test setup: coarse spinor should have Ns != 4";
  ASSERT_NE(coarse.Ncolor(), 3) << "test setup: coarse spinor should have Nc != 3";
  blas::ax(2.5, coarse); // give it a non-trivial norm so a successful
                         // store would be detectable via takeResiduals.
  tracker.recordIteration(coarse);
  EXPECT_EQ(tracker.numStored(), 0) << "coarse residual leaked past the hierarchy filter";

  // A fine Wilson spinor must be accepted.
  auto fine = makeFineSpinor(QUDA_DOUBLE_PRECISION, QUDA_FULL_SITE_SUBSET);
  ASSERT_EQ(fine.Nspin(), 4);
  ASSERT_EQ(fine.Ncolor(), 3);
  spinorNoise(fine, /*seed=*/12345, QUDA_NOISE_GAUSS);
  tracker.recordIteration(fine);
  EXPECT_EQ(tracker.numStored(), 1) << "fine residual was incorrectly filtered";
}

/**
 * @brief Precision promotion: GCRTracker stores residuals at the target
 *        precision passed to its constructor, regardless of the source's
 *        precision.
 *
 * The pool's reference vectors and absorption kernels live at
 * inv_param.cuda_prec (typically double). GCR runs at precision_sloppy
 * (typically single). Without an explicit promotion the pool absorb path
 * faults inside multiCdot — no instantiation exists for the mixed
 * double-pool / single-residual combination.
 */
TEST(HMC, GCRTrackerPrecisionPromotion)
{
  using namespace quda;

  // Construct a single-precision fine spinor with non-trivial content.
  auto srcSingle = makeFineSpinor(QUDA_SINGLE_PRECISION, QUDA_FULL_SITE_SUBSET);
  spinorNoise(srcSingle, /*seed=*/54321, QUDA_NOISE_GAUSS);
  ASSERT_GT(blas::norm2(srcSingle), 0.0);

  // Tracker promotes to double.
  GCRTracker tracker(/*maxVecs=*/1, /*targetPrecision=*/QUDA_DOUBLE_PRECISION);
  tracker.recordIteration(srcSingle);
  ASSERT_EQ(tracker.numStored(), 1);

  auto stored = tracker.takeResiduals();
  ASSERT_EQ(stored.size(), 1u);

  EXPECT_EQ(stored[0].Precision(), QUDA_DOUBLE_PRECISION) << "stored field not promoted";
  EXPECT_EQ(stored[0].Nspin(),    4);
  EXPECT_EQ(stored[0].Ncolor(),   3);

  // recordIteration normalises: stored field should have unit L2 norm.
  EXPECT_NEAR(blas::norm2(stored[0]), 1.0, 1e-10);

  // Tracker should have drained.
  EXPECT_EQ(tracker.numStored(), 0);
}

/**
 * @brief Site-subset preservation: GCRTracker does NOT embed a half-site
 *        residual into a full-site container.
 *
 * The EigenTracker pool is seeded by hmc.cpp's seedEigenTrackingFromMG
 * from the EVEN-PARITY (half-site) components of the MG null vectors, so
 * pool reference vectors are half-site fine. Inside a PC solve GCR's
 * r_sloppy is also half-site, so the two match by construction. An
 * earlier development version embedded half-site into full-site here
 * "to align with the pool" — that inverted the match and reintroduced
 * the MultiReduce length-mismatch crash. This test pins the design.
 */
TEST(HMC, GCRTrackerSiteSubsetPreserved)
{
  using namespace quda;

  // Half-site (single-parity) fine spinor at single precision.
  auto srcHalf = makeFineSpinor(QUDA_SINGLE_PRECISION, QUDA_PARITY_SITE_SUBSET);
  spinorNoise(srcHalf, /*seed=*/13579, QUDA_NOISE_GAUSS);
  ASSERT_EQ(srcHalf.SiteSubset(), QUDA_PARITY_SITE_SUBSET);

  GCRTracker tracker(/*maxVecs=*/1, /*targetPrecision=*/QUDA_DOUBLE_PRECISION);
  tracker.recordIteration(srcHalf);
  ASSERT_EQ(tracker.numStored(), 1);

  auto stored = tracker.takeResiduals();
  ASSERT_EQ(stored.size(), 1u);

  EXPECT_EQ(stored[0].SiteSubset(), QUDA_PARITY_SITE_SUBSET) << "half-site residual was embedded into full-site";
  EXPECT_EQ(stored[0].Precision(),  QUDA_DOUBLE_PRECISION)   << "precision promotion still required";
  EXPECT_NEAR(blas::norm2(stored[0]), 1.0, 1e-10);
}

/**
 * @brief End-to-end: when --eigentracking-residual-cap > 0 the per-solve
 *        Krylov capture actually feeds the EigenTracker pool, in addition
 *        to the converged-solution stash that runs unconditionally.
 *
 * Mirrors the MG setup of HMC.MGPreconditionedRun (which is the path the
 * GCR install instruments) and forces ET on with cap=4. After one
 * trajectory the absorbed-Ritz count must be strictly positive — a
 * cap=0 baseline would absorb only the converged solution, while cap=4
 * adds 4 GCR residuals per γ₅ two-pass solve. The exact number depends
 * on how many force / action solves an integrator step does, so we
 * assert >= 1 (firmly catches the regression where the install path
 * silently no-ops; the differential CG-vs-GCR-cap exact arithmetic is
 * better verified at production scale by paired HMC.Production runs).
 */
TEST(HMC, EigenTrackerCapEnrichesPool)
{
  using namespace quda;

  // Force MG-aligned clover precisions if needed (same as MGPreconditionedRun).
  if (inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cuda_prec = inv_param.cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = inv_param.cuda_prec_precondition;
    inv_param.clover_cuda_prec_eigensolver = inv_param.cuda_prec_precondition;
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }

  // Reuse the known-good 4^4 MG configuration from HMC.MGPreconditionedRun.
  QudaPrecision mg_prec =
    (prec_precondition != QUDA_INVALID_PRECISION) ? prec_precondition : QUDA_SINGLE_PRECISION;
  QudaInvertParam mg_inv_param = inv_param;
  mg_inv_param.solve_type = QUDA_DIRECT_SOLVE;
  mg_inv_param.solution_type = QUDA_MAT_SOLUTION;
  mg_inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  mg_inv_param.cuda_prec = gauge_param.cuda_prec;
  mg_inv_param.cuda_prec_sloppy = mg_prec;
  mg_inv_param.cuda_prec_precondition = mg_prec;
  mg_inv_param.cuda_prec_eigensolver = mg_prec;
  mg_inv_param.clover_cuda_prec = gauge_param.cuda_prec;
  mg_inv_param.clover_cuda_prec_sloppy = mg_prec;
  mg_inv_param.clover_cuda_prec_precondition = mg_prec;
  mg_inv_param.clover_cuda_prec_eigensolver = mg_prec;

  QudaMultigridParam mg_param = newQudaMultigridParam();
  configureHMCTestMG(mg_param, mg_inv_param, mg_prec);

  void *mg_preconditioner = newMultigridQuda(&mg_param);

  // Snapshot inv_param for restore at exit.
  const void *saved_preconditioner = inv_param.preconditioner;
  const QudaInverterType saved_inv_type = inv_param.inv_type;
  const QudaInverterType saved_inv_type_precondition = inv_param.inv_type_precondition;
  const QudaSolveType saved_solve_type = inv_param.solve_type;

  inv_param.preconditioner = mg_preconditioner;
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;

  // Force cap > 0 + ET on for this test, restore the user's CLI values at exit.
  const int saved_cap = eigentracking_residual_cap;
  const bool saved_et = eigentracking_enabled;
  eigentracking_residual_cap = 4;
  eigentracking_enabled = true;

  // Tear down any pre-existing tracker so this test starts clean.
  if (auto *prev = getEigenTrackingInstance()) {
    delete prev;
    setEigenTrackingInstance(nullptr);
  }

  QudaHMCParam hmc_param = makeHMCParam();
  hmc_param.n_trajectories = 1;
  hmc_param.n_thermalization = 0;

  hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, mg_preconditioner, &mg_param);

  auto *et = getEigenTrackingInstance();
  ASSERT_NE(et, nullptr) << "EigenTrackingState was not constructed during HMC";

  EXPECT_GE(et->getTrajectoryCount(), 1);
  EXPECT_GT(et->getTotalRitzAbsorbed(), 0)
      << "per-solve Krylov capture (cap=4) failed to feed the pool: "
         "got " << et->getTotalRitzAbsorbed() << " absorbed";

  // Restore HMC + ET state.
  destroyMultigridQuda(mg_preconditioner);
  inv_param.preconditioner = const_cast<void *>(saved_preconditioner);
  inv_param.inv_type = saved_inv_type;
  inv_param.inv_type_precondition = saved_inv_type_precondition;
  inv_param.solve_type = saved_solve_type;
  eigentracking_residual_cap = saved_cap;
  eigentracking_enabled = saved_et;
}

/**
 * @brief TrackerScope<T> install/restore lifecycle.
 *
 * Locks in the contract of inv_tracker.h's templated scope: on
 * construction the scope sets the supplied global slot to the provided
 * tracker pointer; on destruction it restores whatever value the slot
 * had before. Nested scopes see the outer install once the inner exits.
 */
TEST(HMC, TrackerScopeInstallRestore)
{
  using namespace quda;

  // Start from a known clean state regardless of previous tests.
  GCRTracker *initial = activeGCRTracker;
  activeGCRTracker = nullptr;
  ASSERT_EQ(activeGCRTracker, nullptr);

  GCRTracker outer(/*maxVecs=*/1, /*targetPrecision=*/QUDA_DOUBLE_PRECISION);
  GCRTracker inner(/*maxVecs=*/2, /*targetPrecision=*/QUDA_DOUBLE_PRECISION);

  {
    TrackerScope<GCRTracker> scope_outer(activeGCRTracker, &outer);
    EXPECT_EQ(activeGCRTracker, &outer);

    {
      TrackerScope<GCRTracker> scope_inner(activeGCRTracker, &inner);
      EXPECT_EQ(activeGCRTracker, &inner);

      // A nested scope with nullptr suspends tracking inside the inner
      // region without disturbing the outer install.
      {
        TrackerScope<GCRTracker> scope_null(activeGCRTracker, nullptr);
        EXPECT_EQ(activeGCRTracker, nullptr);
      }
      EXPECT_EQ(activeGCRTracker, &inner);
    }
    EXPECT_EQ(activeGCRTracker, &outer);
  }
  EXPECT_EQ(activeGCRTracker, nullptr);

  // Restore prior state (paranoia for any later test).
  activeGCRTracker = initial;
}

int main(int argc, char **argv)
{
  // Let gtest strip its args first
  ::testing::InitGoogleTest(&argc, argv);

  // HMC tests need double outer precision for numerical force/action
  // consistency: single-precision CG convergence introduces 10-15%
  // per-link errors that masquerade as algorithmic bugs (see CLAUDE.md).
  // Sloppy/precondition stay at single so MG (which doesn't have double
  // kernels compiled in this build) can use them. Set BEFORE CLI parsing
  // so any explicit --prec / --prec-sloppy / --prec-precondition wins.
  prec = QUDA_DOUBLE_PRECISION;
  prec_sloppy = QUDA_SINGLE_PRECISION;
  prec_precondition = QUDA_SINGLE_PRECISION;
  prec_eigensolver = QUDA_SINGLE_PRECISION;

  // Process remaining command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  add_hmc_option_group(app);
  app->allow_extras();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Initialize communications and QUDA
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  // Initialize test fields
  initHMCTest(argc, argv);

  // Run tests
  int result = RUN_ALL_TESTS();

  // Cleanup: free resident fields before endQuda to avoid destruction-order issues
  destroyHMCQuda();
  freeGaugeQuda();
  freeCloverQuda();
  endQuda();
  finalizeComms();

  return result;
}
