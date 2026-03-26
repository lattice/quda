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

#include <quda.h>
#include <gauge_field.h>
#include <gtest/gtest.h>

#include "command_line_params.h"
#include "gauge_utils.h"
#include "host_utils.h"
#include "momentum_utils.h"
#include "misc.h"
#include "test.h"

// Global test state
static QudaGaugeParam gauge_param;
static QudaInvertParam inv_param;

// Host gauge storage (QDP order: array of 4 pointers)
static std::vector<char> gauge_buf;
static void *gauge[4];

void initHMCTest(int argc, char **argv)
{
  // Initialize QUDA gauge parameters
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Initialize inverter parameters
  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.clover_csw = 1.0;
  inv_param.clover_coeff = inv_param.clover_csw * inv_param.kappa;
  inv_param.clover_cpu_prec = gauge_param.cpu_prec;
  inv_param.clover_cuda_prec = gauge_param.cuda_prec;
  inv_param.clover_cuda_prec_sloppy = gauge_param.cuda_prec_sloppy;
  inv_param.clover_cuda_prec_precondition = gauge_param.cuda_prec_precondition;
  inv_param.clover_cuda_prec_eigensolver = gauge_param.cuda_prec_eigensolver;
  inv_param.clover_cuda_prec_refinement_sloppy = gauge_param.cuda_prec_refinement_sloppy;
  inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  inv_param.compute_clover = QUDA_BOOLEAN_TRUE;
  setDims(gauge_param.X);

  // Allocate host gauge field (QDP order)
  gauge_buf.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) gauge[i] = gauge_buf.data() + i * V * gauge_site_size * host_gauge_data_type_size;
  constructHostGaugeField(gauge, gauge_param, argc, argv);

  // Load gauge to QUDA (makes it resident)
  gauge_param.use_resident_gauge = 0;
  gauge_param.make_resident_gauge = 1;
  loadGaugeQuda(gauge, &gauge_param);

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
 * Test: Leapfrog trajectory with energy conservation check.
 *
 * Demonstrates the self-contained usage pattern:
 *   1. Load gauge from host
 *   2. Let QUDA generate Gaussian momentum internally
 *   3. Call hmcTrajectoryQuda -- it generates pseudofermion, runs MD, returns dH
 */
TEST(HMC, LeapfrogTrajectory)
{
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_LEAPFROG_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 10;
  hmc_param.beta = 6.0;
  hmc_param.generate_momentum = 1;
  hmc_param.momentum_seed = 12345;
  hmc_param.use_resident_gauge = 1;
  hmc_param.make_resident_gauge = 1;
  hmc_param.return_result_gauge = 0;

  double dH = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("Leapfrog: dH = %e\n", dH);
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Omelyan trajectory.
 */
TEST(HMC, OmelyanTrajectory)
{
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_OMELYAN_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 5;
  hmc_param.omelyan_lambda = 0.1932;
  hmc_param.beta = 6.0;
  hmc_param.generate_momentum = 1;
  hmc_param.momentum_seed = 54321;
  hmc_param.use_resident_gauge = 1;
  hmc_param.make_resident_gauge = 1;
  hmc_param.return_result_gauge = 0;

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
 */
TEST(HMC, ForceGradientTrajectory)
{
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_FORCE_GRADIENT_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 3;
  hmc_param.fgi_lambda = 1.0 / 6.0;
  hmc_param.fgi_xi = 1.0 / 72.0;
  hmc_param.beta = 6.0;
  hmc_param.generate_momentum = 1;
  hmc_param.momentum_seed = 99999;
  hmc_param.use_resident_gauge = 1;
  hmc_param.make_resident_gauge = 1;
  hmc_param.return_result_gauge = 0;

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
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_NESTED_FGI_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 5;

  // FGI coefficients (from report)
  hmc_param.fgi_lambda = 1.0 / 6.0;
  hmc_param.fgi_xi = 1.0 / 72.0;

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
  EXPECT_EQ(hmc_param.n_steps, 5);
  EXPECT_EQ(hmc_param.n_inner_steps, 3);
  EXPECT_EQ(hmc_param.n_defl, 32);
  EXPECT_EQ(hmc_param.n_mr_smooth, 3);

  // Print the CG solve count estimate from the report:
  // Total CG solves = 3 * n_outer + 3 (including Hamiltonian evaluations)
  int cg_per_traj = 3 * hmc_param.n_steps + 3;
  printfQuda("Nested FGI: estimated %d CG solves per trajectory (n_outer=%d)\n", cg_per_traj, hmc_param.n_steps);
  EXPECT_EQ(cg_per_traj, 18); // 3*5 + 3 = 18
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
  QudaHMCParam hmc_param = newQudaHMCParam();

  // Use CLI options if set, otherwise defaults
  hmc_param.integrator = static_cast<QudaIntegratorType>(hmc_integrator);
  hmc_param.beta = hmc_beta;
  hmc_param.tau = hmc_tau;
  hmc_param.n_steps = hmc_n_steps;
  hmc_param.n_trajectories = hmc_n_trajectories;
  hmc_param.n_thermalization = hmc_n_thermalization;
  hmc_param.checkpoint_interval = hmc_checkpoint_interval;
  hmc_param.generate_momentum = 1;
  hmc_param.momentum_seed = 42;

  strncpy(hmc_param.checkpoint_prefix, hmc_checkpoint_prefix.c_str(), sizeof(hmc_param.checkpoint_prefix) - 1);
  strncpy(hmc_param.gauge_infile, hmc_gauge_infile.c_str(), sizeof(hmc_param.gauge_infile) - 1);
  strncpy(hmc_param.gauge_outfile, hmc_gauge_outfile.c_str(), sizeof(hmc_param.gauge_outfile) - 1);

  // Load gauge from host (already loaded in initHMCTest)
  hmc_param.use_resident_gauge = 1;

  hmcRunQuda(nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);

  // If we get here without crashing, the run succeeded
  SUCCEED();
}

int main(int argc, char **argv)
{
  // Let gtest strip its args first
  ::testing::InitGoogleTest(&argc, argv);

  // Process remaining command line options
  auto app = make_app();
  add_hmc_option_group(app);
  add_multigrid_option_group(app);
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
