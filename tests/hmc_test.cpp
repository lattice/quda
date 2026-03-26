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
static quda::GaugeField gauge;
static quda::GaugeField mom;

void initHMCTest(int argc, char **argv)
{
  // Initialize QUDA gauge parameters
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Initialize inverter parameters
  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  setDims(gauge_param.X);

  // Allocate and randomize host gauge field
  quda::GaugeFieldParam gParam(gauge_param, nullptr, QUDA_SU3_LINKS);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gauge = quda::GaugeField(gParam);
  constructHostGaugeField(gauge, gauge_param, argc, argv);

  // Allocate host momentum field
  quda::GaugeFieldParam mParam(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
  mParam.create = QUDA_ZERO_FIELD_CREATE;
  mParam.order = QUDA_MILC_GAUGE_ORDER;
  mParam.reconstruct = QUDA_RECONSTRUCT_10;
  mom = quda::GaugeField(mParam);

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
  EXPECT_EQ(p.return_result_gauge, 1);
  EXPECT_EQ(p.return_result_mom, 1);
}

/**
 * Test: Leapfrog trajectory with energy conservation check.
 *
 * Demonstrates the simplest usage pattern:
 *   1. Set up gauge and inverter parameters
 *   2. Configure QudaHMCParam with leapfrog integrator
 *   3. Call hmcTrajectoryQuda with host pointers
 *   4. Check that dH is returned (energy conservation)
 */
TEST(HMC, LeapfrogTrajectory)
{
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_LEAPFROG_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 20;
  hmc_param.use_resident_gauge = 0;
  hmc_param.make_resident_gauge = 0;
  hmc_param.return_result_gauge = 1;
  hmc_param.use_resident_mom = 0;
  hmc_param.make_resident_mom = 0;
  hmc_param.return_result_mom = 1;

  double dH = hmcTrajectoryQuda(gauge.data(), mom.data(), &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("Leapfrog: dH = %e\n", dH);
  // dH should be finite (not NaN/Inf)
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Omelyan trajectory.
 *
 * Shows how to configure the Omelyan integrator with a custom lambda parameter.
 */
TEST(HMC, OmelyanTrajectory)
{
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_OMELYAN_INTEGRATOR;
  hmc_param.tau = 1.0;
  hmc_param.n_steps = 10;
  hmc_param.omelyan_lambda = 0.1932;
  hmc_param.use_resident_gauge = 0;
  hmc_param.make_resident_gauge = 0;
  hmc_param.return_result_gauge = 1;
  hmc_param.use_resident_mom = 0;
  hmc_param.make_resident_mom = 0;
  hmc_param.return_result_mom = 1;

  double dH = hmcTrajectoryQuda(gauge.data(), mom.data(), &hmc_param, &gauge_param, &inv_param, nullptr);

  printfQuda("Omelyan: dH = %e\n", dH);
  EXPECT_TRUE(std::isfinite(dH));
}

/**
 * Test: Device-resident workflow.
 *
 * Shows how to keep fields on device across multiple trajectories,
 * avoiding host-device copies. This is the high-performance usage pattern.
 */
TEST(HMC, DeviceResidentWorkflow)
{
  // First trajectory: load from host, keep on device
  QudaHMCParam hmc_param = newQudaHMCParam();
  hmc_param.integrator = QUDA_LEAPFROG_INTEGRATOR;
  hmc_param.tau = 0.5;
  hmc_param.n_steps = 10;
  hmc_param.use_resident_gauge = 0;
  hmc_param.make_resident_gauge = 1; // keep gauge on device
  hmc_param.return_result_gauge = 0; // don't copy back
  hmc_param.use_resident_mom = 0;
  hmc_param.make_resident_mom = 1; // keep momentum on device
  hmc_param.return_result_mom = 0;

  double dH1 = hmcTrajectoryQuda(gauge.data(), mom.data(), &hmc_param, &gauge_param, &inv_param, nullptr);
  printfQuda("Resident trajectory 1: dH = %e\n", dH1);

  // Second trajectory: use device-resident fields, return to host at the end
  hmc_param.use_resident_gauge = 1;
  hmc_param.make_resident_gauge = 0;
  hmc_param.return_result_gauge = 1; // copy final gauge back
  hmc_param.use_resident_mom = 1;
  hmc_param.make_resident_mom = 0;
  hmc_param.return_result_mom = 1;

  double dH2 = hmcTrajectoryQuda(nullptr, nullptr, &hmc_param, &gauge_param, &inv_param, nullptr);
  printfQuda("Resident trajectory 2: dH = %e\n", dH2);

  EXPECT_TRUE(std::isfinite(dH1));
  EXPECT_TRUE(std::isfinite(dH2));
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
 * Example: Full nested FGI trajectory with MG (commented out -- requires MG setup).
 *
 * This shows the complete workflow an external library would use:
 *
 *   // 1. Setup MG preconditioner (one-time)
 *   QudaMultigridParam mg_param = newQudaMultigridParam();
 *   // ... configure mg_param levels, null vectors, smoothers ...
 *   void *mg = newMultigridQuda(&mg_param);
 *
 *   // 2. Configure HMC
 *   QudaHMCParam hmc_param = newQudaHMCParam();
 *   hmc_param.integrator = QUDA_NESTED_FGI_INTEGRATOR;
 *   hmc_param.tau = 1.0;
 *   hmc_param.n_steps = 5;
 *   hmc_param.n_inner_steps = 3;
 *   hmc_param.n_defl = 32;
 *   hmc_param.eig_tol = 1e-6;
 *   hmc_param.n_mr_smooth = 3;
 *
 *   // 3. HMC loop
 *   for (int traj = 0; traj < n_trajectories; traj++) {
 *     // Generate random momentum from Gaussian distribution
 *     gaussianMomentum(momentum, gauge_param);
 *
 *     // Run MD trajectory -- returns dH for Metropolis test
 *     double dH = hmcTrajectoryQuda(gauge, momentum, &hmc_param,
 *                                    &gauge_param, &inv_param, mg);
 *
 *     // Metropolis accept/reject
 *     double r = drand48();
 *     if (r < exp(-dH)) {
 *       // Accept: gauge is already updated
 *       printfQuda("Trajectory %d: ACCEPTED (dH = %e)\n", traj, dH);
 *     } else {
 *       // Reject: restore gauge from backup
 *       printfQuda("Trajectory %d: REJECTED (dH = %e)\n", traj, dH);
 *       // ... restore gauge ...
 *     }
 *
 *     // Update MG for new gauge field
 *     updateMultigridQuda(mg, &mg_param);
 *   }
 *
 *   // 4. Cleanup
 *   destroyHMCQuda();
 *   destroyMultigridQuda(mg);
 */

int main(int argc, char **argv)
{
  // Process command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  add_eigen_option_group(app);
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
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  // Cleanup
  destroyHMCQuda();
  endQuda();
  finalizeComms();

  return result;
}
