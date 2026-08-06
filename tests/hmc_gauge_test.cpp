#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <random>
#include <string>
#include <vector>

#include <quda.h>
#include <util_quda.h>
#include <comm_quda.h>
#include <qio_field.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "gauge_utils.h"

/**
   Pure-gauge HMC driver for the Wilson plaquette action, built entirely
   on the public interface: gaussMomQuda / momActionQuda /
   computeGaugeForceQuda / updateGaugeFieldQuda / plaqQuda.

   This exists as an algorithm cross-check for the correlator
   distribution program: a second exact update scheme generating the
   same target distribution as the heatbath driver, and the scaffold
   onto which pseudofermion forces can be added for dynamical running.
 */

// file-local options, static to avoid collision with the hmc_test option globals
static int hmcg_trajectories = 100;
static int hmcg_therm = 20;
static int hmcg_traj_steps = 25;
static double hmcg_traj_length = 1.0;
static double hmcg_beta = 5.9;
static double hmcg_force_sign = 1.0;
static unsigned long long hmcg_seed = 5551212;
static std::string hmcg_save_prefix = "";
static int hmcg_save_interval = 0;
static int hmcg_config_start = 0;

static void add_hmc_gauge_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  auto opgroup = quda_app->add_option_group("hmc", "Options controlling pure-gauge HMC");
  opgroup->add_option("--hmc-beta", hmcg_beta, "Wilson plaquette action beta (default 5.9)");
  opgroup->add_option("--hmc-trajectories", hmcg_trajectories, "Number of measured trajectories (default 100)");
  opgroup->add_option("--hmc-therm", hmcg_therm, "Number of thermalization trajectories (default 20)");
  opgroup->add_option("--hmc-traj-steps", hmcg_traj_steps, "Leapfrog steps per trajectory (default 25)");
  opgroup->add_option("--hmc-traj-length", hmcg_traj_length, "Trajectory length in MD time (default 1.0)");
  opgroup->add_option("--hmc-seed", hmcg_seed, "RNG seed for momenta and Metropolis (default 5551212)");
  opgroup->add_option("--hmc-force-sign", hmcg_force_sign, "Sign convention of the force accumulation (default +1)");
  opgroup->add_option("--hmc-save-prefix", hmcg_save_prefix,
                      "If set, save accepted configurations to <prefix>_cfg_<n>.lime at the save interval");
  opgroup->add_option("--hmc-save-interval", hmcg_save_interval,
                      "Trajectories between configuration saves; 0 disables (default 0)");
  opgroup->add_option("--hmc-config-start", hmcg_config_start,
                      "Offset added to the trajectory number when naming saved configurations (default 0)");
}

// The six plaquette staples per direction, in the standard path
// convention (7 - dir denotes a backwards link), taken from
// gauge_path_test.cpp
static int plaq_path_x[6][3] = {{1, 7, 6}, {6, 7, 1}, {2, 7, 5}, {5, 7, 2}, {3, 7, 4}, {4, 7, 3}};
static int plaq_path_y[6][3] = {{2, 6, 5}, {5, 6, 2}, {3, 6, 4}, {4, 6, 3}, {0, 6, 7}, {7, 6, 0}};
static int plaq_path_z[6][3] = {{3, 5, 4}, {4, 5, 3}, {0, 5, 7}, {7, 5, 0}, {1, 5, 6}, {6, 5, 1}};
static int plaq_path_t[6][3] = {{0, 4, 7}, {7, 4, 0}, {1, 4, 6}, {6, 4, 1}, {2, 4, 5}, {5, 4, 2}};

int main(int argc, char **argv)
{
  auto app = make_app();
  add_hmc_gauge_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  setQudaPrecisions();
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);
  initRand();

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);

  printfQuda("Pure-gauge HMC: beta = %g, %d+%d trajectories, tau = %g in %d steps (eps = %g)\n", hmcg_beta, hmcg_therm,
             hmcg_trajectories, hmcg_traj_length, hmcg_traj_steps, hmcg_traj_length / hmcg_traj_steps);

  // host gauge field (QDP order) used for start-up and accept/reject backup
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

  if (latfile.size() > 0) {
    read_gauge_field(latfile.c_str(), gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);
    printfQuda("Starting from %s\n", latfile.c_str());
  } else {
    constructHostGaugeField(gauge, gauge_param, argc, argv); // unit or random per --unit-gauge
    printfQuda("Starting from a %s gauge field\n", unit_gauge ? "unit" : "random");
  }
  loadGaugeQuda((void *)gauge, &gauge_param);

  // host momentum field (MILC order, 10 reals per site per direction),
  // zero-filled; only used to seed the resident momentum
  size_t mom_bytes = (size_t)V * 4 * 10 * host_gauge_data_type_size;
  void *mom = safe_malloc(mom_bytes);
  memset(mom, 0, mom_bytes);

  QudaGaugeParam mom_param = gauge_param;
  mom_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  mom_param.type = QUDA_ASQTAD_MOM_LINKS;
  mom_param.reconstruct = QUDA_RECONSTRUCT_10;
  mom_param.make_resident_mom = 1;
  mom_param.use_resident_mom = 0;
  mom_param.return_result_mom = 0;
  momResidentQuda(mom, &mom_param);
  mom_param.use_resident_mom = 1;

  // force paths: six plaquette staples per direction, unit coefficients
  constexpr int num_paths = 6;
  constexpr int path_length = 3;
  int lengths[num_paths] = {3, 3, 3, 3, 3, 3};
  double coeffs[num_paths] = {1, 1, 1, 1, 1, 1};
  int **path_buf[4];
  for (int dir = 0; dir < 4; dir++) {
    path_buf[dir] = (int **)safe_malloc(num_paths * sizeof(int *));
    for (int i = 0; i < num_paths; i++) {
      path_buf[dir][i] = (int *)safe_malloc(path_length * sizeof(int));
      const int(*src)[3] = dir == 0 ? plaq_path_x : dir == 1 ? plaq_path_y : dir == 2 ? plaq_path_z : plaq_path_t;
      memcpy(path_buf[dir][i], src[i], path_length * sizeof(int));
    }
  }

  QudaGaugeParam force_param = gauge_param;
  force_param.use_resident_gauge = 1;
  force_param.use_resident_mom = 1;
  force_param.make_resident_mom = 1;
  force_param.return_result_mom = 0;
  force_param.overwrite_mom = 0;

  QudaGaugeParam update_param = gauge_param;
  update_param.use_resident_gauge = 1;
  update_param.use_resident_mom = 1;
  update_param.make_resident_gauge = 1;
  update_param.make_resident_mom = 1; // keep the momentum resident across the trajectory
  update_param.return_result_gauge = 0;
  update_param.return_result_mom = 0;

  const double vol_global = (double)V * quda::comm_size();
  auto gauge_action = [&]() {
    double plaq[3];
    plaqQuda(plaq);
    return hmcg_beta * 6.0 * vol_global * (1.0 - plaq[0]);
  };

  QudaGaugeObservableParam obs_param = newQudaGaugeObservableParam();
  obs_param.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obs_param.compute_qcharge = QUDA_BOOLEAN_TRUE;

  const double eps = hmcg_traj_length / hmcg_traj_steps;
  const double fdt = hmcg_force_sign * hmcg_beta / 3.0;
  int n_accept = 0, n_measured = 0;
  double exp_mdh_sum = 0.0;

  // save the starting configuration as the accept/reject fallback
  saveGaugeQuda((void *)gauge, &gauge_param);

  for (int traj = 1; traj <= hmcg_therm + hmcg_trajectories; traj++) {
    bool measuring = traj > hmcg_therm;

    // momentum refresh and initial Hamiltonian
    gaussMomQuda(hmcg_seed + traj, 1.0);
    double H_old = gauge_action() + momActionQuda(mom, &mom_param);

    // leapfrog: half-step momentum, alternating full steps, half-step momentum
    computeGaugeForceQuda(mom, gauge, path_buf, lengths, coeffs, num_paths, path_length, 0.5 * eps * fdt, &force_param);
    for (int k = 1; k <= hmcg_traj_steps; k++) {
      updateGaugeFieldQuda(gauge, mom, eps, 0, 1, &update_param);
      if (k < hmcg_traj_steps)
        computeGaugeForceQuda(mom, gauge, path_buf, lengths, coeffs, num_paths, path_length, eps * fdt, &force_param);
    }
    computeGaugeForceQuda(mom, gauge, path_buf, lengths, coeffs, num_paths, path_length, 0.5 * eps * fdt, &force_param);

    double H_new = gauge_action() + momActionQuda(mom, &mom_param);
    double dH = H_new - H_old;

    // Metropolis step, deterministic and identical on all ranks
    std::mt19937_64 rng(hmcg_seed ^ (0x9E3779B97F4A7C15ULL * traj));
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    bool accept = dH <= 0.0 || u < exp(-dH);

    if (accept) {
      saveGaugeQuda((void *)gauge, &gauge_param); // resident -> host backup
    } else {
      loadGaugeQuda((void *)gauge, &gauge_param); // restore host backup -> resident
    }

    if (measuring) {
      n_measured++;
      if (accept) n_accept++;
      exp_mdh_sum += exp(-dH);
      gaugeObservablesQuda(&obs_param);
      printfQuda("traj=%d plaquette = %.8e topological charge = %+.6e dH = %+.6e %s\n", traj - hmcg_therm,
                 obs_param.plaquette[0], obs_param.qcharge, dH, accept ? "ACCEPT" : "REJECT");

      if (hmcg_save_interval > 0 && (traj - hmcg_therm) % hmcg_save_interval == 0 && hmcg_save_prefix.size() > 0) {
#ifdef HAVE_QIO
        std::string fname
          = hmcg_save_prefix + "_cfg_" + std::to_string(hmcg_config_start + (traj - hmcg_therm)) + ".lime";
        write_gauge_field(fname.c_str(), gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);
        printfQuda("Saved configuration to %s\n", fname.c_str());
#else
        errorQuda("--hmc-save-prefix requires QUDA to be built with QUDA_QIO=ON");
#endif
      }
    } else {
      printfQuda("therm=%d dH = %+.6e %s\n", traj, dH, accept ? "ACCEPT" : "REJECT");
    }
  }

  printfQuda("HMC complete: acceptance %d/%d = %.3f, <exp(-dH)> = %.6f (Creutz equality: 1)\n", n_accept, n_measured,
             n_measured ? (double)n_accept / n_measured : 0.0, n_measured ? exp_mdh_sum / n_measured : 0.0);

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < num_paths; i++) host_free(path_buf[dir][i]);
    host_free(path_buf[dir]);
    host_free(gauge[dir]);
  }
  host_free(mom);

  endQuda();
  finalizeComms();
  return 0;
}
