#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include <quda.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <gauge_tools.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "gauge_utils.h"

namespace quda
{
  extern void setTransferGPU(bool);
}

// Local helper functions
//------------------------------------------------------------------------------------
void setReunitarizationConsts()
{
  using namespace quda;
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
  const int reunit_allow_svd = 1;
  const int reunit_svd_only = 0;
  const double svd_rel_error = 1e-6;
  const double svd_abs_error = 1e-6;
  setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error, svd_abs_error);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

quda::RNG randstates;
QudaGaugeParam gauge_param;
quda::GaugeField cpuGauge = {};
quda::GaugeField gauge = {};
quda::GaugeField gaugeEx = {};

// Magic header marking a QUDA RNG checkpoint file (version 1)
constexpr uint64_t rng_checkpoint_magic = 0x514344524e470001ULL;

std::string rng_checkpoint_filename(const std::string &base) { return base + ".rank" + std::to_string(quda::comm_rank()); }

void save_rng_state()
{
  std::vector<char> buffer(randstates.StateBytes());
  randstates.saveState(buffer.data());

  std::string fname = rng_checkpoint_filename(heatbath_rng_save);
  std::ofstream file(fname, std::ios::binary | std::ios::trunc);
  if (!file) errorQuda("Failed to open RNG checkpoint file %s for writing", fname.c_str());
  uint64_t header[2] = {rng_checkpoint_magic, static_cast<uint64_t>(buffer.size())};
  file.write(reinterpret_cast<char *>(header), sizeof(header));
  file.write(buffer.data(), buffer.size());
  if (!file.good()) errorQuda("Failed writing RNG checkpoint file %s", fname.c_str());
  printfQuda("Checkpointed RNG state to %s(.rank<n>)\n", heatbath_rng_save.c_str());
}

void load_rng_state()
{
  std::string fname = rng_checkpoint_filename(heatbath_rng_load);
  std::ifstream file(fname, std::ios::binary);
  if (!file) errorQuda("Failed to open RNG checkpoint file %s for reading", fname.c_str());
  uint64_t header[2];
  file.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!file.good() || header[0] != rng_checkpoint_magic)
    errorQuda("File %s is not a QUDA RNG checkpoint", fname.c_str());
  if (header[1] != randstates.StateBytes())
    errorQuda("RNG checkpoint %s has state size %lu but %lu expected; geometry or process grid mismatch",
              fname.c_str(), (unsigned long)header[1], (unsigned long)randstates.StateBytes());
  std::vector<char> buffer(randstates.StateBytes());
  file.read(buffer.data(), buffer.size());
  if (!file.good()) errorQuda("Failed reading RNG checkpoint file %s", fname.c_str());
  randstates.loadState(buffer.data());
  printfQuda("Restored RNG state from %s(.rank<n>)\n", heatbath_rng_load.c_str());
}

// Save the current configuration, measure flowed observables on it, and
// checkpoint the RNG state, as requested via the command line.
void save_and_measure_config(int config_number)
{
  // copy back to the host
  copyExtendedGauge(gauge, gaugeEx, QUDA_CUDA_FIELD_LOCATION);
  cpuGauge = gauge;

  if (heatbath_save_config_prefix.size() > 0) {
#ifdef HAVE_QIO
    std::string fname = heatbath_save_config_prefix + "_cfg_" + std::to_string(config_number) + ".lime";
    write_gauge_field(fname.c_str(), reinterpret_cast<void **>(cpuGauge.raw_pointer()), gauge_param.cpu_prec,
                      gauge_param.X, 0, (char **)0);
    printfQuda("Saved configuration %d to %s\n", config_number, fname.c_str());
#else
    errorQuda("--heatbath-save-config-prefix requires QUDA to be built with QUDA_QIO=ON");
#endif
  }

  if (heatbath_flow_steps > 0) {
    // Wilson-flowed observables through the public interface, on a copy
    // of the configuration resident in the library
    loadGaugeQuda(cpuGauge.raw_pointer(), &gauge_param);

    QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
    smear_param.n_steps = heatbath_flow_steps;
    smear_param.epsilon = heatbath_flow_epsilon;
    smear_param.meas_interval = heatbath_flow_steps; // measure at start and end only
    smear_param.smear_type = QUDA_GAUGE_SMEAR_WILSON_FLOW;

    std::vector<QudaGaugeObservableParam> obs_param(2, newQudaGaugeObservableParam());
    for (auto &o : obs_param) {
      o.compute_plaquette = QUDA_BOOLEAN_TRUE;
      o.compute_qcharge = QUDA_BOOLEAN_TRUE;
    }
    performWFlowQuda(&smear_param, obs_param.data());

    double flow_time = heatbath_flow_steps * heatbath_flow_epsilon;
    printfQuda("config %d flow time %g: plaquette = %.16e Q = %+.6e E = %.6e\n", config_number, flow_time,
               obs_param.back().plaquette[0], obs_param.back().qcharge, obs_param.back().energy[0]);

    freeGaugeSmearedQuda();
  }

  if (heatbath_rng_save.size() > 0) save_rng_state();
}

void init_gauge(int argc, char **argv)
{
  // *** QUDA parameters begin here.
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  // *** Everything between here and the timer is  application specific.
  setDims(gauge_param.X);

  // Allocate a host CPU field in QDP order
  quda::GaugeFieldParam cpuParam(gauge_param);
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  cpuGauge = quda::GaugeField(cpuParam);

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  // Create a device field
  quda::GaugeFieldParam gParam(gauge_param);
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = gauge_param.type;
  gParam.reconstruct = gauge_param.reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  gauge = quda::GaugeField(gParam);

  // Create the extended device field
  int pad = 0;
  quda::lat_dim_t y;
  quda::lat_dim_t R = {0, 0, 0, 0};
  for (int dir = 0; dir < 4; ++dir)
    if (quda::comm_dim_partitioned(dir)) R[dir] = 2;
  for (int dir = 0; dir < 4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
  GaugeFieldParam gParamEx(y, prec, link_recon, pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.location = QUDA_CUDA_FIELD_LOCATION;
  gParamEx.order = gParam.order;
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gParam.t_boundary;
  gParamEx.nFace = 1;
  for (int dir = 0; dir < 4; ++dir) gParamEx.r[dir] = R[dir];
  gaugeEx = quda::GaugeField(gParamEx);

  // initialize the device-side RNG, restoring a checkpointed state if requested
  randstates = quda::RNG(gaugeEx, 1234);
  if (heatbath_rng_load.size() > 0) load_rng_state();

  if (latfile.size() > 0 || heatbath_initialize_on_host) {
    // Load or fill the gauge fields from the host
    constructHostGaugeField(reinterpret_cast<void **>(cpuGauge.raw_pointer()), gauge_param, argc, argv);

    // copy in the host field
    gauge = cpuGauge;

    // copy the local field into the extended gauge field
    copyExtendedGauge(gaugeEx, gauge, QUDA_CUDA_FIELD_LOCATION);

    // manually handle the exchange
    gaugeEx.exchangeExtendedGhost(gaugeEx.R());
  } else {
    // initialize on the device using QUDA routines
    if (heatbath_coldstart)
      InitGaugeField(gaugeEx);
    else
      InitGaugeField(gaugeEx, randstates);
  }
}

void teardown_gauge()
{
  // Save if output string is specified
  if (gauge_outfile.size() > 0) {
    printfQuda("Saving the gauge field to file %s\n", gauge_outfile.c_str());

    // copy back to the host
    copyExtendedGauge(gauge, gaugeEx, QUDA_CUDA_FIELD_LOCATION);
    cpuGauge = gauge;

    write_gauge_field(gauge_outfile.c_str(), reinterpret_cast<void **>(cpuGauge.raw_pointer()), gauge_param.cpu_prec,
                      gauge_param.X, 0, (char **)0);
  } else {
    printfQuda("No output file specified.\n");
  }

  // clean up the RNG
  randstates = {};

  cpuGauge = {};
  gauge = {};
  gaugeEx = {};
}

void heatbath_test()
{
  using quda::GaugeField;

  quda::quda_ptr num_failures(QUDA_MEMORY_HOST_PINNED, sizeof(int), false);
  int &num_failures_h = *static_cast<int *>(num_failures.data_host());
  int &num_failures_d = *static_cast<int *>(num_failures.data_device());
  num_failures_h = 0;

  // start the timer
  quda::host_timer_t host_timer;

  host_timer.start();
  {
    int nsteps = heatbath_num_steps;
    int nwarm = heatbath_warmup_steps;
    int nhbsteps = heatbath_num_heatbath_per_step;
    int novrsteps = heatbath_num_overrelax_per_step;
    bool coldstart = heatbath_coldstart;
    double beta_value = heatbath_beta_value;

    // Descriptor of how the gauge field was initialized
    std::string start_description;
    if (latfile.size() > 0) {
      start_description = "loaded gauge field";
    } else {
      start_description = coldstart ? "cold start" : "hot start";
      start_description += " initialized on the ";
      start_description += heatbath_initialize_on_host ? "host" : "device";
    }

    printfQuda("Starting heatbath for beta = %f from a %s\n", beta_value, start_description.c_str());
    printfQuda("  %d Heatbath hits and %d overrelaxation hits per step\n", nhbsteps, novrsteps);
    printfQuda("  %d Warmup steps\n", nwarm);
    printfQuda("  %d Measurement steps\n", nsteps);

    QudaGaugeObservableParam param = newQudaGaugeObservableParam();
    param.compute_plaquette = QUDA_BOOLEAN_TRUE;
    param.compute_qcharge = QUDA_BOOLEAN_TRUE;

    // Measure the plaquette and the topological charge
    gaugeObservables(gaugeEx, param);

    printfQuda("Initial gauge field plaquette = %e topological charge = %e\n", param.plaquette[0], param.qcharge);

    // Reunitarization setup
    setReunitarizationConsts();

    // Do a warmup if requested
    if (nwarm > 0) {
      for (int step = 1; step <= nwarm; ++step) {
        Monte(gaugeEx, randstates, beta_value, nhbsteps, novrsteps);
        quda::unitarizeLinks(gaugeEx, &num_failures_d);
        if (num_failures_h > 0) errorQuda("Error in the unitarization, %d failures", num_failures_h);
      }
    }

    gaugeObservables(gaugeEx, param);
    printfQuda("step=0 plaquette = %e topological charge = %e\n", param.plaquette[0], param.qcharge);

    for (int step = 1; step <= nsteps; ++step) {
      Monte(gaugeEx, randstates, beta_value, nhbsteps, novrsteps);

      // Reunitarize gauge links...
      quda::unitarizeLinks(gaugeEx, &num_failures_d);
      if (num_failures_h > 0) errorQuda("Error in the unitarization, %d failures", num_failures_h);

      gaugeObservables(gaugeEx, param);
      printfQuda("step=%d plaquette = %e topological charge = %e\n", step, param.plaquette[0], param.qcharge);

      if (heatbath_save_config_interval > 0 && step % heatbath_save_config_interval == 0)
        save_and_measure_config(heatbath_config_start + step);
    }

    // leave a final RNG checkpoint so the stream can be continued
    if (heatbath_rng_save.size() > 0) save_rng_state();

    // Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    quda::PGaugeExchangeFree();
  }

  // stop the timer
  host_timer.stop();
  double time0 = host_timer.last();

  printfQuda("\nDone, total time = %g secs\n", time0);
}

int main(int argc, char **argv)
{
  // command line options
  auto app = make_app();
  add_heatbath_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // let heatbath_coldstart take precedence over unit_gauge
  if (heatbath_coldstart != unit_gauge) {
    warningQuda("--heatbath-coldstart and --unit-gauge do not agree, --heatbath-coldstart takes precedence.");
    unit_gauge = heatbath_coldstart;
  }

  display_test_info();

  // initialize the QUDA library
  initQuda(device_ordinal);

  // initialize the gauge field
  init_gauge(argc, argv);

  // run the test
  heatbath_test();

  // clean up the gauge field
  teardown_gauge();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
