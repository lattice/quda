#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <array>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <comm_quda.h>
#include <instantiate.h>

#include "timer.h"
#include "util_quda.h"
#include "host_utils.h"
#include "gauge_utils.h"
#include "command_line_params.h"
#include "dslash_reference.h"
#include "gauge_observable_reference.h"
#include "misc.h"
#include "test.h"

#include "su3_test_gtest.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

const quda::GaugeField *test_input = nullptr;

GaugeInputMode default_gauge_input_mode() { return GaugeInputMode::GAUSSIAN_SU3; }

QudaGaugeParam make_gauge_param(QudaPrecision precision, QudaReconstructType reconstruct)
{
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.cuda_prec = precision;
  gauge_param.reconstruct = reconstruct;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);
  return gauge_param;
}

const quda::GaugeField &shared_test_input()
{
  if (!test_input) errorQuda("Shared SU(3) test input is not initialized");
  return *test_input;
}

struct Su3Fields {
  QudaGaugeParam gauge_param;
  const quda::GaugeField &input;
  void *new_gauge[4] {};

  Su3Fields(const quda::GaugeField &input, QudaPrecision precision, QudaReconstructType reconstruct) :
    gauge_param(make_gauge_param(precision, reconstruct)), input(input)
  {
    for (int dir = 0; dir < 4; dir++) new_gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

    const auto input_ptrs = input.data_array<void *>();
    loadGaugeQuda(const_cast<void **>(input_ptrs.data), &gauge_param);
    saveGaugeQuda(new_gauge, &gauge_param);
  }

  ~Su3Fields()
  {
    for (int dir = 0; dir < 4; dir++) host_free(new_gauge[dir]);
    freeGaugeQuda();
  }
};

std::array<double, 3> run_plaquette(Su3Fields &fields, bool verify)
{
  long long flops_plaquette = 6ll * 597 * V;
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  quda::host_timer_t host_timer;

  param.compute_plaquette = QUDA_BOOLEAN_TRUE;
  gaugeObservablesQuda(&param);

  host_timer.start();
  for (int i = 0; i < niter; i++) gaugeObservablesQuda(&param);
  host_timer.stop();
  double secs_plaquette = host_timer.last() / niter;
  double perf_plaquette = flops_plaquette / (secs_plaquette * 1024 * 1024 * 1024);
  printfQuda(
    "Computed plaquette gauge precise is %.16e (spatial = %.16e, temporal = %.16e), done in %g seconds, %g GFLOPS\n",
    param.plaquette[0], param.plaquette[1], param.plaquette[2], secs_plaquette, perf_plaquette);
  param.compute_plaquette = QUDA_BOOLEAN_FALSE;

  std::array<double, 3> deviation {};
  if (verify) {
    const auto reference = plaquette_reference(fields.input);
    for (int i = 0; i < 3; i++) {
      const double scale = std::max(std::abs(param.plaquette[i]), std::abs(reference[i]));
      deviation[i] = scale == 0.0 ? 0.0 : std::abs(param.plaquette[i] - reference[i]) / scale;
    }
    printfQuda(
      "Host plaquette reference is %.16e (spatial = %.16e, temporal = %.16e), relative deviations %.3e %.3e %.3e\n",
      reference[0], reference[1], reference[2], deviation[0], deviation[1], deviation[2]);
  }
  return deviation;
}

std::array<double, 6> run_plaquette_rectangle(Su3Fields &fields, bool verify)
{
  constexpr long long Nc = 3;
  const long long flops = 6ll * V * (10 * Nc * Nc * (8 * Nc - 2) + 2 * Nc);
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  quda::host_timer_t host_timer;

  param.compute_rectangle = QUDA_BOOLEAN_TRUE;
  gaugeObservablesQuda(&param);

  host_timer.start();
  for (int i = 0; i < niter; i++) gaugeObservablesQuda(&param);
  host_timer.stop();
  const double seconds = host_timer.last() / niter;
  const double performance = flops / (seconds * 1024 * 1024 * 1024);
  printfQuda("Computed plaquette + rectangle is\n"
             "  plaquette %.16e (spatial %.16e, temporal %.16e)\n"
             "  rectangle %.16e (spatial %.16e, temporal %.16e)\n"
             "Done in %g seconds, %g GFLOPS\n",
             param.plaquette[0], param.plaquette[1], param.plaquette[2], param.rectangle[0], param.rectangle[1],
             param.rectangle[2], seconds, performance);

  std::array<double, 6> deviation {};
  if (verify) {
    const auto reference = plaquette_rectangle_reference(fields.input);
    for (int i = 0; i < 3; i++) {
      const double plaquette_scale = std::max(std::abs(param.plaquette[i]), std::abs(reference.plaquette[i]));
      deviation[i]
        = plaquette_scale == 0.0 ? 0.0 : std::abs(param.plaquette[i] - reference.plaquette[i]) / plaquette_scale;
      const double rectangle_scale = std::max(std::abs(param.rectangle[i]), std::abs(reference.rectangle[i]));
      deviation[i + 3]
        = rectangle_scale == 0.0 ? 0.0 : std::abs(param.rectangle[i] - reference.rectangle[i]) / rectangle_scale;
    }
    printfQuda("Host plaquette + rectangle relative deviations are\n"
               "  plaquette %.3e %.3e %.3e\n"
               "  rectangle %.3e %.3e %.3e\n",
               deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5]);
  }
  return deviation;
}

void run_polyakov_loop(const Su3Fields &fields)
{
  long long flops_ploop = 198ll * V + 6 * V / fields.gauge_param.X[3];
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  quda::host_timer_t host_timer;

  param.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
  gaugeObservablesQuda(&param);

  host_timer.start();
  for (int i = 0; i < niter; i++) gaugeObservablesQuda(&param);
  host_timer.stop();
  double secs_ploop = host_timer.last() / niter;
  double perf_ploop = flops_ploop / (secs_ploop * 1024 * 1024 * 1024);
  printfQuda("Computed Polyakov loop gauge precise is %.16e +/- I %.16e , done in %g seconds, %g GFLOPS\n",
             param.ploop[0], param.ploop[1], secs_ploop, perf_ploop);
  param.compute_polyakov_loop = QUDA_BOOLEAN_FALSE;
}

void run_topological_charge_and_density()
{
  double q_charge_check = 0.0;
  size_t data_size = prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
  size_t array_size = V * data_size;
  void *qDensity = host_pinned_malloc(array_size);
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  quda::host_timer_t host_timer;

  host_timer.start();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param.qcharge_density = qDensity;
  gaugeObservablesQuda(&param);
  host_timer.stop();
  printfQuda("Computed Etot, Es, Et, Q is\n%.16e %.16e, %.16e %.16e\nDone in %g secs\n", param.energy[0],
             param.energy[1], param.energy[2], param.qcharge, host_timer.last());

  if (prec == QUDA_DOUBLE_PRECISION) {
    for (int i = 0; i < V; i++) q_charge_check += ((double *)qDensity)[i];
  } else {
    for (int i = 0; i < V; i++) q_charge_check += ((float *)qDensity)[i];
  }

  host_free(qDensity);
  quda::comm_allreduce_sum(q_charge_check);
  printfQuda("GPU value %e and host density sum %e. Q charge deviation: %e\n", param.qcharge, q_charge_check,
             param.qcharge - q_charge_check);
}

void run_gauge_smearing_or_flow(Su3Fields &fields)
{
  QudaGaugeObservableParam *obs_param = new QudaGaugeObservableParam[gauge_smear_steps / measurement_interval + 1];
  for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
    obs_param[i] = newQudaGaugeObservableParam();
    obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
    obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  }

  QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
  smear_param.smear_type = gauge_smear_type;
  smear_param.n_steps = gauge_smear_steps;
  smear_param.meas_interval = measurement_interval;
  smear_param.alpha = gauge_smear_alpha;
  smear_param.rho = gauge_smear_rho;
  smear_param.epsilon = gauge_smear_epsilon;
  smear_param.alpha1 = gauge_smear_alpha1;
  smear_param.alpha2 = gauge_smear_alpha2;
  smear_param.alpha3 = gauge_smear_alpha3;
  smear_param.dir_ignore = gauge_smear_dir_ignore;

  quda::host_timer_t host_timer;
  host_timer.start();
  switch (smear_param.smear_type) {
  case QUDA_GAUGE_SMEAR_APE:
  case QUDA_GAUGE_SMEAR_STOUT:
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
  case QUDA_GAUGE_SMEAR_HYP: performGaugeSmearQuda(&smear_param, obs_param); break;
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW:
    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    }
    performWFlowQuda(&smear_param, obs_param);
    break;
  default: errorQuda("Undefined gauge smear type %d given", smear_param.smear_type);
  }
  host_timer.stop();
  printfQuda("Total time for gauge smearing = %g secs\n", host_timer.last());

  if (verify_results) {
    const auto input_ptrs = fields.input.data_array<void *>();
    check_gauge(const_cast<void **>(input_ptrs.data), fields.new_gauge, 1e-3, fields.gauge_param.cpu_prec);
  }
  delete[] obs_param;
}

void run_all()
{
  Su3Fields fields(shared_test_input(), prec, link_recon);
  run_plaquette(fields, false);
  run_plaquette_rectangle(fields, false);
  run_polyakov_loop(fields);
  run_topological_charge_and_density();
  run_gauge_smearing_or_flow(fields);
}
std::array<double, 3> plaquette_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  return run_plaquette(fields, true);
}

std::array<double, 6> plaquette_rectangle_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  return run_plaquette_rectangle(fields, true);
}

void polyakov_loop_test()
{
  Su3Fields fields(shared_test_input(), prec, link_recon);
  run_polyakov_loop(fields);
}

void topological_charge_and_density_test()
{
  Su3Fields fields(shared_test_input(), prec, link_recon);
  run_topological_charge_and_density();
}

void gauge_smearing_or_flow_test()
{
  Su3Fields fields(shared_test_input(), prec, link_recon);
  run_gauge_smearing_or_flow(fields);
}

struct su3_test : quda_test {
  void display_info() const override
  {
    auto sloppy_prec = prec_sloppy == QUDA_INVALID_PRECISION ? prec : prec_sloppy;
    auto sloppy_recon = link_recon_sloppy == QUDA_RECONSTRUCT_INVALID ? link_recon : link_recon_sloppy;

    printfQuda("running the following test:\n");
    printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
    printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
               get_prec_str(sloppy_prec), get_recon_str(link_recon), get_recon_str(sloppy_recon), xdim, ydim, zdim, tdim);

    printfQuda("\n%s smearing\n", get_gauge_smear_str(gauge_smear_type));
    switch (gauge_smear_type) {
    case QUDA_GAUGE_SMEAR_APE: printfQuda(" - alpha %f\n", gauge_smear_alpha); break;
    case QUDA_GAUGE_SMEAR_STOUT: printfQuda(" - rho %f\n", gauge_smear_rho); break;
    case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
      printfQuda(" - rho %f\n", gauge_smear_rho);
      printfQuda(" - epsilon %f\n", gauge_smear_epsilon);
      break;
    case QUDA_GAUGE_SMEAR_HYP:
      printfQuda(" - alpha1 %f\n", gauge_smear_alpha1);
      printfQuda(" - alpha2 %f\n", gauge_smear_alpha2);
      printfQuda(" - alpha3 %f\n", gauge_smear_alpha3);
      break;
    case QUDA_GAUGE_SMEAR_WILSON_FLOW:
    case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: printfQuda(" - epsilon %f\n", gauge_smear_epsilon); break;
    default: errorQuda("Undefined test type %d given", test_type);
    }
    printfQuda(" - smearing steps %d\n", gauge_smear_steps);
    printfQuda(" - smearing ignore direction %d\n", gauge_smear_dir_ignore);
    printfQuda(" - Measurement interval %d\n", measurement_interval);
    const auto input_mode = resolveGaugeInputMode(default_gauge_input_mode());
    printfQuda(" - gauge input %s\n", getGaugeInputStr(input_mode));
    if (input_mode == GaugeInputMode::GAUSSIAN_SU3) printfQuda(" - gauge input width %f\n", gauge_input_width);

    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
  }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    add_su3_option_group(app);
  }

  su3_test(int argc, char **argv) : quda_test("SU(3) Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  su3_test test(argc, argv);
  test.init();
  const auto input_param = make_gauge_param(prec, QUDA_RECONSTRUCT_NO);
  HostGaugeInput input(input_param, test.argc, test.argv, default_gauge_input_mode());
  test_input = &input.field();
  const int result = enable_testing ? test.execute() : (run_all(), 0);
  test_input = nullptr;
  return result;
}
