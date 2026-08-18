#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <quda.h>
#include <tune_quda.h>

#include "gauge_smear_reference.h"
#include "gauge_utils.h"
#include "host_utils.h"
#include "misc.h"
#include "command_line_params.h"
#include "test.h"

// google test
#include "gauge_smear_test_gtest.hpp"

namespace {

QudaGaugeSmearParam make_smear_param(QudaGaugeSmearType type, int dir_ignore, bool use_cli, unsigned int rk_order = 3,
                                     double smear_anisotropy = 1.0, unsigned int n_steps = 1)
{
  QudaGaugeSmearParam param = newQudaGaugeSmearParam();
  param.smear_type = type;
  param.n_steps = n_steps;
  param.meas_interval = n_steps + 1;
  param.rk_order = rk_order;
  param.dir_ignore = dir_ignore;
  param.smear_anisotropy = smear_anisotropy;
  param.restart = QUDA_BOOLEAN_FALSE;
  param.alpha = use_cli ? gauge_smear_alpha : 0.6;
  param.rho = use_cli ? gauge_smear_rho : 0.1;
  param.epsilon = use_cli ? gauge_smear_epsilon : 0.1;
  param.alpha1 = use_cli ? gauge_smear_alpha1 : 0.75;
  param.alpha2 = use_cli ? gauge_smear_alpha2 : 0.6;
  param.alpha3 = use_cli ? gauge_smear_alpha3 : 0.3;
  return param;
}

bool is_flow(QudaGaugeSmearType type)
{
  return type == QUDA_GAUGE_SMEAR_WILSON_FLOW || type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW;
}

GaugeInputMode default_gauge_input_mode()
{
  return GaugeInputMode::GAUSSIAN_SU3;
}

QudaGaugeObservableParam make_disabled_observables()
{
  QudaGaugeObservableParam obs_param = newQudaGaugeObservableParam();
  obs_param.compute_plaquette = QUDA_BOOLEAN_FALSE;
  obs_param.compute_rectangle = QUDA_BOOLEAN_FALSE;
  obs_param.compute_polyakov_loop = QUDA_BOOLEAN_FALSE;
  obs_param.compute_qcharge = QUDA_BOOLEAN_FALSE;
  obs_param.compute_qcharge_density = QUDA_BOOLEAN_FALSE;
  obs_param.su_project = QUDA_BOOLEAN_FALSE;
  return obs_param;
}

void run_smear(QudaGaugeSmearParam &smear_param)
{
  auto obs_param = make_disabled_observables();

  pushVerbosity(QUDA_SILENT);
  if (is_flow(smear_param.smear_type))
    performWFlowQuda(&smear_param, &obs_param);
  else
    performGaugeSmearQuda(&smear_param, &obs_param);
  popVerbosity();
}

QudaGaugeParam make_gauge_param(QudaPrecision precision, QudaReconstructType reconstruct)
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.cuda_prec = precision;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  if (reconstruct != QUDA_RECONSTRUCT_INVALID) gauge_param.reconstruct = reconstruct;
  setDims(gauge_param.X);
  return gauge_param;
}

quda::GaugeFieldParam make_field_param(const QudaGaugeParam &gauge_param)
{
  quda::GaugeFieldParam field_param(gauge_param);
  field_param.location = QUDA_CPU_FIELD_LOCATION;
  field_param.order = QUDA_QDP_GAUGE_ORDER;
  field_param.create = QUDA_NULL_FIELD_CREATE;
  return field_param;
}

struct GaugeSmearFields {
  QudaGaugeParam gauge_param;
  quda::GaugeField input;

  GaugeSmearFields(QudaPrecision precision, QudaReconstructType reconstruct) :
    gauge_param(make_gauge_param(precision, reconstruct)),
    input(make_field_param(gauge_param))
  {
    constructHostGaugeInputField(input, gauge_param, 0, nullptr, default_gauge_input_mode());
    auto input_ptrs = input.data_array<void *>();
    loadGaugeQuda(input_ptrs.data, &gauge_param);
  }
};

void save_smear_result(quda::GaugeField &result, const QudaGaugeParam &gauge_param)
{
  auto result_ptrs = result.data_array<void *>();
  auto save_param = gauge_param;
  save_param.type = QUDA_SMEARED_LINKS;
  save_param.reconstruct = QUDA_RECONSTRUCT_NO;
  saveGaugeQuda(result_ptrs.data, &save_param);
}

int verify_one_step(QudaPrecision precision, QudaGaugeSmearParam smear_param,
                    QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID)
{
  GaugeSmearFields fields(precision, reconstruct);
  run_smear(smear_param);
  quda::GaugeField reference(make_field_param(fields.gauge_param));
  quda::GaugeField result(make_field_param(fields.gauge_param));
  save_smear_result(result, fields.gauge_param);

  gauge_smear_reference(reference, fields.input, smear_param);

  const auto tolerance = getTolerance(precision);
  int check = 1;
  auto max_deviation = 0.0;
  for (int dir = 0; dir < 4; dir++) {
    max_deviation = std::max(max_deviation, compare_floats_v2(result.data(dir), reference.data(dir), V * gauge_site_size,
                                                              std::numeric_limits<double>::infinity(), fields.gauge_param.cpu_prec));
    check &= compare_floats(result.data(dir), reference.data(dir), V * gauge_site_size, tolerance, fields.gauge_param.cpu_prec);
  }
  logQuda(QUDA_SUMMARIZE,
          "%s one-step %s reconstruct=%s rk_order=%u dir_ignore=%d smear_anisotropy=%.1f: max deviation %.3e, "
          "tolerance %.3e\n",
          get_gauge_smear_str(smear_param.smear_type), get_prec_str(precision),
          get_recon_str(fields.gauge_param.reconstruct),
          smear_param.rk_order, smear_param.dir_ignore, smear_param.smear_anisotropy, max_deviation, tolerance);
  auto result_ptrs = result.data_array<void *>();
  auto reference_ptrs = reference.data_array<void *>();
  strong_check_link(result_ptrs.data, "QUDA result:", reference_ptrs.data, "CPU reference:", V, fields.gauge_param.cpu_prec);
  return check;
}

struct SmearMetrics {
  double seconds;
  unsigned long long flops;
  unsigned long long bytes;
};

SmearMetrics measure_smear(QudaGaugeSmearParam smear_param, unsigned int n_steps)
{
  smear_param.n_steps = n_steps;
  smear_param.meas_interval = n_steps + 1; // suppress in-loop measurements so only the fixed initial observable runs

  const auto flops0 = quda::Tunable::flops_global();
  const auto bytes0 = quda::Tunable::bytes_global();

  quda::device_timer_t timer;
  quda::comm_barrier();
  timer.start();
  run_smear(smear_param);
  timer.stop();

  return {timer.last(), quda::Tunable::flops_global() - flops0, quda::Tunable::bytes_global() - bytes0};
}

// Isolate the marginal per-step smearing cost by finite-differencing two runs. The public entry point bundles a
// one-time extended-field build (the CopyGauge/GhostExtractor kernels that copy gaugePrecise into the halo'd field)
// with the step loop, and that build is constant per call regardless of n_steps. Differencing two calls therefore
// cancels it exactly, leaving only the work that scales with n_steps: the smear stencil plus its per-step extended
// boundary refresh (and, when partitioned, the halo pack/exchange). The disabled observables contribute nothing. This
// differential replaces the usual "time only the kernel in a loop" harness, which isn't possible here because the
// extension cannot be hoisted out of the measured call through the public interface.
SmearMetrics benchmark(QudaPrecision precision, QudaGaugeSmearParam smear_param,
                       QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID)
{
  GaugeSmearFields fields(precision, reconstruct);

  const auto steps = smear_param.n_steps;

  auto warmup = smear_param; // tune the extension and smear kernels before any measured run
  warmup.n_steps = 1;
  warmup.meas_interval = 2;
  run_smear(warmup);

  const auto lo = measure_smear(smear_param, 1);
  const auto hi = measure_smear(smear_param, steps + 1);
  return {hi.seconds - lo.seconds, hi.flops - lo.flops, hi.bytes - lo.bytes};
}

void report_benchmark(QudaGaugeSmearType type, int n_steps, const SmearMetrics &metrics)
{
  const auto steps = static_cast<double>(n_steps);
  const auto gflops = 1e-9 * metrics.flops / metrics.seconds;
  const auto gbytes = 1e-9 * metrics.bytes / metrics.seconds;
  const auto intensity = metrics.bytes == 0 ? 0.0 : static_cast<double>(metrics.flops) / metrics.bytes;

  printfQuda("%s kernel benchmark: %.3f us/step, %llu FLOPs/step, %llu bytes/step\n", get_gauge_smear_str(type),
             1e6 * metrics.seconds / steps, metrics.flops / n_steps, metrics.bytes / n_steps);
  printfQuda("Kernel performance: %.3f GFLOP/s, %.3f GB/s, %.3f FLOP/byte\n", gflops, gbytes, intensity);
}

} // namespace

int smear_verify(QudaPrecision precision, QudaReconstructType reconstruct, QudaGaugeSmearType type, int dir_ignore,
                 double smear_anisotropy)
{
  return verify_one_step(precision, make_smear_param(type, dir_ignore, false, 3, smear_anisotropy), reconstruct);
}

int flow_verify(QudaPrecision precision, QudaReconstructType reconstruct, QudaGaugeSmearType type, int dir_ignore,
                unsigned int rk_order, double smear_anisotropy)
{
  return verify_one_step(precision, make_smear_param(type, dir_ignore, false, rk_order, smear_anisotropy), reconstruct);
}

struct gauge_smear_test : quda_test {
  void display_info() const override
  {
    quda_test::display_info();
    printfQuda("\n%s gauge smearing\n", get_gauge_smear_str(gauge_smear_type));
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
    default: errorQuda("Undefined gauge smear type %d", gauge_smear_type);
    }
    printfQuda(" - smearing ignore direction %d\n", gauge_smear_dir_ignore);
    {
      const auto input_mode = resolveGaugeInputMode(default_gauge_input_mode());
      printfQuda(" - gauge input %s\n", getGaugeInputStr(input_mode));
      if (input_mode == GaugeInputMode::GAUSSIAN_SU3) {
        printfQuda(" - gauge input width %f\n", gauge_input_width);
      }
    }
    if (!enable_testing) {
      printfQuda(" - benchmark steps (--niter) %d\n", niter);
      printfQuda(" - one-step verification (--verify) %s\n", verify_results ? "enabled" : "disabled");
    }
  }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    add_su3_option_group(app);
  }

  gauge_smear_test(int argc, char **argv) : quda_test("Gauge Smear Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  gauge_smear_test test(argc, argv);
  test.init();
  if (enable_testing) {
    return test.execute();
  } else {
    if (niter < 1) errorQuda("--niter must be positive");

    const auto verify_param = make_smear_param(gauge_smear_type, gauge_smear_dir_ignore, true);
    if (verify_results && !verify_one_step(prec, verify_param)) {
      freeGaugeQuda();
      return 1;
    }

    const auto benchmark_param
      = make_smear_param(gauge_smear_type, gauge_smear_dir_ignore, true, 3, 1.0, static_cast<unsigned int>(niter));
    const auto metrics = benchmark(prec, benchmark_param);
    report_benchmark(gauge_smear_type, niter, metrics);
    freeGaugeQuda();
    return 0;
  }
}
