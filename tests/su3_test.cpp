#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <vector>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <pgauge_monte.h>
#include <gauge_tools.h>
#include <malloc_quda.h>

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

quda::GaugeFieldParam make_tensor_param(const Su3Fields &fields, QudaFieldLocation location, QudaPrecision precision,
                                        QudaGaugeFieldOrder order)
{
  quda::GaugeFieldParam param(fields.input.X(), precision, QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
  param.location = location;
  param.order = order;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  param.create = QUDA_NULL_FIELD_CREATE;
  return param;
}

struct FieldStrengthFields {
  std::unique_ptr<quda::GaugeField> device_gauge;
  quda::GaugeField device_fmunu;
  quda::GaugeField device_qdp_fmunu;
  quda::GaugeField device_result;
  quda::GaugeField host_reference;

  FieldStrengthFields(const Su3Fields &fields, bool copy_result = false) :
    device_fmunu(
      make_tensor_param(fields, QUDA_CUDA_FIELD_LOCATION, fields.gauge_param.cuda_prec, QUDA_NATIVE_GAUGE_ORDER)),
    device_qdp_fmunu(
      make_tensor_param(fields, QUDA_CUDA_FIELD_LOCATION, fields.gauge_param.cpu_prec, QUDA_QDP_GAUGE_ORDER)),
    device_result(make_tensor_param(fields, QUDA_CPU_FIELD_LOCATION, fields.gauge_param.cpu_prec, QUDA_QDP_GAUGE_ORDER)),
    host_reference(make_tensor_param(fields, QUDA_CPU_FIELD_LOCATION, fields.gauge_param.cpu_prec, QUDA_QDP_GAUGE_ORDER))
  {
    quda::GaugeFieldParam device_param(fields.gauge_param);
    device_param.location = QUDA_CUDA_FIELD_LOCATION;
    device_param.order = QUDA_NATIVE_GAUGE_ORDER;
    device_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    device_param.create = QUDA_NULL_FIELD_CREATE;
    device_param.setPrecision(fields.gauge_param.cuda_prec, true);
    quda::GaugeField device(device_param);
    device.copy(fields.input);

    quda::lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
    static quda::TimeProfile profile("SU3FieldStrengthTest");
    device_gauge.reset(quda::createExtendedGauge(device, R, profile));

    quda::computeFmunu(device_fmunu, *device_gauge);
    if (copy_result) {
      device_qdp_fmunu.copy(device_fmunu);
      const size_t component_bytes = fields.input.Volume() * gauge_site_size * fields.gauge_param.cpu_prec;
      for (int component = 0; component < 6; component++)
        qudaMemcpy(device_result.data(component), device_qdp_fmunu.data(component), component_bytes,
                   qudaMemcpyDeviceToHost);
    }
    compute_fmunu_reference(host_reference, fields.input);
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

std::array<double, 2> run_polyakov_loop(const Su3Fields &fields, bool verify)
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

  std::array<double, 2> deviation {};
  if (verify) {
    const auto reference = polyakov_loop_reference(fields.input);
    for (int i = 0; i < 2; i++) {
      const double scale = std::max(std::abs(param.ploop[i]), std::abs(reference[i]));
      deviation[i] = scale == 0.0 ? 0.0 : std::abs(param.ploop[i] - reference[i]) / scale;
    }
    printfQuda("Host Polyakov loop is %.16e +/- I %.16e, relative deviations %.3e %.3e\n", reference[0], reference[1],
               deviation[0], deviation[1]);
  }
  return deviation;
}

std::array<double, 4> run_determinant_trace(const Su3Fields &fields, bool verify)
{
  quda::GaugeFieldParam field_param(fields.gauge_param);
  field_param.location = QUDA_CUDA_FIELD_LOCATION;
  field_param.order = QUDA_NATIVE_GAUGE_ORDER;
  field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  field_param.create = QUDA_NULL_FIELD_CREATE;
  field_param.setPrecision(fields.gauge_param.cuda_prec, true);
  quda::GaugeField device_gauge(field_param);
  device_gauge.copy(fields.input);

  auto determinant = quda::getLinkDeterminant(device_gauge);
  quda::host_timer_t host_timer;
  host_timer.start();
  for (int i = 0; i < niter; i++) determinant = quda::getLinkDeterminant(device_gauge);
  host_timer.stop();
  const double determinant_seconds = host_timer.last() / niter;

  auto link_trace = quda::getLinkTrace(device_gauge);
  host_timer.start();
  for (int i = 0; i < niter; i++) link_trace = quda::getLinkTrace(device_gauge);
  host_timer.stop();
  const double trace_seconds = host_timer.last() / niter;

  printfQuda("Computed mean link determinant %.16e +/- I %.16e in %g seconds\n", determinant.real(), determinant.imag(),
             determinant_seconds);
  printfQuda("Computed mean link trace %.16e +/- I %.16e in %g seconds\n", link_trace.real(), link_trace.imag(),
             trace_seconds);

  std::array<double, 4> comparison {};
  if (verify) {
    const auto reference = link_determinant_trace_reference(fields.input);
    const double tolerance = getTolerance(fields.gauge_param.cuda_prec);
    comparison[0] = std::abs(determinant - reference.determinant);
    comparison[1] = tolerance * reference.determinant_scale;
    comparison[2] = std::abs(link_trace - reference.trace);
    comparison[3] = tolerance * reference.trace_scale;
    printfQuda("Host determinant %.16e +/- I %.16e, difference %.3e, threshold %.3e\n", reference.determinant.real(),
               reference.determinant.imag(), comparison[0], comparison[1]);
    printfQuda("Host trace %.16e +/- I %.16e, difference %.3e, threshold %.3e\n", reference.trace.real(),
               reference.trace.imag(), comparison[2], comparison[3]);
  }
  return comparison;
}

double field_strength_tensor_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  FieldStrengthFields fmunu(fields, true);
  const auto comparison
    = strong_check_field(fmunu.device_result, "QUDA Fmunu", fmunu.host_reference, "host Fmunu reference");
  return comparison.max_deviation;
}

std::array<double, 16> energy_topological_charge_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  FieldStrengthFields fmunu(fields);
  const auto reference = field_strength_observable_reference(fmunu.host_reference);
  const double tolerance = getTolerance(precision);

  quda::array<quda::real_t, 3> energy {};
  const double qcharge = quda::computeQCharge(energy, fmunu.device_fmunu);

  QudaGaugeObservableParam observable = newQudaGaugeObservableParam();
  observable.compute_qcharge = QUDA_BOOLEAN_TRUE;
  gaugeObservablesQuda(&observable);

  std::array<double, 16> comparison {};
  int offset = 0;
  auto add_comparison = [&](double value, double expected, double scale) {
    comparison[offset++] = std::abs(value - expected);
    comparison[offset++] = tolerance * scale;
  };
  for (int i = 0; i < 3; i++) add_comparison(energy[i], reference.energy[i], std::abs(reference.energy[i]));
  add_comparison(qcharge, reference.qcharge, reference.qcharge_scale);
  for (int i = 0; i < 3; i++) add_comparison(observable.energy[i], reference.energy[i], std::abs(reference.energy[i]));
  add_comparison(observable.qcharge, reference.qcharge, reference.qcharge_scale);
  return comparison;
}

std::vector<double> copy_density_to_double(const void *density, size_t length, QudaPrecision precision,
                                           QudaFieldLocation location)
{
  std::vector<double> result(length);
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (location == QUDA_CUDA_FIELD_LOCATION)
      qudaMemcpy(result.data(), density, length * sizeof(double), qudaMemcpyDeviceToHost);
    else
      std::copy_n(static_cast<const double *>(density), length, result.data());
  } else {
    std::vector<float> temporary(length);
    if (location == QUDA_CUDA_FIELD_LOCATION)
      qudaMemcpy(temporary.data(), density, length * sizeof(float), qudaMemcpyDeviceToHost);
    else
      std::copy_n(static_cast<const float *>(density), length, temporary.data());
    std::copy(temporary.begin(), temporary.end(), result.begin());
  }
  return result;
}

std::array<double, 24> topological_charge_density_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  FieldStrengthFields fmunu(fields);
  const auto reference = field_strength_observable_reference(fmunu.host_reference);
  const double tolerance = getTolerance(precision);
  const size_t length = fields.input.Volume();
  const size_t bytes = length * precision;

  void *device_density = device_malloc(bytes);
  quda::array<quda::real_t, 3> energy {};
  const double qcharge = quda::computeQChargeDensity(energy, device_density, fmunu.device_fmunu);
  const auto device_density_host = copy_density_to_double(device_density, length, precision, QUDA_CUDA_FIELD_LOCATION);
  device_free(device_density);

  std::vector<unsigned char> public_density_storage(bytes);
  QudaGaugeObservableParam observable = newQudaGaugeObservableParam();
  observable.compute_qcharge = QUDA_BOOLEAN_TRUE;
  observable.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  observable.qcharge_density = public_density_storage.data();
  gaugeObservablesQuda(&observable);
  const auto public_density
    = copy_density_to_double(public_density_storage.data(), length, precision, QUDA_CPU_FIELD_LOCATION);

  const auto direct_field
    = strong_check_scalar(device_density_host.data(), "QUDA direct Q density", reference.qdensity.data(),
                          "host Q density", length, QUDA_DOUBLE_PRECISION);
  const auto public_field
    = strong_check_scalar(public_density.data(), "QUDA observable Q density", reference.qdensity.data(),
                          "host Q density", length, QUDA_DOUBLE_PRECISION);

  double device_density_sum = std::accumulate(device_density_host.begin(), device_density_host.end(), 0.0);
  double public_density_sum = std::accumulate(public_density.begin(), public_density.end(), 0.0);
  quda::comm_allreduce_sum(device_density_sum);
  quda::comm_allreduce_sum(public_density_sum);

  std::array<double, 24> comparison {};
  int offset = 0;
  auto add_comparison = [&](double difference, double scale) {
    comparison[offset++] = difference;
    comparison[offset++] = tolerance * scale;
  };
  add_comparison(direct_field.max_deviation, 1.0);
  add_comparison(public_field.max_deviation, 1.0);
  for (int i = 0; i < 3; i++) add_comparison(std::abs(energy[i] - reference.energy[i]), std::abs(reference.energy[i]));
  for (int i = 0; i < 3; i++)
    add_comparison(std::abs(observable.energy[i] - reference.energy[i]), std::abs(reference.energy[i]));
  add_comparison(std::abs(qcharge - reference.qcharge), reference.qcharge_scale);
  add_comparison(std::abs(observable.qcharge - reference.qcharge), reference.qcharge_scale);
  add_comparison(std::abs(device_density_sum - qcharge), reference.qcharge_scale);
  add_comparison(std::abs(public_density_sum - observable.qcharge), reference.qcharge_scale);
  return comparison;
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
  run_polyakov_loop(fields, false);
  run_determinant_trace(fields, false);
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

std::array<double, 2> polyakov_loop_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  return run_polyakov_loop(fields, true);
}

std::array<double, 4> determinant_trace_test(QudaPrecision precision, QudaReconstructType reconstruct)
{
  Su3Fields fields(shared_test_input(), precision, reconstruct);
  return run_determinant_trace(fields, true);
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
