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

using smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int>;
using flow_smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, unsigned int>;
using anisotropic_smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, double>;
using anisotropic_flow_smear_test_t
  = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, unsigned int, double>;

namespace {

constexpr double kAnisotropicSmearValue = 1.3;

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
  return enable_testing ? GaugeInputMode::GAUSSIAN_SU3 : GaugeInputMode::HAAR;
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

SmearMetrics benchmark(QudaPrecision precision, QudaGaugeSmearParam smear_param,
                       QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID)
{
  GaugeSmearFields fields(precision, reconstruct);

  auto warmup_param = smear_param;
  warmup_param.n_steps = 1;
  warmup_param.meas_interval = 2;
  run_smear(warmup_param);

  const auto flops0 = quda::Tunable::flops_global();
  const auto bytes0 = quda::Tunable::bytes_global();

  quda::device_timer_t timer;
  quda::comm_barrier();
  timer.start();
  run_smear(smear_param);
  timer.stop();

  return {timer.last(), quda::Tunable::flops_global() - flops0, quda::Tunable::bytes_global() - bytes0};
}

void report_benchmark(QudaGaugeSmearType type, int n_steps, const SmearMetrics &metrics)
{
  const auto steps = static_cast<double>(n_steps);
  const auto flops_per_step = metrics.flops / n_steps;
  const auto bytes_per_step = metrics.bytes / n_steps;
  const auto gflops = 1e-9 * metrics.flops / metrics.seconds;
  const auto gbytes = 1e-9 * metrics.bytes / metrics.seconds;
  const auto intensity = metrics.bytes == 0 ? 0.0 : static_cast<double>(metrics.flops) / metrics.bytes;

  printfQuda("%s benchmark: %.3f us per step\n", get_gauge_smear_str(type), 1e6 * metrics.seconds / steps);
  printfQuda("Accounted FLOPs: %llu total, %llu per step\n", metrics.flops, flops_per_step);
  printfQuda("Accounted bytes: %llu total, %llu per step\n", metrics.bytes, bytes_per_step);
  printfQuda("Accounted performance: %.3f GFLOP/s, %.3f GB/s, %.3f FLOP/byte\n", gflops, gbytes, intensity);
}

const char *reconstruct_label(QudaReconstructType reconstruct)
{
  switch (reconstruct) {
  case QUDA_RECONSTRUCT_NO: return "r18";
  case QUDA_RECONSTRUCT_12: return "r12";
  default: return "runknown";
  }
}

std::string test_name(testing::TestParamInfo<smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_" + direction;
}

std::string flow_test_name(testing::TestParamInfo<flow_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto rk_order = testing::get<4>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_rk"
         + std::to_string(rk_order) + "_" + direction;
}

std::string anisotropic_test_name(testing::TestParamInfo<anisotropic_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_aniso_" + direction;
}

std::string anisotropic_flow_test_name(testing::TestParamInfo<anisotropic_flow_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto rk_order = testing::get<4>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_rk"
         + std::to_string(rk_order) + "_aniso_" + direction;
}

} // namespace

class GaugeSmearTest : public ::testing::TestWithParam<smear_test_t> {
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeSmearTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(verify_one_step(precision, make_smear_param(type, dir_ignore, false), reconstruct), 1)
    << "CPU and QUDA gauge smearing implementations do not agree";
}

INSTANTIATE_TEST_SUITE_P(
  APE, GaugeSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12), testing::Values(QUDA_GAUGE_SMEAR_APE),
                   testing::Values(-1, 3, 4)),
  test_name);
INSTANTIATE_TEST_SUITE_P(
  Stout, GaugeSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12), testing::Values(QUDA_GAUGE_SMEAR_STOUT),
                   testing::Values(-1, 3, 4)),
  test_name);
INSTANTIATE_TEST_SUITE_P(
  OvrImpStout, GaugeSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_OVRIMP_STOUT), testing::Values(-1, 3, 4)),
  test_name);
INSTANTIATE_TEST_SUITE_P(
  HYP, GaugeSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12), testing::Values(QUDA_GAUGE_SMEAR_HYP),
                   testing::Values(-1, 3, 4)),
  test_name);

class GaugeFlowSmearTest : public ::testing::TestWithParam<flow_smear_test_t> {
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeFlowSmearTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, rk_order] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(verify_one_step(precision, make_smear_param(type, dir_ignore, false, rk_order), reconstruct), 1)
    << "CPU and QUDA gauge smearing implementations do not agree";
}

INSTANTIATE_TEST_SUITE_P(
  WilsonFlow, GaugeFlowSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_WILSON_FLOW), testing::Values(-1), testing::Values(3u, 4u)),
  flow_test_name);
INSTANTIATE_TEST_SUITE_P(
  SymanzikFlow, GaugeFlowSmearTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_SYMANZIK_FLOW), testing::Values(-1), testing::Values(3u, 4u)),
  flow_test_name);

class GaugeSmearAnisotropicTest : public ::testing::TestWithParam<anisotropic_smear_test_t> {
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeSmearAnisotropicTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, smear_anisotropy] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(verify_one_step(precision, make_smear_param(type, dir_ignore, false, 3, smear_anisotropy), reconstruct), 1)
    << "CPU and QUDA gauge smearing implementations do not agree";
}

INSTANTIATE_TEST_SUITE_P(
  APE_Anisotropic, GaugeSmearAnisotropicTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12), testing::Values(QUDA_GAUGE_SMEAR_APE),
                   testing::Values(4), testing::Values(kAnisotropicSmearValue)),
  anisotropic_test_name);
INSTANTIATE_TEST_SUITE_P(
  Stout_Anisotropic, GaugeSmearAnisotropicTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12), testing::Values(QUDA_GAUGE_SMEAR_STOUT),
                   testing::Values(4), testing::Values(kAnisotropicSmearValue)),
  anisotropic_test_name);
INSTANTIATE_TEST_SUITE_P(
  OvrImpStout_Anisotropic, GaugeSmearAnisotropicTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_OVRIMP_STOUT), testing::Values(4),
                   testing::Values(kAnisotropicSmearValue)),
  anisotropic_test_name);

class GaugeFlowSmearAnisotropicTest : public ::testing::TestWithParam<anisotropic_flow_smear_test_t> {
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeFlowSmearAnisotropicTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, rk_order, smear_anisotropy] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(verify_one_step(precision, make_smear_param(type, dir_ignore, false, rk_order, smear_anisotropy), reconstruct),
            1)
    << "CPU and QUDA gauge smearing implementations do not agree";
}

INSTANTIATE_TEST_SUITE_P(
  WilsonFlow_Anisotropic, GaugeFlowSmearAnisotropicTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_WILSON_FLOW), testing::Values(-1), testing::Values(3u, 4u),
                   testing::Values(kAnisotropicSmearValue)),
  anisotropic_flow_test_name);
INSTANTIATE_TEST_SUITE_P(
  SymanzikFlow_Anisotropic, GaugeFlowSmearAnisotropicTest,
  testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                   testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                   testing::Values(QUDA_GAUGE_SMEAR_SYMANZIK_FLOW), testing::Values(-1), testing::Values(3u, 4u),
                   testing::Values(kAnisotropicSmearValue)),
  anisotropic_flow_test_name);

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
