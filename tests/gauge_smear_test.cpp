#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <quda.h>

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
                                     double smear_anisotropy = 1.0)
{
  QudaGaugeSmearParam param = newQudaGaugeSmearParam();
  param.smear_type = type;
  param.n_steps = 1;
  param.meas_interval = 2;
  param.rk_order = rk_order;
  param.dir_ignore = dir_ignore;
  param.smear_anisotropy = smear_anisotropy;
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

int run_one_step(QudaPrecision precision, QudaGaugeSmearParam smear_param,
                 QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID)
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.cuda_prec = precision;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  if (reconstruct != QUDA_RECONSTRUCT_INVALID) gauge_param.reconstruct = reconstruct;
  setDims(gauge_param.X);

  quda::GaugeFieldParam field_param(gauge_param);
  field_param.location = QUDA_CPU_FIELD_LOCATION;
  field_param.order = QUDA_QDP_GAUGE_ORDER;
  field_param.create = QUDA_NULL_FIELD_CREATE;
  quda::GaugeField input(field_param);
  quda::GaugeField reference(field_param);
  quda::GaugeField result(field_param);
  createSiteLinkCPU(input, gauge_param.cpu_prec, SiteLinkType::SITELINK_PHASE_NO);

  auto input_ptrs = input.data_array<void *>();
  auto result_ptrs = result.data_array<void *>();
  loadGaugeQuda(input_ptrs.data, &gauge_param);

  QudaGaugeObservableParam obs_param = newQudaGaugeObservableParam();
  obs_param.compute_plaquette = QUDA_BOOLEAN_FALSE;
  obs_param.compute_rectangle = QUDA_BOOLEAN_FALSE;
  obs_param.compute_polyakov_loop = QUDA_BOOLEAN_FALSE;
  obs_param.compute_qcharge = QUDA_BOOLEAN_FALSE;
  obs_param.compute_qcharge_density = QUDA_BOOLEAN_FALSE;
  obs_param.su_project = QUDA_BOOLEAN_FALSE;

  pushVerbosity(QUDA_SILENT);
  if (is_flow(smear_param.smear_type))
    performWFlowQuda(&smear_param, &obs_param);
  else
    performGaugeSmearQuda(&smear_param, &obs_param);
  popVerbosity();

  auto save_param = gauge_param;
  save_param.type = QUDA_SMEARED_LINKS;
  save_param.reconstruct = QUDA_RECONSTRUCT_NO;
  saveGaugeQuda(result_ptrs.data, &save_param);
  gauge_smear_reference(reference, input, smear_param);

  const auto tolerance = getTolerance(precision);
  int check = 1;
  auto max_deviation = 0.0;
  for (int dir = 0; dir < 4; dir++) {
    max_deviation = std::max(max_deviation, compare_floats_v2(result.data(dir), reference.data(dir), V * gauge_site_size,
                                                               std::numeric_limits<double>::infinity(), gauge_param.cpu_prec));
    check &= compare_floats(result.data(dir), reference.data(dir), V * gauge_site_size, tolerance, gauge_param.cpu_prec);
  }
  logQuda(QUDA_SUMMARIZE,
          "%s one-step %s reconstruct=%s rk_order=%u dir_ignore=%d smear_anisotropy=%.1f: max deviation %.3e, "
          "tolerance %.3e\n",
          get_gauge_smear_str(smear_param.smear_type), get_prec_str(precision), get_recon_str(gauge_param.reconstruct),
          smear_param.rk_order, smear_param.dir_ignore, smear_param.smear_anisotropy, max_deviation, tolerance);
  auto reference_ptrs = reference.data_array<void *>();
  strong_check_link(result_ptrs.data, "QUDA result:", reference_ptrs.data, "CPU reference:", V, gauge_param.cpu_prec);
  return check;
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
  ASSERT_EQ(run_one_step(precision, make_smear_param(type, dir_ignore, false), reconstruct), 1)
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
  ASSERT_EQ(run_one_step(precision, make_smear_param(type, dir_ignore, false, rk_order), reconstruct), 1)
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
  ASSERT_EQ(run_one_step(precision, make_smear_param(type, dir_ignore, false, 3, smear_anisotropy), reconstruct), 1)
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
  ASSERT_EQ(run_one_step(precision, make_smear_param(type, dir_ignore, false, rk_order, smear_anisotropy), reconstruct),
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
    printfQuda("\n%s one-step smearing\n", get_gauge_smear_str(gauge_smear_type));
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
    const auto smear_param = make_smear_param(gauge_smear_type, gauge_smear_dir_ignore, true);
    const auto check = run_one_step(prec, smear_param);
    freeGaugeQuda();
    return check ? 0 : 1;
  }
}
