#pragma once

#include <string>
#include <tuple>

#include <gtest/gtest.h>

using smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int>;
using flow_smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, unsigned int>;
using anisotropic_smear_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, double>;
using anisotropic_flow_smear_test_t
  = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, int, unsigned int, double>;

constexpr double kAnisotropicSmearValue = 1.3;

// Verification workers defined in gauge_smear_test.cpp; return 1 when the host and device fields agree.
int smear_verify(QudaPrecision precision, QudaReconstructType reconstruct, QudaGaugeSmearType type, int dir_ignore,
                 double smear_anisotropy);
int flow_verify(QudaPrecision precision, QudaReconstructType reconstruct, QudaGaugeSmearType type, int dir_ignore,
                unsigned int rk_order, double smear_anisotropy);

inline const char *reconstruct_label(QudaReconstructType reconstruct)
{
  switch (reconstruct) {
  case QUDA_RECONSTRUCT_NO: return "r18";
  case QUDA_RECONSTRUCT_12: return "r12";
  default: return "runknown";
  }
}

inline std::string test_name(testing::TestParamInfo<smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_" + direction;
}

inline std::string flow_test_name(testing::TestParamInfo<flow_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto rk_order = testing::get<4>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_rk"
    + std::to_string(rk_order) + "_" + direction;
}

inline std::string anisotropic_test_name(testing::TestParamInfo<anisotropic_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_aniso_" + direction;
}

inline std::string anisotropic_flow_test_name(testing::TestParamInfo<anisotropic_flow_smear_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto rk_order = testing::get<4>(param.param);
  const auto dir_ignore = testing::get<3>(param.param);
  const auto direction = dir_ignore < 0 ? "default" : "dir" + std::to_string(dir_ignore);
  return std::string(get_prec_str(precision)) + "_" + reconstruct_label(reconstruct) + "_rk"
    + std::to_string(rk_order) + "_aniso_" + direction;
}

class GaugeSmearTest : public ::testing::TestWithParam<smear_test_t>
{
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeSmearTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(smear_verify(precision, reconstruct, type, dir_ignore, 1.0), 1)
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

class GaugeFlowSmearTest : public ::testing::TestWithParam<flow_smear_test_t>
{
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeFlowSmearTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, rk_order] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(flow_verify(precision, reconstruct, type, dir_ignore, rk_order, 1.0), 1)
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

class GaugeSmearAnisotropicTest : public ::testing::TestWithParam<anisotropic_smear_test_t>
{
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeSmearAnisotropicTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, smear_anisotropy] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(smear_verify(precision, reconstruct, type, dir_ignore, smear_anisotropy), 1)
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

class GaugeFlowSmearAnisotropicTest : public ::testing::TestWithParam<anisotropic_flow_smear_test_t>
{
protected:
  void TearDown() override { freeGaugeQuda(); }
};

TEST_P(GaugeFlowSmearAnisotropicTest, OneStep)
{
  const auto [precision, reconstruct, type, dir_ignore, rk_order, smear_anisotropy] = GetParam();
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  ASSERT_EQ(flow_verify(precision, reconstruct, type, dir_ignore, rk_order, smear_anisotropy), 1)
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
