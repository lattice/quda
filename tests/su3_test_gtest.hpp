#pragma once

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

using gauge_observable_test_t = std::tuple<QudaPrecision, QudaReconstructType, int>;
using gauge_smear_observable_test_t = std::tuple<QudaPrecision, QudaReconstructType, QudaGaugeSmearType, bool, int>;

struct DifferenceTolerance {
  double difference;
  double tolerance;
};

struct EnergyTopologicalChargeComparison {
  std::array<DifferenceTolerance, 3> direct_energy;
  DifferenceTolerance direct_qcharge;
  std::array<DifferenceTolerance, 3> observable_energy;
  DifferenceTolerance observable_qcharge;
};

struct TopologicalChargeDensityComparison {
  DifferenceTolerance direct_field;
  DifferenceTolerance observable_field;
  std::array<DifferenceTolerance, 3> direct_energy;
  std::array<DifferenceTolerance, 3> observable_energy;
  DifferenceTolerance direct_qcharge;
  DifferenceTolerance observable_qcharge;
  DifferenceTolerance direct_density_sum;
  DifferenceTolerance observable_density_sum;
};

struct GaugeSmearObservableComparison {
  double field_deviation;
  double field_tolerance;
  std::array<double, 3> plaquette_deviation;
  std::array<double, 3> energy_difference;
  std::array<double, 3> energy_tolerance;
  double qcharge_difference;
  double qcharge_tolerance;
  int projection_failures;
};

#ifdef MULTI_GPU
constexpr std::array gauge_observable_partitions {0, 1, 2, 4, 8, 12, 14, 15};
#else
constexpr std::array gauge_observable_partitions {0};
#endif

inline void set_test_partition(int partition)
{
  for (int dir = 0; dir < 4; dir++) {
    if (partition & (1 << dir)) quda::commDimPartitionedSet(dir);
  }
  updateR();
}

inline void expect_finite(const DifferenceTolerance &comparison, const std::string &description)
{
  EXPECT_FALSE(std::isnan(comparison.difference)) << description << " difference is NaN";
  EXPECT_FALSE(std::isnan(comparison.tolerance)) << description << " tolerance is NaN";
}

std::array<double, 3> plaquette_test(QudaPrecision precision, QudaReconstructType reconstruct);
std::array<double, 6> plaquette_rectangle_test(QudaPrecision precision, QudaReconstructType reconstruct);
std::array<double, 2> polyakov_loop_test(QudaPrecision precision, QudaReconstructType reconstruct);
std::array<double, 4> determinant_trace_test(QudaPrecision precision, QudaReconstructType reconstruct);
double field_strength_tensor_test(QudaPrecision precision, QudaReconstructType reconstruct);
EnergyTopologicalChargeComparison energy_topological_charge_test(QudaPrecision precision,
                                                                 QudaReconstructType reconstruct);
TopologicalChargeDensityComparison topological_charge_density_test(QudaPrecision precision,
                                                                    QudaReconstructType reconstruct);
GaugeSmearObservableComparison run_gauge_smear_observable_test(QudaPrecision precision, QudaReconstructType reconstruct,
                                                               QudaGaugeSmearType type, bool su_project);

inline std::string gauge_observable_test_name(testing::TestParamInfo<gauge_observable_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto partition = testing::get<2>(param.param);
  return std::string(get_prec_str(precision)) + "_r" + get_recon_str(reconstruct) + "_partition"
    + std::to_string(partition);
}

inline std::string gauge_smear_observable_test_name(testing::TestParamInfo<gauge_smear_observable_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto su_project = testing::get<3>(param.param);
  const auto partition = testing::get<4>(param.param);
  return std::string(get_prec_str(precision)) + "_r" + get_recon_str(reconstruct) + "_project"
    + std::to_string(su_project) + "_partition" + std::to_string(partition);
}

class GaugeObservableTest : public ::testing::TestWithParam<gauge_observable_test_t>
{
protected:
  void SetUp() override { set_test_partition(testing::get<2>(GetParam())); }

  void TearDown() override { quda::commDimPartitionedReset(); }
};

TEST_P(GaugeObservableTest, Plaquette)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = plaquette_test(precision, reconstruct);
  for (int i = 0; i < 3; i++) {
    EXPECT_FALSE(std::isnan(deviation[i])) << "Host and QUDA plaquette component " << i << " deviation is NaN";
    EXPECT_LE(deviation[i], getTolerance(precision)) << "Host and QUDA plaquette component " << i << " do not agree";
  }
}

TEST_P(GaugeObservableTest, PlaquetteRectangle)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = plaquette_rectangle_test(precision, reconstruct);
  for (int i = 0; i < 6; i++) {
    EXPECT_FALSE(std::isnan(deviation[i])) << "Host and QUDA plaquette-rectangle component " << i << " deviation is NaN";
    EXPECT_LE(deviation[i], getTolerance(precision))
      << "Host and QUDA plaquette-rectangle component " << i << " do not agree";
  }
}

TEST_P(GaugeObservableTest, PolyakovLoop)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = polyakov_loop_test(precision, reconstruct);
  for (int i = 0; i < 2; i++) {
    EXPECT_FALSE(std::isnan(deviation[i])) << "Host and QUDA Polyakov-loop component " << i << " deviation is NaN";
    EXPECT_LE(deviation[i], getTolerance(precision)) << "Host and QUDA Polyakov-loop component " << i << " do not agree";
  }
}

TEST_P(GaugeObservableTest, DeterminantTrace)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto comparison = determinant_trace_test(precision, reconstruct);
  expect_finite({comparison[0], comparison[1]}, "Host and QUDA mean link determinant");
  EXPECT_LE(comparison[0], comparison[1]) << "Host and QUDA mean link determinant do not agree";
  expect_finite({comparison[2], comparison[3]}, "Host and QUDA mean link trace");
  EXPECT_LE(comparison[2], comparison[3]) << "Host and QUDA mean link trace do not agree";
}

TEST_P(GaugeObservableTest, FieldStrengthTensor)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = field_strength_tensor_test(precision, reconstruct);
  EXPECT_FALSE(std::isnan(deviation)) << "Host and QUDA field-strength tensor deviation is NaN";
  EXPECT_LE(deviation, getTolerance(precision))
    << "Host and QUDA field-strength tensors do not agree";
}

TEST_P(GaugeObservableTest, EnergyAndTopologicalCharge)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto comparison = energy_topological_charge_test(precision, reconstruct);
  for (int i = 0; i < 3; i++) {
    expect_finite(comparison.direct_energy[i], "Host and QUDA direct field-energy component " + std::to_string(i));
    EXPECT_LE(comparison.direct_energy[i].difference, comparison.direct_energy[i].tolerance)
      << "Host and QUDA direct field-energy component " << i << " do not agree";
    expect_finite(comparison.observable_energy[i],
                  "Host and QUDA observable field-energy component " + std::to_string(i));
    EXPECT_LE(comparison.observable_energy[i].difference, comparison.observable_energy[i].tolerance)
      << "Host and QUDA observable field-energy component " << i << " do not agree";
  }
  expect_finite(comparison.direct_qcharge, "Host and QUDA direct topological charge");
  EXPECT_LE(comparison.direct_qcharge.difference, comparison.direct_qcharge.tolerance)
    << "Host and QUDA direct topological charge do not agree";
  expect_finite(comparison.observable_qcharge, "Host and QUDA observable topological charge");
  EXPECT_LE(comparison.observable_qcharge.difference, comparison.observable_qcharge.tolerance)
    << "Host and QUDA observable topological charge do not agree";
}

TEST_P(GaugeObservableTest, TopologicalChargeDensity)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto comparison = topological_charge_density_test(precision, reconstruct);
  expect_finite(comparison.direct_field, "Host and QUDA direct topological-charge density");
  EXPECT_LE(comparison.direct_field.difference, comparison.direct_field.tolerance)
    << "Host and QUDA direct topological-charge density does not agree";
  expect_finite(comparison.observable_field, "Host and QUDA observable topological-charge density");
  EXPECT_LE(comparison.observable_field.difference, comparison.observable_field.tolerance)
    << "Host and QUDA observable topological-charge density does not agree";
  for (int i = 0; i < 3; i++) {
    expect_finite(comparison.direct_energy[i], "Host and QUDA direct field-energy component " + std::to_string(i));
    EXPECT_LE(comparison.direct_energy[i].difference, comparison.direct_energy[i].tolerance)
      << "Host and QUDA direct field-energy component " << i << " does not agree";
    expect_finite(comparison.observable_energy[i],
                  "Host and QUDA observable field-energy component " + std::to_string(i));
    EXPECT_LE(comparison.observable_energy[i].difference, comparison.observable_energy[i].tolerance)
      << "Host and QUDA observable field-energy component " << i << " does not agree";
  }
  expect_finite(comparison.direct_qcharge, "Host and QUDA direct topological charge");
  EXPECT_LE(comparison.direct_qcharge.difference, comparison.direct_qcharge.tolerance)
    << "Host and QUDA direct topological charge does not agree";
  expect_finite(comparison.observable_qcharge, "Host and QUDA observable topological charge");
  EXPECT_LE(comparison.observable_qcharge.difference, comparison.observable_qcharge.tolerance)
    << "Host and QUDA observable topological charge does not agree";
  expect_finite(comparison.direct_density_sum, "Direct topological-charge density sum");
  EXPECT_LE(comparison.direct_density_sum.difference, comparison.direct_density_sum.tolerance)
    << "Direct topological-charge density sum does not agree with direct topological charge";
  expect_finite(comparison.observable_density_sum, "Observable topological-charge density sum");
  EXPECT_LE(comparison.observable_density_sum.difference, comparison.observable_density_sum.tolerance)
    << "Observable topological-charge density sum does not agree with observable topological charge";
}

INSTANTIATE_TEST_SUITE_P(GaugeObservable, GaugeObservableTest,
                         testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                                          testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                                          testing::ValuesIn(gauge_observable_partitions)),
                         gauge_observable_test_name);

class GaugeSmearObservableTest : public ::testing::TestWithParam<gauge_smear_observable_test_t>
{
protected:
  void SetUp() override { set_test_partition(testing::get<4>(GetParam())); }

  void TearDown() override { quda::commDimPartitionedReset(); }
};

TEST_P(GaugeSmearObservableTest, FiveStep)
{
  const auto [precision, reconstruct, type, su_project, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";

  const auto comparison = run_gauge_smear_observable_test(precision, reconstruct, type, su_project);
  EXPECT_EQ(comparison.projection_failures, 0) << "Host SU(3) projection failed";
  EXPECT_FALSE(std::isnan(comparison.field_deviation)) << "Host and QUDA five-step gauge-field deviation is NaN";
  EXPECT_FALSE(std::isnan(comparison.field_tolerance)) << "Host and QUDA five-step gauge-field tolerance is NaN";
  EXPECT_LE(comparison.field_deviation, comparison.field_tolerance)
    << "Host and QUDA five-step gauge fields do not agree";
  for (int i = 0; i < 3; i++) {
    EXPECT_FALSE(std::isnan(comparison.plaquette_deviation[i]))
      << "Host and QUDA five-step plaquette component " << i << " deviation is NaN";
    EXPECT_LE(comparison.plaquette_deviation[i], getTolerance(precision))
      << "Host and QUDA five-step plaquette component " << i << " do not agree";
    EXPECT_FALSE(std::isnan(comparison.energy_difference[i]))
      << "Host and QUDA five-step field-energy component " << i << " difference is NaN";
    EXPECT_FALSE(std::isnan(comparison.energy_tolerance[i]))
      << "Host and QUDA five-step field-energy component " << i << " tolerance is NaN";
    EXPECT_LE(comparison.energy_difference[i], comparison.energy_tolerance[i])
      << "Host and QUDA five-step field-energy component " << i << " do not agree";
  }
  EXPECT_FALSE(std::isnan(comparison.qcharge_difference)) << "Host and QUDA five-step topological charge difference is NaN";
  EXPECT_FALSE(std::isnan(comparison.qcharge_tolerance)) << "Host and QUDA five-step topological charge tolerance is NaN";
  EXPECT_LE(comparison.qcharge_difference, comparison.qcharge_tolerance)
    << "Host and QUDA five-step topological charge do not agree";
}

#define INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(prefix, type)                                                          \
  INSTANTIATE_TEST_SUITE_P(prefix, GaugeSmearObservableTest,                                                           \
                           testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),             \
                                            testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),                 \
                                            testing::Values(type), testing::Bool(),                                    \
                                            testing::ValuesIn(gauge_observable_partitions)),                           \
                           gauge_smear_observable_test_name)

INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(APE, QUDA_GAUGE_SMEAR_APE);
INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(Stout, QUDA_GAUGE_SMEAR_STOUT);
INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(OvrImpStout, QUDA_GAUGE_SMEAR_OVRIMP_STOUT);
INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(HYP, QUDA_GAUGE_SMEAR_HYP);
INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(WilsonFlow, QUDA_GAUGE_SMEAR_WILSON_FLOW);
INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST(SymanzikFlow, QUDA_GAUGE_SMEAR_SYMANZIK_FLOW);

#undef INSTANTIATE_GAUGE_SMEAR_OBSERVABLE_TEST
