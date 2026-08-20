#pragma once

#include <array>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

using plaquette_test_t = std::tuple<QudaPrecision, QudaReconstructType, int>;

#ifdef MULTI_GPU
constexpr std::array plaquette_partitions {0, 1, 2, 4, 8, 12, 14, 15};
#else
constexpr std::array plaquette_partitions {0};
#endif

std::array<double, 3> plaquette_test(QudaPrecision precision, QudaReconstructType reconstruct);
std::array<double, 6> plaquette_rectangle_test(QudaPrecision precision, QudaReconstructType reconstruct);
void polyakov_loop_test();
void topological_charge_and_density_test();
void gauge_smearing_or_flow_test();

inline std::string plaquette_test_name(testing::TestParamInfo<plaquette_test_t> param)
{
  const auto precision = testing::get<0>(param.param);
  const auto reconstruct = testing::get<1>(param.param);
  const auto partition = testing::get<2>(param.param);
  return std::string(get_prec_str(precision)) + "_r" + get_recon_str(reconstruct) + "_partition"
    + std::to_string(partition);
}

class PlaquetteTest : public ::testing::TestWithParam<plaquette_test_t>
{
protected:
  void SetUp() override
  {
    const auto partition = testing::get<2>(GetParam());
    for (int dir = 0; dir < 4; dir++) {
      if (partition & (1 << dir)) quda::commDimPartitionedSet(dir);
    }
    updateR();
  }

  void TearDown() override { quda::commDimPartitionedReset(); }
};

TEST_P(PlaquetteTest, Verify)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = plaquette_test(precision, reconstruct);
  for (int i = 0; i < 3; i++)
    EXPECT_LE(deviation[i], getTolerance(precision)) << "Host and QUDA plaquette component " << i << " do not agree";
}

TEST_P(PlaquetteTest, PlaquetteRectangle)
{
  const auto [precision, reconstruct, partition] = GetParam();
  static_cast<void>(partition);
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  if ((QUDA_RECONSTRUCT & getReconstructNibble(reconstruct)) == 0) GTEST_SKIP();
  if (!verify_results) GTEST_SKIP() << "CPU reference verification disabled";
  const auto deviation = plaquette_rectangle_test(precision, reconstruct);
  for (int i = 0; i < 6; i++)
    EXPECT_LE(deviation[i], getTolerance(precision))
      << "Host and QUDA plaquette-rectangle component " << i << " do not agree";
}

INSTANTIATE_TEST_SUITE_P(Plaquette, PlaquetteTest,
                         testing::Combine(testing::Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION),
                                          testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12),
                                          testing::ValuesIn(plaquette_partitions)),
                         plaquette_test_name);

TEST(SU3Test, PolyakovLoop) { polyakov_loop_test(); }

TEST(SU3Test, TopologicalChargeAndDensity) { topological_charge_and_density_test(); }

TEST(SU3Test, GaugeSmearingOrFlow) { gauge_smearing_or_flow_test(); }
