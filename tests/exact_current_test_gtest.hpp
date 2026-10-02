#include <gtest/gtest.h>
#include <quda_arch.h>

using test_t = ::testing::tuple<QudaPrecision, int>;

bool skip_test(test_t param)
{
  auto prec = ::testing::get<0>(param);
  if (!quda::is_enabled(prec)) return true;
  return false;
}

class ExactCurrentTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  ExactCurrentTest() : param(GetParam()) { }
};

double exact_current_test(test_t param);

TEST_P(ExactCurrentTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  auto prec = ::testing::get<0>(GetParam());
  double deviation = exact_current_test(GetParam());
  double tol = getTolerance(prec);

  ASSERT_FALSE(std::isinf(deviation)) << "NaN in exactCurrentQuda or the host reference";
  ASSERT_LE(deviation, tol) << "CPU and QUDA exact currents do not agree";
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string str("exact_current_");
  str += get_prec_str(::testing::get<0>(param.param));
  str += "_nmass" + std::to_string(::testing::get<1>(param.param));
  return str;
}

using ::testing::Combine;
using ::testing::Values;

auto precisions = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);
auto nmasses_values = Values(1, 2, 3);

INSTANTIATE_TEST_SUITE_P(ExactCurrent, ExactCurrentTest, Combine(precisions, nmasses_values), gettestname);
