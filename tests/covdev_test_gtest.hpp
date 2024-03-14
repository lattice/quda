#include <gtest/gtest.h>
#include <quda_arch.h>
#include <cmath>

using test_t = ::testing::tuple<QudaPrecision, QudaDagType>;

class CovDevTest : public ::testing::TestWithParam<test_t>
{
  protected:
    test_t param;

  public:
    CovDevTest() : param(GetParam()) { }
};

bool skip_test(test_t param)
{
  //auto prec             = ::testing::get<0>(param);
  //auto dag              = ::testing::get<1>(param);  

  //should we keep for all options?

  return false;
}

std::array<double, 2> covdev_test(test_t param);

TEST_P(CovDevTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  std::array<double, 2> test_results = covdev_test(param);
 
  double deviation = test_results[0];
  double tol       = test_results[1];

  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}



std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string str("covdev_");

  str += get_prec_str(::testing::get<0>(param.param));
  str += std::string("_") + get_dag_str(::testing::get<1>(param.param));

  return str; 
}

using ::testing::Combine;
using ::testing::Values;

auto precisions = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION);
auto dagger_opt = Values(QUDA_DAG_YES, QUDA_DAG_NO);

INSTANTIATE_TEST_SUITE_P(covdevtst, CovDevTest, Combine(precisions, dagger_opt), gettestname);

