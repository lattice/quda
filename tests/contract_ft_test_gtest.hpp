#include <gtest/gtest.h>
#include <quda_arch.h>
#include <cmath>

using test_t = ::testing::tuple<QudaContractType, QudaPrecision>;

class ContractFTTest : public ::testing::TestWithParam<test_t>
{
  test_t param;

public:
  ContractFTTest() : param(GetParam()) { }
};

bool skip_test(test_t param)
{
  auto contract_type = ::testing::get<0>(param);
  auto prec = ::testing::get<1>(param);

  // skip spin 4 cases
  if (contract_type == QUDA_CONTRACT_TYPE_DR_FT_T or contract_type == QUDA_CONTRACT_TYPE_DR_FT_Z) return true;
  if (prec < QUDA_SINGLE_PRECISION) return true; // outer precision >= sloppy precision
  if (!(QUDA_PRECISION & prec)) return true;     // precision not enabled so skip it

  return false;
}

int contract(test_t param);

TEST_P(ContractFTTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  auto faults = contract(GetParam());
  EXPECT_EQ(faults, 0) << "CPU and GPU implementations do not agree";
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string str("contract_");

  str += get_contract_str(::testing::get<0>(param.param));
  str += std::string("_") + get_prec_str(::testing::get<1>(param.param));

  return str;
}

using ::testing::Combine;
using ::testing::Values;

auto contract_types = Values(QUDA_CONTRACT_TYPE_STAGGERED_FT_T); // FIXME : extend if needed

auto precisions = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);

INSTANTIATE_TEST_SUITE_P(contraction_ft, ContractFTTest, Combine(contract_types, precisions), gettestname);
