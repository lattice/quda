#include <gtest/gtest.h>
#include <quda_arch.h>
#include <cmath>

using test_t = ::testing::tuple<QudaContractType, QudaPrecision>;

class ContractFTTest : public ::testing::TestWithParam<test_t>
{
  protected:
    test_t param;

  public:
    ContractFTTest() : param(GetParam()) { }
};

bool skip_test(test_t param)
{
  auto contract_type = ::testing::get<0>(param);
  auto prec          = ::testing::get<1>(param);

  //FIXME : remove for spin 4
  if(contract_type == QUDA_CONTRACT_TYPE_DR_FT_T or contract_type == QUDA_CONTRACT_TYPE_DR_FT_Z) return true; //skip spin 4 cases

  if (prec == QUDA_HALF_PRECISION) return true; // outer precision >= sloppy precision
  if (!(QUDA_PRECISION & prec))    return true; // precision not enabled so skip it

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
  auto contractType = ::testing::get<0>(param.param);	
  auto prec         = ::testing::get<1>(param.param);

  const char *names[] = {"DegrandRossi_FT_t","DegrandRossi_FT_z", "Staggered_FT_t"};

  std::string str(names[contractType]);
  str += std::string("_");
  str += std::string(get_prec_str(getPrecision(prec)));

  return str; 
}

using ::testing::Combine;
using ::testing::Values;

auto contract_types = Values(QUDA_CONTRACT_TYPE_STAGGERED_FT_T); //FIXME : extend if needed

auto precisions     = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);

INSTANTIATE_TEST_SUITE_P(QUDA, ContractFTTest, Combine(contract_types, precisions), gettestname);

