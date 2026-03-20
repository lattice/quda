#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include <gauge_field.h>

#include <gtest/gtest.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "test.h"

using namespace quda;

using test_t = ::testing::tuple<QudaPrecision, QudaReconstructType, bool>;

class HisqEndToEndTest : public ::testing::TestWithParam<test_t>
{
protected:
  bool skip()
  {
    QudaPrecision precision = ::testing::get<0>(GetParam());
    QudaReconstructType recon = ::testing::get<1>(GetParam());
    if ((QUDA_PRECISION & precision) == 0 || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) return true;
    return false;
  }

public:
  void SetUp() override
  {
    if (skip()) GTEST_SKIP();
  }

  void TearDown() override
  {
    if (skip()) GTEST_SKIP();
  }
};

TEST_P(HisqEndToEndTest, placeholder) { FAIL() << "not yet implemented"; }

std::string getHisqEndToEndTestName(testing::TestParamInfo<test_t> param)
{
  const QudaPrecision prec = ::testing::get<0>(param.param);
  const QudaReconstructType recon = ::testing::get<1>(param.param);
  const bool has_naik = ::testing::get<2>(param.param);
  std::stringstream ss;
  ss << get_prec_str(prec);
  ss << "_r" << recon;
  if (has_naik) ss << "_naik";
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(QUDA, HisqEndToEndTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13),
                                            ::testing::Bool()),
                         getHisqEndToEndTestName);

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  setVerbosity(verbosity);
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();
  if (test_rc != 0) warningQuda("Tests failed");

  endQuda();
  finalizeComms();

  return test_rc;
}
