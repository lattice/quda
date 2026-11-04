#include <sstream>

#include "test.h"
#include "staggered_oprod_test_utils.h"

using namespace quda;

static bool ctest_all_partitions = false;

using test_t = ::testing::tuple<QudaPrecision, int, int, int>;

class StaggeredOprodTest : public ::testing::TestWithParam<test_t>
{
protected:
  StaggeredOprodTestWrapper staggered_oprod_test_wrapper;

  bool skip()
  {
    if ((QUDA_PRECISION & ::testing::get<0>(GetParam())) == 0) return true;
    const std::array<bool, 16> partition_enabled {true, true,  true,  false, true, false, false, false,
                                                  true, false, false, false, true, false, true,  true};
    if (!ctest_all_partitions && !partition_enabled[::testing::get<3>(GetParam())]) return true;
    return false;
  }

public:
  void SetUp() override
  {
    if (skip()) GTEST_SKIP();

    const int partition = ::testing::get<3>(GetParam());
    commDimPartitionedReset();
    for (int j = 0; j < 4; j++) {
      if (partition & (1 << j)) { commDimPartitionedSet(j); }
    }

    staggered_oprod_test_wrapper.init_ctest(::testing::get<0>(GetParam()), ::testing::get<1>(GetParam()),
                                            ::testing::get<2>(GetParam()));
  }

  void TearDown() override { staggered_oprod_test_wrapper.end(); }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  static void TearDownTestCase() { endQuda(); }
};

TEST_P(StaggeredOprodTest, benchmark) { staggered_oprod_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_P(StaggeredOprodTest, verify)
{
  expect_oprod_results(staggered_oprod_test_wrapper.verify(), staggered_oprod_test_wrapper.n_out(),
                       ::testing::get<0>(GetParam()));
}

std::string getStaggeredOprodTestName(testing::TestParamInfo<test_t> param)
{
  const QudaPrecision prec = ::testing::get<0>(param.param);
  const int nFace = ::testing::get<1>(param.param);
  const int nsrc = ::testing::get<2>(param.param);
  const int part = ::testing::get<3>(param.param);
  std::stringstream ss;
  ss << get_prec_str(prec);
  ss << "_nface" << nFace;
  ss << "_nsrc" << nsrc;
  ss << "_partition" << part;
  return ss.str();
}

constexpr int n_partitions = is_multi_gpu() ? 16 : 1;

INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredOprodTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(1, 3), ::testing::Values(1, 3),
                                            ::testing::Range(0, n_partitions)),
                         getStaggeredOprodTestName);

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  auto app = make_app();
  app->add_option("--all-partitions", ctest_all_partitions, "Test all instead of reduced combination of partitions");
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  setVerbosity(verbosity);
  initComms(argc, argv, gridsize_from_cmdline);

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();

  finalizeComms();

  return test_rc;
}
