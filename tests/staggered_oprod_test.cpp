#include "test.h"
#include "staggered_oprod_test_utils.h"

using namespace quda;

class StaggeredOprodTest : public ::testing::Test
{
protected:
  StaggeredOprodTestWrapper staggered_oprod_test_wrapper;

  void display_test_info()
  {
    printfQuda("running the following test:\n");
    printfQuda("prec    nFace    nsrc    space_dimension    T_dimension\n");
    printfQuda("%s      %d        %d       %d/%d/%d           %d\n", get_prec_str(prec),
               staggered_oprod_test_wrapper.nFace, Nsrc, xdim, ydim, zdim, tdim);
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
  }

public:
  void SetUp() override
  {
    staggered_oprod_test_wrapper.init_test();
    display_test_info();
  }

  void TearDown() override { staggered_oprod_test_wrapper.end(); }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  static void TearDownTestCase() { endQuda(); }
};

TEST_F(StaggeredOprodTest, benchmark) { staggered_oprod_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_F(StaggeredOprodTest, verify)
{
  if (!verify_results) GTEST_SKIP();

  expect_oprod_results(staggered_oprod_test_wrapper.verify(), staggered_oprod_test_wrapper.n_out(), prec);
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // defaults: HISQ/asqtad (nFace = 3) outer products of three quark fields
  dslash_type = QUDA_ASQTAD_DSLASH;
  Nsrc = 3;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  // Parse command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION) errorQuda("Invalid precision %d", prec);

  setVerbosity(verbosity);
  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();

  finalizeComms();

  return test_rc;
}
