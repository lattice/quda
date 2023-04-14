#include "staggered_gsmear_test_utils.h"

using namespace quda;

int argc_copy;
char **argv_copy;

class StaggeredGSmearTest : public ::testing::Test
{
protected:
  StaggeredGSmearTestWrapper gsmear_test_wrapper;

  void display_test_info()
  {
    printfQuda("running the following test:\n");
    printfQuda("prec     recon    test_type     S_dim         T_dimension\n");
    printfQuda("%s   %s       %s       %d/%d/%d      %d \n", get_prec_str(prec), get_recon_str(link_recon),
               get_string(gtest_type_map, gtest_type).c_str(), xdim, ydim, zdim, tdim);
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
  }

public:
  StaggeredGSmearTest() = default;

  virtual void SetUp()
  {
    gsmear_test_wrapper.init_test(argc_copy, argv_copy);
    display_test_info();
  }

  virtual void TearDown() { gsmear_test_wrapper.end(); }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  static void TearDownTestCase() { endQuda(); }
};

TEST_F(StaggeredGSmearTest, benchmark) { gsmear_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_F(StaggeredGSmearTest, verify)
{
  if (!verify_results) GTEST_SKIP();

  gsmear_test_wrapper.staggeredGSmearRef();
  gsmear_test_wrapper.run_test(2);

  double deviation = gsmear_test_wrapper.verify();
  double tol = getTolerance(gsmear_test_wrapper.inv_param.cuda_prec);
  ASSERT_LE(deviation, tol) << "reference and QUDA implementations do not agree";
}


int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  app->add_option("--test", gtest_type, "Test method")->transform(CLI::CheckedTransformer(gtest_type_map));
  add_quark_smear_option_group(app);
  add_comms_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // Same approach as in Staggered DslashTest
  argc_copy = argc;
  argv_copy = argv;

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  if (link_recon != QUDA_RECONSTRUCT_NO) {
    printfQuda("Error: link reconstruction is currently not supported.\n");
    exit(0);
  }

  int test_rc = RUN_ALL_TESTS();

  finalizeComms();
  return test_rc;
}
