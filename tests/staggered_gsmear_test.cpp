#include "staggered_gsmear_test_utils.h"

using namespace quda;

StaggeredGSmearTestWrapper gsmear_test_wrapper;

static int gsmearTest()
{
  // return code for google test
  int test_rc = 0;
  gsmear_test_wrapper.init_test();

  gsmear_test_wrapper.staggeredGSmearRef();
  int attempts = 1;
  for (int i = 0; i < attempts; i++) {
    gsmear_test_wrapper.run_test(niter, /**print_metrics =*/true);
    if (verify_results) {
      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0) warningQuda("Tests failed");
    }
  }
  gsmear_test_wrapper.end();

  return test_rc;
}

TEST(gsmear, verify)
{
  double deviation = gsmear_test_wrapper.verify();
  double tol = getTolerance(prec);
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

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

int main(int argc, char **argv)
{
  gsmear_test_wrapper.argc_copy = argc;
  gsmear_test_wrapper.argv_copy = argv;

  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  app->add_option("--test", gtest_type, "Test method")->transform(CLI::CheckedTransformer(gtest_type_map));
  add_quark_smear_option_group(app);
  add_su3_option_group(app);
  add_comms_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  initQuda(device_ordinal);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  display_test_info();

  if (link_recon != QUDA_RECONSTRUCT_NO) {
    printfQuda("Error: link reconstruction is currently not supported.\n");
    exit(0);
  }

  // return result of RUN_ALL_TESTS
  int test_rc = gsmearTest();

  endQuda();

  finalizeComms();

  return test_rc;
}
