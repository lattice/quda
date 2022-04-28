#include "dslash_test_utils.h"

using namespace quda;

dslash_test_type dtest_type = dslash_test_type::Dslash;

int argc_copy;
char **argv_copy;

class DslashTest : public ::testing::Test
{
protected:
  DslashTestWrapper dslash_test_wrapper;

  void display_test_info()
  {
    printfQuda("running the following test:\n");

    printfQuda("prec    recon   dtest_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension "
               "dslash_type    niter\n");
    printfQuda("%6s   %2s       %s           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n",
               get_prec_str(prec), get_recon_str(link_recon),
               get_string(dtest_type_map, dslash_test_wrapper.dtest_type).c_str(), get_matpc_str(matpc_type), dagger,
               xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type), niter);
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));

    if (dslash_test_wrapper.test_split_grid) {
      printfQuda("Testing with split grid: %d  %d  %d  %d\n", grid_partition[0], grid_partition[1], grid_partition[2],
                 grid_partition[3]);
    }
  }

public:
  DslashTest() : dslash_test_wrapper(dtest_type) { }

  virtual void SetUp()
  {
    dslash_test_wrapper.init_test(argc_copy, argv_copy);
    display_test_info();
  }

  virtual void TearDown() { dslash_test_wrapper.end(); }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() { endQuda(); }
};

TEST_F(DslashTest, benchmark) { dslash_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_F(DslashTest, verify)
{
  if (!verify_results) GTEST_SKIP();

  dslash_test_wrapper.dslashRef();
  dslash_test_wrapper.run_test(2);

  double deviation = dslash_test_wrapper.verify();
  double tol = getTolerance(dslash_test_wrapper.inv_param.cuda_prec);
  // If we are using tensor core we tolerate a greater deviation
  if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && dslash_test_wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal)
    tol *= 10;
  if (dslash_test_wrapper.gauge_param.reconstruct == QUDA_RECONSTRUCT_8
      && dslash_test_wrapper.inv_param.cuda_prec >= QUDA_HALF_PRECISION)
    tol *= 10; // if recon 8, we tolerate a greater deviation

  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  add_eofa_option_group(app);
  add_comms_option_group(app);

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // The 'SetUp()' method of the Google Test class from which DslashTest
  // in derived has no arguments, but QUDA's implementation requires the
  // use of argc and argv to set up the test via the function 'init'.
  // As a workaround, we declare argc_copy and argv_copy as global pointers
  // so that they are visible inside the 'init' function.
  argc_copy = argc;
  argv_copy = argv;

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  test_rc = RUN_ALL_TESTS();

  finalizeComms();
  return test_rc;
}
