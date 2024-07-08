#include "dslash_test_utils.h"

using namespace quda;

// For loading the gauge fields
int argc_copy;
char **argv_copy;
dslash_test_type dtest_type = dslash_test_type::Dslash;
bool ctest_all_partitions = false;
bool ctest_domain_decomposition = false;

// For googletest names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class DslashTest : public ::testing::TestWithParam<::testing::tuple<int, int, int, int, int>>
{
protected:
  ::testing::tuple<int, int, int, int, int> param;

  bool skip()
  {
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if ((QUDA_PRECISION & getPrecision(::testing::get<0>(GetParam()))) == 0
        || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) {
      return true;
    }

    if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && dslash_test_wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal
        && (::testing::get<0>(GetParam()) == 2 || ::testing::get<0>(GetParam()) == 3)) {
      warningQuda("Only fixed precision supported for MatPCDagMatPCLocal operator, skipping...");
      return true;
    }

    const std::array<bool, 16> partition_enabled {true, true, true,  false,  true,  false, false, false,
                                                  true, false, false, false, true, false, true, true};
    if (!ctest_all_partitions && !partition_enabled[::testing::get<2>(GetParam())]) return true;

    if (!ctest_domain_decomposition && ::testing::get<3>(GetParam())>0) return true;

    return false;
  }

  DslashTestWrapper dslash_test_wrapper;
  void display_test_info(int precision, QudaReconstructType link_recon)
  {
    auto prec = getPrecision(precision);
    // printfQuda("running the following test:\n");

    printfQuda("prec    recon   test_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension "
               "dslash_type    niter\n");
    printfQuda("%6s   %2s       %s           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n",
               get_prec_str(prec), get_recon_str(link_recon),
               get_string(dtest_type_map, dslash_test_wrapper.dtest_type).c_str(), get_matpc_str(matpc_type), dagger,
               xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type), niter);

    if (dslash_test_wrapper.test_split_grid) {
      printfQuda("Testing with split grid: %d  %d  %d  %d\n", grid_partition[0], grid_partition[1], grid_partition[2],
                 grid_partition[3]);
    }

    if (dslash_test_wrapper.test_domain_decomposition) {
      if (dd_red_black)
        printfQuda("Testing DD Red Black with block: %d  %d  %d  %d\n", dd_block_size[0], dd_block_size[1],
                   dd_block_size[2], dd_block_size[3]);
    }
  }

public:
  virtual void SetUp()
  {
    int prec = ::testing::get<0>(GetParam());
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if (skip()) GTEST_SKIP();

    int value = ::testing::get<2>(GetParam());
    for (int j = 0; j < 4; j++) {
      if (value & (1 << j)) { commDimPartitionedSet(j); }
    }
    updateR();

    int dd_value = ::testing::get<3>(GetParam());
    int dd_color = ::testing::get<4>(GetParam());
    dslash_test_wrapper.init_ctest(argc_copy, argv_copy, prec, recon, dd_value, dd_color);
    display_test_info(prec, recon);
  }

  virtual void TearDown()
  {
    if (skip()) GTEST_SKIP();
    dslash_test_wrapper.end();
    commDimPartitionedReset();
  }

  static void SetUpTestCase()
  {
    DslashTestWrapper::dtest_type = dtest_type;
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase()
  {
    DslashTestWrapper::destroy();
  }
};

TEST_P(DslashTest, verify)
{
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

  ASSERT_LE(deviation, tol) << "Reference and QUDA implementations do not agree";
}

TEST_P(DslashTest, benchmark) { dslash_test_wrapper.run_test(niter, /**show_metrics =*/true); }

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);
  // command line options
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  app->add_option("--all-partitions", ctest_all_partitions, "Test all instead of reduced combination of partitions");
  app->add_option("--domain-decomposition", ctest_domain_decomposition, "Test domain decomposition");
  add_comms_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  // The 'SetUp()' method of the Google Test class from which DslashTest
  // in derived has no arguments, but QUDA's implementation requires the
  // use of argc and argv to set up the test via the function 'init'.
  // As a workaround, we declare argc_copy and argv_copy as global pointers
  // so that they are visible inside the 'init' function.
  argc_copy = argc;
  argv_copy = argv;

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();

  endQuda();
  finalizeComms();
  return test_rc;
}

std::string getdslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int, int, int>> param)
{
  const int prec = ::testing::get<0>(param.param);
  const int recon = ::testing::get<1>(param.param);
  const int part = ::testing::get<2>(param.param);
  const int dd = ::testing::get<3>(param.param);
  const int col = ::testing::get<4>(param.param);
  std::stringstream ss;
  ss << get_prec_str(getPrecision(prec));
  ss << "_r" << recon;
  ss << "_partition" << part;
  if (dd > 0) {
    switch (dd) {
    case 1: ss << "_dd_local"; break;
    case 2: ss << "_dd_global"; break;
    }
    switch (col) {
    case 0: ss << "_red_red"; break;
    case 1: ss << "_black_red"; break;
    case 2: ss << "_red_black"; break;
    case 3: ss << "_black_black"; break;
    }
  }
  return ss.str();
}

#ifdef MULTI_GPU
#define N_PARTITIONS 16
#else
#define N_PARTITIONS 1
#endif

// regular tests
INSTANTIATE_TEST_SUITE_P(Regular, DslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, N_PARTITIONS), ::testing::Values(0), ::testing::Values(0)),
                         getdslashtestname);

#ifdef GPU_DD_DIRAC
#define N_DD_TESTS 3

// DD tests
INSTANTIATE_TEST_SUITE_P(DD, DslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, N_PARTITIONS), Range(1, N_DD_TESTS), Range(0, 4)),
                         getdslashtestname);

#endif

#undef N_PARTITIONS
#undef N_DD_TESTS
