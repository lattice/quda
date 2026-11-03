#include "dslash_test_utils.h"
#include <communicator_quda.h>

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

    if (dslash_test_wrapper.test_domain_decomposition) {
      if (dd_red_black)
        printfQuda("Testing DD Red Black with block: %d  %d  %d  %d\n", dd_block_size[0], dd_block_size[1],
                   dd_block_size[2], dd_block_size[3]);
    }
  }

public:
  virtual void SetUp()
  {
    dslash_test_wrapper.init_test(argc_copy, argv_copy);
    display_test_info();
  }

  virtual void TearDown() { dslash_test_wrapper.end(); }

  static void SetUpTestCase()
  {
    initQuda(device_ordinal);
    DslashTestWrapper::dtest_type = dtest_type;
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase()
  {
    DslashTestWrapper::destroy();
    endQuda();
  }
};

TEST_F(DslashTest, benchmark) { dslash_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_F(DslashTest, verify)
{
  if (!verify_results) GTEST_SKIP();

  if (not dslash_test_wrapper.test_domain_decomposition) dslash_test_wrapper.dslashRef();
  dslash_test_wrapper.run_test(2);

  double deviation = dslash_test_wrapper.verify();
  double tol = getTolerance(dslash_test_wrapper.inv_param.cuda_prec);
  // If we are using tensor core we tolerate a greater deviation
  if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && dslash_test_wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal)
    tol *= 10;

  ASSERT_FALSE(std::isnan(deviation)) << "Nan has propagated into the result";
  tol = checkReasonableHostDeviation(deviation, tol, dslash_test_wrapper.inv_param.cuda_prec,
                                     dslash_test_wrapper.gauge_param.reconstruct);
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

TEST_F(DslashTest, communicator_halo_phase_restore)
{
  if (!verify_results || DslashTestWrapper::test_split_grid || DslashTestWrapper::test_domain_decomposition
      || dslash_type != QUDA_WILSON_DSLASH || comm_size() < 2)
    GTEST_SKIP() << "Requires a global Wilson test on multiple ranks";

  dslash_test_wrapper.dslashRef();
  // Populate the global tuning cache before switching. Check the very first
  // post-switch call below: repeated identical inputs can hide a wrong-slot
  // read once both buffers have eventually received the same halo data.
  dslash_test_wrapper.run_test(2);
  CommKey split = default_comm_key;
  int dim = 3;
  while (dim > 0 && comm_dim(dim) == 1) dim--;
  split[dim] = comm_dim(dim);
  push_communicator(split);
  // Model an odd/even difference in the number of dslashes executed by
  // independent subgrids, without swapping any resident gauge fields.
  LatticeField::bufferIndex = comm_coord_global(dim) % 2;
  push_communicator(default_comm_key);
  int unaligned = LatticeField::bufferIndex != 0;
  comm_allreduce_int(unaligned);

  dslash_test_wrapper.dslashCUDA(1);
  dslash_test_wrapper.spinorOut = dslash_test_wrapper.cudaSpinorOut;
  double deviation = dslash_test_wrapper.verify();
  comm_allreduce_max(deviation);
  EXPECT_EQ(unaligned, 0) << "Halo buffers remained out of phase after communicator restoration";
  ASSERT_TRUE(std::isfinite(deviation));
  ASSERT_LE(deviation, getTolerance(dslash_test_wrapper.inv_param.cuda_prec));
}

TEST_F(DslashTest, split_grid_policy_restore)
{
  if (!verify_results || !DslashTestWrapper::test_split_grid || DslashTestWrapper::test_domain_decomposition
      || dslash_type != QUDA_WILSON_DSLASH || !getTuning())
    GTEST_SKIP() << "Requires Wilson, a split grid and tuning";

  dslash_test_wrapper.dslashRef();
  // Do not retain an operator alias while the split-grid API switches the
  // resident gauge fields. Rebuild global fields/operator for each check.
  dslash_test_wrapper.end();
  double deviation = 0.0;
  bool finite_results = true;
  for (int cycle = 0; cycle < 2; cycle++) {
    printfQuda("Split/global cycle %d: split-grid dslash\n", cycle);
    dslash_test_wrapper.run_test(2);
    double measured = dslash_test_wrapper.verify();
    finite_results &= std::isfinite(measured);
    deviation = std::max(deviation, measured);
    dslash_test_wrapper.cudaSpinor.clear();
    dslash_test_wrapper.cudaSpinorOut.clear();
    dslash_test_wrapper.cudaSpinorTmp.clear();
    dslash_test_wrapper.init();
    DslashTestWrapper::test_split_grid = false;
    printfQuda("Split/global cycle %d: restored global-grid dslash\n", cycle);
    dslash_test_wrapper.run_test(2);
    measured = dslash_test_wrapper.verify();
    finite_results &= std::isfinite(measured);
    deviation = std::max(deviation, measured);
    DslashTestWrapper::test_split_grid = true;
    dslash_test_wrapper.end();
  }

  bool found_global_policy = false;
  bool valid_policies = true;
  const int mask = comm_peer2peer_enabled_global();
  const bool force = comm_peer2peer_force();
  for (const auto &entry : getTuneCache()) {
    if (strncmp(entry.first.aux, "policy,", 7) != 0) continue;
    if (strstr(entry.first.aux, comm_config_string())) {
      found_global_policy = true;
      valid_policies &= ((!force || !mask) && entry.second.aux.y == 0)
        || (entry.second.aux.y == 1 && (mask & 1)) || (entry.second.aux.y == 2 && (mask & 2));
    }
    if (strstr(entry.first.aux, ",p2p=0,")) valid_policies &= entry.second.aux.y == 0;
  }
  EXPECT_TRUE(found_global_policy);
  EXPECT_TRUE(valid_policies) << "Policy set was not refreshed across split/global communicators";
  comm_allreduce_max(deviation);
  ASSERT_TRUE(finite_results);
  ASSERT_LE(deviation, getTolerance(dslash_test_wrapper.inv_param.cuda_prec));
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
  add_dd_option_group(app);
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
