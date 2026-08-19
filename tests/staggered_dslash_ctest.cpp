#include "staggered_dslash_test_utils.h"

using namespace quda;

bool ctest_all_partitions = false;
bool ctest_domain_decomposition = false;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class StaggeredDslashTest
  : public ::testing::TestWithParam<::testing::tuple<int, int, int, QudaDomainDecompositionType, QudaDomainDecompositionColor>>
{
protected:
  ::testing::tuple<int, int, int, QudaDomainDecompositionType, QudaDomainDecompositionColor> param;

  bool skip()
  {
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if ((QUDA_PRECISION & getPrecision(::testing::get<0>(GetParam()))) == 0
        || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0)
      return true;

    if (is_laplace(dslash_type) && (::testing::get<0>(GetParam()) == 0 || ::testing::get<0>(GetParam()) == 1))
      return true;

    const std::array<bool, 16> partition_enabled {true, true, true,  false,  true,  false, false, false,
                                                  true, false, false, false, true, false, true, true};
    if (!ctest_all_partitions && !partition_enabled[::testing::get<2>(GetParam())]) return true;

    if (::testing::get<2>(GetParam()) > 0 && dslash_test_wrapper.test_split_grid) { return true; }

    if (::testing::get<3>(GetParam()) == 0 && ::testing::get<4>(GetParam()) > 0) return true;
    if (!ctest_domain_decomposition && ::testing::get<3>(GetParam()) > 0) return true;

    return false;
  }

  StaggeredDslashTestWrapper dslash_test_wrapper;
  void display_test_info(int precision, QudaReconstructType link_recon)
  {
    auto prec = getPrecision(precision);

    printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
    printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec),
               get_recon_str(link_recon), get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
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

    int partition = ::testing::get<2>(GetParam());
    for (int j = 0; j < 4; j++) {
      if (partition & (1 << j)) { commDimPartitionedSet(j); }
    }
    updateR();

    QudaDomainDecompositionType dd_value = ::testing::get<3>(GetParam());
    QudaDomainDecompositionColor dd_color = ::testing::get<4>(GetParam());

    dslash_test_wrapper.init_ctest(prec, recon, dd_value, dd_color);
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
    initQuda(device_ordinal);
    check_require_p2p(); // enforce --require-p2p (stream-gated P2P regression)
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase()
  {
    StaggeredDslashTestWrapper::destroy();
    endQuda();
  }
};

TEST_P(StaggeredDslashTest, verify)
{
  dslash_test_wrapper.staggeredDslashRef();
  dslash_test_wrapper.run_test(2);

  double deviation = dslash_test_wrapper.verify();
  double tol = getTolerance(dslash_test_wrapper.inv_param.cuda_prec);

  ASSERT_FALSE(std::isnan(deviation)) << "Nan has propagated into the result";
  tol = checkReasonableHostDeviation(deviation, tol, dslash_test_wrapper.inv_param.cuda_prec,
                                     dslash_test_wrapper.gauge_param.reconstruct);
  ASSERT_LE(deviation, tol) << "Reference CPU and QUDA implementations do not agree";
}

TEST_P(StaggeredDslashTest, benchmark) { dslash_test_wrapper.run_test(niter, true); }

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // override the default dslash from Wilson
  dslash_type = QUDA_ASQTAD_DSLASH;

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

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // Only these fermions are supported in this file
  if constexpr (is_enabled_laplace()) {
    if (!is_staggered(dslash_type) && !is_laplace(dslash_type))
      errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  } else {
    if (is_laplace(dslash_type))
      errorQuda("The Laplace dslash is not enabled, cmake configure with -DQUDA_DIRAC_LAPLACE=ON");
    if (!is_staggered(dslash_type)) errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  }

  // Sanity check: if you pass in a gauge field, want to test the asqtad/hisq dslash, and don't
  // ask to build the fat/long links... it doesn't make sense.
  if (latfile.size() > 0 && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH)
    errorQuda(
      "Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");

  // Set n_naiks to 2 if eps_naik != 0.0
  if (eps_naik != 0.0) {
    if (compute_fatlong)
      n_naiks = 2;
    else
      eps_naik = 0.0; // to avoid potential headaches
  }

  if (is_laplace(dslash_type) && dtest_type != dslash_test_type::Mat)
    errorQuda("Test type %s is not supported for the Laplace operator", get_string(dtest_type_map, dtest_type).c_str());

  int test_rc = RUN_ALL_TESTS();

  finalizeComms();

  return test_rc;
}

std::string getstaggereddslashtestname(
  testing::TestParamInfo<::testing::tuple<int, int, int, QudaDomainDecompositionType, QudaDomainDecompositionColor>> param)
{
  const int prec = ::testing::get<0>(param.param);
  const int recon = ::testing::get<1>(param.param);
  const int part = ::testing::get<2>(param.param);
  const QudaDomainDecompositionType dd = ::testing::get<3>(param.param);
  const QudaDomainDecompositionColor col = ::testing::get<4>(param.param);
  std::stringstream ss;
  // ss << get_dslash_str(dslash_type) << "_";
  ss << get_prec_str(getPrecision(prec));
  ss << "_r" << recon;
  ss << "_partition" << part;
  if (dd != QUDA_NO_DD) {
    switch (dd) {
    case QUDA_DDBLOCK_HALFLOCALL: ss << "_dd_local"; break;
    case QUDA_DDBLOCK_HALFGLOBALL: ss << "_dd_global"; break;
    default: break;
    }
    switch (col) {
    case QUDA_DD_COLOR_RED_RED: ss << "_red_red"; break;
    case QUDA_DD_COLOR_BLACK_RED: ss << "_black_red"; break;
    case QUDA_DD_COLOR_RED_BLACK: ss << "_red_black"; break;
    case QUDA_DD_COLOR_BLACK_BLACK: ss << "_black_black"; break;
    }
  } else if (col > 0) {
    ss << "_skipped" << col;
  }
  return ss.str();
}

#ifdef MULTI_GPU
#define N_PARTITIONS 16
#else
#define N_PARTITIONS 1
#endif

#if QUDA_DOMAIN_DECOMPOSITION > 0
INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, N_PARTITIONS),
                                 ::testing::Values(QUDA_NO_DD, QUDA_DDBLOCK_HALFLOCALL, QUDA_DDBLOCK_HALFGLOBALL),
                                 ::testing::Values(QUDA_DD_COLOR_RED_RED, QUDA_DD_COLOR_BLACK_RED,
                                                   QUDA_DD_COLOR_RED_BLACK, QUDA_DD_COLOR_BLACK_BLACK)),
                         getstaggereddslashtestname);
#else
INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, N_PARTITIONS), ::testing::Values(QUDA_NO_DD),
                                 ::testing::Values(QUDA_DD_COLOR_RED_RED)),
                         getstaggereddslashtestname);
#endif

#undef N_PARTITIONS
