#include "staggered_dslash_test_utils.h"

using namespace quda;

bool ctest_all_partitions = false;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class StaggeredDslashTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>>
{
protected:
  ::testing::tuple<int, int, int> param;

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
    return false;
  }

  StaggeredDslashTestWrapper dslash_test_wrapper;
  void display_test_info(int precision, QudaReconstructType link_recon)
  {
    auto prec = getPrecision(precision);

    printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
    printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec),
               get_recon_str(link_recon), get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
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

    dslash_test_wrapper.init_ctest(prec, recon);
    display_test_info(prec, recon);
  }

  virtual void TearDown()
  {
    if (skip()) GTEST_SKIP();
    dslash_test_wrapper.end();
  }

  static void SetUpTestCase() { initQuda(device_ordinal); }

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

  if (dslash_test_wrapper.gauge_param.reconstruct == QUDA_RECONSTRUCT_9
      && dslash_test_wrapper.inv_param.cuda_prec >= QUDA_HALF_PRECISION)
    tol *= 10; // if recon 9, we tolerate a greater deviation

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

std::string getstaggereddslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int>> param)
{
  const int prec = ::testing::get<0>(param.param);
  const int recon = ::testing::get<1>(param.param);
  const int part = ::testing::get<2>(param.param);
  std::stringstream ss;
  // ss << get_dslash_str(dslash_type) << "_";
  ss << get_prec_str(getPrecision(prec));
  ss << "_r" << recon;
  ss << "_partition" << part;
  return ss.str();
}

#ifdef MULTI_GPU
INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, 16)),
                         getstaggereddslashtestname);
#else
INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 ::testing::Values(0)),
                         getstaggereddslashtestname);
#endif
