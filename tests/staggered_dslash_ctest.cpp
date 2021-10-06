#include "staggered_dslash_test_utils.h"

using namespace quda;

StaggeredDslashTestWrapper dslash_test_wrapper;

bool gauge_loaded = false;

const char *prec_str[] = {"quarter", "half", "single", "double"};
const char *recon_str[] = {"r18", "r13", "r9"};

void display_test_info(int precision, QudaReconstructType link_recon)
{
  auto prec = precision == 2 ? QUDA_DOUBLE_PRECISION : precision == 1 ? QUDA_SINGLE_PRECISION : QUDA_HALF_PRECISION;

  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec), get_recon_str(link_recon),
             get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
}

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
        || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) {
      return true;
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong
        && (::testing::get<0>(GetParam()) == 0 || ::testing::get<0>(GetParam()) == 1)) {
      warningQuda("Fixed precision unsupported in fat/long compute, skipping...");
      return true;
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong && (getReconstructNibble(recon) & 1)) {
      warningQuda("Reconstruct 9 unsupported in fat/long compute, skipping...");
      return true;
    }

    if (dslash_type == QUDA_LAPLACE_DSLASH && (::testing::get<0>(GetParam()) == 0 || ::testing::get<0>(GetParam()) == 1)) {
      warningQuda("Fixed precision unsupported for Laplace operator, skipping...");
      return true;
    }

    if (::testing::get<2>(GetParam()) > 0 && dslash_test_wrapper.test_split_grid) { return true; }
    return false;
  }

public:
  virtual ~StaggeredDslashTest() { }
  virtual void SetUp() {
    int prec = ::testing::get<0>(GetParam());
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if (skip()) GTEST_SKIP();

    int partition = ::testing::get<2>(GetParam());
    for (int j = 0; j < 4; j++) {
      if (partition & (1 << j)) { commDimPartitionedSet(j); }
    }
    updateR();

    dslash_test_wrapper.init_ctest(prec, link_recon);
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
  static void TearDownTestCase() { endQuda(); }
};

TEST_P(StaggeredDslashTest, verify)
{
  double deviation = 1.0;
  double tol = getTolerance(dslash_test_wrapper.inv_param.cuda_prec);
  // check for skip_kernel
  dslash_test_wrapper.staggeredDslashRef();
  if (dslash_test_wrapper.spinorRef != nullptr) {
    dslash_test_wrapper.run_test(2);
    deviation = dslash_test_wrapper.verify();
  }
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

TEST_P(StaggeredDslashTest, benchmark) { dslash_test_wrapper.run_test(niter, true); }

int main(int argc, char **argv)
{
  // hack for loading gauge fields
  dslash_test_wrapper.argc_copy = argc;
  dslash_test_wrapper.argv_copy = argv;

  dslash_test_wrapper.init_ctest_once();

  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
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

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Sanity check: if you pass in a gauge field, want to test the asqtad/hisq dslash, and don't
  // ask to build the fat/long links... it doesn't make sense.
  if (strcmp(latfile, "") && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH) {
    errorQuda(
      "Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");
    compute_fatlong = true;
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (dtest_type != dslash_test_type::Mat) {
      errorQuda("Test type %s is not supported for the Laplace operator.\n",
                get_string(dtest_type_map, dtest_type).c_str());
    }
  }

  // return result of RUN_ALL_TESTS
  int test_rc = RUN_ALL_TESTS();

  dslash_test_wrapper.end_ctest_once();

  finalizeComms();

  return test_rc;
}

  std::string getstaggereddslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int>> param){
   const int prec = ::testing::get<0>(param.param);
   const int recon = ::testing::get<1>(param.param);
   const int part = ::testing::get<2>(param.param);
   std::stringstream ss;
   // ss << get_dslash_str(dslash_type) << "_";
   ss << prec_str[prec];
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
