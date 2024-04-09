#include "hisq_stencil_test_utils.h"

using namespace quda;

bool ctest_all_partitions = false;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class HisqStencilTest : public ::testing::TestWithParam<::testing::tuple<QudaPrecision, QudaReconstructType, bool, int>>
{
protected:
  ::testing::tuple<QudaPrecision, QudaReconstructType, bool, int> param;

  HisqStencilTestWrapper hisq_stencil_test_wrapper;

  bool skip()
  {
    QudaPrecision precision = static_cast<QudaPrecision>(::testing::get<0>(GetParam()));
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if ((QUDA_PRECISION & precision) == 0 || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) return true;

    const std::array<bool, 16> partition_enabled {true, true,  true,  false, true, false, false, false,
                                                  true, false, false, false, true, false, true,  true};
    if (!ctest_all_partitions && !partition_enabled[::testing::get<3>(GetParam())]) return true;

    return false;
  }

  void display_test_info(QudaPrecision prec, QudaReconstructType link_recon, bool has_naik)
  {
    printfQuda("running the following test:\n");
    printfQuda(
      "link_precision           link_reconstruct           space_dimension        T_dimension       Ordering\n");
    printfQuda("%s                       %s                         %d/%d/%d/                  %d             %s \n",
               get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim, get_gauge_order_str(gauge_order));
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
    printfQuda("Number of Naiks: %d\n", has_naik ? 2 : 1);
  }

public:
  virtual void SetUp()
  {
    QudaPrecision prec = static_cast<QudaPrecision>(::testing::get<0>(GetParam()));
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));
    bool has_naik = ::testing::get<2>(GetParam());

    if (skip()) GTEST_SKIP();

    int partition = ::testing::get<3>(GetParam());
    for (int j = 0; j < 4; j++) {
      if (partition & (1 << j)) { commDimPartitionedSet(j); }
    }
    updateR();

    hisq_stencil_test_wrapper.init_ctest(prec, recon, has_naik);
    display_test_info(prec, recon, has_naik);
  }

  virtual void TearDown()
  {
    if (skip()) GTEST_SKIP();
    hisq_stencil_test_wrapper.end();
  }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase()
  {
    HisqStencilTestWrapper::destroy();
    endQuda();
  }
};

TEST_P(HisqStencilTest, benchmark) { hisq_stencil_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_P(HisqStencilTest, verify)
{
  hisq_stencil_test_wrapper.run_test(2);

  std::array<double, 2> res = hisq_stencil_test_wrapper.verify();

  // extra factor of 10 b/c the norm isn't normalized
  double max_dev = 10. * getTolerance(prec);

  // fat link
  EXPECT_LE(res[0], max_dev) << "Reference CPU and QUDA implementations of fat link do not agree";

  // long link
  EXPECT_LE(res[1], max_dev) << "Reference CPU and QUDA implementations of long link do not agree";
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // for speed
  xdim = ydim = zdim = tdim = 8;

  // default to 18 reconstruct
  link_recon = QUDA_RECONSTRUCT_NO;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  // Parse command line options
  auto app = make_app();
  app->add_option("--all-partitions", ctest_all_partitions, "Test all instead of reduced combination of partitions");
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec == QUDA_HALF_PRECISION || prec == QUDA_QUARTER_PRECISION)
    errorQuda("Precision %d is unsupported in some link fattening routines\n", prec);

  if (link_recon != QUDA_RECONSTRUCT_NO)
    errorQuda("Reconstruct %d is unsupported in some link fattening routines\n", link_recon);

  if (gauge_order != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported gauge order %d", gauge_order);

  if (eps_naik != 0.0) { n_naiks = 2; }

  setVerbosity(verbosity);
  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();

  finalizeComms();

  return test_rc;
}

std::string
gethisqstenciltestname(testing::TestParamInfo<::testing::tuple<QudaPrecision, QudaReconstructType, bool, int>> param)
{
  const QudaPrecision prec = static_cast<QudaPrecision>(::testing::get<0>(param.param));
  const QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(param.param));
  const bool has_naik = ::testing::get<2>(param.param);
  const int part = ::testing::get<3>(param.param);
  std::stringstream ss;
  // ss << get_dslash_str(dslash_type) << "_";
  ss << get_prec_str(prec);
  ss << "_r" << recon;
  if (has_naik) ss << "_naik";
  ss << "_partition" << part;
  return ss.str();
}

#ifdef MULTI_GPU
INSTANTIATE_TEST_SUITE_P(QUDA, HisqStencilTest,
                         Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO), ::testing::Bool(), Range(0, 16)),
                         gethisqstenciltestname);
#else
INSTANTIATE_TEST_SUITE_P(QUDA, HisqStencilTest,
                         Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO), ::testing::Bool(), ::testing::Values(0)),
                         gethisqstenciltestname);
#endif
