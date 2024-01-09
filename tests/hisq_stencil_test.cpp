#include "hisq_stencil_test_utils.h"

using namespace quda;

class HisqStencilTest : public ::testing::Test
{
protected:
  HisqStencilTestWrapper hisq_stencil_test_wrapper;

  void display_test_info()
  {
    printfQuda("running the following test:\n");
    printfQuda(
      "link_precision           link_reconstruct           space_dimension        T_dimension       Ordering\n");
    printfQuda("%s                       %s                         %d/%d/%d/                  %d             %s \n",
               get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim, get_gauge_order_str(gauge_order));
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
    printfQuda("Number of Naiks: %d\n", n_naiks);
  }

public:
  virtual void SetUp()
  {
    hisq_stencil_test_wrapper.init_test();
    display_test_info();
  }

  virtual void TearDown() { hisq_stencil_test_wrapper.end(); }

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

TEST_F(HisqStencilTest, benchmark) { hisq_stencil_test_wrapper.run_test(niter, /**show_metrics =*/true); }

TEST_F(HisqStencilTest, verify)
{
  if (!verify_results) GTEST_SKIP();

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
