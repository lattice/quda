#include "staggered_gsmear_test_utils.h"

using namespace quda;

int argc_copy;
char **argv_copy;

using test_t = ::testing::tuple<QudaPrecision, gsmear_test_type>;

class StaggeredGSmearTest : public ::testing::TestWithParam<test_t>
{
protected:
  StaggeredGSmearTestWrapper gsmear_test_wrapper;

public:
  StaggeredGSmearTest() = default;

  virtual void SetUp()
  {
    prec = ::testing::get<0>(GetParam());
    gtest_type = ::testing::get<1>(GetParam());
    if (!quda::is_enabled(prec)) GTEST_SKIP();
    gsmear_test_wrapper.init_test(argc_copy, argv_copy);
  }

  virtual void TearDown()
  {
    if (!quda::is_enabled(prec)) GTEST_SKIP();
    gsmear_test_wrapper.end();
  }

  static void SetUpTestCase() { }

  static void TearDownTestCase() { }
};

TEST_P(StaggeredGSmearTest, verify)
{
  prec = ::testing::get<0>(GetParam());
  gtest_type = ::testing::get<1>(GetParam());
  if (!quda::is_enabled(prec)) GTEST_SKIP();

  switch (gtest_type) {
  case gsmear_test_type::TwoLink:
    laplace3D = 4;
    smear_t0 = -1;
    break;
  case gsmear_test_type::GaussianSmear:
    laplace3D = 3;
    smear_t0 = 1;
    break;
  default: errorQuda("Unexpected gsmear_type = %s", get_string(gtest_type_map, gtest_type).c_str());
  }

  gsmear_test_wrapper.staggeredGSmearRef();
  gsmear_test_wrapper.run_test(2);

  double deviation = gsmear_test_wrapper.verify();
  double tol = getTolerance(gsmear_test_wrapper.inv_param.cuda_prec);
  ASSERT_LE(deviation, tol) << "reference and QUDA implementations do not agree";
}

struct gsmear_test : public quda_test {
  void display_info() const override
  {
    printfQuda("prec     recon    test_type     S_dim         T_dimension\n");
    printfQuda("%s   %s       %s       %d/%d/%d      %d \n", get_prec_str(prec), get_recon_str(link_recon),
               get_string(gtest_type_map, gtest_type).c_str(), xdim, ydim, zdim, tdim);
  }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    app->add_option("--test", gtest_type, "Test method")->transform(CLI::CheckedTransformer(gtest_type_map));
    add_quark_smear_option_group(app);
    add_su3_option_group(app);
  }

  gsmear_test(int argc, char **argv) : quda_test("gsmear_test", argc, argv) { }
};

auto test_str = [](testing::TestParamInfo<test_t> param) {
  return std::string(get_prec_str(::testing::get<0>(param.param))) + "_"
    + get_string(gtest_type_map, ::testing::get<1>(param.param));
};

using ::testing::Combine;
using ::testing::Values;

INSTANTIATE_TEST_SUITE_P(, StaggeredGSmearTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(gsmear_test_type::TwoLink, gsmear_test_type::GaussianSmear)),
                         test_str);

int main(int argc, char **argv)
{
  gsmear_test test(argc, argv);
  test.init();

  // Same approach as in Staggered DslashTest
  argc_copy = argc;
  argv_copy = argv;

  if (link_recon != QUDA_RECONSTRUCT_NO) errorQuda("Error: link reconstruction is currently not supported");

  int test_rc = 0;
  if (!enable_testing) {

  } else {
    test_rc = test.execute();
  }

  return test_rc;
}
