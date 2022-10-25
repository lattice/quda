// QUDA headers
#include <quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <instantiate.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#include <gtest/gtest.h>

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    prec_sloppy   matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension "
             "Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
             get_prec_str(prec), get_prec_str(prec_sloppy), get_matpc_str(matpc_type), get_recon_str(link_recon),
             get_recon_str(link_recon_sloppy), get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
             get_dslash_str(dslash_type), get_mass_normalization_str(normalization));
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

using test_t = ::testing::tuple<QudaSiteSubset, QudaDilutionType, int>;

class DilutionTest : public ::testing::TestWithParam<test_t>
{
protected:
  QudaSiteSubset site_subset;
  QudaDilutionType dilution_type;
  int nSpin;

public:
  DilutionTest() :
    site_subset(::testing::get<0>(GetParam())),
    dilution_type(testing::get<1>(GetParam())),
    nSpin(testing::get<2>(GetParam()))
  {
  }
};

TEST_P(DilutionTest, verify)
{
  using namespace quda;

  if (!is_enabled_spin(nSpin)) GTEST_SKIP();

  // Set some parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  setWilsonGaugeParam(gauge_param);
  setInvertParam(inv_param);

  ColorSpinorParam param;
  constructWilsonTestSpinorParam(&param, &inv_param, &gauge_param);
  param.siteSubset = site_subset;
  param.nSpin = nSpin;
  param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true); // change order to native order
  param.location = QUDA_CUDA_FIELD_LOCATION;
  param.create = QUDA_NULL_FIELD_CREATE;
  ColorSpinorField src(param);

  RNG rng(src, 1234);

  for (int i = 0; i < Nsrc; i++) {
    spinorNoise(src, rng, QUDA_NOISE_GAUSS); // Populate the host spinor with random numbers.

    size_t size = 0;
    switch (dilution_type) {
    case QUDA_DILUTION_SPIN: size = src.Nspin(); break;
    case QUDA_DILUTION_COLOR: size = src.Ncolor(); break;
    case QUDA_DILUTION_SPIN_COLOR: size = src.Nspin() * src.Ncolor(); break;
    case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: size = src.Nspin() * src.Ncolor() * src.SiteSubset(); break;
    default: errorQuda("Invalid dilution type %d", dilution_type);
    }

    std::vector<ColorSpinorField> v(size, param);
    spinorDilute(v, src, dilution_type);

    param.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField sum(param);
    blas::axpy(std::vector<double>(v.size(), 1.0), v, sum); // reassemble the vector

    { // check its norm matches the original
      auto src2 = blas::norm2(src);
      auto sum2 = blas::norm2(sum);
      EXPECT_EQ(sum2, src2);
    }

    { // check for component-by-component matching
      auto sum2 = blas::xmyNorm(src, sum);
      EXPECT_EQ(sum2, 0.0);
    }
  }
}

using ::testing::Combine;
using ::testing::Values;

INSTANTIATE_TEST_SUITE_P(
  WilsonFull, DilutionTest,
  Combine(Values(QUDA_FULL_SITE_SUBSET),
          Values(QUDA_DILUTION_SPIN, QUDA_DILUTION_COLOR, QUDA_DILUTION_SPIN_COLOR, QUDA_DILUTION_SPIN_COLOR_EVEN_ODD),
          Values(4)),
  [](testing::TestParamInfo<test_t> param) { return get_dilution_type_str(::testing::get<1>(param.param)); });

INSTANTIATE_TEST_SUITE_P(WilsonParity, DilutionTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET),
                                 Values(QUDA_DILUTION_SPIN, QUDA_DILUTION_COLOR, QUDA_DILUTION_SPIN_COLOR), Values(4)),
                         [](testing::TestParamInfo<test_t> param) {
                           return get_dilution_type_str(::testing::get<1>(param.param));
                         });

INSTANTIATE_TEST_SUITE_P(
  StaggeredFull, DilutionTest,
  Combine(Values(QUDA_FULL_SITE_SUBSET),
          Values(QUDA_DILUTION_SPIN, QUDA_DILUTION_COLOR, QUDA_DILUTION_SPIN_COLOR, QUDA_DILUTION_SPIN_COLOR_EVEN_ODD),
          Values(1)),
  [](testing::TestParamInfo<test_t> param) { return get_dilution_type_str(::testing::get<1>(param.param)); });

INSTANTIATE_TEST_SUITE_P(StaggeredParity, DilutionTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET),
                                 Values(QUDA_DILUTION_SPIN, QUDA_DILUTION_COLOR, QUDA_DILUTION_SPIN_COLOR), Values(1)),
                         [](testing::TestParamInfo<test_t> param) {
                           return get_dilution_type_str(::testing::get<1>(param.param));
                         });

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  // Parse command line options
  auto app = make_app();
  add_comms_option_group(app);
  add_testing_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Initialize the QUDA library
  initQuda(device_ordinal);

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
  int result = RUN_ALL_TESTS();

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  return result;
}
