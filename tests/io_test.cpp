#include <cstdio>
#include <limits>

#include <instantiate.h>
#include <color_spinor_field.h>
#include <misc.h>
#include <qio_field.h> // for QIO routines
#include <vector_io.h>
#include <blas_quda.h>
#include <quda.h>
#include <test.h>

// tuple types: precision
using gauge_test_t = ::testing::tuple<QudaPrecision>;

class GaugeIOTest : public ::testing::TestWithParam<gauge_test_t>
{
protected:
  gauge_test_t param;

public:
  GaugeIOTest() : param(GetParam()) { }
};

// test write/read of a gauge field yields identical lattice
TEST_P(GaugeIOTest, verify)
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  gauge_param.cpu_prec = ::testing::get<0>(param);
  if (!quda::is_enabled(gauge_param.cpu_prec)) GTEST_SKIP();
  gauge_param.cuda_prec = gauge_param.cpu_prec;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, 0, nullptr);

  auto get_plaq = [&]() {
    // Load the gauge field to the device
    loadGaugeQuda((void *)gauge, &gauge_param);
    std::array<double, 3> plaq;
    plaqQuda(plaq.data());
    freeGaugeQuda();
    return plaq;
  };

  auto plaq_old = get_plaq();

  auto file = "dummy.lat";

  // write out the gauge field
  write_gauge_field(file, gauge, gauge_param.cpu_prec, gauge_param.X, 0, nullptr);

  // read it back
  read_gauge_field(file, gauge, gauge_param.cpu_prec, gauge_param.X, 0, nullptr);

  auto plaq_new = get_plaq();

  // test the plaquette is identical
  for (int i = 0; i < 3; i++) EXPECT_EQ(plaq_old[i], plaq_new[i]);

  // cleanup after ourselves and delete the dummy lattice
  if (::quda::comm_rank() == 0 && remove(file) != 0) errorQuda("Error deleting file");

  // release memory
  for (int dir = 0; dir < 4; dir++) { host_free(gauge[dir]); }
}

using cs_test_t = ::testing::tuple<QudaSiteSubset, bool, QudaPrecision, QudaPrecision, int, QudaFieldLocation>;

class ColorSpinorIOTest : public ::testing::TestWithParam<cs_test_t>
{
protected:
  QudaSiteSubset site_subset;
  bool inflate;
  QudaPrecision prec;
  QudaPrecision prec_io;
  int nSpin;
  QudaFieldLocation location;

public:
  ColorSpinorIOTest() :
    site_subset(::testing::get<0>(GetParam())),
    inflate(::testing::get<1>(GetParam())),
    prec(::testing::get<2>(GetParam())),
    prec_io(::testing::get<3>(GetParam())),
    nSpin(::testing::get<4>(GetParam())),
    location(::testing::get<5>(GetParam()))
  {
  }
};

constexpr double get_tolerance(QudaPrecision prec, QudaPrecision prec_io)
{
  // converting half precision field to float field and back doesn't
  // seem to be exactly area preserving
  if (prec == QUDA_HALF_PRECISION && prec_io > prec) return 3 * std::numeric_limits<float>::epsilon();
  switch (prec_io) {
  case QUDA_DOUBLE_PRECISION: return std::numeric_limits<double>::epsilon();
  case QUDA_SINGLE_PRECISION: return std::numeric_limits<float>::epsilon();
  default: return 0.0;
  }
}

TEST_P(ColorSpinorIOTest, verify)
{
  using namespace quda;
  if ((!is_enabled(prec)) || (!is_enabled_spin(nSpin))
      || (prec < QUDA_SINGLE_PRECISION && location == QUDA_CPU_FIELD_LOCATION)
      || (prec < QUDA_SINGLE_PRECISION && nSpin == 2))
    GTEST_SKIP();

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  ColorSpinorParam param;
  setWilsonGaugeParam(gauge_param);
  setInvertParam(inv_param);
  constructWilsonTestSpinorParam(&param, &inv_param, &gauge_param);
  param.siteSubset = site_subset;
  param.suggested_parity = QUDA_EVEN_PARITY;
  param.nSpin = nSpin;
  param.setPrecision(prec, prec, true); // change order to native order
  param.location = location;
  if (location == QUDA_CPU_FIELD_LOCATION) param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.create = QUDA_NULL_FIELD_CREATE;

  // create some random vectors
  auto n_vector = 1;
  std::vector<ColorSpinorField> v(n_vector, param);
  std::vector<ColorSpinorField> u(n_vector, param);

  RNG rng(v[0], 1234);
  for (auto &vi : v) spinorNoise(vi, rng, QUDA_NOISE_GAUSS);

  auto file = "dummy.cs";

  VectorIO io(file, inflate);

  io.save({v.begin(), v.end()}, prec_io, n_vector);
  io.load(u);

  for (auto i = 0u; i < v.size(); i++) {
    auto dev = blas::max_deviation(u[i], v[i]);
    if (prec == prec_io)
      EXPECT_EQ(dev[0], 0.0);
    else
      EXPECT_LE(dev[0], get_tolerance(prec, prec_io));
  }

  // cleanup after ourselves and delete the dummy lattice
  if (::quda::comm_rank() == 0 && remove(file) != 0) errorQuda("Error deleting file");
}

int main(int argc, char **argv)
{
  quda_test test("IO Test", argc, argv);
  test.init();
  return test.execute();
}

using ::testing::Combine;
using ::testing::Values;

// gauge IO test
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeIOTest, Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION)),
                         [](testing::TestParamInfo<gauge_test_t> param) {
                           return get_prec_str(::testing::get<0>(param.param));
                         });

// colorspinor full field IO test
INSTANTIATE_TEST_SUITE_P(Full, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(false),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<2>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<3>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<4>(param.param));
                           name += ::testing::get<5>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           return name;
                         });

// colorspinor parity field IO test
INSTANTIATE_TEST_SUITE_P(Parity, ColorSpinorIOTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET), Values(false, true),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           if (::testing::get<1>(param.param)) name += std::string("inflate_");
                           name += get_prec_str(::testing::get<2>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<3>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<4>(param.param));
                           name += ::testing::get<5>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           return name;
                         });
