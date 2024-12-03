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
  gauge_param.cuda_prec_sloppy = gauge_param.cpu_prec;
  gauge_param.cuda_prec_precondition = gauge_param.cpu_prec;
  gauge_param.cuda_prec_eigensolver = gauge_param.cpu_prec;
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

using cs_test_t = ::testing::tuple<QudaSiteSubset, QudaSiteSubset, QudaParity, bool, QudaPrecision, QudaPrecision, int,
                                   bool, QudaFieldLocation>;

class ColorSpinorIOTest : public ::testing::TestWithParam<cs_test_t>
{
protected:
  QudaSiteSubset site_subset_saved;
  QudaSiteSubset site_subset_loaded;
  QudaParity parity;
  bool inflate;
  QudaPrecision prec;
  QudaPrecision prec_io;
  int nSpin;
  bool partfile;
  QudaFieldLocation location;

public:
  ColorSpinorIOTest() :
    site_subset_saved(::testing::get<0>(GetParam())),
    site_subset_loaded(::testing::get<1>(GetParam())),
    parity(::testing::get<2>(GetParam())),
    inflate(::testing::get<3>(GetParam())),
    prec(::testing::get<4>(GetParam())),
    prec_io(::testing::get<5>(GetParam())),
    nSpin(::testing::get<6>(GetParam())),
    partfile(::testing::get<7>(GetParam())),
    location(::testing::get<8>(GetParam()))
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

  // There's no meaningful way to save a full parity spinor field and load a
  // single-parity field
  if (site_subset_saved == QUDA_FULL_SITE_SUBSET && site_subset_loaded == QUDA_PARITY_SITE_SUBSET) GTEST_SKIP();

  // Site subsets must be the same except for inflation tests
  if (site_subset_saved != site_subset_loaded && !inflate) GTEST_SKIP();

  // Unique test: saving an inflated single-parity field and loading the full-parity field
  bool mixed_inflated_full_test
    = (site_subset_saved == QUDA_PARITY_SITE_SUBSET && site_subset_loaded == QUDA_FULL_SITE_SUBSET && inflate);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  setWilsonGaugeParam(gauge_param);
  setInvertParam(inv_param);

  // We create a separate parameter struct for saved and loaded ColorSpinorFields
  ColorSpinorParam param_save, param_load;

  // Saved
  constructWilsonTestSpinorParam(&param_save, &inv_param, &gauge_param);
  param_save.siteSubset = site_subset_saved;
  param_save.suggested_parity = parity; // ignored for full parity
  param_save.nSpin = nSpin;
  param_save.setPrecision(prec, prec, true); // change order to native order
  param_save.location = location;
  if (location == QUDA_CPU_FIELD_LOCATION) param_save.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param_save.create = QUDA_NULL_FIELD_CREATE;

  // Loaded
  param_load = param_save;
  param_load.siteSubset = site_subset_loaded;

  // Update ColorSpinorField size appropriately depending on the site subset
  if (site_subset_saved == QUDA_PARITY_SITE_SUBSET) param_save.x[0] /= 2;
  if (site_subset_loaded == QUDA_PARITY_SITE_SUBSET) param_load.x[0] /= 2;

  // create some random vectors
  auto n_vector = 1;
  std::vector<ColorSpinorField> v(n_vector, param_save);
  std::vector<ColorSpinorField> u(n_vector, param_load);

  RNG rng(v[0], 1234);
  for (auto &vi : v) spinorNoise(vi, rng, QUDA_NOISE_GAUSS);

  auto file = "dummy.cs";

  // create a separate VectorIO for saving and loading
  // this lets us test saving a single-parity field with inflation
  // then loading the full parity field.
  VectorIO io_save(file, inflate, partfile);
  VectorIO io_load(file, inflate, partfile);

  io_save.save({v.begin(), v.end()}, prec_io, n_vector);
  io_load.load(u);

  // This lambda simplifies returning the correct subset of `u` by reference if we're doing
  // tests where we save an inflated single-parity vector and load a full-parity vector.
  auto get_u_subset = [&](int i) -> const ColorSpinorField & {
    if (mixed_inflated_full_test)
      if (this->parity == QUDA_EVEN_PARITY) // lambdas don't automatically capture `this`
        return u[i].Even();
      else
        return u[i].Odd();
    else
      return u[i];
  };

  for (auto i = 0u; i < v.size(); i++) {
    // grab the correct subset of u if we're doing mixed parity/full tests
    const ColorSpinorField &u_subset = get_u_subset(i);

    auto dev = blas::max_deviation(u_subset, v[i]);
    if (prec == prec_io)
      EXPECT_EQ(dev[0], 0.0);
    else
      EXPECT_LE(dev[0], get_tolerance(prec, prec_io));
  }

  // cleanup after ourselves and delete the dummy lattice
  if (partfile && ::quda::comm_size() > 1) {
    // each rank created its own file, we need to generate the custom filename
    // an exception is single-rank runs where QIO skips appending the volume string
    char volstr[9];
    sprintf(volstr, ".vol%04d", ::quda::comm_rank());
    std::string part_filename = std::string(file) + volstr;
    if (remove(part_filename.c_str()) != 0) errorQuda("Error deleting file");
  } else {
    if (::quda::comm_rank() == 0 && remove(file) != 0) errorQuda("Error deleting file");
  }
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
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_INVALID_PARITY), Values(false),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<4>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<5>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<6>(param.param));
                           name += ::testing::get<7>(param.param) ? "_singlefile" : "_partfile";
                           name += ::testing::get<8>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           return name;
                         });

// colorspinor parity field IO test
INSTANTIATE_TEST_SUITE_P(Parity, ColorSpinorIOTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET), Values(QUDA_PARITY_SITE_SUBSET, QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_EVEN_PARITY, QUDA_ODD_PARITY), Values(false, true),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           name += ::testing::get<1>(param.param) == QUDA_PARITY_SITE_SUBSET ? "load_parity_" :
                                                                                               "load_full_";
                           name += ::testing::get<2>(param.param) == QUDA_EVEN_PARITY ? "even_" : "odd_";
                           if (::testing::get<3>(param.param)) name += std::string("inflate_");
                           name += get_prec_str(::testing::get<4>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<5>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<6>(param.param));
                           name += ::testing::get<7>(param.param) ? "_singlefile" : "_partfile";
                           name += ::testing::get<8>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           return name;
                         });
