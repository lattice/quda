#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <unistd.h>

#include <comm_quda.h>
#include <instantiate.h>
#include <color_spinor_field.h>
#include <io/lime.h>
#include <io/site_io.h>
#include <io/vector_io.h>
#include <blas_quda.h>
#include <quda.h>

#include "test.h"
#include "misc.h"
#include "io_test_utils.h"

namespace
{
  // file formats that vector fields can be written in and read back from in this build
  std::vector<QudaFileFormat> vector_formats()
  {
#ifdef HAVE_QIO
    return {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_QIO_EXTERNAL};
#else
    return {QUDA_FILE_FORMAT_QIO};
#endif
  }

  // parameters of vectors on the test lattice: full or single parity, on the host or the device, 4-d or 5-d
  quda::ColorSpinorParam vector_param(QudaSiteSubset subset, QudaParity parity, int nSpin, QudaPrecision prec,
                                      QudaFieldLocation location, int Ls = 1)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    QudaInvertParam inv_param = newQudaInvertParam();
    setWilsonGaugeParam(gauge_param);
    setInvertParam(inv_param);

    quda::ColorSpinorParam param;
    constructWilsonTestSpinorParam(&param, &inv_param, &gauge_param);
    param.siteSubset = subset;
    param.suggested_parity = parity; // ignored for full parity
    param.nSpin = nSpin;
    param.setPrecision(prec, prec, true); // change order to native order
    param.location = location;
    if (location == QUDA_CPU_FIELD_LOCATION) param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    if (subset == QUDA_PARITY_SITE_SUBSET) param.x[0] /= 2;
    if (Ls > 1) {
      param.nDim = 5;
      param.x[4] = Ls;
    }
    return param;
  }

  std::vector<quda::ColorSpinorField> random_vectors(int n, const quda::ColorSpinorParam &param)
  {
    std::vector<quda::ColorSpinorField> v(n, param);
    quda::RNG rng(v[0], 1234);
    for (auto &vi : v) quda::spinorNoise(vi, rng, QUDA_NOISE_GAUSS);
    return v;
  }

  std::vector<quda::ColorSpinorField> zero_vectors(int n, quda::ColorSpinorParam param)
  {
    param.create = QUDA_ZERO_FIELD_CREATE;
    return std::vector<quda::ColorSpinorField>(n, param);
  }

  void expect_same_vectors(const std::vector<quda::ColorSpinorField> &u, const std::vector<quda::ColorSpinorField> &v)
  {
    ASSERT_EQ(u.size(), v.size());
    for (auto i = 0u; i < v.size(); i++) EXPECT_EQ(quda::blas::max_deviation(u[i], v[i])[0], 0.0) << "vector " << i;
  }
} // namespace

using cs_test_t = ::testing::tuple<QudaSiteSubset, QudaSiteSubset, QudaParity, bool, QudaPrecision, QudaPrecision, int,
                                   bool, QudaFieldLocation, QudaFileFormat, QudaFileIOMode, int, int>;

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
  QudaFileFormat format;
  QudaFileIOMode io_mode;
  int n_vector;
  int Ls;

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
    location(::testing::get<8>(GetParam())),
    format(::testing::get<9>(GetParam())),
    io_mode(::testing::get<10>(GetParam())),
    n_vector(::testing::get<11>(GetParam())),
    Ls(::testing::get<12>(GetParam()))
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

  // only the external QIO backend writes partfiles so far
  if (partfile && format != QUDA_FILE_FORMAT_QIO_EXTERNAL) GTEST_SKIP();

  // There's no meaningful way to save a full parity spinor field and load a
  // single-parity field
  if (site_subset_saved == QUDA_FULL_SITE_SUBSET && site_subset_loaded == QUDA_PARITY_SITE_SUBSET) GTEST_SKIP();

  // Site subsets must be the same except for inflation tests
  if (site_subset_saved != site_subset_loaded && !inflate) GTEST_SKIP();

  // Unique test: saving an inflated single-parity field and loading the full-parity field
  bool mixed_inflated_full_test
    = (site_subset_saved == QUDA_PARITY_SITE_SUBSET && site_subset_loaded == QUDA_FULL_SITE_SUBSET && inflate);

  // We create a separate parameter struct for saved and loaded ColorSpinorFields
  auto param_save = vector_param(site_subset_saved, parity, nSpin, prec, location, Ls);
  auto param_load = vector_param(site_subset_loaded, parity, nSpin, prec, location, Ls);

  // create some random vectors
  auto v = random_vectors(n_vector, param_save);
  std::vector<ColorSpinorField> u(n_vector, param_load);

  auto file = "dummy.cs";

  auto io_param = file_io_param(format, io_mode);
  io_param.parity_inflate = inflate;
  io_param.partfile = partfile;

  // create a separate VectorIO for saving and loading
  // this lets us test saving a single-parity field with inflation
  // then loading the full parity field.
  VectorIO io_save(file, io_param);
  VectorIO io_load(file, io_param);

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

  remove_saved_file(file);
}

#ifdef HAVE_QIO
// how the vectors cover the lattice: full, a single parity stored on the half-x lattice, or a single parity
// inflated to the full lattice
enum class VectorSubset { FULL, EVEN, ODD, INFLATED };

const char *get_vector_subset_str(VectorSubset subset)
{
  switch (subset) {
  case VectorSubset::FULL: return "full";
  case VectorSubset::EVEN: return "even";
  case VectorSubset::ODD: return "odd";
  case VectorSubset::INFLATED: return "inflated";
  default: return "unknown";
  }
}

// tuple types: precision, number of spins, subset, I/O mode
using backend_test_t = ::testing::tuple<QudaPrecision, int, VectorSubset, QudaFileIOMode>;

class VectorIOBackendTest : public ::testing::TestWithParam<backend_test_t>
{
};

// the native and external QIO backends write the same file, apart from the date in the private record XML, and
// each reads the other's file
TEST_P(VectorIOBackendTest, compare)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  auto nSpin = ::testing::get<1>(GetParam());
  auto subset = ::testing::get<2>(GetParam());
  auto io_mode = ::testing::get<3>(GetParam());
  if (!is_enabled(prec) || !is_enabled_spin(nSpin)) GTEST_SKIP();

  const auto site_subset = subset == VectorSubset::FULL ? QUDA_FULL_SITE_SUBSET : QUDA_PARITY_SITE_SUBSET;
  const auto parity = subset == VectorSubset::ODD ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  const auto param = vector_param(site_subset, parity, nSpin, prec, QUDA_CPU_FIELD_LOCATION);
  const int n = 3;
  auto v = random_vectors(n, param);

  const std::string native_file = "native.cs", external_file = "external.cs";
  // serial I/O of a single file goes through rank 0 whatever the grid
  std::vector<std::array<int, 4>> grids = {{}};
  if (io_mode == QUDA_FILE_IO_PARALLEL) grids.push_back({1, 1, 1, 1});
  for (auto grid : grids) {
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
    auto native = file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, grid);
    auto external = file_io_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, io_mode, grid);
    native.parity_inflate = external.parity_inflate = subset == VectorSubset::INFLATED;

    VectorIO(native_file, native).save({v.begin(), v.end()}, prec, n);
    VectorIO(external_file, external).save({v.begin(), v.end()}, prec, n);
    if (comm_rank() == 0) expect_same_saved_file(native_file, external_file, false);

    for (auto [file, p] : {std::make_pair(native_file, external), std::make_pair(external_file, native)}) {
      SCOPED_TRACE("loading " + file);
      auto u = zero_vectors(n, param);
      VectorIO(file, p).load(u);
      expect_same_vectors(u, v);
    }
  }
  remove_saved_file(native_file);
  remove_saved_file(external_file);
}
#endif

// tuple types: file format, site subset
using vector_partition_test_t = ::testing::tuple<QudaFileFormat, QudaSiteSubset>;

class VectorPartitionTest : public ::testing::TestWithParam<vector_partition_test_t>
{
};

// the I/O partition grid changes which ranks access a single file, never the file or the loaded vectors
TEST_P(VectorPartitionTest, grids)
{
  using namespace quda;
  auto format = ::testing::get<0>(GetParam());
  auto subset = ::testing::get<1>(GetParam());
  const bool external = format == QUDA_FILE_FORMAT_QIO_EXTERNAL;
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;

  const auto param = vector_param(subset, QUDA_EVEN_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION);
  auto v = random_vectors(n, param);
  auto save
    = [&](const std::string &file, const FileIOParam &p) { VectorIO(file, p).save({v.begin(), v.end()}, prec, n); };
  auto expect_load = [&](const std::string &file, const FileIOParam &p) {
    auto u = zero_vectors(n, param);
    VectorIO(file, p).load(u);
    expect_same_vectors(u, v);
  };

  const std::string ref_file = "partition_ref.cs", serial_ref_file = "partition_serial_ref.cs";
  const std::string file = "partition.cs", serial_file = "partition_serial.cs", other_file = "partition_other.cs";
  save(ref_file, file_io_param(format, QUDA_FILE_IO_PARALLEL));
  save(serial_ref_file, file_io_param(format, QUDA_FILE_IO_SERIAL));

  const auto grids = partition_grids();
  for (auto grid : grids) {
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
    auto parallel = file_io_param(format, QUDA_FILE_IO_PARALLEL, grid);
    save(file, parallel);
    if (comm_rank() == 0) expect_same_saved_file(ref_file, file, false);
    expect_load(ref_file, parallel);

    // the native backend with the same grid writes the same file as the external one
    if (external) {
      save(other_file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL, grid));
      if (comm_rank() == 0) expect_same_saved_file(file, other_file, false);
    }
  }

  // serial I/O goes through rank 0 whatever the grid
  auto serial = file_io_param(format, QUDA_FILE_IO_SERIAL, grids.back());
  save(serial_file, serial);
  if (comm_rank() == 0) expect_same_saved_file(serial_ref_file, serial_file, false);
  expect_load(ref_file, serial);

  for (auto &f : {ref_file, serial_ref_file, file, serial_file}) remove_saved_file(f);
  if (external) remove_saved_file(other_file);

  // external partfiles: one volume per partition, named after its leader, in either mode
  if (external) {
    const std::string part_file = "partition_part.cs";
    for (auto grid : grids) {
      for (auto io_mode : {QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL}) {
        if (io_mode == QUDA_FILE_IO_SERIAL && grid != grids.back()) continue;
        SCOPED_TRACE("partfile, I/O partition grid " + grid_str(grid) + ", " + get_io_mode_str(io_mode));
        auto p = file_io_param(format, io_mode, grid);
        p.partfile = true;
        save(part_file, p);
        // on one rank QIO writes a single file
        if (comm_size() > 1) {
          const bool leader = io::Partitions(QUDA_FILE_IO_PARALLEL, grid).is_leader();
          EXPECT_EQ(access(volume_file(part_file, comm_rank()).c_str(), F_OK) == 0, leader);
        }
        expect_load(part_file, p);
        remove_saved_file(part_file);
      }
    }
  }
}

// full 5-d vectors, or single-parity 5-d vectors inflated to the full lattice
enum class FiveDCase { FULL, INFLATED_EVEN, INFLATED_ODD };

// tuple types: preconditioning type, field, file format
using five_d_test_t = ::testing::tuple<QudaPCType, FiveDCase, QudaFileFormat>;

class FiveDLayoutTest : public ::testing::TestWithParam<five_d_test_t>
{
};

namespace
{
  // the value of real k of slice s of vector i at the global 4-d coordinates g: different for every (i, s, g, k),
  // never zero, and exact in double precision
  double encoded_value(int i, int s, const std::array<int, 4> &g, int k, int Ls, int reals, const std::array<int, 4> &N)
  {
    size_t v = static_cast<size_t>(i) * Ls + s;
    for (int d = 3; d >= 0; d--) v = v * N[d] + g[d];
    return static_cast<double>(v * reals + k + 1);
  }

  // fill one parity of a 5-d double-precision host field, stored [s][4-d checkerboard site], with encoded values;
  // with 5-d preconditioning the parity includes s, so slice s holds the 4-d parity parity + s
  void fill_parity(quda::ColorSpinorField &f, int parity, int i, QudaPCType pc_type)
  {
    const int Ls = f.X(4), reals = 2 * f.Nspin() * f.Ncolor();
    const std::array<int, 4> X = {2 * f.X(0), f.X(1), f.X(2), f.X(3)};
    std::array<int, 4> N;
    for (int d = 0; d < 4; d++) N[d] = X[d] * quda::comm_dim(d);
    const size_t half_volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3] / 2;
    auto data = f.data<double *>();
    for (int s = 0; s < Ls; s++) {
      const int p4 = (parity + (pc_type == QUDA_5D_PC ? s : 0)) % 2;
      for (size_t cb = 0; cb < half_volume; cb++) {
        // QUDA's 4-d checkerboard order
        std::array<int, 4> x;
        size_t za = cb / (X[0] / 2);
        x[1] = za % X[1];
        x[2] = (za / X[1]) % X[2];
        x[3] = za / (static_cast<size_t>(X[1]) * X[2]);
        x[0] = 2 * (cb % (X[0] / 2)) + ((x[1] + x[2] + x[3] + p4) & 1);
        std::array<int, 4> g;
        for (int d = 0; d < 4; d++) g[d] = x[d] + quda::comm_coord(d) * X[d];
        for (int k = 0; k < reals; k++)
          data[(s * half_volume + cb) * reals + k] = encoded_value(i, s, g, k, Ls, reals, N);
      }
    }
  }
} // namespace

// a saved 5-d vector file holds, in item i * Ls + s of every site, slice s of vector i at that site, so that the file
// means what the format says
TEST_P(FiveDLayoutTest, file_contents)
{
  using namespace quda;
  auto pc_type = ::testing::get<0>(GetParam());
  auto kind = ::testing::get<1>(GetParam());
  auto format = ::testing::get<2>(GetParam());
  const bool inflated = kind != FiveDCase::FULL;
  const QudaParity parity = kind == FiveDCase::INFLATED_ODD ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  const int n = 2, Ls = 4, nSpin = 4;
  const int reals = 2 * nSpin * 3;

  auto param = vector_param(inflated ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET, parity, nSpin,
                            QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION, Ls);
  param.pc_type = pc_type;
  std::vector<ColorSpinorField> v(n, param);
  for (int i = 0; i < n; i++) {
    if (inflated) {
      fill_parity(v[i], parity == QUDA_ODD_PARITY, i, pc_type);
    } else {
      fill_parity(v[i].Even(), 0, i, pc_type);
      fill_parity(v[i].Odd(), 1, i, pc_type);
    }
  }

  const std::string file = "five_d.cs";
  auto p = file_io_param(format, QUDA_FILE_IO_PARALLEL);
  p.parity_inflate = inflated;
  VectorIO(file, p).save({v.begin(), v.end()}, QUDA_DOUBLE_PRECISION, n);

  // every rank takes part in reading the record list
  auto records = lime::read_records(file);
  if (comm_rank() == 0) {
    const lime::Record *data = nullptr;
    for (auto &r : records)
      if (r.type == "scidac-binary-data") data = &r;
    ASSERT_NE(data, nullptr) << "no binary data in " << file;

    std::array<int, 4> N;
    const auto X = v[0].X();
    for (int d = 0; d < 4; d++) N[d] = (d == 0 && inflated ? 2 * X[0] : X[d]) * comm_dim(d);
    const size_t sites = static_cast<size_t>(N[0]) * N[1] * N[2] * N[3];
    const size_t items = static_cast<size_t>(n) * Ls;
    ASSERT_EQ(data->data_bytes, sites * items * reals * sizeof(double));

    auto contents = read_file(file);
    ASSERT_GE(contents.size(), data->data_offset + data->data_bytes);
    const char *bytes = contents.data() + data->data_offset;
    size_t mismatches = 0;
    for (size_t site = 0; site < sites; site++) {
      std::array<int, 4> g;
      size_t r = site;
      for (int d = 0; d < 4; d++) {
        g[d] = r % N[d];
        r /= N[d];
      }
      const int site_parity = (g[0] + g[1] + g[2] + g[3]) % 2;
      for (int i = 0; i < n; i++) {
        for (int s = 0; s < Ls; s++) {
          // an inflated field holds the 4-d parity parity + s (5-d preconditioning) or parity; the rest is zero
          const int held = (parity == QUDA_ODD_PARITY) + (pc_type == QUDA_5D_PC ? s : 0);
          const bool present = !inflated || site_parity == held % 2;
          for (int k = 0; k < reals; k++) {
            uint64_t word;
            memcpy(&word, bytes + ((site * items + i * Ls + s) * reals + k) * sizeof(double), sizeof(word));
            word = __builtin_bswap64(word); // big endian
            double value;
            memcpy(&value, &word, sizeof(value));
            if (value != (present ? encoded_value(i, s, g, k, Ls, reals, N) : 0.0)) mismatches++;
          }
        }
      }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << sites * items * reals << " values in " << file;
  }

  // and it loads back as it was saved
  auto u = zero_vectors(n, param);
  VectorIO(file, p).load(u);
  expect_same_vectors(u, v);

  remove_saved_file(file);
}

int main(int argc, char **argv)
{
  quda_test test("Vector IO Test", argc, argv);
  test.init();
  return test.execute();
}

using ::testing::Combine;
using ::testing::Values;
using ::testing::ValuesIn;

namespace
{
  std::string cs_test_name(const testing::TestParamInfo<cs_test_t> &param)
  {
    std::string name;
    if (::testing::get<0>(param.param) == QUDA_PARITY_SITE_SUBSET) {
      name += ::testing::get<1>(param.param) == QUDA_PARITY_SITE_SUBSET ? "load_parity_" : "load_full_";
      name += ::testing::get<2>(param.param) == QUDA_EVEN_PARITY ? "even_" : "odd_";
      if (::testing::get<3>(param.param)) name += std::string("inflate_");
    }
    name += get_prec_str(::testing::get<4>(param.param)) + std::string("_");
    name += get_prec_str(::testing::get<5>(param.param)) + std::string("_");
    name += std::string("spin") + std::to_string(::testing::get<6>(param.param));
    name += ::testing::get<7>(param.param) ? "_partfile" : "_singlefile";
    name += ::testing::get<8>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
    name += std::string("_") + quda::get_file_format_str(::testing::get<9>(param.param));
    name += std::string("_") + get_io_mode_str(::testing::get<10>(param.param));
    name += std::string("_nvec") + std::to_string(::testing::get<11>(param.param));
    if (::testing::get<12>(param.param) > 1)
      name += std::string("_Ls") + std::to_string(::testing::get<12>(param.param));
    return name;
  }
} // namespace

// colorspinor full field IO test
INSTANTIATE_TEST_SUITE_P(Full, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_INVALID_PARITY), Values(false),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION),
                                 ValuesIn(vector_formats()), Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL),
                                 Values(1), Values(1)),
                         cs_test_name);

// colorspinor parity field IO test
INSTANTIATE_TEST_SUITE_P(Parity, ColorSpinorIOTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET), Values(QUDA_PARITY_SITE_SUBSET, QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_EVEN_PARITY, QUDA_ODD_PARITY), Values(false, true),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION),
                                 ValuesIn(vector_formats()), Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL),
                                 Values(1), Values(1)),
                         cs_test_name);

// several vectors interleaved at each site, in fewer combinations (VectorIOBackendTest and VectorPartitionTest also
// save several vectors, including single-parity ones)
INSTANTIATE_TEST_SUITE_P(Interleaved, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_INVALID_PARITY),
                                 Values(false), Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION), ValuesIn(vector_formats()),
                                 Values(QUDA_FILE_IO_PARALLEL), Values(3), Values(1)),
                         cs_test_name);

// AUTO detects the format on load and saves as QIO
INSTANTIATE_TEST_SUITE_P(Auto, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_INVALID_PARITY), Values(false), Values(QUDA_DOUBLE_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION), Values(4), Values(false), Values(QUDA_CPU_FIELD_LOCATION),
                                 Values(QUDA_FILE_FORMAT_AUTO), Values(QUDA_FILE_IO_PARALLEL), Values(3), Values(1)),
                         cs_test_name);

// 5-d fields, stored as Ls 4-d items per vector
INSTANTIATE_TEST_SUITE_P(FiveD, ColorSpinorIOTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET), Values(QUDA_PARITY_SITE_SUBSET, QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_EVEN_PARITY, QUDA_ODD_PARITY), Values(false, true),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(4), Values(false),
                                 Values(QUDA_CUDA_FIELD_LOCATION), ValuesIn(vector_formats()),
                                 Values(QUDA_FILE_IO_PARALLEL), Values(2), Values(4)),
                         cs_test_name);

INSTANTIATE_TEST_SUITE_P(FiveDFull, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_INVALID_PARITY),
                                 Values(false), Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(4), Values(false),
                                 Values(QUDA_CUDA_FIELD_LOCATION), ValuesIn(vector_formats()),
                                 Values(QUDA_FILE_IO_PARALLEL), Values(2), Values(4)),
                         cs_test_name);

#ifdef HAVE_QIO
// comparison of the native and external QIO backends
INSTANTIATE_TEST_SUITE_P(Vector, VectorIOBackendTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 4),
                                 Values(VectorSubset::FULL, VectorSubset::EVEN, VectorSubset::ODD, VectorSubset::INFLATED),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<backend_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<1>(param.param)) + "_";
                           name += get_vector_subset_str(::testing::get<2>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<3>(param.param));
                           return name;
                         });
#endif

// what the files of 5-d vectors hold
INSTANTIATE_TEST_SUITE_P(Vector, FiveDLayoutTest,
                         Combine(Values(QUDA_4D_PC, QUDA_5D_PC),
                                 Values(FiveDCase::FULL, FiveDCase::INFLATED_EVEN, FiveDCase::INFLATED_ODD),
                                 ValuesIn(vector_formats())),
                         [](testing::TestParamInfo<five_d_test_t> param) {
                           std::string name = ::testing::get<0>(param.param) == QUDA_5D_PC ? "pc5d_" : "pc4d_";
                           switch (::testing::get<1>(param.param)) {
                           case FiveDCase::FULL: name += "full_"; break;
                           case FiveDCase::INFLATED_EVEN: name += "inflated_even_"; break;
                           case FiveDCase::INFLATED_ODD: name += "inflated_odd_"; break;
                           }
                           return name + quda::get_file_format_str(::testing::get<2>(param.param));
                         });

// I/O partition grids
INSTANTIATE_TEST_SUITE_P(Vector, VectorPartitionTest,
                         Combine(ValuesIn(vector_formats()), Values(QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET)),
                         [](testing::TestParamInfo<vector_partition_test_t> param) {
                           std::string name = quda::get_file_format_str(::testing::get<0>(param.param));
                           name += ::testing::get<1>(param.param) == QUDA_FULL_SITE_SUBSET ? "_full" : "_parity";
                           return name;
                         });
