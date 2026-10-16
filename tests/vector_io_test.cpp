#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <unistd.h>

#include <comm_quda.h>
#include <instantiate.h>
#include <color_spinor_field.h>
#include <io/lime.h>
#include <io/scidac_io.h>
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

  // expect the first vectors of v to be u, or, with a parity, that parity of them
  void expect_first_vectors(const std::vector<quda::ColorSpinorField> &u, const std::vector<quda::ColorSpinorField> &v,
                            QudaParity parity = QUDA_INVALID_PARITY)
  {
    ASSERT_LE(u.size(), v.size());
    for (auto i = 0u; i < u.size(); i++) {
      const auto &w = parity == QUDA_EVEN_PARITY ? v[i].Even() : parity == QUDA_ODD_PARITY ? v[i].Odd() : v[i];
      EXPECT_EQ(quda::blas::max_deviation(u[i], w)[0], 0.0) << "vector " << i;
    }
  }

  std::array<int, 4> lattice_coords(size_t site, const std::array<int, 4> &N)
  {
    std::array<int, 4> g;
    for (int d = 0; d < 4; d++) {
      g[d] = site % N[d];
      site /= N[d];
    }
    return g;
  }

  // the records of a LIME file, listed on the calling rank alone
  std::vector<quda::lime::Record> local_records(const std::string &file)
  {
    quda::io::File f(file, quda::io::File::Mode::READ);
    return quda::lime::list_records(f, file);
  }

  // the data of a record, in a file's contents, without its trailing NULs
  std::string record_string(const std::vector<char> &contents, const quda::lime::Record *r)
  {
    if (!r) return "";
    std::string s(contents.data() + r->data_offset, r->data_bytes);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
  }

  // the binary records of a SciDAC single file, as big-endian doubles, read on the calling rank alone
  std::vector<std::vector<double>> binary_records(const std::string &file)
  {
    const auto contents = read_file(file);
    std::vector<std::vector<double>> data;
    for (auto &r : local_records(file)) {
      if (r.type != quda::scidac::binary_data) continue;
      std::vector<double> values(r.data_bytes / sizeof(double));
      for (auto j = 0u; j < values.size(); j++) {
        uint64_t word;
        memcpy(&word, contents.data() + r.data_offset + j * sizeof(word), sizeof(word));
        word = __builtin_bswap64(word);
        memcpy(&values[j], &word, sizeof(word));
      }
      data.push_back(values);
    }
    return data;
  }

  // the user file XML of a SciDAC file, and the private and user record XML of each of its binary records, read on
  // the calling rank alone
  struct SavedStrings {
    std::string user_file;
    std::vector<std::string> private_record;
    std::vector<std::string> user_record;
  };

  SavedStrings saved_strings(const std::string &file)
  {
    using namespace quda::scidac;
    const auto contents = read_file(file);
    const auto records = local_records(file);
    SavedStrings s;
    s.user_file = record_string(contents, find_record(records, file_xml));
    for (auto &m : group_messages(records)) {
      if (!find_record(m, binary_data)) continue;
      s.private_record.push_back(record_string(contents, find_record(m, private_record_xml)));
      s.user_record.push_back(record_string(contents, find_record(m, record_xml)));
    }
    return s;
  }

  // the number of binary records in a SciDAC file, counted on the calling rank alone
  size_t count_binary_records(const std::string &file)
  {
    size_t n = 0;
    for (auto &r : local_records(file)) n += r.type == quda::scidac::binary_data;
    return n;
  }

  // eigenvalues and residuals to save with n vectors, using every bit of a double's mantissa
  std::vector<quda::complex_t> test_evals(int n)
  {
    std::vector<quda::complex_t> evals;
    for (int i = 0; i < n; i++) evals.emplace_back(1.0 / 3.0 + i * std::sqrt(2.0), 0.25 * i);
    return evals;
  }

  std::vector<quda::real_t> test_resids(int n)
  {
    std::vector<quda::real_t> resids;
    for (int i = 0; i < n; i++) resids.push_back((i + 1) / 3.0e11);
    return resids;
  }

  std::string format_value(const char *fmt, double value)
  {
    char buf[64];
    snprintf(buf, sizeof(buf), fmt, value);
    return buf;
  }

  // the loaded eigenvalues are the real parts of those saved
  void expect_loaded_evals(const std::vector<quda::complex_t> &loaded, const std::vector<quda::complex_t> &saved, int n)
  {
    ASSERT_GE(loaded.size(), static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
      EXPECT_EQ(static_cast<double>(loaded[i].real()), static_cast<double>(saved[i].real())) << "eigenvalue " << i;
      EXPECT_EQ(static_cast<double>(loaded[i].imag()), 0.0) << "eigenvalue " << i;
    }
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

  // There's no meaningful way to save a full parity spinor field and load a
  // single-parity field
  if (site_subset_saved == QUDA_FULL_SITE_SUBSET && site_subset_loaded == QUDA_PARITY_SITE_SUBSET) GTEST_SKIP();

  // Site subsets must be the same except for inflation tests
  if (site_subset_saved != site_subset_loaded && !inflate) GTEST_SKIP();

  // partfiles are saved the same way in either mode, and VectorPartfileTest loads them in serial mode
  if (partfile && io_mode == QUDA_FILE_IO_SERIAL) GTEST_SKIP();

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

#ifdef HAVE_QIO
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
}

namespace
{
  // expect two partfiles to be the same apart from the date in their first volume's metadata
  void expect_same_partfile(const std::string &a, const std::string &b)
  {
    auto u = saved_volumes(a), v = saved_volumes(b);
    ASSERT_EQ(u.size(), v.size()) << a << " and " << b << " have different numbers of volumes";
    for (auto i = 0u; i < u.size(); i++) {
      if (i == 0)
        expect_same_saved_file(u[i], v[i], false);
      else
        expect_same_file(u[i], v[i]);
    }
  }
} // namespace

// tuple types: site subset, I/O mode
using vector_partfile_test_t = ::testing::tuple<QudaSiteSubset, QudaFileIOMode>;

class VectorPartfileTest : public ::testing::TestWithParam<vector_partfile_test_t>
{
};

// partfiles have one volume per I/O partition, named after its leader, in either mode, and load with any grid; the
// native and external backends save the same volumes, and each loads the other's partfiles
TEST_P(VectorPartfileTest, grids)
{
  using namespace quda;
  auto subset = ::testing::get<0>(GetParam());
  auto io_mode = ::testing::get<1>(GetParam());
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

  const std::string file = "partfile.cs";
  const auto grids = partition_grids();
  for (auto i = 0u; i < grids.size(); i++) {
    const auto &grid = grids[i];
    const auto &next_grid = grids[(i + 1) % grids.size()];
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
    save(file, partfile_param(QUDA_FILE_FORMAT_QIO, io_mode, grid));
    const bool single = comm_size() == 1;
    const auto volumes = single ? std::vector<std::string>() : expected_volumes(file, grid);
    if (comm_rank() == 0) {
      EXPECT_EQ(access(file.c_str(), F_OK) == 0, single) << file;
      EXPECT_EQ(saved_volumes(file), volumes);
    }

    // serial loads go through rank 0 whatever the grid
    std::vector<std::array<int, 4>> load_grids = {grid};
    if (io_mode == QUDA_FILE_IO_PARALLEL)
      for (auto g : {std::array<int, 4> {}, std::array<int, 4> {1, 1, 1, 1}, next_grid})
        if (std::find(load_grids.begin(), load_grids.end(), g) == load_grids.end()) load_grids.push_back(g);
    for (auto &g : load_grids) {
      SCOPED_TRACE("loading with I/O partition grid " + grid_str(g));
      expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, g));
    }

#ifdef HAVE_QIO
    const std::string external_file = "partfile_external.cs";
    save(external_file, partfile_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, io_mode, grid));
    if (comm_rank() == 0) {
      if (comm_size() == 1)
        expect_same_saved_file(file, external_file, false);
      else
        expect_same_partfile(file, external_file);
    }
    {
      SCOPED_TRACE("native load of the external partfile with I/O partition grid " + grid_str(next_grid));
      expect_load(external_file, file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, next_grid));
    }
    {
      // QIO reads partfiles only with the rank layout and grid that saved them
      SCOPED_TRACE("external load of the native partfile");
      expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, io_mode, grid));
    }
    remove_saved_file(external_file);
#endif
    remove_saved_file(file);
  }
}

// tuple types: site subset, the grid of volumes
using vector_layout_test_t = ::testing::tuple<QudaSiteSubset, std::array<int, 4>>;

class VectorPartfileLayoutTest : public ::testing::TestWithParam<vector_layout_test_t>
{
};

// partfiles saved as if by another rank layout, whose volumes need not line up with this layout's ranks, load with
// this layout and any grid
TEST_P(VectorPartfileLayoutTest, load)
{
  using namespace quda;
  auto subset = ::testing::get<0>(GetParam());
  auto volumes = ::testing::get<1>(GetParam());
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;

  const auto param = vector_param(subset, QUDA_ODD_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION);
  auto v = random_vectors(n, param);
  auto save
    = [&](const std::string &file, const FileIOParam &p) { VectorIO(file, p).save({v.begin(), v.end()}, prec, n); };

  const std::string file = "partfile_layout.cs", serial_file = "partfile_layout_serial.cs";
  auto p = partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  p.partfile_grid = volumes;
  save(file, p);
  p.io_mode = QUDA_FILE_IO_SERIAL;
  save(serial_file, p);
  if (comm_rank() == 0) {
    std::vector<std::string> expected;
    for (int i = 0; i < volumes[0] * volumes[1] * volumes[2] * volumes[3]; i++)
      expected.push_back(volume_file(file, i));
    EXPECT_EQ(saved_volumes(file), expected);
    expect_same_partfile(file, serial_file);
  }

  for (auto io_mode : {QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL}) {
    for (auto grid : {std::array<int, 4> {}, std::array<int, 4> {1, 1, 1, 1}, partition_grids().back()}) {
      if (io_mode == QUDA_FILE_IO_SERIAL && grid != std::array<int, 4> {}) continue;
      SCOPED_TRACE(std::string("loading in ") + get_io_mode_str(io_mode) + " mode with I/O partition grid "
                   + grid_str(grid));
      auto u = zero_vectors(n, param);
      VectorIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, grid)).load(u);
      expect_same_vectors(u, v);
    }
  }
  remove_saved_file(file);
  remove_saved_file(serial_file);
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

  // fill one parity of a 4-d or 5-d double-precision host field, stored [s][4-d checkerboard site], with encoded
  // values; with 5-d preconditioning the parity includes s, so slice s holds the 4-d parity parity + s
  void fill_parity(quda::ColorSpinorField &f, int parity, int i, QudaPCType pc_type)
  {
    const int Ls = f.Ndim() == 5 ? f.X(4) : 1, reals = 2 * f.Nspin() * f.Ncolor();
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

// a saved 5-d vector file holds, in item s of every site of record i, slice s of vector i at that site, so that the
// file means what the format says
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

  if (comm_rank() == 0) {
    const auto data = binary_records(file);
    ASSERT_EQ(data.size(), static_cast<size_t>(n)) << "binary records in " << file;

    std::array<int, 4> N;
    const auto X = v[0].X();
    for (int d = 0; d < 4; d++) N[d] = (d == 0 && inflated ? 2 * X[0] : X[d]) * comm_dim(d);
    const size_t sites = static_cast<size_t>(N[0]) * N[1] * N[2] * N[3];
    size_t mismatches = 0;
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(data[i].size(), sites * Ls * reals) << "values in binary record " << i << " of " << file;
      for (size_t site = 0; site < sites; site++) {
        const auto g = lattice_coords(site, N);
        const int site_parity = (g[0] + g[1] + g[2] + g[3]) % 2;
        for (int s = 0; s < Ls; s++) {
          // an inflated field holds the 4-d parity parity + s (5-d preconditioning) or parity; the rest is zero
          const int held = (parity == QUDA_ODD_PARITY) + (pc_type == QUDA_5D_PC ? s : 0);
          const bool present = !inflated || site_parity == held % 2;
          for (int k = 0; k < reals; k++)
            if (data[i][(site * Ls + s) * reals + k] != (present ? encoded_value(i, s, g, k, Ls, reals, N) : 0.0))
              mismatches++;
        }
      }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << n * sites * Ls * reals << " values in " << file;
  }

  // and it loads back as it was saved
  auto u = zero_vectors(n, param);
  VectorIO(file, p).load(u);
  expect_same_vectors(u, v);

  remove_saved_file(file);
}

namespace
{
  // PartfileCrossRun's options
  std::string cross_run_save;
  std::string cross_run_load;
  bool cross_run_external = false;
  std::array<int, 4> cross_run_volumes = {};
  std::string cross_run_layout = "quda";

  // 4-d vectors whose values depend only on the global site, the vector and the component
  std::vector<quda::ColorSpinorField> encoded_vectors(int n, const quda::ColorSpinorParam &param)
  {
    std::vector<quda::ColorSpinorField> v(n, param);
    for (int i = 0; i < n; i++) {
      if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
        fill_parity(v[i], param.suggested_parity == QUDA_ODD_PARITY, i, QUDA_4D_PC);
      } else {
        fill_parity(v[i].Even(), 0, i, QUDA_4D_PC);
        fill_parity(v[i].Odd(), 1, i, QUDA_4D_PC);
      }
    }
    return v;
  }
} // namespace

// tuple types: site subset, partfile, number of vectors
using interleaved_test_t = ::testing::tuple<QudaSiteSubset, bool, int>;

class InterleavedFileTest : public ::testing::TestWithParam<interleaved_test_t>
{
};

// files of QUDA's layout before one record per vector, with all the vectors in one record (saved by the test-only
// interleaved writer), load with both backends, also into fewer vectors
TEST_P(InterleavedFileTest, load)
{
  using namespace quda;
  auto subset = ::testing::get<0>(GetParam());
  auto partfile = ::testing::get<1>(GetParam());
  auto n = ::testing::get<2>(GetParam());
  const auto prec = QUDA_DOUBLE_PRECISION;
  const auto param = vector_param(subset, QUDA_EVEN_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION);
  auto v = random_vectors(n, param);

  const std::string file = "interleaved.cs";
  auto p = partfile ? partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL) :
                      file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  p.interleaved_vectors = true;
  VectorIO(file, p).save({v.begin(), v.end()}, prec, n);
  if (comm_rank() == 0) {
    const bool volumes = partfile && comm_size() > 1;
    const auto s = saved_strings(volumes ? volume_file(file, 0) : file);
    EXPECT_EQ(s.user_file, "Dummy user file XML");
    ASSERT_EQ(s.private_record.size(), 1u);
    EXPECT_EQ(scidac::xml_value(s.private_record[0], "datacount").value_or(""), std::to_string(n));
    for (auto &volume : saved_volumes(file)) EXPECT_EQ(count_binary_records(volume), 1u) << volume;
  }

  for (auto format : vector_formats()) {
    for (int count : {n, 1}) {
      SCOPED_TRACE(std::string("loading ") + std::to_string(count) + " vectors with " + get_file_format_str(format));
      auto u = zero_vectors(count, param);
      const complex_t unset(-1.0, -1.0);
      std::vector<complex_t> evals(count, unset);
      VectorIO(file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(u, &evals);
      expect_first_vectors(u, v);
      // such files have no eigenvalues
      for (auto &e : evals) EXPECT_TRUE(e == unset);
    }
  }
  remove_saved_file(file);
}

// tuple types: site subset, partfile
using eigenvalue_test_t = ::testing::tuple<QudaSiteSubset, bool>;

class VectorEigenvalueTest : public ::testing::TestWithParam<eigenvalue_test_t>
{
};

// QUDA's files hold an eigenvalue and a residual in each vector's record when they are given, and loads give the
// eigenvalues back
TEST_P(VectorEigenvalueTest, records)
{
  using namespace quda;
  auto subset = ::testing::get<0>(GetParam());
  auto partfile = ::testing::get<1>(GetParam());
  if (!is_enabled_spin(4)) GTEST_SKIP();
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;
  const auto param = vector_param(subset, QUDA_ODD_PARITY, 4, prec, QUDA_CPU_FIELD_LOCATION);
  auto v = random_vectors(n, param);
  const auto evals = test_evals(n);
  const auto resids = test_resids(n);
  const bool volumes = partfile && comm_size() > 1;

  std::vector<std::string> files;
  for (auto format : vector_formats()) {
    const std::string file = std::string("evals_") + get_file_format_str(format) + ".cs";
    files.push_back(file);
    auto p = partfile ? partfile_param(format, QUDA_FILE_IO_PARALLEL) : file_io_param(format, QUDA_FILE_IO_PARALLEL);
    VectorIO(file, p).save({v.begin(), v.end()}, prec, n, &evals, &resids);
    if (comm_rank() == 0) {
      SCOPED_TRACE(file);
      const auto s = saved_strings(volumes ? volume_file(file, 0) : file);
      EXPECT_EQ(s.user_file, "Dummy user file XML");
      ASSERT_EQ(s.private_record.size(), static_cast<size_t>(n));
      for (int i = 0; i < n; i++) {
        EXPECT_EQ(scidac::xml_value(s.private_record[i], "datatype").value_or(""), "QUDA_DNs4Nc3_ColorSpinorField");
        EXPECT_EQ(scidac::xml_value(s.private_record[i], "datacount").value_or(""), "1");
        EXPECT_EQ(scidac::xml_value(s.user_record[i], "eigVal").value_or(""),
                  format_value("%.18e", static_cast<double>(evals[i].real())));
        EXPECT_EQ(scidac::xml_value(s.user_record[i], "resid").value_or(""),
                  format_value("%e", static_cast<double>(resids[i])));
      }
      for (auto &volume : saved_volumes(file)) EXPECT_EQ(count_binary_records(volume), static_cast<size_t>(n));
    }
  }
#ifdef HAVE_QIO
  if (comm_rank() == 0) {
    if (volumes)
      expect_same_partfile(files[0], files[1]);
    else
      expect_same_saved_file(files[0], files[1], false);
  }
#endif

  for (auto &file : files) {
    for (auto format : vector_formats()) {
      SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format));
      auto u = zero_vectors(n, param);
      std::vector<complex_t> loaded;
      VectorIO(file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(u, &loaded);
      expect_same_vectors(u, v);
      expect_loaded_evals(loaded, evals, n);
    }
  }

  // without eigenvalues the records have none, and loads leave the eigenvalues alone
  const std::string plain = "evals_none.cs";
  VectorIO(plain, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL)).save({v.begin(), v.end()}, prec, n);
  if (comm_rank() == 0) {
    for (auto &xml : saved_strings(plain).user_record) EXPECT_FALSE(scidac::xml_value(xml, "eigVal")) << xml;
  }
  {
    auto u = zero_vectors(n, param);
    const complex_t unset(-1.0, -1.0);
    std::vector<complex_t> loaded(n, unset);
    VectorIO(plain, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL)).load(u, &loaded);
    for (auto &e : loaded) EXPECT_TRUE(e == unset);
  }

  for (auto &file : files) remove_saved_file(file);
  remove_saved_file(plain);
}

namespace
{
  // the metadata of MILC's eigenvector files, written out here rather than with the library's builders
  void expect_milc_strings(const std::string &file, int n, QudaPrecision prec,
                           const std::vector<quda::complex_t> &evals, const std::vector<quda::real_t> &resids)
  {
    using quda::scidac::xml_value;
    const auto s = saved_strings(file);
    EXPECT_EQ(s.user_file,
              "<?xml version=\"1.0\" encoding=\"UTF-8\"?><info><title>KS eigenvalues and vectors</title><Nvecs>"
                + std::to_string(n) + "</Nvecs></info>");
    ASSERT_EQ(s.private_record.size(), static_cast<size_t>(n));
    const std::string p = prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
    for (int i = 0; i < n; i++) {
      const auto &r = s.private_record[i];
      EXPECT_EQ(xml_value(r, "recordtype").value_or(""), "0");
      EXPECT_EQ(xml_value(r, "datatype").value_or(""), "USQCD_" + p + "3_ColorVector");
      EXPECT_EQ(xml_value(r, "precision").value_or(""), p);
      EXPECT_EQ(xml_value(r, "colors").value_or(""), "3");
      EXPECT_FALSE(xml_value(r, "spins")) << r;
      EXPECT_EQ(xml_value(r, "typesize").value_or(""), std::to_string(6 * prec));
      EXPECT_EQ(xml_value(r, "datacount").value_or(""), "1");
      EXPECT_EQ(s.user_record[i],
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><info><eigVal>"
                  + format_value("%.18e", static_cast<double>(evals[i].real())) + "</eigVal><resid>"
                  + format_value("%e", static_cast<double>(resids[i])) + "</resid></info>");
    }
  }

  // the data of a MILC eigenvector file of encoded vectors: every site of the full lattice, with the other parity
  // zero for single-parity vectors
  void expect_milc_data(const std::string &file, const quda::ColorSpinorField &v0, int n)
  {
    const bool single_parity = v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET;
    const int held = v0.SuggestedParity() == QUDA_ODD_PARITY;
    std::array<int, 4> N;
    for (int d = 0; d < 4; d++) N[d] = (d == 0 && single_parity ? 2 : 1) * v0.X(d) * quda::comm_dim(d);
    const size_t sites = static_cast<size_t>(N[0]) * N[1] * N[2] * N[3];
    const auto data = binary_records(file);
    ASSERT_EQ(data.size(), static_cast<size_t>(n)) << "binary records in " << file;
    size_t mismatches = 0;
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(data[i].size(), sites * 6) << "values in binary record " << i << " of " << file;
      for (size_t site = 0; site < sites; site++) {
        const auto g = lattice_coords(site, N);
        const bool present = !single_parity || (g[0] + g[1] + g[2] + g[3]) % 2 == held;
        for (int k = 0; k < 6; k++)
          if (data[i][site * 6 + k] != (present ? encoded_value(i, 0, g, k, 1, 6, N) : 0.0)) mismatches++;
      }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << n * sites * 6 << " values in " << file;
  }
} // namespace

// tuple types: subset, partfile, I/O mode
using milc_test_t = ::testing::tuple<VectorSubset, bool, QudaFileIOMode>;

class MILCEigenvectorTest : public ::testing::TestWithParam<milc_test_t>
{
};

// MILC's eigenvector format: what the files hold, the same files from both backends, and loads into full fields,
// each parity and fewer vectors, with the eigenvalues
TEST_P(MILCEigenvectorTest, save_load)
{
  using namespace quda;
  auto kind = ::testing::get<0>(GetParam());
  auto partfile = ::testing::get<1>(GetParam());
  auto io_mode = ::testing::get<2>(GetParam());
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const bool full = kind == VectorSubset::FULL;
  const auto parity = kind == VectorSubset::ODD ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;
  const auto param
    = vector_param(full ? QUDA_FULL_SITE_SUBSET : QUDA_PARITY_SITE_SUBSET, parity, 1, prec, QUDA_CPU_FIELD_LOCATION);
  const auto v = encoded_vectors(n, param);
  const auto evals = test_evals(n);
  const auto resids = test_resids(n);
  const bool volumes = partfile && comm_size() > 1;

  // MILC's files are on the full lattice, whether parity_inflate is set or not
  std::vector<std::string> files;
  for (auto format : vector_formats()) {
    const std::string file = std::string("milc_") + get_file_format_str(format) + ".cs";
    files.push_back(file);
    auto p = partfile ? partfile_param(format, io_mode) : file_io_param(format, io_mode);
    p.milc_eigenvectors = true;
    p.parity_inflate = format == QUDA_FILE_FORMAT_QIO_EXTERNAL;
    VectorIO(file, p).save({v.begin(), v.end()}, prec, n, &evals, &resids);
    if (comm_rank() == 0) {
      SCOPED_TRACE(file);
      expect_milc_strings(volumes ? volume_file(file, 0) : file, n, prec, evals, resids);
      if (volumes) {
        for (auto &volume : saved_volumes(file)) EXPECT_EQ(count_binary_records(volume), static_cast<size_t>(n));
      } else {
        expect_milc_data(file, v[0], n);
      }
    }
  }
#ifdef HAVE_QIO
  if (comm_rank() == 0) {
    if (volumes)
      expect_same_partfile(files[0], files[1]);
    else
      expect_same_saved_file(files[0], files[1], false);
  }
#endif

  for (auto &file : files) {
    for (auto format : vector_formats()) {
      SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format));
      auto lp = file_io_param(format, io_mode);
      lp.parity_inflate = format == QUDA_FILE_FORMAT_QIO_EXTERNAL;
      {
        auto u = zero_vectors(n, param);
        std::vector<complex_t> loaded;
        VectorIO(file, lp).load(u, &loaded);
        expect_same_vectors(u, v);
        expect_loaded_evals(loaded, evals, n);
      }
      {
        auto u = zero_vectors(n - 1, param);
        VectorIO(file, lp).load(u);
        expect_first_vectors(u, v);
      }
      if (full) {
        for (auto p : {QUDA_EVEN_PARITY, QUDA_ODD_PARITY}) {
          SCOPED_TRACE(p == QUDA_EVEN_PARITY ? "into even-parity vectors" : "into odd-parity vectors");
          auto u = zero_vectors(n, vector_param(QUDA_PARITY_SITE_SUBSET, p, 1, prec, QUDA_CPU_FIELD_LOCATION));
          VectorIO(file, lp).load(u);
          expect_first_vectors(u, v, p);
        }
      }
    }

    // natively, partfiles load with any grid
    if (partfile && io_mode == QUDA_FILE_IO_PARALLEL) {
      for (auto grid : {std::array<int, 4> {1, 1, 1, 1}, partition_grids().back()}) {
        SCOPED_TRACE("loading " + file + " natively with I/O partition grid " + grid_str(grid));
        auto u = zero_vectors(n, param);
        VectorIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, grid)).load(u);
        expect_same_vectors(u, v);
      }
    }
  }
  for (auto &file : files) remove_saved_file(file);
}

class MILCEigenvectorLayoutTest : public ::testing::TestWithParam<VectorSubset>
{
};

// MILC partfiles saved as if by another rank layout load with this layout and any grid
TEST_P(MILCEigenvectorLayoutTest, load)
{
  using namespace quda;
  auto kind = GetParam();
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const bool full = kind == VectorSubset::FULL;
  const auto parity = kind == VectorSubset::ODD ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;
  const auto param
    = vector_param(full ? QUDA_FULL_SITE_SUBSET : QUDA_PARITY_SITE_SUBSET, parity, 1, prec, QUDA_CPU_FIELD_LOCATION);
  const auto v = encoded_vectors(n, param);
  const auto evals = test_evals(n);
  const auto resids = test_resids(n);

  // the volumes divide the full lattice of every layout run_io_test.sh uses (4 6 8 10 per rank)
  const std::array<int, 4> grid = {2, 3, 2, 2};
  const std::string file = "milc_layout.cs";
  auto p = partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  p.milc_eigenvectors = true;
  p.partfile_grid = grid;
  VectorIO(file, p).save({v.begin(), v.end()}, prec, n, &evals, &resids);
  if (comm_rank() == 0) {
    const auto saved = saved_volumes(file);
    EXPECT_EQ(saved.size(), static_cast<size_t>(grid[0] * grid[1] * grid[2] * grid[3]));
    for (auto &volume : saved) EXPECT_EQ(count_binary_records(volume), static_cast<size_t>(n));
    expect_milc_strings(volume_file(file, 0), n, prec, evals, resids);
  }

  for (auto io_mode : {QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL}) {
    for (auto g : {std::array<int, 4> {}, std::array<int, 4> {1, 1, 1, 1}, partition_grids().back()}) {
      if (io_mode == QUDA_FILE_IO_SERIAL && g != std::array<int, 4> {}) continue;
      SCOPED_TRACE(std::string("loading in ") + get_io_mode_str(io_mode) + " mode with I/O partition grid " + grid_str(g));
      auto u = zero_vectors(n, param);
      std::vector<complex_t> loaded;
      VectorIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, g)).load(u, &loaded);
      expect_same_vectors(u, v);
      expect_loaded_evals(loaded, evals, n);
    }
  }
  remove_saved_file(file);
}

// MILC's eigenvector format in single precision
TEST(MILCEigenvectorPrecisionTest, single)
{
  using namespace quda;
  if (!is_enabled(QUDA_SINGLE_PRECISION) || !is_enabled_spin(1)) GTEST_SKIP();
  const auto prec = QUDA_SINGLE_PRECISION;
  const int n = 3;
  const auto param = vector_param(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION);
  auto v = random_vectors(n, param);
  const auto evals = test_evals(n);
  const auto resids = test_resids(n);

  std::vector<std::string> files;
  for (auto format : vector_formats()) {
    const std::string file = std::string("milc_single_") + get_file_format_str(format) + ".cs";
    files.push_back(file);
    auto p = file_io_param(format, QUDA_FILE_IO_PARALLEL);
    p.milc_eigenvectors = true;
    VectorIO(file, p).save({v.begin(), v.end()}, prec, n, &evals, &resids);
    if (comm_rank() == 0) expect_milc_strings(file, n, prec, evals, resids);
  }
#ifdef HAVE_QIO
  if (comm_rank() == 0) expect_same_saved_file(files[0], files[1], false);
#endif
  for (auto &file : files) {
    for (auto format : vector_formats()) {
      SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format));
      auto u = zero_vectors(n, param);
      std::vector<complex_t> loaded;
      VectorIO(file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(u, &loaded);
      expect_same_vectors(u, v);
      expect_loaded_evals(loaded, evals, n);
    }
  }
  for (auto &file : files) remove_saved_file(file);
}

// partfiles across runs with different rank layouts (run_partfile_cross.sh): --partfile-save saves full and
// single-parity vectors to <prefix>_full and <prefix>_parity, and --partfile-load loads them; --partfile-layout
// chooses QUDA's files, MILC's eigenvector format or QUDA's old interleaved layout
TEST(PartfileCrossRun, vectors)
{
  using namespace quda;
  if (cross_run_save.empty() && cross_run_load.empty()) GTEST_SKIP();
  const bool milc = cross_run_layout == "milc", interleaved = cross_run_layout == "interleaved";
  ASSERT_TRUE(milc || interleaved || cross_run_layout == "quda") << "unknown layout " << cross_run_layout;
  const auto prec = QUDA_DOUBLE_PRECISION;
  const int n = 3;
  const auto evals = test_evals(n);
  const auto resids = test_resids(n);
  for (auto subset : {QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET}) {
    const auto param = vector_param(subset, QUDA_EVEN_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION);
    const auto v = encoded_vectors(n, param);
    const std::string suffix = subset == QUDA_FULL_SITE_SUBSET ? "_full" : "_parity";

    if (!cross_run_save.empty()) {
      auto p = partfile_param(cross_run_external ? QUDA_FILE_FORMAT_QIO_EXTERNAL : QUDA_FILE_FORMAT_QIO,
                              QUDA_FILE_IO_PARALLEL);
      p.partfile_grid = cross_run_volumes;
      p.milc_eigenvectors = milc;
      p.interleaved_vectors = interleaved;
      VectorIO(cross_run_save + suffix, p).save({v.begin(), v.end()}, prec, n, &evals, &resids);
    }

    if (!cross_run_load.empty()) {
      for (auto io_mode : {QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL}) {
        for (auto grid : {std::array<int, 4> {}, std::array<int, 4> {1, 1, 1, 1}}) {
          if (io_mode == QUDA_FILE_IO_SERIAL && grid != std::array<int, 4> {}) continue;
          SCOPED_TRACE(cross_run_load + suffix + ", " + get_io_mode_str(io_mode) + " mode, I/O partition grid "
                       + grid_str(grid));
          const auto p = file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, grid);
          auto u = zero_vectors(n, param);
          std::vector<complex_t> loaded;
          VectorIO(cross_run_load + suffix, p).load(u, &loaded);
          expect_same_vectors(u, v);
          if (!interleaved) expect_loaded_evals(loaded, evals, n);
          if (interleaved) {
            auto w = zero_vectors(1, param);
            VectorIO(cross_run_load + suffix, p).load(w);
            expect_first_vectors(w, v);
          }
          if (milc && subset == QUDA_FULL_SITE_SUBSET) {
            auto w
              = zero_vectors(n, vector_param(QUDA_PARITY_SITE_SUBSET, QUDA_ODD_PARITY, 1, prec, QUDA_CPU_FIELD_LOCATION));
            VectorIO(cross_run_load + suffix, p).load(w);
            expect_first_vectors(w, v, QUDA_ODD_PARITY);
          }
        }
      }
    }
  }
}

struct vector_io_test : quda_test {
  using quda_test::quda_test;

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    auto group = app->add_option_group("Partfile cross-run");
    group->add_option("--partfile-save", cross_run_save,
                      "Save partfiles of vectors set by their global sites to <prefix>_full and <prefix>_parity "
                      "(PartfileCrossRun)");
    group->add_option("--partfile-load", cross_run_load,
                      "Load and check the vectors of --partfile-save, natively (PartfileCrossRun)");
    group->add_flag("--partfile-save-external", cross_run_external,
                    "Save with the external QIO backend (PartfileCrossRun)");
    group
      ->add_option("--partfile-volumes", cross_run_volumes,
                   "Save natively as if on a process grid of this shape, one volume per rank (PartfileCrossRun; "
                   "default 0 0 0 0, the real layout)")
      ->expected(4);
    group->add_option("--partfile-layout", cross_run_layout,
                      "quda (default), milc (MILC's eigenvector format) or interleaved (QUDA's old layout, saved "
                      "natively) (PartfileCrossRun)");
  }
};

int main(int argc, char **argv)
{
  vector_io_test test("Vector IO Test", argc, argv);
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

// several vectors, a record each, in fewer combinations (VectorIOBackendTest and VectorPartitionTest also save several
// vectors, including single-parity ones)
INSTANTIATE_TEST_SUITE_P(Several, ColorSpinorIOTest,
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

// partfiles with I/O partition grids
INSTANTIATE_TEST_SUITE_P(Vector, VectorPartfileTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<vector_partfile_test_t> param) {
                           std::string name
                             = ::testing::get<0>(param.param) == QUDA_FULL_SITE_SUBSET ? "full_" : "parity_";
                           return name + get_io_mode_str(::testing::get<1>(param.param));
                         });

// partfiles saved as if by other rank layouts, with volumes that divide the stored local lattice of every layout
// run_io_test.sh uses (4 6 8 10 per rank, half the x extent for a single parity)
INSTANTIATE_TEST_SUITE_P(Vector, VectorPartfileLayoutTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET),
                                 Values(std::array<int, 4> {1, 1, 1, 1}, std::array<int, 4> {1, 3, 1, 5},
                                        std::array<int, 4> {2, 1, 4, 1}, std::array<int, 4> {2, 3, 2, 2})),
                         [](testing::TestParamInfo<vector_layout_test_t> param) {
                           std::string name = ::testing::get<0>(param.param) == QUDA_FULL_SITE_SUBSET ? "full" : "parity";
                           for (auto n : ::testing::get<1>(param.param)) name += "_" + std::to_string(n);
                           return name;
                         });

// files of QUDA's old layout, all vectors in one record
INSTANTIATE_TEST_SUITE_P(Vector, InterleavedFileTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET), Values(false, true),
                                 Values(2, 3)),
                         [](testing::TestParamInfo<interleaved_test_t> param) {
                           std::string name = ::testing::get<0>(param.param) == QUDA_FULL_SITE_SUBSET ? "full" : "parity";
                           name += ::testing::get<1>(param.param) ? "_partfile" : "_singlefile";
                           return name + "_nvec" + std::to_string(::testing::get<2>(param.param));
                         });

// eigenvalues and residuals in QUDA's files
INSTANTIATE_TEST_SUITE_P(Vector, VectorEigenvalueTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET, QUDA_PARITY_SITE_SUBSET), Values(false, true)),
                         [](testing::TestParamInfo<eigenvalue_test_t> param) {
                           std::string name = ::testing::get<0>(param.param) == QUDA_FULL_SITE_SUBSET ? "full" : "parity";
                           return name + (::testing::get<1>(param.param) ? "_partfile" : "_singlefile");
                         });

// MILC's eigenvector format
INSTANTIATE_TEST_SUITE_P(Vector, MILCEigenvectorTest,
                         Combine(Values(VectorSubset::FULL, VectorSubset::EVEN, VectorSubset::ODD), Values(false, true),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<milc_test_t> param) {
                           std::string name = get_vector_subset_str(::testing::get<0>(param.param));
                           name += ::testing::get<1>(param.param) ? "_partfile_" : "_singlefile_";
                           return name + get_io_mode_str(::testing::get<2>(param.param));
                         });

INSTANTIATE_TEST_SUITE_P(Vector, MILCEigenvectorLayoutTest, Values(VectorSubset::FULL, VectorSubset::ODD),
                         [](testing::TestParamInfo<VectorSubset> param) {
                           return std::string(get_vector_subset_str(param.param));
                         });
