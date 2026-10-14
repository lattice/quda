#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#include <comm_quda.h>
#include <instantiate.h>
#include <color_spinor_field.h>
#include <io/gauge_io.h>
#include <io/site_io.h>
#include <io/vector_io.h>
#include <blas_quda.h>
#include <quda.h>

#include "gauge_utils.h"
#include "test.h"
#include "misc.h"

namespace
{
  // file formats that can be written and read back in this build, for each field type
  std::vector<QudaFileFormat> gauge_formats()
  {
    std::vector<QudaFileFormat> formats
      = {QUDA_FILE_FORMAT_AUTO, QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG, QUDA_FILE_FORMAT_MILC};
#ifdef HAVE_QIO
    formats.insert(formats.end(), {QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_FORMAT_ILDG_EXTERNAL});
#endif
    return formats;
  }

  // the format a file is saved in: AUTO saves as QIO, and the external backend writes the same formats
  QudaFileFormat saved_format(QudaFileFormat format)
  {
    switch (format) {
    case QUDA_FILE_FORMAT_AUTO:
    case QUDA_FILE_FORMAT_QIO_EXTERNAL: return QUDA_FILE_FORMAT_QIO;
    case QUDA_FILE_FORMAT_ILDG_EXTERNAL: return QUDA_FILE_FORMAT_ILDG;
    default: return format;
    }
  }

  std::vector<char> read_file(const std::string &file)
  {
    std::ifstream in(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }

  bool write_file(const std::string &file, const std::vector<char> &contents)
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(contents.data(), contents.size());
    return static_cast<bool>(out);
  }

  // change the stored checksum of a gauge file, keeping its size, so that it no longer matches the data
  void corrupt_checksum(const std::string &file, QudaFileFormat format)
  {
    if (quda::comm_rank() == 0) {
      auto contents = read_file(file);
      if (format == QUDA_FILE_FORMAT_MILC) {
        ASSERT_GE(contents.size(), 96u);
        contents[88] ^= 1; // the low byte of sum29
      } else {
        const std::string tag = "<suma>";
        auto it = std::find_end(contents.begin(), contents.end(), tag.begin(), tag.end());
        ASSERT_NE(it, contents.end()) << "no checksum record in " << file;
        auto &digit = *(it + tag.size());
        digit = digit == '1' ? '2' : '1';
      }
      EXPECT_TRUE(write_file(file, contents)) << "could not change the checksum in " << file;
    }
    quda::comm_barrier();
  }

  std::vector<QudaFileFormat> vector_formats()
  {
#ifdef HAVE_QIO
    return {QUDA_FILE_FORMAT_AUTO, QUDA_FILE_FORMAT_QIO};
#else
    return {};
#endif
  }

  const char *get_io_mode_str(QudaFileIOMode io_mode) { return io_mode == QUDA_FILE_IO_SERIAL ? "serial" : "parallel"; }

  double get_element(const void *data, QudaPrecision prec, size_t i)
  {
    return prec == QUDA_DOUBLE_PRECISION ? static_cast<const double *>(data)[i] : static_cast<const float *>(data)[i];
  }

  // check a saved MILC v5 gauge file independently of the library's reader: its header, where
  // each of this rank's links is, and (on rank 0) its checksums and those in its .info file
  void check_milc_gauge_file(const std::string &file, void *const gauge[], QudaPrecision prec, const int *X)
  {
    using namespace quda;
    constexpr size_t header_bytes = 96;
    constexpr size_t site_reals = 4 * 18;

    int N[4];
    size_t volume = 1, global_volume = 1;
    for (int d = 0; d < 4; d++) {
      N[d] = X[d] * comm_dim(d);
      volume *= X[d];
      global_volume *= N[d];
    }

    FILE *fp = fopen(file.c_str(), "rb");
    ASSERT_NE(fp, nullptr);
    uint32_t header[header_bytes / 4];
    ASSERT_EQ(fread(header, 4, header_bytes / 4, fp), header_bytes / 4);
    EXPECT_EQ(header[0], 0x4e87u);
    for (int d = 0; d < 4; d++) EXPECT_EQ(header[1 + d], static_cast<uint32_t>(N[d]));
    EXPECT_EQ(header[21], 0u); // order
    fseek(fp, 0, SEEK_END);
    EXPECT_EQ(static_cast<size_t>(ftell(fp)), header_bytes + global_volume * site_reals * sizeof(float));

    size_t mismatches = 0;
    std::vector<float> site(site_reals);
    for (int t = 0; t < X[3]; t++) {
      for (int z = 0; z < X[2]; z++) {
        for (int y = 0; y < X[1]; y++) {
          for (int x = 0; x < X[0]; x++) {
            int g[4]
              = {comm_coord(0) * X[0] + x, comm_coord(1) * X[1] + y, comm_coord(2) * X[2] + z, comm_coord(3) * X[3] + t};
            size_t global_lex
              = g[0] + N[0] * (g[1] + static_cast<size_t>(N[1]) * (g[2] + static_cast<size_t>(N[2]) * g[3]));
            size_t lex = x + X[0] * (y + static_cast<size_t>(X[1]) * (z + static_cast<size_t>(X[2]) * t));
            size_t index = lex / 2 + ((g[0] + g[1] + g[2] + g[3]) % 2) * (volume / 2);
            fseek(fp, header_bytes + global_lex * site_reals * sizeof(float), SEEK_SET);
            if (fread(site.data(), sizeof(float), site_reals, fp) != site_reals) {
              mismatches++;
              continue;
            }
            for (int mu = 0; mu < 4; mu++)
              for (int i = 0; i < 18; i++)
                if (site[mu * 18 + i] != static_cast<float>(get_element(gauge[mu], prec, index * 18 + i))) mismatches++;
          }
        }
      }
    }
    EXPECT_EQ(mismatches, 0u);

    if (comm_rank() == 0) {
      std::vector<uint32_t> words(global_volume * site_reals);
      fseek(fp, header_bytes, SEEK_SET);
      ASSERT_EQ(fread(words.data(), 4, words.size(), fp), words.size());
      uint32_t sum29 = 0, sum31 = 0;
      for (size_t i = 0; i < words.size(); i++) {
        int r29 = i % 29, r31 = i % 31;
        sum29 ^= r29 == 0 ? words[i] : (words[i] << r29) | (words[i] >> (32 - r29));
        sum31 ^= r31 == 0 ? words[i] : (words[i] << r31) | (words[i] >> (32 - r31));
      }
      EXPECT_EQ(sum29, header[22]);
      EXPECT_EQ(sum31, header[23]);

      char expected[64];
      snprintf(expected, sizeof(expected), "checksums = \"%x %x\"", sum29, sum31);
      std::ifstream info(file + ".info");
      bool found = false;
      for (std::string line; std::getline(info, line);) found = found || line == expected;
      EXPECT_TRUE(found) << "no line '" << expected << "' in " << file << ".info";
    }
    fclose(fp);
  }
} // namespace

// the conversion from the public QudaFileIOParam is the single place defaults and validation are applied
TEST(FileIOParamTest, conversion)
{
  using namespace quda;

  auto expect_equal = [](const FileIOParam &a, const FileIOParam &b) {
    EXPECT_EQ(a.format, b.format);
    EXPECT_EQ(a.external_qio, b.external_qio);
    EXPECT_EQ(a.io_mode, b.io_mode);
    EXPECT_EQ(a.partfile, b.partfile);
    EXPECT_EQ(a.parity_inflate, b.parity_inflate);
    EXPECT_EQ(a.ildg_lfn, b.ildg_lfn);
    EXPECT_EQ(a.write_usqcd_info, b.write_usqcd_info);
    EXPECT_EQ(a.allow_checksum_mismatch, b.allow_checksum_mismatch);
  };

  const FileIOParam defaults;
  expect_equal(FileIOParam(nullptr), defaults);

  QudaFileIOParam param = newQudaFileIOParam();
  expect_equal(FileIOParam(&param), defaults);

  param.struct_size = sizeof(param);
  param.format = QUDA_FILE_FORMAT_MILC;
  param.io_mode = QUDA_FILE_IO_SERIAL;
  strcpy(param.ildg_lfn, "lfn://test");
  param.write_usqcd_info = QUDA_BOOLEAN_FALSE;
  param.allow_checksum_mismatch = QUDA_BOOLEAN_TRUE;

  FileIOParam expected;
  expected.format = QUDA_FILE_FORMAT_MILC;
  expected.io_mode = QUDA_FILE_IO_SERIAL;
  expected.ildg_lfn = "lfn://test";
  expected.write_usqcd_info = false;
  expected.allow_checksum_mismatch = true;
  expect_equal(FileIOParam(&param), expected);

  // the _EXTERNAL formats become the plain format and the external QIO backend
  for (auto f : {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG}) {
    param.format = f;
    expected.format = f;
    expected.external_qio = false;
    expect_equal(FileIOParam(&param), expected);
#ifdef HAVE_QIO
    param.format = f == QUDA_FILE_FORMAT_QIO ? QUDA_FILE_FORMAT_QIO_EXTERNAL : QUDA_FILE_FORMAT_ILDG_EXTERNAL;
    expected.external_qio = true;
    expect_equal(FileIOParam(&param), expected);
#endif
  }
}

// the CRC-32 used in SciDAC checksums is that of zlib
TEST(CRC32Test, check_value)
{
  const char data[] = "123456789";
  EXPECT_EQ(quda::io::crc32(data, 9), 0xcbf43926u);
  EXPECT_EQ(quda::io::crc32(data + 4, 5, quda::io::crc32(data, 4)), 0xcbf43926u);
  EXPECT_EQ(quda::io::crc32(data, 0), 0u);
}

// tuple types: precision, file format, io mode, whether to write the usqcdInfo record
using gauge_test_t = ::testing::tuple<QudaPrecision, QudaFileFormat, QudaFileIOMode, bool>;

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
  using namespace quda;
  if (!is_enabled(prec)) GTEST_SKIP();

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  gauge_param.cpu_prec = ::testing::get<0>(param);
  if (!quda::is_enabled(gauge_param.cpu_prec)) GTEST_SKIP();
  gauge_param.cuda_prec = gauge_param.cpu_prec;
  gauge_param.cuda_prec_sloppy = gauge_param.cpu_prec;
  gauge_param.cuda_prec_precondition = gauge_param.cpu_prec;
  gauge_param.cuda_prec_eigensolver = gauge_param.cpu_prec;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  // the usqcdInfo record does not apply to MILC files
  if (::testing::get<1>(param) == QUDA_FILE_FORMAT_MILC && ::testing::get<3>(param)) GTEST_SKIP();

  // the file options go through the public interface struct
  QudaFileIOParam io_param = newQudaFileIOParam();
  io_param.struct_size = sizeof(io_param);
  io_param.format = ::testing::get<1>(param);
  io_param.io_mode = ::testing::get<2>(param);
  io_param.write_usqcd_info = ::testing::get<3>(param) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  gauge_param.file_io_param = &io_param;

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

  std::vector<std::vector<char>> original(4);
  for (int dir = 0; dir < 4; dir++) {
    auto bytes = V * gauge_site_size * host_gauge_data_type_size;
    original[dir].assign(static_cast<char *>(gauge[dir]), static_cast<char *>(gauge[dir]) + bytes);
  }

  // MILC files are always single precision
  auto format = saved_format(io_param.format);
  auto file_prec = format == QUDA_FILE_FORMAT_MILC ? QUDA_SINGLE_PRECISION : gauge_param.cpu_prec;

  auto file = "dummy.lat";
  GaugeIO io(file, FileIOParam(gauge_param.file_io_param));

  // write out the gauge field
  io.save(gauge, gauge_param.cpu_prec, gauge_param.X);
  if (format == QUDA_FILE_FORMAT_MILC) check_milc_gauge_file(file, gauge, gauge_param.cpu_prec, gauge_param.X);

  // read it back
  auto info = io.load(gauge, gauge_param.cpu_prec, gauge_param.X);

  // test the links are identical, or as close as the file precision allows
  double link_tol = file_prec == gauge_param.cpu_prec ? 0.0 : std::numeric_limits<float>::epsilon();
  double max_dev = 0.0;
  for (int dir = 0; dir < 4; dir++)
    for (size_t i = 0; i < V * gauge_site_size; i++)
      max_dev = std::max(max_dev,
                         std::abs(get_element(gauge[dir], gauge_param.cpu_prec, i)
                                  - get_element(original[dir].data(), gauge_param.cpu_prec, i)));
  EXPECT_LE(max_dev, link_tol);

  auto plaq_new = get_plaq();

  // test the plaquette is identical, or as close as the file precision allows
  for (int i = 0; i < 3; i++) {
    if (file_prec == gauge_param.cpu_prec)
      EXPECT_EQ(plaq_old[i], plaq_new[i]);
    else
      EXPECT_NEAR(plaq_old[i], plaq_new[i], 1e-6);
  }

  // test the metadata found in the file
  EXPECT_EQ(info.format, format);
  EXPECT_EQ(info.file_prec, file_prec);
  EXPECT_EQ(info.ildg, format == QUDA_FILE_FORMAT_ILDG);
  EXPECT_EQ(info.has_plaq_linktr, ::testing::get<3>(param));
  if (info.has_plaq_linktr) {
    EXPECT_TRUE(info.plaq_linktr_matches);
    auto tol = gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION ? 1e-12 : 1e-6;
    EXPECT_NEAR(info.plaq, plaq_new[0], tol);
  }
  EXPECT_EQ(info.has_info_file, format == QUDA_FILE_FORMAT_MILC);
  EXPECT_TRUE(info.has_checksum);
  EXPECT_TRUE(info.checksum_matches);
  EXPECT_FALSE(info.byte_swapped);

  // a file whose stored checksum is wrong still loads when that is allowed
  corrupt_checksum(file, format);
  FileIOParam allow(gauge_param.file_io_param);
  allow.allow_checksum_mismatch = true;
  auto corrupt_info = GaugeIO(file, allow).load(gauge, gauge_param.cpu_prec, gauge_param.X);
  EXPECT_FALSE(corrupt_info.checksum_matches);
  EXPECT_EQ(corrupt_info.sum29, info.sum29);
  EXPECT_EQ(corrupt_info.sum31, info.sum31);

  // cleanup after ourselves and delete the dummy lattice
  if (::quda::comm_rank() == 0 && remove(file) != 0) errorQuda("Error deleting file");
  if (format == QUDA_FILE_FORMAT_MILC && ::quda::comm_rank() == 0 && remove((std::string(file) + ".info").c_str()) != 0)
    errorQuda("Error deleting file");

  // release memory
  for (int dir = 0; dir < 4; dir++) { host_free(gauge[dir]); }
}

#ifdef HAVE_QIO
class GaugeIOBackendTest : public ::testing::TestWithParam<gauge_test_t>
{
};

// the native and external QIO backends write the same file, apart from the date in the private record XML, and
// each reads the other's file
TEST_P(GaugeIOBackendTest, compare)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  if (!is_enabled(prec)) GTEST_SKIP();

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.cpu_prec = prec;
  gauge_param.cuda_prec = prec;
  gauge_param.cuda_prec_sloppy = prec;
  gauge_param.cuda_prec_precondition = prec;
  gauge_param.cuda_prec_eigensolver = prec;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  const size_t bytes = V * gauge_site_size * prec;
  void *gauge[4], *loaded[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = safe_malloc(bytes);
    loaded[dir] = safe_malloc(bytes);
  }
  constructHostGaugeField(gauge, gauge_param, 0, nullptr);

  FileIOParam native;
  native.format = ::testing::get<1>(GetParam());
  native.io_mode = ::testing::get<2>(GetParam());
  native.write_usqcd_info = ::testing::get<3>(GetParam());
  native.ildg_lfn = "lfn://quda/io_test"; // the default, the file name, differs between the two files
  FileIOParam external = native;
  external.external_qio = true;

  const std::string native_file = "native.lat", external_file = "external.lat";
  GaugeIO(native_file, native).save(gauge, prec, gauge_param.X);
  GaugeIO(external_file, external).save(gauge, prec, gauge_param.X);

  if (comm_rank() == 0) {
    auto mask_date = [](std::vector<char> &contents) {
      const std::string open = "<date>", close = "</date>";
      auto start = std::search(contents.begin(), contents.end(), open.begin(), open.end());
      auto end = std::search(start, contents.end(), close.begin(), close.end());
      EXPECT_NE(end, contents.end()) << "no <date> element";
      if (start != contents.end()) std::fill(start + open.size(), end, '#');
    };
    auto a = read_file(native_file), b = read_file(external_file);
    mask_date(a);
    mask_date(b);
    EXPECT_EQ(a.size(), b.size());
    auto diff = std::mismatch(a.begin(), a.begin() + std::min(a.size(), b.size()), b.begin());
    EXPECT_EQ(diff.first - a.begin(), static_cast<ptrdiff_t>(std::min(a.size(), b.size())))
      << "the files first differ at byte " << diff.first - a.begin();
  }

  std::vector<GaugeFileInfo> infos;
  for (auto [file, param] : {std::make_pair(native_file, external), std::make_pair(external_file, native)}) {
    for (int dir = 0; dir < 4; dir++) memset(loaded[dir], 0, bytes);
    infos.push_back(GaugeIO(file, param).load(loaded, prec, gauge_param.X));
    for (int dir = 0; dir < 4; dir++)
      EXPECT_EQ(memcmp(loaded[dir], gauge[dir], bytes), 0) << file << " direction " << dir;
    EXPECT_TRUE(infos.back().has_checksum);
    EXPECT_TRUE(infos.back().checksum_matches);
    EXPECT_EQ(infos.back().ildg, native.format == QUDA_FILE_FORMAT_ILDG);
    EXPECT_EQ(infos.back().has_plaq_linktr, native.write_usqcd_info);
  }
  EXPECT_EQ(infos[0].sum29, infos[1].sum29);
  EXPECT_EQ(infos[0].sum31, infos[1].sum31);

  if (comm_rank() == 0 && (remove(native_file.c_str()) != 0 || remove(external_file.c_str()) != 0))
    errorQuda("Error deleting file");
  for (int dir = 0; dir < 4; dir++) {
    host_free(gauge[dir]);
    host_free(loaded[dir]);
  }
}
#endif

using cs_test_t = ::testing::tuple<QudaSiteSubset, QudaSiteSubset, QudaParity, bool, QudaPrecision, QudaPrecision, int,
                                   bool, QudaFieldLocation, QudaFileFormat, QudaFileIOMode>;

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
    io_mode(::testing::get<10>(GetParam()))
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

  FileIOParam io_param;
  io_param.format = format;
  io_param.io_mode = io_mode;
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
using ::testing::ValuesIn;

// gauge IO test
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeIOTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), ValuesIn(gauge_formats()),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL), Values(true, false)),
                         [](testing::TestParamInfo<gauge_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += quda::get_file_format_str(::testing::get<1>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<2>(param.param));
                           name += ::testing::get<3>(param.param) ? "_usqcd_info" : "_no_usqcd_info";
                           return name;
                         });

#ifdef HAVE_QIO
// comparison of the native and external QIO backends
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeIOBackendTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL), Values(true, false)),
                         [](testing::TestParamInfo<gauge_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += quda::get_file_format_str(::testing::get<1>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<2>(param.param));
                           name += ::testing::get<3>(param.param) ? "_usqcd_info" : "_no_usqcd_info";
                           return name;
                         });
#endif

// colorspinor full field IO test
INSTANTIATE_TEST_SUITE_P(Full, ColorSpinorIOTest,
                         Combine(Values(QUDA_FULL_SITE_SUBSET), Values(QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_INVALID_PARITY), Values(false),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION),
                                 ValuesIn(vector_formats()), Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<4>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<5>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<6>(param.param));
                           name += ::testing::get<7>(param.param) ? "_partfile" : "_singlefile";
                           name += ::testing::get<8>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           name += std::string("_") + quda::get_file_format_str(::testing::get<9>(param.param));
                           name += std::string("_") + get_io_mode_str(::testing::get<10>(param.param));
                           return name;
                         });

// colorspinor parity field IO test
INSTANTIATE_TEST_SUITE_P(Parity, ColorSpinorIOTest,
                         Combine(Values(QUDA_PARITY_SITE_SUBSET), Values(QUDA_PARITY_SITE_SUBSET, QUDA_FULL_SITE_SUBSET),
                                 Values(QUDA_EVEN_PARITY, QUDA_ODD_PARITY), Values(false, true),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 2, 4),
                                 Values(false, true), Values(QUDA_CUDA_FIELD_LOCATION, QUDA_CPU_FIELD_LOCATION),
                                 ValuesIn(vector_formats()), Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<cs_test_t> param) {
                           std::string name;
                           name += ::testing::get<1>(param.param) == QUDA_PARITY_SITE_SUBSET ? "load_parity_" :
                                                                                               "load_full_";
                           name += ::testing::get<2>(param.param) == QUDA_EVEN_PARITY ? "even_" : "odd_";
                           if (::testing::get<3>(param.param)) name += std::string("inflate_");
                           name += get_prec_str(::testing::get<4>(param.param)) + std::string("_");
                           name += get_prec_str(::testing::get<5>(param.param)) + std::string("_");
                           name += std::string("spin") + std::to_string(::testing::get<6>(param.param));
                           name += ::testing::get<7>(param.param) ? "_partfile" : "_singlefile";
                           name += ::testing::get<8>(param.param) == QUDA_CUDA_FIELD_LOCATION ? "_device" : "_host";
                           name += std::string("_") + quda::get_file_format_str(::testing::get<9>(param.param));
                           name += std::string("_") + get_io_mode_str(::testing::get<10>(param.param));
                           return name;
                         });
