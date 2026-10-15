#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <comm_quda.h>
#include <instantiate.h>
#include <io/gauge_io.h>
#include <io/reunitarize.h>
#include <io/site_io.h>
#include <quda.h>

#include "gauge_utils.h"
#include "test.h"
#include "misc.h"
#include "io_test_utils.h"

namespace
{
  // file formats that gauge fields can be written in and read back from in this build
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

  // the LIME formats (SciDAC and ILDG) with each backend available in this build
  std::vector<QudaFileFormat> lime_formats()
  {
    std::vector<QudaFileFormat> formats = {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG};
#ifdef HAVE_QIO
    formats.insert(formats.end(), {QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_FORMAT_ILDG_EXTERNAL});
#endif
    return formats;
  }

  // file options for a format, mode and I/O partition grid, through the public interface struct, without the
  // usqcdInfo record
  quda::FileIOParam file_io_param(QudaFileFormat format, QudaFileIOMode io_mode,
                                  const std::array<int, 4> &io_partition_grid = {})
  {
    QudaFileIOParam param = newQudaFileIOParam();
    param.struct_size = sizeof(param);
    param.format = format;
    param.io_mode = io_mode;
    param.write_usqcd_info = QUDA_BOOLEAN_FALSE;
    for (int d = 0; d < 4; d++) param.io_partition_grid[d] = io_partition_grid[d];
    return quda::FileIOParam(&param);
  }

  // a host gauge field in QDP order, one array per direction
  struct HostGauge {
    QudaPrecision prec;
    size_t bytes;
    void *data[4];

    HostGauge(QudaPrecision prec) : prec(prec), bytes(V * gauge_site_size * prec)
    {
      for (auto &d : data) d = safe_malloc(bytes);
    }
    ~HostGauge()
    {
      for (auto &d : data) host_free(d);
    }
    HostGauge(const HostGauge &) = delete;
    HostGauge &operator=(const HostGauge &) = delete;

    bool operator==(const HostGauge &other) const
    {
      for (int dir = 0; dir < 4; dir++)
        if (memcmp(data[dir], other.data[dir], bytes) != 0) return false;
      return true;
    }

    void clear()
    {
      for (auto &d : data) memset(d, 0, bytes);
    }
  };

  // the I/O partition grids to test: every grid whose values divide the process grid if there are at most 8;
  // otherwise a single partition, a partition spanning each split dimension alone, and two 2-d blocks
  std::vector<std::array<int, 4>> partition_grids()
  {
    std::vector<std::array<int, 4>> all = {{1, 1, 1, 1}};
    for (int d = 0; d < 4; d++) {
      std::vector<std::array<int, 4>> more;
      for (auto grid : all) {
        for (int n = 1; n <= quda::comm_dim(d); n++) {
          if (quda::comm_dim(d) % n != 0) continue;
          grid[d] = n;
          more.push_back(grid);
        }
      }
      all = more;
    }
    if (all.size() <= 8) return all;

    const std::array<int, 4> full = {quda::comm_dim(0), quda::comm_dim(1), quda::comm_dim(2), quda::comm_dim(3)};
    std::vector<std::array<int, 4>> grids = {{1, 1, 1, 1}};
    auto add = [&](const std::array<int, 4> &grid) {
      if (grid != full && std::find(grids.begin(), grids.end(), grid) == grids.end()) grids.push_back(grid);
    };
    for (int d = 0; d < 4; d++) {
      if (full[d] == 1) continue;
      auto grid = full;
      grid[d] = 1;
      add(grid);
    }
    add({1, 1, full[2], full[3]});
    add({full[0], full[1], 1, 1});
    return grids;
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

  // gauge parameters for a host field of the given precision, with every device precision set to match
  QudaGaugeParam gauge_param_for(QudaPrecision prec)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    gauge_param.cpu_prec = prec;
    gauge_param.cuda_prec = prec;
    gauge_param.cuda_prec_sloppy = prec;
    gauge_param.cuda_prec_precondition = prec;
    gauge_param.cuda_prec_eigensolver = prec;
    gauge_param.t_boundary = QUDA_PERIODIC_T;
    return gauge_param;
  }

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
    EXPECT_EQ(a.io_partition_grid, b.io_partition_grid);
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
  const int grid[4] = {1, 2, 1, 4};
  for (int d = 0; d < 4; d++) param.io_partition_grid[d] = grid[d];

  FileIOParam expected;
  expected.format = QUDA_FILE_FORMAT_MILC;
  expected.io_mode = QUDA_FILE_IO_SERIAL;
  expected.ildg_lfn = "lfn://test";
  expected.write_usqcd_info = false;
  expected.allow_checksum_mismatch = true;
  expected.io_partition_grid = {1, 2, 1, 4};
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

// strings and vectors broadcast from rank 0 take rank 0's contents and size on every rank
TEST(BroadcastTest, resize)
{
  using namespace quda;
  const bool root = comm_rank() == 0;

  std::string expected_string(1000, '\0');
  for (auto i = 0u; i < expected_string.size(); i++) expected_string[i] = static_cast<char>(i * 7);
  std::string s = root ? expected_string : std::string(5, 'x');
  io::broadcast(s);
  EXPECT_EQ(s, expected_string);

  std::vector<int64_t> expected_vector(333);
  for (auto i = 0u; i < expected_vector.size(); i++) expected_vector[i] = (int64_t(1) << 40) + i;
  std::vector<int64_t> v = root ? expected_vector : std::vector<int64_t>(5000, -1);
  io::broadcast(v);
  EXPECT_EQ(v, expected_vector);

  s = root ? std::string() : expected_string;
  io::broadcast(s);
  EXPECT_TRUE(s.empty());
  v = root ? std::vector<int64_t>() : expected_vector;
  io::broadcast(v);
  EXPECT_TRUE(v.empty());
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

  QudaGaugeParam gauge_param = gauge_param_for(::testing::get<0>(param));
  if (!quda::is_enabled(gauge_param.cpu_prec)) GTEST_SKIP();

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

// tuple types: precision, io mode
using milc_gauge_test_t = ::testing::tuple<QudaPrecision, QudaFileIOMode>;

class MILCGaugeFileTest : public ::testing::TestWithParam<milc_gauge_test_t>
{
};

// a saved MILC v5 file has the layout, links and checksums that MILC expects
TEST_P(MILCGaugeFileTest, layout)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  if (!is_enabled(prec)) GTEST_SKIP();

  QudaGaugeParam gauge_param = gauge_param_for(prec);
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, 0, nullptr);

  FileIOParam param;
  param.format = QUDA_FILE_FORMAT_MILC;
  param.io_mode = ::testing::get<1>(GetParam());

  const std::string file = "milc.lat";
  GaugeIO(file, param).save(gauge, prec, gauge_param.X);
  check_milc_gauge_file(file, gauge, prec, gauge_param.X);

  // every rank reads the file on its own
  comm_barrier();
  if (comm_rank() == 0 && (remove(file.c_str()) != 0 || remove((file + ".info").c_str()) != 0))
    errorQuda("Error deleting file");
  for (int dir = 0; dir < 4; dir++) host_free(gauge[dir]);
}

// tuple types: precision, LIME file format (SciDAC or ILDG, with either backend), io mode
using conversion_test_t = ::testing::tuple<QudaPrecision, QudaFileFormat, QudaFileIOMode>;

class GaugeConversionTest : public ::testing::TestWithParam<conversion_test_t>
{
protected:
  QudaPrecision prec;
  quda::FileIOParam milc;
  quda::FileIOParam lime;
  QudaGaugeParam gauge_param;

public:
  GaugeConversionTest() :
    prec(::testing::get<0>(GetParam())),
    milc(file_io_param(QUDA_FILE_FORMAT_MILC, ::testing::get<2>(GetParam()))),
    lime(file_io_param(::testing::get<1>(GetParam()), ::testing::get<2>(GetParam()))),
    gauge_param(gauge_param_for(prec))
  {
  }

  quda::GaugeFileInfo load(const std::string &file, const quda::FileIOParam &param, HostGauge &gauge)
  {
    return quda::GaugeIO(file, param).load(gauge.data, prec, gauge_param.X);
  }

  void save(const std::string &file, const quda::FileIOParam &param, const HostGauge &gauge)
  {
    quda::GaugeIO(file, param).save(gauge.data, prec, gauge_param.X);
  }

  void remove_files(const std::vector<std::string> &files)
  {
    if (quda::comm_rank() == 0)
      for (auto &file : files)
        if (remove(file.c_str()) != 0) errorQuda("Error deleting file %s", file.c_str());
  }
};

// a MILC file converted to SciDAC or ILDG and back gives the same links and the same MILC file
TEST_P(GaugeConversionTest, milc_to_lime_and_back)
{
  using namespace quda;
  if (!is_enabled(prec)) GTEST_SKIP();

  HostGauge original(prec), from_milc(prec), from_lime(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  const std::string milc_file = "conv.milc", lime_file = "conv.lat", milc_again = "conv_again.milc";
  save(milc_file, milc, original);
  auto milc_info = load(milc_file, milc, from_milc);

  // the LIME file is in the host precision, so it holds the MILC file's values exactly
  save(lime_file, lime, from_milc);
  auto lime_info = load(lime_file, lime, from_lime);
  EXPECT_EQ(lime_info.file_prec, prec);
  EXPECT_TRUE(lime_info.checksum_matches);
  EXPECT_TRUE(from_lime == from_milc);

  save(milc_again, milc, from_lime);
  auto again_info = load(milc_again, milc, from_milc);
  EXPECT_TRUE(from_milc == from_lime);
  EXPECT_TRUE(again_info.checksum_matches);
  EXPECT_EQ(again_info.sum29, milc_info.sum29);
  EXPECT_EQ(again_info.sum31, milc_info.sum31);

  if (comm_rank() == 0) expect_same_saved_file(milc_file, milc_again, true);

  remove_files({milc_file, milc_file + ".info", lime_file, milc_again, milc_again + ".info"});
}

// a SciDAC or ILDG file converted to MILC gives its links rounded to single precision
TEST_P(GaugeConversionTest, lime_to_milc)
{
  using namespace quda;
  if (!is_enabled(prec)) GTEST_SKIP();

  HostGauge original(prec), from_lime(prec), from_milc(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  const std::string lime_file = "conv.lat", milc_file = "conv.milc";
  save(lime_file, lime, original);
  load(lime_file, lime, from_lime);
  EXPECT_TRUE(from_lime == original);

  save(milc_file, milc, from_lime);
  auto milc_info = load(milc_file, milc, from_milc);
  EXPECT_EQ(milc_info.file_prec, QUDA_SINGLE_PRECISION);
  EXPECT_TRUE(milc_info.checksum_matches);

  size_t mismatches = 0;
  for (int dir = 0; dir < 4; dir++)
    for (size_t i = 0; i < V * gauge_site_size; i++) {
      double expected = static_cast<float>(get_element(from_lime.data[dir], prec, i));
      if (get_element(from_milc.data[dir], prec, i) != expected) mismatches++;
    }
  EXPECT_EQ(mismatches, 0u);

  remove_files({lime_file, milc_file, milc_file + ".info"});
}

// the deviation from unitarity is MILC's measure: the row norm errors and the overlaps between rows
TEST(UnitarityDeviationTest, measure)
{
  constexpr size_t volume = 2;
  std::vector<std::vector<double>> links(4, std::vector<double>(volume * 18, 0.0));
  for (auto &dir : links)
    for (size_t s = 0; s < volume; s++)
      for (int i = 0; i < 3; i++) dir[s * 18 + 8 * i] = 1.0;
  const void *gauge[4] = {links[0].data(), links[1].data(), links[2].data(), links[3].data()};
  EXPECT_EQ(quda::io::unitarity_deviation(gauge, QUDA_DOUBLE_PRECISION, volume), 0.0);

  // row 1 of one link longer by 1e-3
  for (int j = 0; j < 6; j++) links[2][18 + 6 + j] *= 1.001;
  EXPECT_NEAR(quda::io::unitarity_deviation(gauge, QUDA_DOUBLE_PRECISION, volume), 1e-3, 1e-12);

  // element (1, 0) of another link 1e-2, so row 1 overlaps row 0 by 1e-2 (its norm grows by only 5e-5)
  links[3][6] = 1e-2;
  EXPECT_NEAR(quda::io::unitarity_deviation(gauge, QUDA_DOUBLE_PRECISION, volume), 1e-2, 1e-12);

  links[0][0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(quda::io::unitarity_deviation(gauge, QUDA_DOUBLE_PRECISION, volume), std::numeric_limits<double>::infinity());
}

// tuple types: precision, reunitarization method
using reunitarize_test_t = ::testing::tuple<QudaPrecision, QudaGaugeReunitarize>;

class GaugeReunitarizeTest : public ::testing::TestWithParam<reunitarize_test_t>
{
};

// a MILC file, which is single precision, loaded with each reunitarization method
TEST_P(GaugeReunitarizeTest, milc_file)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  auto method = ::testing::get<1>(GetParam());
  if (!is_enabled(prec)) GTEST_SKIP();
  // reunitarizing as MILC's double-precision builds do is an error for a single-precision host field
  if (method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2 && prec == QUDA_SINGLE_PRECISION) GTEST_SKIP();

  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), raw(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  const std::string file = "reunitarize.milc";
  auto param = file_io_param(QUDA_FILE_FORMAT_MILC, QUDA_FILE_IO_PARALLEL);
  GaugeIO(file, param).save(original.data, prec, gauge_param.X);
  auto raw_info = GaugeIO(file, param).load(raw.data, prec, gauge_param.X);
  param.reunitarize = method;
  auto info = GaugeIO(file, param).load(loaded.data, prec, gauge_param.X);

  // the deviation is measured when reunitarizing, when the file is less precise than the host field, or at
  // verbose verbosity
  const bool measured = prec == QUDA_DOUBLE_PRECISION || getVerbosity() >= QUDA_VERBOSE;
  EXPECT_EQ(raw_info.unitarity_deviation >= 0.0, measured);
  EXPECT_EQ(info.unitarity_deviation >= 0.0, measured || method != QUDA_GAUGE_REUNITARIZE_NONE);
  if (info.unitarity_deviation >= 0.0) {
    EXPECT_GT(info.unitarity_deviation, 0.0);
    EXPECT_LT(info.unitarity_deviation, 1e-5);
  }

  // as MILC's single-precision builds, PRECISION_1 only checks the links, which in a MILC file are already single
  // precision
  if (method == QUDA_GAUGE_REUNITARIZE_NONE || method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1) {
    EXPECT_LT(info.reunitarized_deviation, 0.0);
    EXPECT_TRUE(loaded == raw);
  } else {
    const double tol = prec == QUDA_DOUBLE_PRECISION ? 1e-14 : 1e-6;
    EXPECT_GE(info.reunitarized_deviation, 0.0);
    EXPECT_LE(info.reunitarized_deviation, tol);
    EXPECT_LE(io::unitarity_deviation(loaded.data, prec, V), tol);

    // reunitarization moves each link by about its deviation from unitarity; Gram-Schmidt keeps the direction
    // of row 0, while the polar projection does not
    double max_change = 0.0, max_row0 = 0.0;
    for (int dir = 0; dir < 4; dir++) {
      for (size_t s = 0; s < static_cast<size_t>(V); s++) {
        double norm = 0.0;
        for (int k = 0; k < 6; k++) norm += std::pow(get_element(raw.data[dir], prec, s * 18 + k), 2);
        norm = std::sqrt(norm);
        for (int k = 0; k < 18; k++) {
          double a = get_element(loaded.data[dir], prec, s * 18 + k), b = get_element(raw.data[dir], prec, s * 18 + k);
          max_change = std::max(max_change, std::abs(a - b));
          if (k < 6) max_row0 = std::max(max_row0, std::abs(a - b / norm));
        }
      }
    }
    EXPECT_LE(max_change, 1e-5);
    if (method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2) {
      EXPECT_LE(max_row0, 1e-15);
    } else if (prec == QUDA_DOUBLE_PRECISION) {
      EXPECT_GT(max_row0, 1e-9);
    }
  }

  if (comm_rank() == 0 && (remove(file.c_str()) != 0 || remove((file + ".info").c_str()) != 0))
    errorQuda("Error deleting file");
}

// as MILC's single-precision builds, PRECISION_1 rounds a double-precision file's links to single precision
TEST(GaugeMILCPrecisionTest, precision_1_rounds)
{
  using namespace quda;
  if (!is_enabled(QUDA_DOUBLE_PRECISION)) GTEST_SKIP();

  QudaGaugeParam gauge_param = gauge_param_for(QUDA_DOUBLE_PRECISION);
  HostGauge original(QUDA_DOUBLE_PRECISION), loaded(QUDA_DOUBLE_PRECISION);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  const std::string file = "reunitarize.lat";
  auto param = file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  GaugeIO(file, param).save(original.data, QUDA_DOUBLE_PRECISION, gauge_param.X);
  param.reunitarize = QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1;
  auto info = GaugeIO(file, param).load(loaded.data, QUDA_DOUBLE_PRECISION, gauge_param.X);

  EXPECT_EQ(info.file_prec, QUDA_DOUBLE_PRECISION);
  EXPECT_GT(info.unitarity_deviation, 0.0);
  EXPECT_LT(info.unitarity_deviation, 1e-5);
  EXPECT_LT(info.reunitarized_deviation, 0.0);

  size_t mismatches = 0;
  for (int dir = 0; dir < 4; dir++)
    for (size_t i = 0; i < V * gauge_site_size; i++) {
      double expected = static_cast<float>(static_cast<const double *>(original.data[dir])[i]);
      if (static_cast<const double *>(loaded.data[dir])[i] != expected) mismatches++;
    }
  EXPECT_EQ(mismatches, 0u);

  if (comm_rank() == 0 && remove(file.c_str()) != 0) errorQuda("Error deleting file");
}

// tuple types: precision, file format
using partition_test_t = ::testing::tuple<QudaPrecision, QudaFileFormat>;

class GaugePartitionTest : public ::testing::TestWithParam<partition_test_t>
{
};

// the I/O partition grid changes which ranks access a file, never the file or the loaded links
TEST_P(GaugePartitionTest, grids)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  auto format = ::testing::get<1>(GetParam());
  if (!is_enabled(prec)) GTEST_SKIP();
  const bool milc = format == QUDA_FILE_FORMAT_MILC;
  const bool external = format == QUDA_FILE_FORMAT_QIO_EXTERNAL || format == QUDA_FILE_FORMAT_ILDG_EXTERNAL;

  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), reference(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  auto param = [&](QudaFileFormat f, QudaFileIOMode io_mode, const std::array<int, 4> &grid) {
    auto p = file_io_param(f, io_mode, grid);
    p.ildg_lfn = "lfn://quda/partition_test"; // the default, the file name, differs between files
    return p;
  };
  auto save
    = [&](const std::string &file, const FileIOParam &p) { GaugeIO(file, p).save(original.data, prec, gauge_param.X); };
  auto load = [&](const std::string &file, const FileIOParam &p, HostGauge &gauge) {
    gauge.clear();
    return GaugeIO(file, p).load(gauge.data, prec, gauge_param.X);
  };

  const std::string ref_file = "partition_ref.lat", serial_ref_file = "partition_serial_ref.lat";
  const std::string file = "partition.lat", serial_file = "partition_serial.lat", other_file = "partition_other.lat";
  save(ref_file, param(format, QUDA_FILE_IO_PARALLEL, {}));
  save(serial_ref_file, param(format, QUDA_FILE_IO_SERIAL, {}));
  auto ref_info = load(ref_file, param(format, QUDA_FILE_IO_PARALLEL, {}), reference);
  EXPECT_TRUE(ref_info.checksum_matches);

  // a file equal to the reference loads as the reference does, so each grid's file need not be loaded
  const auto grids = partition_grids();
  for (auto grid : grids) {
    SCOPED_TRACE("I/O partition grid " + std::to_string(grid[0]) + " " + std::to_string(grid[1]) + " "
                 + std::to_string(grid[2]) + " " + std::to_string(grid[3]));
    auto parallel = param(format, QUDA_FILE_IO_PARALLEL, grid);
    save(file, parallel);
    if (comm_rank() == 0) expect_same_saved_file(ref_file, file, milc);

    auto info = load(ref_file, parallel, loaded);
    EXPECT_TRUE(loaded == reference);
    EXPECT_TRUE(info.checksum_matches);
    EXPECT_EQ(info.sum29, ref_info.sum29);
    EXPECT_EQ(info.sum31, ref_info.sum31);

    // the native backend with the same grid writes the same file as the external one
    if (external) {
      save(other_file, param(saved_format(format), QUDA_FILE_IO_PARALLEL, grid));
      if (comm_rank() == 0) expect_same_saved_file(file, other_file, false);
    }
  }

  // serial I/O goes through rank 0 whatever the grid
  auto serial = param(format, QUDA_FILE_IO_SERIAL, grids.back());
  save(serial_file, serial);
  if (comm_rank() == 0) expect_same_saved_file(serial_ref_file, serial_file, milc);
  load(ref_file, serial, loaded);
  EXPECT_TRUE(loaded == reference);

  if (comm_rank() == 0) {
    std::vector<std::string> files = {ref_file, serial_ref_file, file, serial_file};
    if (external) files.push_back(other_file);
    if (milc)
      for (auto i = 0u, n = static_cast<unsigned>(files.size()); i < n; i++) files.push_back(files[i] + ".info");
    for (auto &f : files)
      if (remove(f.c_str()) != 0) errorQuda("Error deleting file %s", f.c_str());
  }
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

  QudaGaugeParam gauge_param = gauge_param_for(prec);

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

  if (comm_rank() == 0) expect_same_saved_file(native_file, external_file, false);

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

int main(int argc, char **argv)
{
  quda_test test("Gauge IO Test", argc, argv);
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

// layout of saved MILC v5 files
INSTANTIATE_TEST_SUITE_P(Gauge, MILCGaugeFileTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<milc_gauge_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<1>(param.param));
                           return name;
                         });

// conversions between MILC and the LIME formats
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeConversionTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), ValuesIn(lime_formats()),
                                 Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<conversion_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += quda::get_file_format_str(::testing::get<1>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<2>(param.param));
                           return name;
                         });

// reunitarization of a loaded gauge field
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeReunitarizeTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_GAUGE_REUNITARIZE_NONE, QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1,
                                        QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2,
                                        QUDA_GAUGE_REUNITARIZE_POLAR)),
                         [](testing::TestParamInfo<reunitarize_test_t> param) {
                           std::string name = get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           switch (::testing::get<1>(param.param)) {
                           case QUDA_GAUGE_REUNITARIZE_NONE: return name + "none";
                           case QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1: return name + "milc_precision_1";
                           case QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2: return name + "milc_precision_2";
                           case QUDA_GAUGE_REUNITARIZE_POLAR: return name + "polar";
                           default: return name + "unknown";
                           }
                         });

// I/O partition grids, with every format but AUTO
INSTANTIATE_TEST_SUITE_P(Gauge, GaugePartitionTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), ValuesIn([] {
                                   auto formats = gauge_formats();
                                   formats.erase(std::find(formats.begin(), formats.end(), QUDA_FILE_FORMAT_AUTO));
                                   return formats;
                                 }())),
                         [](testing::TestParamInfo<partition_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += quda::get_file_format_str(::testing::get<1>(param.param));
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
