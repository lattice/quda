#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <vector>
#include <unistd.h>

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

  // change the stored checksum of a gauge file, keeping its size, so that it no longer matches the data
  void corrupt_checksum(const std::string &file, QudaFileFormat format)
  {
    if (quda::comm_rank() == 0) {
      auto contents = read_file(file);
      bool changed = false;
      if (format == QUDA_FILE_FORMAT_MILC) {
        EXPECT_GE(contents.size(), 96u);
        if (contents.size() >= 96) {
          contents[88] ^= 1; // the low byte of sum29
          changed = true;
        }
      } else {
        const std::string tag = "<suma>";
        auto it = std::find_end(contents.begin(), contents.end(), tag.begin(), tag.end());
        EXPECT_NE(it, contents.end()) << "no checksum record in " << file;
        if (it != contents.end()) {
          auto &digit = *(it + tag.size());
          digit = digit == '1' ? '2' : '1';
          changed = true;
        }
      }
      if (changed) { EXPECT_TRUE(write_file(file, contents)) << "could not change the checksum in " << file; }
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

  // MILC's NERSC checksum of a host gauge field, summed here independently of the library
  uint32_t nersc_checksum(const HostGauge &gauge)
  {
    uint64_t sum = 0;
    for (int dir = 0; dir < 4; dir++)
      for (size_t s = 0; s < static_cast<size_t>(V); s++)
        for (int i = 0; i < 12; i++) {
          const float f = static_cast<float>(get_element(gauge.data[dir], gauge.prec, s * 18 + i));
          uint32_t bits;
          memcpy(&bits, &f, sizeof(bits));
          sum += bits;
        }
    size_t total = sum;
    quda::comm_allreduce_sum(total);
    return static_cast<uint32_t>(total);
  }

  // the lines of MILC's .info keys that QUDA computes, for the links of a gauge field rounded to float as a MILC
  // file holds them: in MILC's formats (write_gauge_info_item in generic/io_lat_utils.c), each value as a float
  std::vector<std::string> computed_info_lines(const void *const gauge[], QudaPrecision prec, const int *X)
  {
    HostGauge rounded(QUDA_DOUBLE_PRECISION);
    for (int dir = 0; dir < 4; dir++)
      for (size_t i = 0; i < V * gauge_site_size; i++)
        static_cast<double *>(rounded.data[dir])[i] = static_cast<float>(get_element(gauge[dir], prec, i));
    const auto obs = quda::gauge_observables(rounded.data, QUDA_DOUBLE_PRECISION, X);
    char lines[4][96];
    snprintf(lines[0], sizeof(lines[0]), "gauge.ssplaq = %f", static_cast<float>(3.0 * obs.plaq_spatial));
    snprintf(lines[1], sizeof(lines[1]), "gauge.stplaq = %f", static_cast<float>(3.0 * obs.plaq_temporal));
    snprintf(lines[2], sizeof(lines[2]), "gauge.nersc_linktr = %e", static_cast<float>(obs.linktr));
    snprintf(lines[3], sizeof(lines[3]), "gauge.nersc_checksum = %u", nersc_checksum(rounded));
    return {lines[0], lines[1], lines[2], lines[3]};
  }

  // the lines of a .info file, read on the calling rank
  std::vector<std::string> info_lines(const std::string &file)
  {
    std::ifstream in(file + ".info");
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
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
    EXPECT_EQ(a.milc_eigenvectors, b.milc_eigenvectors);
  };

  const FileIOParam defaults;
  expect_equal(FileIOParam(nullptr), defaults);

  QudaFileIOParam param = newQudaFileIOParam();
  EXPECT_EQ(param.partfile, QUDA_BOOLEAN_FALSE);
  expect_equal(FileIOParam(&param), defaults);

  param.struct_size = sizeof(param);
  param.format = QUDA_FILE_FORMAT_MILC;
  param.io_mode = QUDA_FILE_IO_SERIAL;
  strcpy(param.ildg_lfn, "lfn://test");
  param.write_usqcd_info = QUDA_BOOLEAN_FALSE;
  param.allow_checksum_mismatch = QUDA_BOOLEAN_TRUE;
  const int grid[4] = {1, 2, 1, 4};
  for (int d = 0; d < 4; d++) param.io_partition_grid[d] = grid[d];
  param.milc_eigenvectors = QUDA_BOOLEAN_TRUE;
  param.partfile = QUDA_BOOLEAN_TRUE;

  FileIOParam expected;
  expected.format = QUDA_FILE_FORMAT_MILC;
  expected.io_mode = QUDA_FILE_IO_SERIAL;
  expected.ildg_lfn = "lfn://test";
  expected.write_usqcd_info = false;
  expected.allow_checksum_mismatch = true;
  expected.io_partition_grid = {1, 2, 1, 4};
  expected.milc_eigenvectors = true;
  expected.partfile = true;
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

  // the MPI-IO mode needs an MPI or QMP build and the native backend
  if (mpi_io_enabled()) {
    param.format = QUDA_FILE_FORMAT_QIO;
    param.io_mode = QUDA_FILE_IO_MPI_IO;
    expected.format = QUDA_FILE_FORMAT_QIO;
    expected.external_qio = false;
    expected.io_mode = QUDA_FILE_IO_MPI_IO;
    expect_equal(FileIOParam(&param), expected);
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
  // usqcdInfo, or for MILC files the plaquettes and link trace in the .info file
  EXPECT_EQ(info.has_plaq_linktr, ::testing::get<3>(param));
  if (info.has_plaq_linktr) {
    EXPECT_TRUE(info.plaq_linktr_matches);
    // MILC's .info files print the plaquettes to six decimal places
    auto tol = format == QUDA_FILE_FORMAT_MILC ? 1e-5 : gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION ? 1e-12 : 1e-6;
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

  // the .info file: QUDA's lines, then MILC's four keys that QUDA computes, of the links as saved and in MILC's
  // formats; a load verifies the plaquettes and link trace
  const auto computed = computed_info_lines(gauge, prec, gauge_param.X);
  const char *keys[] = {"magic_number", "time_stamp", "checksums", "nx", "ny", "nz", "nt"};
  const std::regex formats[]
    = {std::regex("gauge\\.ssplaq = -?[0-9]+\\.[0-9]{6}"), std::regex("gauge\\.stplaq = -?[0-9]+\\.[0-9]{6}"),
       std::regex("gauge\\.nersc_linktr = -?[0-9]\\.[0-9]{6}e[-+][0-9]{2}"),
       std::regex("gauge\\.nersc_checksum = [0-9]+")};
  if (comm_rank() == 0) {
    const auto lines = info_lines(file);
    EXPECT_EQ(lines.size(), 11u);
    for (auto i = 0u; i < std::min<size_t>(lines.size(), 7); i++)
      EXPECT_EQ(lines[i].substr(0, lines[i].find(' ')), keys[i]) << lines[i];
    for (auto i = 7u; i < std::min<size_t>(lines.size(), 11); i++) {
      EXPECT_EQ(lines[i], computed[i - 7]);
      EXPECT_TRUE(std::regex_match(lines[i], formats[i - 7])) << lines[i];
    }
  }
  HostGauge loaded(prec);
  auto info = GaugeIO(file, param).load(loaded.data, prec, gauge_param.X);
  EXPECT_TRUE(info.has_plaq_linktr);
  EXPECT_TRUE(info.plaq_linktr_matches);

  // without write_usqcd_info, only QUDA's lines
  auto plain = param;
  plain.write_usqcd_info = false;
  GaugeIO(file, plain).save(gauge, prec, gauge_param.X);
  if (comm_rank() == 0) { EXPECT_EQ(info_lines(file).size(), 7u); }
  info = GaugeIO(file, plain).load(loaded.data, prec, gauge_param.X);
  EXPECT_FALSE(info.has_plaq_linktr);

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
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
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

class GaugePartfileTest : public ::testing::TestWithParam<QudaPrecision>
{
};

// SciDAC gauge partfiles, saved through QudaFileIOParam::partfile, load natively with any grid and rank layout; the
// external backend saves the same volumes, loads the native ones with the grid that saved them, and the native backend
// loads its partfiles
TEST_P(GaugePartfileTest, grids)
{
  using namespace quda;
  auto prec = GetParam();
  if (!is_enabled(prec)) GTEST_SKIP();

  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);
  auto save
    = [&](const std::string &file, const FileIOParam &p) { GaugeIO(file, p).save(original.data, prec, gauge_param.X); };
  auto expect_load = [&](const std::string &file, const FileIOParam &p, bool usqcd_info = false) {
    loaded.clear();
    auto info = GaugeIO(file, p).load(loaded.data, prec, gauge_param.X);
    EXPECT_TRUE(loaded == original);
    EXPECT_TRUE(info.checksum_matches);
    EXPECT_EQ(info.has_plaq_linktr, usqcd_info);
    if (usqcd_info) { EXPECT_TRUE(info.plaq_linktr_matches); }
  };
#ifdef HAVE_QIO
  // on one rank a partfile is a single file
  auto expect_same = [](const std::string &a, const std::string &b) {
    if (comm_rank() != 0) return;
    if (comm_size() == 1)
      expect_same_saved_file(a, b, false);
    else
      expect_same_partfile(a, b);
  };
#endif

  const std::string file = "partfile.lat", external_file = "partfile_external.lat";
  const auto grids = partition_grids();
  for (auto i = 0u; i < grids.size(); i++) {
    const auto &grid = grids[i];
    const auto &next_grid = grids[(i + 1) % grids.size()];
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
    save(file, partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL, grid));
    const bool single = comm_size() == 1;
    const auto volumes = single ? std::vector<std::string>() : expected_volumes(file, grid);
    if (comm_rank() == 0) {
      EXPECT_EQ(access(file.c_str(), F_OK) == 0, single) << file;
      EXPECT_EQ(saved_volumes(file), volumes);
    }
    expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL, next_grid));
#ifdef HAVE_QIO
    save(external_file, partfile_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_IO_PARALLEL, grid));
    expect_same(file, external_file);
    {
      SCOPED_TRACE("native load of the external partfile with I/O partition grid " + grid_str(next_grid));
      expect_load(external_file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL, next_grid));
    }
    {
      // QIO reads partfiles only with the rank layout and grid that saved them
      SCOPED_TRACE("external load of the native partfile");
      expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_IO_PARALLEL, grid));
    }
    remove_saved_file(external_file);
#endif
    remove_saved_file(file);
  }

  {
    // the usqcdInfo record goes in the first volume, with the record's other metadata
    SCOPED_TRACE("with the usqcdInfo record");
    auto with_info = [](QudaFileFormat format) {
      auto p = partfile_param(format, QUDA_FILE_IO_PARALLEL);
      p.write_usqcd_info = true;
      return p;
    };
    save(file, with_info(QUDA_FILE_FORMAT_QIO));
    expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL), true);
#ifdef HAVE_QIO
    save(external_file, with_info(QUDA_FILE_FORMAT_QIO_EXTERNAL));
    expect_same(file, external_file);
    expect_load(external_file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL), true);
    expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_IO_PARALLEL), true);
    remove_saved_file(external_file);
#endif
    remove_saved_file(file);
  }

  // volumes as if saved by another rank layout, crossing this layout's rank boundaries where the lattice allows
  auto p = partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  for (int d = 0; d < 4; d++) {
    const int N = gauge_param.X[d] * comm_dim(d);
    p.partfile_grid[d] = N % 3 == 0 ? 3 : N % 2 == 0 ? 2 : 1;
  }
  SCOPED_TRACE("volumes " + grid_str(p.partfile_grid));
  save(file, p);
  expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL));
  expect_load(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_SERIAL));
  remove_saved_file(file);
}

namespace
{
  // links of a type on the device at the host precision, without reconstruction, so they hold the host's exactly
  QudaGaugeParam read_gauge_param(QudaPrecision prec, QudaLinkType type = QUDA_WILSON_LINKS)
  {
    QudaGaugeParam gauge_param = gauge_param_for(prec);
    gauge_param.type = type;
    gauge_param.cuda_prec_refinement_sloppy = prec;
    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
    return gauge_param;
  }

  // readGaugeQuda of a file, copied back to the host, gives the links GaugeIO loads from it with the same options
  void expect_read_as_loaded(const std::string &file, QudaGaugeParam gauge_param, QudaFileIOParam io_param)
  {
    using namespace quda;
    const auto prec = gauge_param.cpu_prec;
    HostGauge expected(prec), read(prec);
    GaugeIO(file, FileIOParam(&io_param)).load(expected.data, prec, gauge_param.X);

    gauge_param.file_io_param = &io_param;
    readGaugeQuda(file.c_str(), &gauge_param);
    read.clear();
    saveGaugeQuda(read.data, &gauge_param);
    freeGaugeQuda();
    EXPECT_TRUE(read == expected);
  }
} // namespace

// tuple types: file format, io mode
using read_gauge_test_t = ::testing::tuple<QudaFileFormat, QudaFileIOMode>;

class ReadGaugeQudaTest : public ::testing::TestWithParam<read_gauge_test_t>
{
};

// readGaugeQuda loads a file into the resident gauge field as GaugeIO loads it, for every format, single files and
// partfiles
TEST_P(ReadGaugeQudaTest, load)
{
  using namespace quda;
  auto format = ::testing::get<0>(GetParam());
  auto io_mode = ::testing::get<1>(GetParam());
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();

  auto gauge_param = read_gauge_param(prec);
  HostGauge original(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);
  auto save = [&](const std::string &file, QudaFileIOParam io_param) {
    GaugeIO(file, FileIOParam(&io_param)).save(original.data, prec, gauge_param.X);
  };

  const std::string file = "read_gauge.lat";
  const auto io_param = quda_file_io_param(format, io_mode);
  save(file, io_param);
  expect_read_as_loaded(file, gauge_param, io_param);
  remove_saved_file(file);
  if (saved_format(format) == QUDA_FILE_FORMAT_MILC) remove_saved_file(file + ".info");

  if (saved_format(format) == QUDA_FILE_FORMAT_QIO) {
    const auto grid = partition_grids().back();
    SCOPED_TRACE("partfile with I/O partition grid " + grid_str(grid));
    const auto partfile = quda_file_io_param(format, io_mode, grid, true);
    save(file, partfile);
    expect_read_as_loaded(file, gauge_param, partfile);
    remove_saved_file(file);
  }
}

// writeGaugeQuda then readGaugeQuda gives back the resident links exactly, for each type of links, in single files and
// partfiles; writeGaugeQuda saves the same partfile as GaugeIO
TEST(ReadGaugeQudaWriteTest, round_trip)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();

  HostGauge original(prec), read(prec);
  constructHostGaugeField(original.data, read_gauge_param(prec), 0, nullptr);
  const std::string file = "write_gauge.lat", internal_file = "write_gauge_internal.lat";
  for (auto type : {QUDA_WILSON_LINKS, QUDA_ASQTAD_FAT_LINKS, QUDA_ASQTAD_LONG_LINKS, QUDA_SMEARED_LINKS}) {
    SCOPED_TRACE("link type " + std::to_string(type));
    auto gauge_param = read_gauge_param(prec, type);
    for (bool partfile : {false, true}) {
      SCOPED_TRACE(partfile ? "partfile" : "single file");
      auto io_param = quda_file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL, {}, partfile);
      gauge_param.file_io_param = &io_param;
      loadGaugeQuda(original.data, &gauge_param);
      writeGaugeQuda(file.c_str(), &gauge_param);
      freeGaugeQuda();

      readGaugeQuda(file.c_str(), &gauge_param);
      read.clear();
      saveGaugeQuda(read.data, &gauge_param);
      freeGaugeQuda();
      EXPECT_TRUE(read == original);

      if (partfile && type == QUDA_WILSON_LINKS) {
        GaugeIO(internal_file, FileIOParam(&io_param)).save(original.data, prec, gauge_param.X);
        if (comm_rank() == 0) {
          if (comm_size() == 1)
            expect_same_saved_file(file, internal_file, false);
          else
            expect_same_partfile(file, internal_file);
        }
        remove_saved_file(internal_file);
      }
      remove_saved_file(file);
    }
  }
}

// readGaugeQuda reunitarizes as GaugeIO does
TEST(ReadGaugeQudaWriteTest, reunitarize)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();

  auto gauge_param = read_gauge_param(prec);
  HostGauge original(prec), raw(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  // a MILC file is single precision, so reunitarizing in double precision changes its links
  const std::string file = "read_gauge_reunitarize.milc";
  auto io_param = quda_file_io_param(QUDA_FILE_FORMAT_MILC, QUDA_FILE_IO_PARALLEL);
  GaugeIO(file, FileIOParam(&io_param)).save(original.data, prec, gauge_param.X);
  GaugeIO(file, FileIOParam(&io_param)).load(raw.data, prec, gauge_param.X);
  io_param.reunitarize = QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2;
  expect_read_as_loaded(file, gauge_param, io_param);

  HostGauge reunitarized(prec);
  GaugeIO(file, FileIOParam(&io_param)).load(reunitarized.data, prec, gauge_param.X);
  EXPECT_FALSE(reunitarized == raw);

  remove_saved_file(file);
  remove_saved_file(file + ".info");
}

namespace
{
  // the backends that save and load a LIME format (QIO or ILDG) in this build
  std::vector<QudaFileFormat> lime_backends(QudaFileFormat format)
  {
    std::vector<QudaFileFormat> formats = {format};
#ifdef HAVE_QIO
    formats.push_back(format == QUDA_FILE_FORMAT_ILDG ? QUDA_FILE_FORMAT_ILDG_EXTERNAL : QUDA_FILE_FORMAT_QIO_EXTERNAL);
#endif
    return formats;
  }

  // QUDA's record XML of a gauge field, without usqcdInfo, written out here rather than with the library's builders
  std::string expected_gauge_record_xml(const std::string &version_and_type)
  {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><qudaGaugeFieldFile>" + version_and_type
      + "<info><subset>full</subset><parity>full</parity><nColor>3</nColor><nSpin>0</nSpin></info>"
        "</qudaGaugeFieldFile>";
  }

  // the private record XML of a gauge field holds these values
  void expect_gauge_private_record(const std::string &xml, const std::string &datatype, QudaPrecision prec)
  {
    using quda::scidac::xml_value;
    EXPECT_EQ(xml_value(xml, "datatype").value_or(""), datatype) << xml;
    EXPECT_EQ(xml_value(xml, "colors").value_or(""), "3") << xml;
    EXPECT_FALSE(xml_value(xml, "spins")) << xml;
    EXPECT_EQ(xml_value(xml, "typesize").value_or(""), std::to_string(18 * prec)) << xml;
    EXPECT_EQ(xml_value(xml, "datacount").value_or(""), "4") << xml;
  }
} // namespace

class GaugeStringsTest : public ::testing::TestWithParam<QudaFileFormat>
{
};

// QUDA's strings in gauge files, against literals, for both backends: its file XML, USQCD's datatype for SciDAC and
// ILDG files, and its record XML without usqcdInfo, in single files and in the first volume of partfiles
TEST_P(GaugeStringsTest, saved)
{
  using namespace quda;
  auto format = GetParam();
  const std::string file = "strings.lat";
  for (auto prec : {QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION}) {
    if (!is_enabled(prec)) continue;
    QudaGaugeParam gauge_param = gauge_param_for(prec);
    HostGauge gauge(prec);
    constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
    const std::string datatype = prec == QUDA_DOUBLE_PRECISION ? "USQCD_D3_ColorMatrix" : "USQCD_F3_ColorMatrix";
    for (auto backend : lime_backends(format)) {
      const char *kinds[] = {"single file", "partfile", "single file with usqcdInfo"};
      for (int k = 0; k < (format == QUDA_FILE_FORMAT_QIO ? 3 : 1); k++) {
        const bool partfile = k == 1, usqcd_info = k == 2;
        SCOPED_TRACE(std::string(get_file_format_str(backend)) + ", " + get_prec_str(prec) + ", " + kinds[k]);
        auto p = file_io_param(backend, QUDA_FILE_IO_PARALLEL, {}, partfile);
        p.write_usqcd_info = usqcd_info;
        GaugeIO(file, p).save(gauge.data, prec, gauge_param.X);
        if (comm_rank() == 0) {
          const auto s = saved_strings(partfile && comm_size() > 1 ? volume_file(file, 0) : file);
          EXPECT_EQ(s.user_file, expected_file_xml("gauge field"));
          EXPECT_EQ(s.private_record.size(), 1u);
          if (!s.private_record.empty()) {
            expect_gauge_private_record(s.private_record[0], datatype, prec);
            if (usqcd_info)
              EXPECT_TRUE(scidac::xml_value(s.user_record[0], "usqcdInfo")) << s.user_record[0];
            else
              EXPECT_EQ(s.user_record[0], expected_gauge_record_xml("<version>1.0</version>"));
          }
        }
        remove_saved_file(file);
      }
    }
  }
}

class GaugeLegacyStringsTest : public ::testing::TestWithParam<QudaFileFormat>
{
};

// gauge files with the strings QUDA saved before (FileIOParam::legacy_strings), single files and partfiles, are
// the same from both backends and load exactly with both
TEST_P(GaugeLegacyStringsTest, load)
{
  using namespace quda;
  auto format = GetParam();
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);
  // QIO only writes the ILDG records with USQCD's datatype
  const std::string datatype = format == QUDA_FILE_FORMAT_ILDG ? "USQCD_D3_ColorMatrix" : "QUDA_DNc3_GaugeField";

  // ILDG files have no partfiles
  for (bool partfile : {false, true}) {
    if (partfile && format == QUDA_FILE_FORMAT_ILDG) continue;
    SCOPED_TRACE(partfile ? "partfile" : "single file");
    const bool volumes = partfile && comm_size() > 1;
    std::vector<std::string> files;
    for (auto backend : lime_backends(format)) {
      const std::string file = std::string("legacy_") + get_file_format_str(backend) + ".lat";
      files.push_back(file);
      auto p = file_io_param(backend, QUDA_FILE_IO_PARALLEL, {}, partfile);
      p.legacy_strings = true;
      p.ildg_lfn = "lfn://quda/io_test"; // the default, the file name, differs between the files
      GaugeIO(file, p).save(original.data, prec, gauge_param.X);
      if (comm_rank() == 0) {
        SCOPED_TRACE(file);
        const auto s = saved_strings(volumes ? volume_file(file, 0) : file);
        EXPECT_EQ(s.user_file, "Dummy user file XML");
        EXPECT_EQ(s.private_record.size(), 1u);
        if (!s.private_record.empty()) {
          expect_gauge_private_record(s.private_record[0], datatype, prec);
          EXPECT_EQ(s.user_record[0], expected_gauge_record_xml("<version>BETA</version><type>" + datatype + "</type>"));
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
      for (auto backend : lime_backends(format)) {
        SCOPED_TRACE("loading " + file + " with " + get_file_format_str(backend));
        loaded.clear();
        auto info = GaugeIO(file, file_io_param(backend, QUDA_FILE_IO_PARALLEL)).load(loaded.data, prec, gauge_param.X);
        EXPECT_TRUE(loaded == original);
        EXPECT_TRUE(info.checksum_matches);
        EXPECT_EQ(info.ildg, format == QUDA_FILE_FORMAT_ILDG);
      }
    }
    for (auto &file : files) remove_saved_file(file);
  }
}

// tuple types: precision, file format, whether to write the usqcdInfo record
using gauge_mpi_io_test_t = ::testing::tuple<QudaPrecision, QudaFileFormat, bool>;

class GaugeMPIIOTest : public ::testing::TestWithParam<gauge_mpi_io_test_t>
{
};

// an MPI-IO save is the file a parallel save writes, and MPI-IO loads of either give the parallel load exactly
TEST_P(GaugeMPIIOTest, save_load)
{
  using namespace quda;
  auto prec = ::testing::get<0>(GetParam());
  auto format = ::testing::get<1>(GetParam());
  auto usqcd_info = ::testing::get<2>(GetParam());
  if (!mpi_io_enabled() || !is_enabled(prec)) GTEST_SKIP();
  const auto saved = saved_format(format);
  const bool milc = saved == QUDA_FILE_FORMAT_MILC;

  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), reference(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  auto param = [&](QudaFileFormat f, QudaFileIOMode io_mode) {
    auto p = file_io_param(f, io_mode);
    p.write_usqcd_info = usqcd_info;
    p.ildg_lfn = "lfn://quda/mpi_io_test"; // the default, the file name, differs between files
    return p;
  };
  auto load = [&](const std::string &file, const FileIOParam &p, HostGauge &gauge) {
    gauge.clear();
    return GaugeIO(file, p).load(gauge.data, prec, gauge_param.X);
  };

  const std::string ref_file = "mpi_io_ref.lat", file = "mpi_io.lat";
  GaugeIO(ref_file, param(format, QUDA_FILE_IO_PARALLEL)).save(original.data, prec, gauge_param.X);
  GaugeIO(file, param(format, QUDA_FILE_IO_MPI_IO)).save(original.data, prec, gauge_param.X);
  if (comm_rank() == 0) expect_same_saved_file(ref_file, file, milc);

  const auto ref_info = load(ref_file, param(format, QUDA_FILE_IO_PARALLEL), reference);
  EXPECT_TRUE(ref_info.checksum_matches);
  EXPECT_EQ(ref_info.has_plaq_linktr, usqcd_info);
  if (!milc) { EXPECT_TRUE(reference == original); }

  auto expect_reference = [&](const GaugeFileInfo &info) {
    EXPECT_TRUE(loaded == reference);
    EXPECT_EQ(info.format, ref_info.format);
    EXPECT_EQ(info.file_prec, ref_info.file_prec);
    EXPECT_EQ(info.ildg, ref_info.ildg);
    EXPECT_EQ(info.has_info_file, ref_info.has_info_file);
    EXPECT_TRUE(info.has_checksum);
    EXPECT_TRUE(info.checksum_matches);
    EXPECT_EQ(info.sum29, ref_info.sum29);
    EXPECT_EQ(info.sum31, ref_info.sum31);
    EXPECT_EQ(info.has_plaq_linktr, ref_info.has_plaq_linktr);
    if (info.has_plaq_linktr) {
      EXPECT_TRUE(info.plaq_linktr_matches);
      EXPECT_EQ(info.plaq, ref_info.plaq);
      EXPECT_EQ(info.linktr, ref_info.linktr);
    }
  };
  {
    SCOPED_TRACE("MPI-IO load of the MPI-IO file");
    expect_reference(load(file, param(format, QUDA_FILE_IO_MPI_IO), loaded));
  }
  {
    SCOPED_TRACE("MPI-IO load of the parallel file");
    expect_reference(load(ref_file, param(format, QUDA_FILE_IO_MPI_IO), loaded));
  }
#ifdef HAVE_QIO
  if (!milc) {
    SCOPED_TRACE("external load of the MPI-IO file");
    auto external = saved == QUDA_FILE_FORMAT_ILDG ? QUDA_FILE_FORMAT_ILDG_EXTERNAL : QUDA_FILE_FORMAT_QIO_EXTERNAL;
    expect_reference(load(file, param(external, QUDA_FILE_IO_PARALLEL), loaded));
  }
#endif

  // a wrong stored checksum is still found
  corrupt_checksum(file, saved);
  auto allow = param(format, QUDA_FILE_IO_MPI_IO);
  allow.allow_checksum_mismatch = true;
  auto corrupt_info = load(file, allow, loaded);
  EXPECT_TRUE(loaded == reference);
  EXPECT_FALSE(corrupt_info.checksum_matches);
  EXPECT_EQ(corrupt_info.sum29, ref_info.sum29);
  EXPECT_EQ(corrupt_info.sum31, ref_info.sum31);

  for (auto &f : {ref_file, file}) {
    remove_saved_file(f);
    if (milc && comm_rank() == 0 && remove((f + ".info").c_str()) != 0) errorQuda("Error deleting %s.info", f.c_str());
  }
}

// in MPI-IO mode a grid equal to the process grid groups no ranks, so it is accepted
TEST(GaugeMPIIOGridTest, process_grid)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!mpi_io_enabled() || !is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), reference(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);
  const std::array<int, 4> grid = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};

  for (auto format : {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_MILC}) {
    SCOPED_TRACE(get_file_format_str(format));
    const bool milc = format == QUDA_FILE_FORMAT_MILC;
    const std::string ref_file = "mpi_io_grid_ref.lat", file = "mpi_io_grid.lat";
    GaugeIO(ref_file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).save(original.data, prec, gauge_param.X);
    GaugeIO(file, file_io_param(format, QUDA_FILE_IO_MPI_IO, grid)).save(original.data, prec, gauge_param.X);
    if (comm_rank() == 0) expect_same_saved_file(ref_file, file, milc);

    reference.clear();
    loaded.clear();
    auto ref_info
      = GaugeIO(ref_file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(reference.data, prec, gauge_param.X);
    auto info = GaugeIO(file, file_io_param(format, QUDA_FILE_IO_MPI_IO, grid)).load(loaded.data, prec, gauge_param.X);
    EXPECT_TRUE(loaded == reference);
    EXPECT_TRUE(info.checksum_matches);
    EXPECT_EQ(info.sum29, ref_info.sum29);
    EXPECT_EQ(info.sum31, ref_info.sum31);

    for (auto &f : {ref_file, file}) {
      remove_saved_file(f);
      if (milc && comm_rank() == 0 && remove((f + ".info").c_str()) != 0)
        errorQuda("Error deleting %s.info", f.c_str());
    }
  }
}

// an MPI-IO save over a larger file truncates it: the file is the one a fresh parallel save writes
TEST(GaugeMPIIOOverwriteTest, larger_file)
{
  using namespace quda;
  if (!mpi_io_enabled() || !is_enabled(QUDA_DOUBLE_PRECISION) || !is_enabled(QUDA_SINGLE_PRECISION)) GTEST_SKIP();
  QudaGaugeParam large_param = gauge_param_for(QUDA_DOUBLE_PRECISION),
                 small_param = gauge_param_for(QUDA_SINGLE_PRECISION);
  HostGauge large(QUDA_DOUBLE_PRECISION), small(QUDA_SINGLE_PRECISION);
  constructHostGaugeField(large.data, large_param, 0, nullptr);
  constructHostGaugeField(small.data, small_param, 0, nullptr);

  for (auto format : {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_MILC}) {
    SCOPED_TRACE(get_file_format_str(format));
    const bool milc = format == QUDA_FILE_FORMAT_MILC;
    const std::string ref_file = "mpi_io_overwrite_ref.lat", file = "mpi_io_overwrite.lat";
    // double-precision SciDAC, larger than either single-precision file
    GaugeIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL))
      .save(large.data, QUDA_DOUBLE_PRECISION, large_param.X);
    const size_t old_size = comm_rank() == 0 ? std::filesystem::file_size(file) : 0;

    GaugeIO(file, file_io_param(format, QUDA_FILE_IO_MPI_IO)).save(small.data, QUDA_SINGLE_PRECISION, small_param.X);
    GaugeIO(ref_file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).save(small.data, QUDA_SINGLE_PRECISION, small_param.X);
    if (comm_rank() == 0) {
      EXPECT_GT(old_size, std::filesystem::file_size(ref_file));
      expect_same_saved_file(ref_file, file, milc);
    }
    HostGauge loaded(QUDA_SINGLE_PRECISION);
    loaded.clear();
    auto info
      = GaugeIO(file, file_io_param(format, QUDA_FILE_IO_MPI_IO)).load(loaded.data, QUDA_SINGLE_PRECISION, small_param.X);
    EXPECT_TRUE(loaded == small);
    EXPECT_TRUE(info.checksum_matches);

    for (auto &f : {ref_file, file}) {
      remove_saved_file(f);
      if (milc && comm_rank() == 0 && remove((f + ".info").c_str()) != 0)
        errorQuda("Error deleting %s.info", f.c_str());
    }
  }
}

// partfiles go through POSIX I/O as in parallel mode, so an MPI-IO partfile save writes the parallel volumes, and
// MPI-IO loads them exactly; on one rank a partfile is a single file, saved through MPI-IO
TEST(GaugeMPIIOPartfileTest, volumes)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!mpi_io_enabled() || !is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge original(prec), loaded(prec);
  constructHostGaugeField(original.data, gauge_param, 0, nullptr);

  auto check = [&](FileIOParam parallel) {
    auto mpi_io = parallel;
    mpi_io.io_mode = QUDA_FILE_IO_MPI_IO;
    const std::string ref_file = "mpi_io_partfile_ref.lat", file = "mpi_io_partfile.lat";
    GaugeIO(ref_file, parallel).save(original.data, prec, gauge_param.X);
    GaugeIO(file, mpi_io).save(original.data, prec, gauge_param.X);
    const bool volumes = comm_size() > 1 || parallel.partfile_grid != std::array<int, 4> {};
    if (comm_rank() == 0) {
      EXPECT_EQ(saved_volumes(file).empty(), !volumes);
      if (volumes)
        expect_same_partfile(ref_file, file);
      else
        expect_same_saved_file(ref_file, file, false);
    }
    loaded.clear();
    auto info
      = GaugeIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_MPI_IO)).load(loaded.data, prec, gauge_param.X);
    EXPECT_TRUE(loaded == original);
    EXPECT_TRUE(info.checksum_matches);
    remove_saved_file(ref_file);
    remove_saved_file(file);
  };

  check(partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL));

  // volumes as if saved by another rank layout
  auto p = partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  for (int d = 0; d < 4; d++) {
    const int N = gauge_param.X[d] * comm_dim(d);
    p.partfile_grid[d] = N % 3 == 0 ? 3 : N % 2 == 0 ? 2 : 1;
  }
  SCOPED_TRACE("volumes " + grid_str(p.partfile_grid));
  check(p);
}

namespace
{
  const std::string xml_declaration = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";

  // MILC's user file XML of gauge fields (generic/io_scidac.c)
  const std::string milc_file_xml = xml_declaration + "<title>MILC ILDG archival gauge configuration</title>";

  // the lines of a MILC .info file as create_MILC_info (ks_imp_dyn/gauge_info.c) puts them in usqcdInfo's <info>,
  // from configs/24x24x24x24.chklat.info
  const std::string milc_info_lines
    = "action.description = \"Gauge plus fermion\"\n"
      "gauge.description = \"Symanzik 1x1 + 1x2 + 1x1x1 action with HISQ quark loops\"\n"
      "gauge.nloops = 3\n"
      "gauge.nreps = 1\n"
      "gauge.beta11 = 6.300000\n"
      "gauge.tadpole.u0 = 0.890000\n"
      "gauge.ssplaq = 1.623083\n"
      "gauge.stplaq = 1.624976\n"
      "gauge.nersc_linktr = -2.204722e-04\n"
      "gauge.nersc_checksum = 2510161697\n"
      "quark.description = \"HISQ action version 1\"\n"
      "quark.dyn_flavors =[3] 2 1 1\n"
      "quark.dyn_mass =[3] 0.001000 0.050000 0.500000\n";

  // a usqcdInfo record as QIO's encoder writes it, which MILC uses (create_QCDML), with the plaquette and link
  // trace in a format and the content of <info> (none: no <info> element)
  std::string usqcd_record(const char *fmt, double plaq, double linktr, const std::optional<std::string> &info)
  {
    char p[32], l[32];
    snprintf(p, sizeof(p), fmt, plaq);
    snprintf(l, sizeof(l), fmt, linktr);
    return xml_declaration + "<usqcdInfo><version>1.0</version><plaq>" + p + "</plaq><linktr>" + l + "</linktr>"
      + (info ? "<info>" + *info + "</info>" : "") + "</usqcdInfo>";
  }

  // the <date> of the private record XML and the ILDG LFN of a file's field, read from the file on the calling rank
  struct FieldStrings {
    std::optional<std::string> date;
    std::optional<std::string> lfn;
  };

  FieldStrings field_strings(const std::string &file)
  {
    using namespace quda::scidac;
    const auto contents = read_file(file);
    FieldStrings s;
    for (auto &m : group_messages(local_records(file))) {
      if (!find_record(m, binary_data) && !find_record(m, ildg_binary_data)) continue;
      if (auto r = find_record(m, private_record_xml)) {
        const auto xml = record_string(contents, r);
        const auto start = xml.find("<date>"), end = xml.find("</date>");
        if (start != std::string::npos && end != std::string::npos) s.date = xml.substr(start + 6, end - start - 6);
      }
      if (auto r = find_record(m, ildg_data_lfn)) s.lfn = record_string(contents, r);
      break;
    }
    return s;
  }

  void expect_same_metadata(const quda::GaugeMetadata &a, const quda::GaugeMetadata &b)
  {
    EXPECT_EQ(a.user_file_xml, b.user_file_xml);
    EXPECT_EQ(a.user_record_xml, b.user_record_xml);
    EXPECT_EQ(a.has_usqcd_info, b.has_usqcd_info);
    EXPECT_EQ(a.usqcd_info, b.usqcd_info);
    EXPECT_EQ(a.ildg_lfn, b.ildg_lfn);
    EXPECT_EQ(a.record_date, b.record_date);
    EXPECT_EQ(a.milc_time_stamp, b.milc_time_stamp);
    EXPECT_EQ(a.milc_info.size(), b.milc_info.size());
    for (auto i = 0u; i < std::min(a.milc_info.size(), b.milc_info.size()); i++) {
      EXPECT_EQ(a.milc_info[i].key, b.milc_info[i].key);
      EXPECT_EQ(a.milc_info[i].line, b.milc_info[i].line);
    }
  }

  // load a gauge field with each backend that reads LIME files, expecting the links given; returns each load's info
  std::vector<quda::GaugeFileInfo> load_lime_backends(const std::string &file, const HostGauge &expected,
                                                      bool external = true)
  {
    using namespace quda;
    std::vector<GaugeFileInfo> infos;
    HostGauge loaded(expected.prec);
    for (auto backend : lime_backends(QUDA_FILE_FORMAT_QIO)) {
      if (backend != QUDA_FILE_FORMAT_QIO && !external) continue;
      SCOPED_TRACE(std::string("loaded with ") + get_file_format_str(backend));
      loaded.clear();
      infos.push_back(GaugeIO(file, file_io_param(backend, QUDA_FILE_IO_PARALLEL))
                        .load(loaded.data, expected.prec, gauge_param_for(expected.prec).X));
      EXPECT_TRUE(loaded == expected);
    }
    if (infos.size() == 2) {
      SCOPED_TRACE("native and external loads");
      expect_same_metadata(infos[0].metadata, infos[1].metadata);
    }
    return infos;
  }

  // a double-precision gauge field's links rounded to float, as a single-precision file holds them
  void round_to_float(const HostGauge &gauge, HostGauge &rounded)
  {
    for (int dir = 0; dir < 4; dir++)
      for (size_t i = 0; i < V * gauge_site_size; i++)
        static_cast<double *>(rounded.data[dir])[i] = static_cast<float>(static_cast<double *>(gauge.data[dir])[i]);
  }

  // write a gauge field file with the given strings through the native backend's writer, in single precision
  void write_gauge_with_strings(const std::string &file, const HostGauge &gauge, const std::string &user_file_xml,
                                const std::string &user_record_xml, bool ildg = false,
                                const std::optional<std::string> &lfn = {}, const quda::FileIOParam &param = {})
  {
    using namespace quda;
    scidac::GaugeInfo info;
    info.ildg = ildg;
    auto g = scidac::gauge_records(QUDA_SINGLE_PRECISION, info);
    g.record.user_xml = user_record_xml;
    g.record.ildg_lfn = lfn;
    const void *const *data = gauge.data;
    scidac::write_fields(file, user_file_xml, {g.record}, {io::HostField(data, 4, gauge.prec, 18)},
                         gauge_param_for(gauge.prec).X, QUDA_INVALID_PARITY, param);
    comm_barrier();
  }

  // keep only the ILDG records of a LIME file, in one message, as in ILDG files from other codes (rank 0)
  void keep_ildg_records(const std::string &in, const std::string &out)
  {
    using namespace quda;
    const auto contents = read_file(in);
    std::vector<lime::Record> kept;
    for (auto &r : local_records(in))
      if (r.type == scidac::ildg_format || r.type == scidac::ildg_data_lfn || r.type == scidac::ildg_binary_data)
        kept.push_back(r);
    EXPECT_EQ(kept.size(), 3u);
    std::vector<char> result;
    for (auto i = 0u; i < kept.size(); i++) {
      const auto &r = kept[i];
      const auto start = contents.begin() + (r.data_offset - lime::header_bytes);
      const auto end = contents.begin() + (r.data_offset + r.data_bytes + lime::padding(r.data_bytes));
      const auto header = result.size();
      result.insert(result.end(), start, end);
      result[header + 6] = static_cast<char>((i == 0 ? 0x80 : 0) | (i + 1 == kept.size() ? 0x40 : 0));
    }
    EXPECT_TRUE(write_file(out, result));
  }
} // namespace

// the metadata of QUDA's own SciDAC and ILDG files, saved by either backend and loaded by both
TEST(GaugeMetadataTest, quda_files)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);

  const std::string file = "metadata.lat";
  struct Case {
    QudaFileFormat format;
    bool usqcd_info;
    std::string lfn;
  };
  const Case cases[] = {{QUDA_FILE_FORMAT_QIO, false, ""},
                        {QUDA_FILE_FORMAT_QIO, true, ""},
                        {QUDA_FILE_FORMAT_ILDG, false, ""},
                        {QUDA_FILE_FORMAT_ILDG, false, "lfn://quda/metadata_test"},
                        {QUDA_FILE_FORMAT_ILDG, true, "lfn://quda/metadata_test"}};
  for (auto &c : cases) {
    for (auto saver : lime_backends(c.format)) {
      SCOPED_TRACE(std::string("saved as ") + get_file_format_str(saver) + (c.usqcd_info ? " with usqcdInfo" : "")
                   + (c.lfn.empty() ? "" : " with an LFN"));
      auto p = file_io_param(saver, QUDA_FILE_IO_PARALLEL);
      p.write_usqcd_info = c.usqcd_info;
      p.ildg_lfn = c.lfn;
      GaugeIO(file, p).save(gauge.data, prec, gauge_param.X);
      const auto strings = field_strings(file);
      const bool ildg = c.format == QUDA_FILE_FORMAT_ILDG;

      for (auto &info : load_lime_backends(file, gauge)) {
        const auto &m = info.metadata;
        EXPECT_EQ(m.user_file_xml, expected_file_xml("gauge field"));
        EXPECT_EQ(m.has_usqcd_info, c.usqcd_info);
        if (c.usqcd_info) {
          // the plaquette and link trace of usqcdInfo are those of gauge_observables
          EXPECT_EQ(m.user_record_xml, usqcd_record("%.16e", obs.plaq, obs.linktr, "QUDA"));
          EXPECT_EQ(m.usqcd_info, std::optional<std::string>("QUDA"));
          EXPECT_EQ(info.plaq, obs.plaq);
          EXPECT_EQ(info.linktr, obs.linktr);
        } else {
          EXPECT_EQ(m.user_record_xml, expected_gauge_record_xml("<version>1.0</version>"));
          EXPECT_FALSE(m.usqcd_info);
        }
        if (ildg)
          EXPECT_EQ(m.ildg_lfn, std::optional<std::string>(c.lfn.empty() ? file : c.lfn));
        else
          EXPECT_FALSE(m.ildg_lfn);
        EXPECT_EQ(m.ildg_lfn, strings.lfn);
        // asctime() in UTC, with its newline replaced by " UTC", as QIO dates records
        EXPECT_TRUE(strings.date);
        EXPECT_EQ(m.record_date, strings.date);
        const auto date = strings.date.value_or("");
        EXPECT_EQ(date.size(), 28u) << date;
        EXPECT_EQ(date.substr(date.size() - std::min<size_t>(date.size(), 4)), " UTC") << date;
        EXPECT_TRUE(m.milc_time_stamp.empty());
        EXPECT_TRUE(m.milc_info.empty());
      }
      remove_saved_file(file);
    }
  }
}

// files with MILC's strings, and with strings of other codes, written through the native writer: every string is
// read back exactly by both backends
TEST(GaugeMetadataTest, foreign_strings)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec), rounded(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  // the files are single precision, as MILC writes them
  round_to_float(gauge, rounded);
  const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);
  const std::string milc_record = usqcd_record("%.6e", obs.plaq, obs.linktr, milc_info_lines);

  // a Chroma-like file and record XML, with white space and nested elements of the same name
  const std::string other_file_xml = "<?xml version=\"1.0\"?>\n<HMC>\n  <ProgramInfo>\n    <code>chroma</code>\n"
                                     "  </ProgramInfo>\n</HMC>\n";
  const std::string other_record_xml = "<?xml version=\"1.0\"?>\n<Params>\n  <Params><info>beta 6.3</info></Params>\n"
                                       "  <MesPlq>0.54</MesPlq>\n</Params>\n";

  struct Case {
    std::string name;
    std::string file_xml;
    std::string record_xml;
    bool ildg;
    std::optional<std::string> lfn;
    bool usqcd_info;
    std::optional<std::string> info;
  };
  const Case cases[] = {
    {"MILC SciDAC", milc_file_xml, milc_record, false, {}, true, milc_info_lines},
    {"MILC ILDG", milc_file_xml, milc_record, true, "lfn://USQCD/MILC/test/metadata", true, milc_info_lines},
    {"empty <info>", milc_file_xml, usqcd_record("%.6e", obs.plaq, obs.linktr, ""), false, {}, true, ""},
    {"no <info>", milc_file_xml, usqcd_record("%.6e", obs.plaq, obs.linktr, {}), false, {}, true, {}},
    {"other code", other_file_xml, other_record_xml, true, "lfn://other/metadata", false, {}},
  };
  const std::string file = "metadata_foreign.lat";
  for (auto &c : cases) {
    SCOPED_TRACE(c.name);
    write_gauge_with_strings(file, gauge, c.file_xml, c.record_xml, c.ildg, c.lfn);
    const auto strings = field_strings(file);
    const auto infos = load_lime_backends(file, rounded);
    for (auto &info : infos) {
      const auto &m = info.metadata;
      EXPECT_EQ(m.user_file_xml, c.file_xml);
      EXPECT_EQ(m.user_record_xml, c.record_xml);
      EXPECT_EQ(m.has_usqcd_info, c.usqcd_info);
      EXPECT_EQ(m.usqcd_info, c.info);
      EXPECT_EQ(m.ildg_lfn, c.lfn);
      EXPECT_TRUE(strings.date);
      EXPECT_EQ(m.record_date, strings.date);
      EXPECT_EQ(info.ildg, c.ildg);
      EXPECT_EQ(info.has_plaq_linktr, c.usqcd_info);
      if (info.has_plaq_linktr) { EXPECT_TRUE(info.plaq_linktr_matches); }
    }
    if (mpi_io_enabled()) {
      SCOPED_TRACE("MPI-IO load");
      HostGauge loaded(prec);
      loaded.clear();
      auto info
        = GaugeIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_MPI_IO)).load(loaded.data, prec, gauge_param.X);
      EXPECT_TRUE(loaded == rounded);
      expect_same_metadata(info.metadata, infos[0].metadata);
    }
    remove_saved_file(file);
  }
}

// the metadata of a partfile comes from its first volume
TEST(GaugeMetadataTest, partfile)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec), rounded(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  round_to_float(gauge, rounded);
  const std::string file = "metadata_part.lat";

  // QUDA's partfile, one volume per rank (a single file on one rank), saved and loaded by either backend
  for (auto saver : lime_backends(QUDA_FILE_FORMAT_QIO)) {
    SCOPED_TRACE(std::string("saved as ") + get_file_format_str(saver));
    auto p = partfile_param(saver, QUDA_FILE_IO_PARALLEL);
    p.write_usqcd_info = true;
    GaugeIO(file, p).save(gauge.data, prec, gauge_param.X);
    const auto strings = field_strings(comm_size() > 1 ? volume_file(file, 0) : file);
    for (auto &info : load_lime_backends(file, gauge)) {
      EXPECT_EQ(info.metadata.user_file_xml, expected_file_xml("gauge field"));
      EXPECT_TRUE(info.metadata.has_usqcd_info);
      EXPECT_EQ(info.metadata.usqcd_info, std::optional<std::string>("QUDA"));
      EXPECT_TRUE(strings.date);
      EXPECT_EQ(info.metadata.record_date, strings.date);
    }
    remove_saved_file(file);
  }

  // MILC's strings in a partfile of two volumes in t, as if saved by another rank layout (which only the native
  // backend reads)
  auto p = partfile_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  p.partfile_grid = {1, 1, 1, 2};
  const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);
  const auto record = usqcd_record("%.6e", obs.plaq, obs.linktr, milc_info_lines);
  write_gauge_with_strings(file, gauge, milc_file_xml, record, false, {}, p);
  EXPECT_EQ(saved_volumes(file).size(), 2u);
  const auto strings = field_strings(volume_file(file, 0));
  for (auto &info : load_lime_backends(file, rounded, false)) {
    EXPECT_EQ(info.metadata.user_file_xml, milc_file_xml);
    EXPECT_EQ(info.metadata.user_record_xml, record);
    EXPECT_EQ(info.metadata.usqcd_info, std::optional<std::string>(milc_info_lines));
    EXPECT_TRUE(strings.date);
    EXPECT_EQ(info.metadata.record_date, strings.date);
  }
  remove_saved_file(file);
}

// an ILDG file from another code, with only the ILDG records: an LFN, but no SciDAC strings or date; QIO reads no
// LFN from such files
TEST(GaugeMetadataTest, ildg_only)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const std::string saved = "metadata_ildg.lat", file = "metadata_ildg_only.lat";
  auto p = file_io_param(QUDA_FILE_FORMAT_ILDG, QUDA_FILE_IO_PARALLEL);
  p.ildg_lfn = "lfn://other/ildg_only";
  GaugeIO(saved, p).save(gauge.data, prec, gauge_param.X);
  if (comm_rank() == 0) keep_ildg_records(saved, file);
  comm_barrier();

  const auto infos = load_lime_backends(file, gauge, false);
#ifdef HAVE_QIO
  HostGauge loaded(prec);
  loaded.clear();
  auto external = GaugeIO(file, file_io_param(QUDA_FILE_FORMAT_QIO_EXTERNAL, QUDA_FILE_IO_PARALLEL))
                    .load(loaded.data, prec, gauge_param.X);
  EXPECT_TRUE(loaded == gauge);
  EXPECT_FALSE(external.metadata.ildg_lfn);
  auto expected = infos[0].metadata;
  expected.ildg_lfn.reset();
  expect_same_metadata(external.metadata, expected);
#endif
  for (auto &info : infos) {
    const auto &m = info.metadata;
    EXPECT_TRUE(m.user_file_xml.empty());
    EXPECT_TRUE(m.user_record_xml.empty());
    EXPECT_FALSE(m.has_usqcd_info);
    EXPECT_EQ(m.ildg_lfn, std::optional<std::string>("lfn://other/ildg_only"));
    EXPECT_FALSE(m.record_date);
    EXPECT_TRUE(info.ildg);
    EXPECT_FALSE(info.has_checksum);
  }
  remove_saved_file(saved);
  remove_saved_file(file);
}

// a MILC file's header time stamp, and its .info lines in file order as they are written, whatever their order
// and form
TEST(GaugeMetadataTest, milc_info)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec), loaded(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const std::string file = "metadata.milc";
  auto p = file_io_param(QUDA_FILE_FORMAT_MILC, QUDA_FILE_IO_PARALLEL);
  p.write_usqcd_info = true;
  GaugeIO(file, p).save(gauge.data, prec, gauge_param.X);

  // the header's time stamp, and the .info file's lines by key
  auto contents = read_file(file);
  EXPECT_GE(contents.size(), 96u);
  contents.resize(std::max<size_t>(contents.size(), 96));
  const std::string stamp(contents.data() + 20, strnlen(contents.data() + 20, 64));
  EXPECT_FALSE(stamp.empty());
  std::map<std::string, std::string> saved;
  {
    std::ifstream in(file + ".info");
    for (std::string line; std::getline(in, line);) saved[line.substr(0, line.find(' '))] = line;
  }
  // QUDA's lines and the four computed keys
  EXPECT_EQ(saved.size(), 11u);

  // MILC's array lines, a "key value" line, an indented one, one without a key, blank lines and a CR
  const std::vector<std::pair<std::string, std::string>> items = {
    {"quark.dyn_flavors", "quark.dyn_flavors =[3] 2 1 1"},
    {"nt", saved["nt"]},
    {"gauge.beta11", "  gauge.beta11 6.300000"},
    {"checksums", saved["checksums"]},
    {"", "= 1"},
    {"magic_number", saved["magic_number"]},
    {"time_stamp", saved["time_stamp"]},
    {"quark.dyn_mass", "quark.dyn_mass =[3] 0.001000 0.050000 0.500000\r"},
    {"nz", saved["nz"]},
    {"ny", saved["ny"]},
    {"nx", saved["nx"]},
    {"action.description", "action.description = \"Gauge plus fermion\""},
  };
  // every rank has read the saved .info file before it is replaced
  comm_barrier();
  if (comm_rank() == 0) {
    std::ofstream out(file + ".info", std::ios::trunc);
    for (auto i = 0u; i < items.size(); i++) out << items[i].second << (i % 4 == 1 ? "\n\n  \n" : "\n");
  }
  comm_barrier();

  loaded.clear();
  auto info
    = GaugeIO(file, file_io_param(QUDA_FILE_FORMAT_MILC, QUDA_FILE_IO_PARALLEL)).load(loaded.data, prec, gauge_param.X);
  EXPECT_TRUE(info.has_info_file);
  EXPECT_TRUE(info.checksum_matches);
  const auto &m = info.metadata;
  EXPECT_EQ(m.milc_time_stamp, stamp);
  EXPECT_EQ(m.milc_info.size(), items.size());
  for (auto i = 0u; i < std::min(items.size(), m.milc_info.size()); i++) {
    EXPECT_EQ(m.milc_info[i].key, items[i].first);
    EXPECT_EQ(m.milc_info[i].line, items[i].second);
  }
  EXPECT_TRUE(m.user_file_xml.empty());
  EXPECT_TRUE(m.user_record_xml.empty());
  EXPECT_FALSE(m.ildg_lfn);
  EXPECT_FALSE(m.record_date);

  remove_saved_file(file);
  remove_saved_file(file + ".info");
}

namespace
{
  // a string of XML longer than QIO's own XML strings (QIO_STRINGALLOC, 1024 bytes)
  std::string long_xml(const std::string &element)
  {
    std::string xml = xml_declaration + "<" + element + ">\n";
    for (int i = 0; i < 40; i++) xml += "  <line>" + std::string(40, static_cast<char>('a' + i % 26)) + "</line>\n";
    return xml + "</" + element + ">\n";
  }

  // the metadata of a save, and what a load of the file must find
  struct SaveCase {
    std::string name;
    QudaFileFormat format; // QIO or ILDG
    bool write_usqcd_info;
    quda::GaugeMetadata given;
    quda::GaugeMetadata expected; // the record date is that of the file when none is given
    bool partfile = false;
  };

  // save a gauge field with each backend, given the metadata, and expect the same files from both, byte for byte
  // when the date is given; returns the files saved
  std::vector<std::string> save_lime_backends(const SaveCase &c, const HostGauge &gauge)
  {
    using namespace quda;
    std::vector<std::string> files;
    for (auto saver : lime_backends(c.format)) {
      const std::string file = std::string("save_metadata_") + get_file_format_str(saver) + ".lat";
      auto p = file_io_param(saver, QUDA_FILE_IO_PARALLEL, {}, c.partfile);
      p.write_usqcd_info = c.write_usqcd_info;
      p.ildg_lfn = "lfn://quda/param"; // the default, the file name, differs between the files
      GaugeIO(file, p).save(gauge.data, gauge.prec, gauge_param_for(gauge.prec).X, c.given);
      files.push_back(file);
    }
    if (files.size() == 2 && comm_rank() == 0) {
      SCOPED_TRACE("native and external saves");
      const bool volumes = c.partfile && comm_size() > 1;
      auto a = volumes ? saved_volumes(files[0]) : std::vector<std::string> {files[0]};
      auto b = volumes ? saved_volumes(files[1]) : std::vector<std::string> {files[1]};
      EXPECT_EQ(a.size(), b.size());
      for (auto i = 0u; i < std::min(a.size(), b.size()); i++) {
        if (c.given.record_date || i > 0)
          expect_same_file(a[i], b[i]);
        else
          expect_same_saved_file(a[i], b[i], false);
      }
    }
    return files;
  }
} // namespace

// metadata given to a SciDAC or ILDG save is written by either backend, the same bytes from both when the date is
// given, and comes back exactly when the file is loaded by either backend
TEST(GaugeSaveMetadataTest, lime)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);
  auto usqcd = [&](const std::string &info) { return usqcd_record("%.16e", obs.plaq, obs.linktr, info); };

  const std::string other_file_xml = "<?xml version=\"1.0\"?>\n<HMC>\n  <ProgramInfo>\n    <code>chroma</code>\n"
                                     "  </ProgramInfo>\n</HMC>\n";
  const std::string other_record_xml = "<?xml version=\"1.0\"?>\n<Params>\n  <Params><info>beta 6.3</info></Params>\n"
                                       "  <MesPlq>0.54</MesPlq>\n</Params>\n";
  // MILC-style lines with an element of usqcdInfo's own name in them, which QIO's XML parser can't nest
  const std::string nested_info = "gauge.description = \"<info>nested</info>\"\n<info>gauge.nloops = 3</info>\n";
  const std::string date = "Sun Dec  4 20:49:06 2005 UTC";

  // QUDA's strings, with a usqcdInfo record with the given <info>, and the other fields as given
  auto quda_strings = [&](const std::optional<std::string> &info) {
    GaugeMetadata m;
    m.user_file_xml = expected_file_xml("gauge field");
    m.user_record_xml = info ? usqcd(*info) : expected_gauge_record_xml("<version>1.0</version>");
    m.has_usqcd_info = info.has_value();
    m.usqcd_info = info;
    return m;
  };

  std::vector<SaveCase> cases;
  {
    SaveCase c {"file XML", QUDA_FILE_FORMAT_QIO, true, {}, quda_strings("QUDA")};
    c.given.user_file_xml = other_file_xml;
    c.expected.user_file_xml = other_file_xml;
    cases.push_back(c);
  }
  {
    SaveCase c {"long file and record XML", QUDA_FILE_FORMAT_QIO, true, {}, {}};
    c.given.user_file_xml = long_xml("file");
    c.given.user_record_xml = long_xml("record");
    c.expected = c.given;
    cases.push_back(c);
  }
  {
    SaveCase c {"record XML, LFN and date", QUDA_FILE_FORMAT_ILDG, true, {}, {}};
    c.given.user_record_xml = other_record_xml;
    c.given.ildg_lfn = "lfn://other/given";
    c.given.record_date = date;
    c.expected = c.given;
    c.expected.user_file_xml = expected_file_xml("gauge field");
    cases.push_back(c);
  }
  for (auto format : {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG}) {
    // as MILC writes them, with usqcdInfo's old plaquette and link trace, which are computed afresh
    SaveCase c {"MILC's strings", format, true, {}, {}};
    c.given.user_file_xml = milc_file_xml;
    c.given.user_record_xml = usqcd_record("%.6e", 0.5, 0.1, milc_info_lines);
    c.given.has_usqcd_info = true;
    c.given.usqcd_info = milc_info_lines;
    c.given.record_date = date;
    if (format == QUDA_FILE_FORMAT_ILDG) c.given.ildg_lfn = "lfn://USQCD/MILC/test/save_metadata";
    c.expected = c.given;
    c.expected.user_record_xml = usqcd(milc_info_lines);
    cases.push_back(c);
  }
  {
    SaveCase c {"nested <info>", QUDA_FILE_FORMAT_QIO, true, {}, quda_strings(nested_info)};
    c.given.has_usqcd_info = true;
    c.given.usqcd_info = nested_info;
    cases.push_back(c);
  }
  {
    SaveCase c {"usqcdInfo with an empty <info>", QUDA_FILE_FORMAT_QIO, true, {}, quda_strings("")};
    c.given.has_usqcd_info = true;
    c.given.usqcd_info = "";
    cases.push_back(c);
  }
  {
    // no <info> means QUDA's
    SaveCase c {"usqcdInfo without <info>", QUDA_FILE_FORMAT_QIO, true, {}, quda_strings("QUDA")};
    c.given.has_usqcd_info = true;
    cases.push_back(c);
  }
  {
    SaveCase c {"usqcdInfo without write_usqcd_info", QUDA_FILE_FORMAT_QIO, false, {}, quda_strings({})};
    c.given.has_usqcd_info = true;
    c.given.usqcd_info = milc_info_lines;
    cases.push_back(c);
  }
  {
    SaveCase c {"LFN", QUDA_FILE_FORMAT_ILDG, false, {}, quda_strings({})};
    c.given.ildg_lfn = "lfn://other/given";
    c.expected.ildg_lfn = c.given.ildg_lfn;
    cases.push_back(c);
  }
  {
    // an empty LFN counts as none
    SaveCase c {"empty LFN", QUDA_FILE_FORMAT_ILDG, false, {}, quda_strings({})};
    c.given.ildg_lfn = "";
    c.expected.ildg_lfn = "lfn://quda/param";
    cases.push_back(c);
  }
  {
    // the metadata of the first volume, which differs between the backends only by the date
    SaveCase c {"partfile", QUDA_FILE_FORMAT_QIO, true, {}, {}, true};
    c.given.user_file_xml = milc_file_xml;
    c.given.has_usqcd_info = true;
    c.given.usqcd_info = milc_info_lines;
    c.given.record_date = date;
    c.expected = c.given;
    c.expected.user_record_xml = usqcd(milc_info_lines);
    cases.push_back(c);
  }

  for (auto &c : cases) {
    SCOPED_TRACE(c.name + " (" + get_file_format_str(c.format) + ")");
    for (auto &file : save_lime_backends(c, gauge)) {
      SCOPED_TRACE("saved as " + file);
      const auto strings = field_strings(c.partfile && comm_size() > 1 ? volume_file(file, 0) : file);
      auto expected = c.expected;
      if (!expected.record_date) expected.record_date = strings.date;
      EXPECT_TRUE(strings.date);
      for (auto &info : load_lime_backends(file, gauge)) {
        expect_same_metadata(info.metadata, expected);
        EXPECT_EQ(info.ildg, c.format == QUDA_FILE_FORMAT_ILDG);
        EXPECT_EQ(info.has_plaq_linktr, c.expected.has_usqcd_info);
        if (info.has_plaq_linktr) {
          EXPECT_TRUE(info.plaq_linktr_matches);
          EXPECT_EQ(info.plaq, obs.plaq);
          EXPECT_EQ(info.linktr, obs.linktr);
        }
      }
      remove_saved_file(file);
    }
  }
}

// a MILC save given a time stamp and .info lines: the time stamp in the header and the .info file, the lines after
// QUDA's, and QUDA's lines in place of given lines of the same keys, so that MILC's own order is kept
TEST(GaugeSaveMetadataTest, milc)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec), loaded(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const auto computed = computed_info_lines(gauge.data, prec, gauge_param.X);
  const std::string file = "save_metadata.milc";

  // QUDA's lines by key, for a time stamp and the checksums of the file
  auto quda_line = [&](const std::string &key, const std::string &stamp, const GaugeFileInfo &info) {
    const char *dims[] = {"nx", "ny", "nz", "nt"};
    for (int d = 0; d < 4; d++)
      if (key == dims[d]) return key + " = " + std::to_string(gauge_param.X[d] * comm_dim(d));
    const char *computed_keys[] = {"gauge.ssplaq", "gauge.stplaq", "gauge.nersc_linktr", "gauge.nersc_checksum"};
    for (int k = 0; k < 4; k++)
      if (key == computed_keys[k]) return computed[k];
    if (key == "magic_number") return std::string("magic_number = 20103");
    if (key == "time_stamp") return "time_stamp = \"" + stamp + "\"";
    char sums[64];
    snprintf(sums, sizeof(sums), "checksums = \"%x %x\"", info.sum29, info.sum31);
    return std::string(sums);
  };
  const std::vector<std::string> quda_keys = {"magic_number", "time_stamp", "checksums", "nx", "ny", "nz", "nt"};
  const std::vector<std::string> computed_keys
    = {"gauge.ssplaq", "gauge.stplaq", "gauge.nersc_linktr", "gauge.nersc_checksum"};

  // configs/24x24x24x24.chklat.info, as MILC wrote it
  std::vector<MILCInfoItem> milc_lines = {{"magic_number", "magic_number = 20103"},
                                          {"time_stamp", "time_stamp = \"Sat Nov 18 17:30:04 2017\""},
                                          {"checksums", "checksums = \"fc4e1f56 74b0fe7c\""},
                                          {"nx", "nx = 24"},
                                          {"ny", "ny = 24"},
                                          {"nz", "nz = 24"},
                                          {"nt", "nt = 24"}};
  {
    std::istringstream in(milc_info_lines);
    for (std::string line; std::getline(in, line);) milc_lines.push_back({line.substr(0, line.find(' ')), line});
  }
  EXPECT_EQ(milc_lines.size(), 20u);

  // lines out of MILC's order: QUDA's keys given once or twice, array lines, a "key value" line, one without a
  // key, a line ending in a CR, and a key of MILC's given twice
  const std::vector<MILCInfoItem> other_lines = {{"quark.dyn_flavors", "quark.dyn_flavors =[3] 2 1 1"},
                                                 {"nx", "nx = 99"},
                                                 {"", "= 1"},
                                                 {"gauge.ssplaq", "gauge.ssplaq = 9.999999"},
                                                 {"nx", "nx = 98"},
                                                 {"quark.dyn_mass", "quark.dyn_mass =[3] 0.001000 0.050000 0.500000\r"},
                                                 {"gauge.beta11", "  gauge.beta11 6.300000"},
                                                 {"action.description", "action.description = \"Gauge plus fermion\""},
                                                 {"action.description", "action.description = \"twice\""}};

  struct Case {
    std::string name;
    bool write_usqcd_info;
    std::string stamp;
    std::vector<MILCInfoItem> lines;
  };
  const Case cases[] = {{"MILC's .info", true, "Sat Nov 18 17:30:04 2017", milc_lines},
                        {"other lines", true, "Thu Oct 22 18:00:00 2026", other_lines},
                        {"other lines without write_usqcd_info", false, std::string(64, 'x'), other_lines}};
  for (auto &c : cases) {
    SCOPED_TRACE(c.name);
    GaugeMetadata given;
    given.milc_time_stamp = c.stamp;
    given.milc_info = c.lines;
    auto p = file_io_param(QUDA_FILE_FORMAT_MILC, QUDA_FILE_IO_PARALLEL);
    p.write_usqcd_info = c.write_usqcd_info;
    GaugeIO(file, p).save(gauge.data, prec, gauge_param.X, given);
    loaded.clear();
    const auto info = GaugeIO(file, p).load(loaded.data, prec, gauge_param.X);
    EXPECT_TRUE(info.checksum_matches);
    EXPECT_EQ(info.has_plaq_linktr, c.write_usqcd_info);
    if (info.has_plaq_linktr) { EXPECT_TRUE(info.plaq_linktr_matches); }

    // QUDA's lines whose keys aren't given, then the given ones, with QUDA's line for the first of each of its keys
    auto keys = quda_keys;
    if (c.write_usqcd_info) keys.insert(keys.end(), computed_keys.begin(), computed_keys.end());
    auto given_key = [&](const std::string &key) {
      return std::any_of(c.lines.begin(), c.lines.end(), [&](const MILCInfoItem &i) { return i.key == key; });
    };
    std::vector<MILCInfoItem> expected;
    for (auto &key : keys)
      if (!given_key(key)) expected.push_back({key, quda_line(key, c.stamp, info)});
    std::vector<std::string> placed;
    for (auto &line : c.lines) {
      const bool quda = std::find(keys.begin(), keys.end(), line.key) != keys.end();
      const bool dropped = std::find(computed_keys.begin(), computed_keys.end(), line.key) != computed_keys.end();
      if (quda && std::find(placed.begin(), placed.end(), line.key) == placed.end()) {
        expected.push_back({line.key, quda_line(line.key, c.stamp, info)});
        placed.push_back(line.key);
      } else if (!quda && !dropped) {
        expected.push_back(line);
      }
    }
    if (c.name == "MILC's .info") {
      // MILC's order, with this field's values
      EXPECT_EQ(expected.size(), milc_lines.size());
      for (auto i = 0u; i < std::min(expected.size(), milc_lines.size()); i++)
        EXPECT_EQ(expected[i].key, milc_lines[i].key);
    }

    const auto &m = info.metadata;
    EXPECT_EQ(m.milc_time_stamp, c.stamp);
    EXPECT_EQ(m.milc_info.size(), expected.size());
    for (auto i = 0u; i < std::min(expected.size(), m.milc_info.size()); i++) {
      EXPECT_EQ(m.milc_info[i].key, expected[i].key);
      EXPECT_EQ(m.milc_info[i].line, expected[i].line);
    }
    if (comm_rank() == 0) {
      auto contents = read_file(file);
      contents.resize(std::max<size_t>(contents.size(), 96));
      std::string header_stamp(contents.data() + 20, 64);
      EXPECT_EQ(header_stamp, c.stamp + std::string(64 - c.stamp.size(), '\0'));
    }
    remove_saved_file(file);
    remove_saved_file(file + ".info");
  }
}

namespace
{
  // a value of a link that depends only on its global site, its direction and the component
  double global_value(size_t global, int dir, int k)
  {
    uint64_t h = global * 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(dir) << 48) ^ (static_cast<uint64_t>(k) << 56);
    h ^= h >> 31;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 29;
    return static_cast<double>(h >> 11) / static_cast<double>(1ull << 53) * 2.0 - 1.0;
  }
} // namespace

// a unit field: plaquettes and link trace 1, and the checksum of the 1s of rows 0 and 1
TEST(GaugeObservablesTest, unit)
{
  using namespace quda;
  for (auto prec : {QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION}) {
    if (!is_enabled(prec)) continue;
    SCOPED_TRACE(get_prec_str(prec));
    QudaGaugeParam gauge_param = gauge_param_for(prec);
    HostGauge gauge(prec);
    gauge.clear();
    for (int dir = 0; dir < 4; dir++)
      for (size_t s = 0; s < static_cast<size_t>(V); s++)
        for (int k : {0, 8, 16}) {
          if (prec == QUDA_DOUBLE_PRECISION)
            static_cast<double *>(gauge.data[dir])[s * 18 + k] = 1.0;
          else
            static_cast<float *>(gauge.data[dir])[s * 18 + k] = 1.0f;
        }
    const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);
    EXPECT_NEAR(obs.plaq, 1.0, 1e-14);
    EXPECT_NEAR(obs.plaq_spatial, 1.0, 1e-14);
    EXPECT_NEAR(obs.plaq_temporal, 1.0, 1e-14);
    EXPECT_NEAR(obs.linktr, 1.0, 1e-14);
    EXPECT_EQ(obs.nersc_checksum, nersc_checksum(gauge));
    // rows 0 and 1 hold one 1.0f (0x3f800000) each
    const uint64_t global_links = 4 * static_cast<uint64_t>(V) * comm_size();
    EXPECT_EQ(obs.nersc_checksum, static_cast<uint32_t>(global_links * 2 * 0x3f800000ull));
  }
}

// the checksum of a field whose values depend only on the global site is the same on every rank layout: it equals
// the sum over the global lattice, computed here on every rank alone
TEST(GaugeObservablesTest, checksum_layout)
{
  using namespace quda;
  const QudaGaugeParam gauge_param = gauge_param_for(QUDA_DOUBLE_PRECISION);
  const int *X = gauge_param.X;
  std::array<int, 4> N;
  for (int d = 0; d < 4; d++) N[d] = X[d] * comm_dim(d);
  uint64_t expected = 0;
  for (size_t g = 0; g < static_cast<size_t>(N[0]) * N[1] * N[2] * N[3]; g++)
    for (int dir = 0; dir < 4; dir++)
      for (int k = 0; k < 12; k++) {
        const float f = static_cast<float>(global_value(g, dir, k));
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        expected += bits;
      }

  for (auto prec : {QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION}) {
    if (!is_enabled(prec)) continue;
    SCOPED_TRACE(get_prec_str(prec));
    HostGauge gauge(prec);
    for (int t = 0; t < X[3]; t++)
      for (int z = 0; z < X[2]; z++)
        for (int y = 0; y < X[1]; y++)
          for (int x = 0; x < X[0]; x++) {
            const int c[4]
              = {comm_coord(0) * X[0] + x, comm_coord(1) * X[1] + y, comm_coord(2) * X[2] + z, comm_coord(3) * X[3] + t};
            const size_t g = c[0] + N[0] * (c[1] + static_cast<size_t>(N[1]) * (c[2] + static_cast<size_t>(N[2]) * c[3]));
            const size_t lex = x + X[0] * (y + static_cast<size_t>(X[1]) * (z + static_cast<size_t>(X[2]) * t));
            const size_t index = lex / 2 + ((c[0] + c[1] + c[2] + c[3]) % 2) * (V / 2);
            for (int dir = 0; dir < 4; dir++)
              for (int k = 0; k < 18; k++) {
                const double v = global_value(g, dir, k);
                if (prec == QUDA_DOUBLE_PRECISION)
                  static_cast<double *>(gauge.data[dir])[index * 18 + k] = v;
                else
                  static_cast<float *>(gauge.data[dir])[index * 18 + k] = static_cast<float>(v);
              }
          }
    const auto obs = gauge_observables(gauge.data, prec, X);
    EXPECT_EQ(obs.nersc_checksum, static_cast<uint32_t>(expected));
    EXPECT_EQ(obs.nersc_checksum, nersc_checksum(gauge));
  }
}

// the spatial and temporal plaquettes average to the plaquette, which is QUDA's
TEST(GaugeObservablesTest, plaquettes)
{
  using namespace quda;
  const auto prec = QUDA_DOUBLE_PRECISION;
  if (!is_enabled(prec)) GTEST_SKIP();
  QudaGaugeParam gauge_param = gauge_param_for(prec);
  HostGauge gauge(prec);
  constructHostGaugeField(gauge.data, gauge_param, 0, nullptr);
  const auto obs = gauge_observables(gauge.data, prec, gauge_param.X);
  EXPECT_NEAR(obs.plaq_spatial + obs.plaq_temporal, 2.0 * obs.plaq, 1e-14);
  EXPECT_GT(std::abs(obs.plaq_spatial - obs.plaq_temporal), 0.0);

  loadGaugeQuda(gauge.data, &gauge_param);
  std::array<double, 3> plaq;
  plaqQuda(plaq.data());
  freeGaugeQuda();
  EXPECT_NEAR(obs.plaq, plaq[0], 1e-12);
  EXPECT_NEAR(obs.plaq_spatial, plaq[1], 1e-12);
  EXPECT_NEAR(obs.plaq_temporal, plaq[2], 1e-12);
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

// SciDAC gauge partfiles
INSTANTIATE_TEST_SUITE_P(Gauge, GaugePartfileTest, Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                         [](testing::TestParamInfo<QudaPrecision> param) {
                           return std::string(get_prec_str(param.param));
                         });

// MPI-IO, with the native formats
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeMPIIOTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_FILE_FORMAT_AUTO, QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG,
                                        QUDA_FILE_FORMAT_MILC),
                                 Values(true, false)),
                         [](testing::TestParamInfo<gauge_mpi_io_test_t> param) {
                           std::string name;
                           name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
                           name += quda::get_file_format_str(::testing::get<1>(param.param));
                           name += ::testing::get<2>(param.param) ? "_usqcd_info" : "_no_usqcd_info";
                           return name;
                         });

// readGaugeQuda
INSTANTIATE_TEST_SUITE_P(Gauge, ReadGaugeQudaTest,
                         Combine(ValuesIn(gauge_formats()), Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL)),
                         [](testing::TestParamInfo<read_gauge_test_t> param) {
                           std::string name;
                           name += quda::get_file_format_str(::testing::get<0>(param.param)) + std::string("_");
                           name += get_io_mode_str(::testing::get<1>(param.param));
                           return name;
                         });

// QUDA's strings, and those it saved before
INSTANTIATE_TEST_SUITE_P(Gauge, GaugeStringsTest, Values(QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG),
                         [](testing::TestParamInfo<QudaFileFormat> param) {
                           return std::string(quda::get_file_format_str(param.param));
                         });

INSTANTIATE_TEST_SUITE_P(Gauge, GaugeLegacyStringsTest, Values(QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_ILDG),
                         [](testing::TestParamInfo<QudaFileFormat> param) {
                           return std::string(quda::get_file_format_str(param.param));
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
