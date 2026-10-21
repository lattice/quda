#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <comm_quda.h>
#include <instantiate.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <staggered_sign_convention.h>
#include <io/ks_io.h>
#include <io/lime.h>
#include <io/scidac_io.h>
#include <io/site_io.h>
#include <io/vector_io.h>
#ifdef HAVE_QIO
#include <io/qio_field.h>
#endif
#include <quda.h>

#include "test.h"
#include "misc.h"
#include "host_utils.h"
#include "staggered_source_utils.h"
#include "io_test_utils.h"

namespace
{
  using box_t = std::optional<quda::io::Hypercube>;

  // file formats that staggered sources and propagators load in, in this build
  std::vector<QudaFileFormat> ks_formats()
  {
#ifdef HAVE_QIO
    return {QUDA_FILE_FORMAT_QIO, QUDA_FILE_FORMAT_QIO_EXTERNAL};
#else
    return {QUDA_FILE_FORMAT_QIO};
#endif
  }

  // whether the files are also written with the external QIO backend
  std::vector<bool> writers()
  {
#ifdef HAVE_QIO
    return {false, true};
#else
    return {false};
#endif
  }

  // parameters of staggered fields on the test lattice: full or single parity, on the host or the device
  quda::ColorSpinorParam ks_param(QudaSiteSubset subset, QudaParity parity, QudaPrecision prec, QudaFieldLocation location)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    QudaInvertParam inv_param = newQudaInvertParam();
    setWilsonGaugeParam(gauge_param);
    setInvertParam(inv_param);

    quda::ColorSpinorParam param;
    constructWilsonTestSpinorParam(&param, &inv_param, &gauge_param);
    param.siteSubset = subset;
    param.suggested_parity = parity;
    param.nSpin = 1;
    param.setPrecision(prec, prec, true);
    param.location = location;
    if (location == QUDA_CPU_FIELD_LOCATION) param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.create = QUDA_ZERO_FIELD_CREATE;
    if (subset == QUDA_PARITY_SITE_SUBSET) param.x[0] /= 2;
    return param;
  }

  // the local dimensions of the full lattice
  std::array<int, 4> local_dims()
  {
    auto p = ks_param(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION);
    return {p.x[0], p.x[1], p.x[2], p.x[3]};
  }

  std::array<int, 4> global_dims() { return quda::io::global_dims(local_dims().data()); }

  quda::io::Hypercube box(const std::array<int, 4> &lower, const std::array<int, 4> &upper) { return {lower, upper}; }

  // the time slice t of the lattice, as MILC writes its sources
  quda::io::Hypercube time_slice(int t)
  {
    const auto N = global_dims();
    return box({0, 0, 0, t}, {N[0] - 1, N[1] - 1, N[2] - 1, t});
  }

  bool in_box(const std::array<int, 4> &g, const box_t &b)
  {
    if (!b) return true;
    for (int d = 0; d < 4; d++)
      if (g[d] < b->lower[d] || g[d] > b->upper[d]) return false;
    return true;
  }

  // the value of real k of record r at the global coordinates g: different for every (r, g, k), never zero, and
  // exact in single precision on the test lattices
  double encoded(int r, const std::array<int, 4> &g, int k, int reals)
  {
    const auto N = global_dims();
    size_t v = r;
    for (int d = 3; d >= 0; d--) v = v * N[d] + g[d];
    return static_cast<double>(v * reals + k + 1);
  }

  /**
     @brief The global coordinates of the site at host index h of a host field on the full local lattice X: QUDA's
     checkerboard order, even sites first for full fields, or the sites of the field's parity
   */
  std::array<int, 4> host_coords(size_t h, const std::array<int, 4> &X, QudaSiteSubset subset, QudaParity parity)
  {
    const size_t half = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3] / 2;
    const int p = subset == QUDA_FULL_SITE_SUBSET ? h >= half : parity == QUDA_ODD_PARITY;
    const size_t cb = h % half;
    std::array<int, 4> x;
    size_t za = cb / (X[0] / 2);
    x[1] = za % X[1];
    x[2] = (za / X[1]) % X[2];
    x[3] = za / (static_cast<size_t>(X[1]) * X[2]);
    // the test layouts have even local extents, so a site's local and global parities agree
    x[0] = 2 * (cb % (X[0] / 2)) + ((x[1] + x[2] + x[3] + p) & 1);
    for (int d = 0; d < 4; d++) x[d] += quda::comm_coord(d) * X[d];
    return x;
  }

  /**
     @brief A record of a test file: its metadata, and its values on every local site of the full lattice in
     QUDA's host order (only those in its box are written)
   */
  struct TestRecord {
    quda::scidac::FieldMetadata m;
    std::vector<double> data;
  };

  const std::string xml_head = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";

  // MILC's file XML text (create_ks_XML() in ks_spectrum, as in a file it wrote)
  const std::string milc_info = "\nDerived MILC KS field\ngauge.filename lat.sample.l8888\n"
                                "propagator.boundary_conditions space: periodic time: antiperiodic\n"
                                "asqtad.u0 1.00000\naction.fn.coeff[0] 1.000000e+00\n\ninv_arg.rsqprop 0.000000e+00\n";

  // the strings of MILC's files, as QIO's encoders write them (QIO_info_usqcd_ksprop.c)
  std::string vv_file_xml()
  {
    return xml_head + "<usqcdKSPropFile><version>1.0</version><type>USQCD_ColorVector_Source_Sink_Pairs</type><info>"
      + milc_info + "</info></usqcdKSPropFile>";
  }

  // the source record of a VV-pair file, without <color>
  std::string pair_source_xml(const std::string &info)
  {
    return xml_head + "<usqcdSourceInfo><version>1.0</version><info>" + info + "</info></usqcdSourceInfo>";
  }

  std::string solution_xml(int color)
  {
    return xml_head + "<usqcdKSPropInfo><version>1.0</version><color>" + std::to_string(color)
      + "</color><info></info></usqcdKSPropInfo>";
  }

  // the record of a color-vector source file, as MILC's w_source_ks writes it
  std::string source_xml(int color)
  {
    return xml_head + "<usqcdSourceInfo><version>1.0</version><color>" + std::to_string(color)
      + "</color><info></info></usqcdSourceInfo>";
  }

  TestRecord test_record(int r, const std::string &datatype, int reals, int colors, QudaPrecision prec, const box_t &b,
                         const std::string &user_xml)
  {
    TestRecord t;
    t.m.datatype = datatype;
    t.m.prec = prec;
    t.m.colors = colors;
    t.m.typesize = reals * prec;
    t.m.datacount = 1;
    t.m.user_xml = user_xml;
    t.m.hypercube = b;
    const auto X = local_dims();
    const size_t volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3];
    t.data.resize(volume * reals);
    for (size_t h = 0; h < volume; h++) {
      const auto g = host_coords(h, X, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
      for (int k = 0; k < reals; k++) t.data[h * reals + k] = encoded(r, g, k, reals);
    }
    return t;
  }

  TestRecord color_vector_record(int r, QudaPrecision prec, const box_t &b, const std::string &user_xml)
  {
    const std::string p = prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
    return test_record(r, "USQCD_" + p + "3_ColorVector", 6, 3, prec, b, user_xml);
  }

  TestRecord complex_record(int r, QudaPrecision prec, const box_t &b, const std::string &user_xml)
  {
    const std::string p = prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
    return test_record(r, "QLA_" + p + "_Complex", 2, 0, prec, b, user_xml);
  }

  /**
     @brief Write a SciDAC file of records as MILC writes them, natively or with the external QIO backend
   */
  void write_test_file(const std::string &file, const std::string &file_xml, const std::vector<TestRecord> &records,
                       bool external, QudaFileIOMode io_mode = QUDA_FILE_IO_PARALLEL, const std::array<int, 4> &grid = {})
  {
    using namespace quda;
    std::vector<scidac::FieldMetadata> metadata;
    std::vector<io::HostField> fields;
    for (auto &r : records) {
      metadata.push_back(r.m);
      const void *data = r.data.data();
      fields.emplace_back(&data, 1, QUDA_DOUBLE_PRECISION, r.m.typesize / r.m.prec);
    }
    const auto X = local_dims();
    if (external) {
#ifdef HAVE_QIO
      write_spinor_field(file.c_str(), {file_xml, metadata}, fields, X.data(), QUDA_FULL_SITE_SUBSET, false, io_mode,
                         grid);
#endif
    } else {
      scidac::write_fields(file, file_xml, metadata, fields, X.data(), QUDA_INVALID_PARITY,
                           file_io_param(QUDA_FILE_FORMAT_QIO, io_mode, grid));
    }
    // rank 0 appends each record's checksum after the data
    comm_barrier();
  }

  std::string writer_str(bool external) { return external ? "external" : "native"; }

  /**
     @brief The number of reals of loaded fields, on all ranks, that differ from expected(field, g, k)
   */
  int mismatches(const std::vector<quda::ColorSpinorField> &fields,
                 const std::function<double(int, const std::array<int, 4> &, int)> &expected)
  {
    using namespace quda;
    const auto X = local_dims();
    int n = 0;
    for (auto i = 0u; i < fields.size(); i++) {
      ColorSpinorParam p(fields[i]);
      p.location = QUDA_CPU_FIELD_LOCATION;
      p.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      p.setPrecision(QUDA_DOUBLE_PRECISION);
      p.create = QUDA_NULL_FIELD_CREATE;
      ColorSpinorField h(p);
      h = fields[i];
      auto data = h.data<const double *>();
      for (size_t s = 0; s < h.Volume(); s++) {
        const auto g = host_coords(s, X, h.SiteSubset(), h.SuggestedParity());
        for (int k = 0; k < 6; k++) n += data[s * 6 + k] != expected(i, g, k);
      }
    }
    comm_allreduce_int(n);
    return n;
  }

  // what fields are loaded into in a test
  struct Target {
    QudaSiteSubset subset;
    QudaParity parity;
    QudaPrecision prec;
    QudaFieldLocation location;
  };

  std::string target_str(const Target &t)
  {
    std::string s = t.subset == QUDA_FULL_SITE_SUBSET ? "full" : t.parity == QUDA_EVEN_PARITY ? "even" : "odd";
    s += std::string(" ") + get_prec_str(t.prec);
    return s + (t.location == QUDA_CUDA_FIELD_LOCATION ? " device" : " host");
  }

  // full double host fields, even device fields and odd host fields in the precision given
  std::vector<Target> targets(QudaPrecision prec)
  {
    std::vector<Target> t
      = {{QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION}};
    if (quda::is_enabled(prec)) {
      t.push_back({QUDA_PARITY_SITE_SUBSET, QUDA_EVEN_PARITY, prec, QUDA_CUDA_FIELD_LOCATION});
      t.push_back({QUDA_PARITY_SITE_SUBSET, QUDA_ODD_PARITY, prec, QUDA_CPU_FIELD_LOCATION});
    }
    return t;
  }

  std::vector<quda::ColorSpinorField> zero_fields(int n, const Target &t)
  {
    return std::vector<quda::ColorSpinorField>(n, ks_param(t.subset, t.parity, t.prec, t.location));
  }

  // the content of every <date> element replaced by a single #
  std::string mask_dates(std::string s)
  {
    const std::string open = "<date>", close = "</date>";
    for (auto start = s.find(open); start != std::string::npos; start = s.find(open, start + 1)) {
      const auto end = s.find(close, start);
      if (end == std::string::npos) break;
      s.replace(start + open.size(), end - start - open.size(), "#");
    }
    return s;
  }

  std::string int_list(const std::array<int, 4> &v)
  {
    std::string s;
    for (auto n : v) s += std::to_string(n) + " ";
    return s;
  }

  void remove_files(const std::vector<std::string> &files)
  {
    for (auto &f : files) remove_saved_file(f);
  }
} // namespace

class HypercubeTest : public ::testing::TestWithParam<QudaFileIOMode>
{
};

namespace
{
  /**
     @brief The boxes to save: time slices on both sides of every rank boundary in t and at the ends of the lattice,
     a box across a rank boundary in every split dimension, one inside the last rank (with the grid 1 1 1 1, inside
     a partition whose leader holds none of it), a single site and the whole lattice
   */
  std::vector<quda::io::Hypercube> test_boxes()
  {
    const auto X = local_dims();
    const auto N = global_dims();
    std::vector<quda::io::Hypercube> boxes;
    auto add = [&](const quda::io::Hypercube &b) {
      if (std::find(boxes.begin(), boxes.end(), b) == boxes.end()) boxes.push_back(b);
    };
    for (int t : {0, X[3] / 2, X[3] - 1, X[3], X[3] + X[3] / 2, N[3] - 1})
      if (t < N[3]) add(time_slice(t));
    quda::io::Hypercube across, last;
    for (int d = 0; d < 4; d++) {
      const bool split = quda::comm_dim(d) > 1;
      across.lower[d] = split ? X[d] - 1 : 1;
      across.upper[d] = split ? X[d] : N[d] - 2;
      last.lower[d] = N[d] - X[d] + 1;
      last.upper[d] = N[d] - 2;
    }
    add(across);
    add(last);
    add(box({N[0] - 1, N[1] - 1, N[2] - 1, N[3] - 1}, {N[0] - 1, N[1] - 1, N[2] - 1, N[3] - 1}));
    add(box({0, 0, 0, 0}, {N[0] - 1, N[1] - 1, N[2] - 1, N[3] - 1}));
    return boxes;
  }
} // namespace

// hypercube records: a color-vector source file with a record per box, saved with every I/O partition grid, holds
// the box's sites in the record's size and loads exactly into the box, with zeros elsewhere, on every rank
TEST_P(HypercubeTest, boxes)
{
  using namespace quda;
  const auto io_mode = GetParam();
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const auto boxes = test_boxes();
  const int n = boxes.size();
  std::vector<TestRecord> records;
  for (int k = 0; k < n; k++) records.push_back(color_vector_record(k, QUDA_DOUBLE_PRECISION, boxes[k], source_xml(k)));
  const Target full = {QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION};
  auto expected
    = [&](int i, const std::array<int, 4> &g, int k) { return in_box(g, boxes[i]) ? encoded(i, g, k, 6) : 0.0; };

  // serial mode ignores the grid
  std::vector<std::array<int, 4>> grids = {{}};
  if (io_mode == QUDA_FILE_IO_PARALLEL)
    for (auto &g : partition_grids()) grids.push_back(g);

  for (auto &grid : grids) {
    SCOPED_TRACE("I/O partition grid " + grid_str(grid));
    std::vector<std::string> files;
    for (bool external : writers()) {
      files.push_back("hypercube_" + writer_str(external) + ".src");
      write_test_file(files.back(), milc_info, records, external, io_mode, grid);
    }

    if (comm_rank() == 0) {
      SCOPED_TRACE(files[0]);
      const auto s = saved_strings(files[0]);
      ASSERT_EQ(s.private_record.size(), static_cast<size_t>(n));
      for (int k = 0; k < n; k++)
        EXPECT_EQ(mask_dates(s.private_record[k]),
                  xml_head + "<scidacRecord><version>1.1</version><date>#</date><recordtype>2</recordtype>"
                    + "<spacetime>4</spacetime><hyperlower>" + int_list(boxes[k].lower) + "</hyperlower><hyperupper>"
                    + int_list(boxes[k].upper) + "</hyperupper><datatype>USQCD_D3_ColorVector</datatype>"
                    + "<precision>D</precision><colors>3</colors><typesize>48</typesize><datacount>1</datacount>"
                    + "</scidacRecord>")
          << "record " << k;
      int k = 0;
      for (auto &r : local_records(files[0]))
        if (r.type == scidac::binary_data) {
          ASSERT_LT(k, n);
          EXPECT_EQ(r.data_bytes, boxes[k].volume() * 48) << "record " << k;
          k++;
        }
      EXPECT_EQ(k, n);
      if (files.size() > 1) expect_same_saved_file(files[0], files[1], false);
    }

    for (auto &file : files) {
      for (auto format : ks_formats()) {
        SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format));
        auto u = zero_fields(n, full);
        auto info = KSSourceIO(file, file_io_param(format, io_mode, grid)).load(u);
        EXPECT_EQ(mismatches(u, expected), 0);
        EXPECT_EQ(info.type, KSFileType::VectorSource);
        EXPECT_EQ(info.file_info, milc_info);
        ASSERT_EQ(info.boxes.size(), static_cast<size_t>(n));
        for (int i = 0; i < n; i++) EXPECT_EQ(info.boxes[i], box_t(boxes[i])) << "record " << i;
      }
    }
    remove_files(files);
  }
}

// tuple types: file precision, number of colors (pairs), whether the sources are time slices
using prop_test_t = ::testing::tuple<QudaPrecision, int, bool>;

class KSPropagatorLoadTest : public ::testing::TestWithParam<prop_test_t>
{
};

// VV-pair files as MILC writes them load into full and single-parity fields with both backends, with and without
// their sources, and fewer pairs than they hold
TEST_P(KSPropagatorLoadTest, load)
{
  using namespace quda;
  const auto prec = ::testing::get<0>(GetParam());
  const int nc = ::testing::get<1>(GetParam());
  const bool slice = ::testing::get<2>(GetParam());
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const auto X = local_dims();
  const auto N = global_dims();
  // the first slice of the second rank in t, or the middle of the lattice
  const box_t source_box = slice ? box_t(time_slice(X[3] < N[3] ? X[3] : N[3] / 2)) : std::nullopt;
  const std::string source_info = slice ? "point" : "even_wall";

  std::vector<TestRecord> records;
  for (int c = 0; c < nc; c++) {
    records.push_back(color_vector_record(2 * c, prec, source_box, pair_source_xml(source_info)));
    records.push_back(color_vector_record(2 * c + 1, prec, std::nullopt, solution_xml(c)));
  }
  auto solution = [&](int i, const std::array<int, 4> &g, int k) { return encoded(2 * i + 1, g, k, 6); };
  auto source
    = [&](int i, const std::array<int, 4> &g, int k) { return in_box(g, source_box) ? encoded(2 * i, g, k, 6) : 0.0; };

  std::vector<std::string> files;
  for (bool external : writers()) {
    files.push_back("vv_" + writer_str(external) + ".prop");
    write_test_file(files.back(), vv_file_xml(), records, external);
  }
  if (comm_rank() == 0 && files.size() > 1) expect_same_saved_file(files[0], files[1], false);

  const Target full = {QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION};
  for (auto &file : files) {
    for (auto format : ks_formats()) {
      const auto p = file_io_param(format, QUDA_FILE_IO_PARALLEL);
      for (auto &t : targets(prec)) {
        SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format) + " into " + target_str(t) + " fields");
        auto x = zero_fields(nc, t), b = zero_fields(nc, t);
        auto info = KSPropagatorIO(file, p).load(x, b);
        EXPECT_EQ(mismatches(x, solution), 0);
        EXPECT_EQ(mismatches(b, source), 0);
        EXPECT_EQ(info.type, KSFileType::Propagator);
        EXPECT_EQ(info.file_prec, prec);
        EXPECT_EQ(info.file_info, milc_info);
        EXPECT_EQ(info.boxes, std::vector<box_t>(nc, source_box));
        EXPECT_EQ(info.source_info, std::vector<std::string>(nc, source_info));
        EXPECT_EQ(info.solution_info, std::vector<std::string>(nc, ""));
      }
      {
        SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format) + " without the sources");
        auto x = zero_fields(nc, full);
        auto info = KSPropagatorIO(file, p).load(x, {});
        EXPECT_EQ(mismatches(x, solution), 0);
        EXPECT_TRUE(info.boxes.empty());
        EXPECT_TRUE(info.source_info.empty());
        EXPECT_EQ(info.solution_info.size(), static_cast<size_t>(nc));
      }
      if (nc > 1) {
        SCOPED_TRACE("loading " + std::to_string(nc - 1) + " pairs of " + file + " with " + get_file_format_str(format));
        auto x = zero_fields(nc - 1, full), b = zero_fields(nc - 1, full);
        auto info = KSPropagatorIO(file, p).load(x, b);
        EXPECT_EQ(mismatches(x, solution), 0);
        EXPECT_EQ(mismatches(b, source), 0);
        EXPECT_EQ(info.boxes.size(), static_cast<size_t>(nc - 1));
      }
    }
  }
  remove_files(files);
}

enum class SourceKind { COLOR, NO_COLOR, OLD_STYLE, PAIRS, COMPLEX, COMPLEX_SLICE };

const char *source_kind_str(SourceKind kind)
{
  switch (kind) {
  case SourceKind::COLOR: return "color";
  case SourceKind::NO_COLOR: return "no_color";
  case SourceKind::OLD_STYLE: return "old_style";
  case SourceKind::PAIRS: return "pairs";
  case SourceKind::COMPLEX: return "complex";
  case SourceKind::COMPLEX_SLICE: return "complex_slice";
  }
  return "";
}

class KSSourceLoadTest : public ::testing::TestWithParam<SourceKind>
{
};

// MILC's source files load through KSSourceIO: color-vector source files with and without <color>, with the older
// usqcdKSPropInfo record XML and with time slices; the sources of a VV-pair file; and complex fields, promoted into
// color c of source c
TEST_P(KSSourceLoadTest, load)
{
  using namespace quda;
  const auto kind = GetParam();
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const auto N = global_dims();
  const auto prec = QUDA_DOUBLE_PRECISION;
  const bool complex = kind == SourceKind::COMPLEX || kind == SourceKind::COMPLEX_SLICE;

  // the records, and the source each holds with its box and <info>
  std::string file_xml = milc_info;
  std::vector<TestRecord> records;
  std::vector<int> source_records;
  std::vector<box_t> boxes;
  std::vector<std::string> infos;
  auto add_source = [&](const TestRecord &r, const std::string &info) {
    source_records.push_back(records.size());
    boxes.push_back(r.m.hypercube);
    infos.push_back(info);
    records.push_back(r);
  };
  switch (kind) {
  case SourceKind::COLOR:
    add_source(color_vector_record(0, prec, time_slice(0), source_xml(0)), "");
    add_source(color_vector_record(1, prec, time_slice(N[3] - 1), source_xml(1)), "");
    add_source(color_vector_record(2, prec, std::nullopt, source_xml(2)), "");
    break;
  case SourceKind::NO_COLOR:
    add_source(color_vector_record(0, prec, std::nullopt, pair_source_xml("random_color_wall")), "random_color_wall");
    break;
  case SourceKind::OLD_STYLE:
    add_source(color_vector_record(0, prec, time_slice(1), solution_xml(0)), "");
    add_source(color_vector_record(1, prec, time_slice(1), solution_xml(1)), "");
    break;
  case SourceKind::PAIRS:
    file_xml = vv_file_xml();
    for (int c = 0; c < 3; c++) {
      add_source(color_vector_record(2 * c, prec, time_slice(N[3] / 2), pair_source_xml("evenandodd_wall")),
                 "evenandodd_wall");
      records.push_back(color_vector_record(2 * c + 1, prec, std::nullopt, solution_xml(c)));
    }
    break;
  case SourceKind::COMPLEX:
    file_xml = "MILC complex field";
    add_source(complex_record(0, prec, std::nullopt, "modulation"), "");
    break;
  case SourceKind::COMPLEX_SLICE:
    // a second record, which is not loaded
    file_xml = "MILC complex field";
    add_source(complex_record(0, QUDA_SINGLE_PRECISION, time_slice(N[3] - 1), "modulation"), "");
    records.push_back(complex_record(1, QUDA_SINGLE_PRECISION, std::nullopt, "modulation"));
    break;
  }
  const int n_records = source_records.size();
  const auto type = complex   ? KSFileType::ComplexSource :
    kind == SourceKind::PAIRS ? KSFileType::Propagator :
                                KSFileType::VectorSource;
  const std::string file_info = kind == SourceKind::PAIRS ? milc_info : file_xml;

  std::vector<std::string> files;
  for (bool external : writers()) {
    files.push_back(std::string("source_") + source_kind_str(kind) + "_" + writer_str(external) + ".src");
    write_test_file(files.back(), file_xml, records, external);
  }
  if (comm_rank() == 0 && files.size() > 1) expect_same_saved_file(files[0], files[1], false);

  // complex files load into 1 to 3 sources, the others as many sources as they hold, and fewer
  std::vector<int> counts;
  if (complex)
    counts = {1, 2, 3};
  else
    for (int n = n_records; n >= 1 && counts.size() < 2; n--) counts.push_back(n);
  auto expected = [&](int i, const std::array<int, 4> &g, int k) {
    if (!complex) return in_box(g, boxes[i]) ? encoded(source_records[i], g, k, 6) : 0.0;
    return k / 2 == i && in_box(g, boxes[0]) ? encoded(0, g, k % 2, 2) : 0.0;
  };

  for (auto &file : files) {
    for (auto format : ks_formats()) {
      for (int n : counts) {
        for (auto &t : targets(QUDA_SINGLE_PRECISION)) {
          if (t.subset == QUDA_PARITY_SITE_SUBSET && n != counts[0]) continue;
          SCOPED_TRACE("loading " + std::to_string(n) + " sources of " + file + " with " + get_file_format_str(format)
                       + " into " + target_str(t) + " fields");
          auto u = zero_fields(n, t);
          auto info = KSSourceIO(file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(u);
          EXPECT_EQ(mismatches(u, expected), 0);
          EXPECT_EQ(info.type, type);
          EXPECT_EQ(info.file_prec, records[source_records[0]].m.prec);
          EXPECT_EQ(info.file_info, file_info);
          ASSERT_EQ(info.boxes.size(), static_cast<size_t>(n));
          ASSERT_EQ(info.source_info.size(), static_cast<size_t>(n));
          for (int i = 0; i < n; i++) {
            EXPECT_EQ(info.boxes[i], boxes[complex ? 0 : i]) << "source " << i;
            EXPECT_EQ(info.source_info[i], infos[complex ? 0 : i]) << "source " << i;
          }
          EXPECT_TRUE(info.solution_info.empty());
        }
      }
    }
  }
  remove_files(files);
}

// QUDA's own staggered vector files and MILC's eigenvector files are not taken for staggered source files, and
// find_ks_file tells the three kinds apart
TEST(KSDetectionTest, kinds)
{
  using namespace quda;
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const auto prec = QUDA_DOUBLE_PRECISION;
  const auto param = ks_param(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, prec, QUDA_CPU_FIELD_LOCATION);
  std::vector<ColorSpinorField> v(2, param);
  const std::vector<complex_t> evals = {0.5, 0.75};
  const std::vector<real_t> resids = {1e-12, 2e-12};

  const std::string quda_file = "detect_quda.cs", milc_file = "detect_milc.cs";
  VectorIO(quda_file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL)).save({v.begin(), v.end()});
  auto mp = file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL);
  mp.milc_eigenvectors = true;
  VectorIO(milc_file, mp).save({v.begin(), v.end()}, prec, 2, &evals, &resids);
  for (auto &file : {quda_file, milc_file}) {
    SCOPED_TRACE(file);
    EXPECT_FALSE(scidac::ks_file_type(scidac::find_field(file)));
    // and they still load as vectors
    std::vector<ColorSpinorField> u(2, param);
    VectorIO(file, file_io_param(QUDA_FILE_FORMAT_QIO, QUDA_FILE_IO_PARALLEL)).load(u);
    for (int i = 0; i < 2; i++) EXPECT_EQ(blas::max_deviation(u[i], v[i])[0], 0.0);
  }

  const std::string source_file = "detect.src", vv_file = "detect.prop", complex_file = "detect_complex.src";
  write_test_file(source_file, milc_info, {color_vector_record(0, prec, time_slice(0), source_xml(0))}, false);
  write_test_file(vv_file, vv_file_xml(),
                  {color_vector_record(0, prec, time_slice(0), pair_source_xml("point")),
                   color_vector_record(1, prec, std::nullopt, solution_xml(0))},
                  false);
  write_test_file(complex_file, "MILC complex field", {complex_record(0, prec, std::nullopt, "")}, false);
  const std::vector<std::pair<std::string, KSFileType>> kinds = {{source_file, KSFileType::VectorSource},
                                                                 {vv_file, KSFileType::Propagator},
                                                                 {complex_file, KSFileType::ComplexSource}};
  for (auto &[file, type] : kinds) {
    SCOPED_TRACE(file);
    EXPECT_EQ(scidac::ks_file_type(scidac::find_field(file)), std::optional<KSFileType>(type));
    const auto f = scidac::find_ks_file(file);
    EXPECT_EQ(f.type, type);
    EXPECT_EQ(f.file_info, type == KSFileType::ComplexSource ? "MILC complex field" : milc_info);
  }
  EXPECT_EQ(scidac::find_ks_file(vv_file).info, std::vector<std::string>({"point", ""}));

  remove_files({quda_file, milc_file, source_file, vv_file, complex_file});
}

namespace
{
  // full staggered fields of the target's precision and location, field i holding encoded(i, g, k)
  std::vector<quda::ColorSpinorField> encoded_fields(int n, const Target &t)
  {
    using namespace quda;
    const auto X = local_dims();
    ColorSpinorField h(
      ks_param(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION));
    auto data = h.data<double *>();
    std::vector<ColorSpinorField> fields;
    for (int i = 0; i < n; i++) {
      for (size_t s = 0; s < h.Volume(); s++) {
        const auto g = host_coords(s, X, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
        for (int k = 0; k < 6; k++) data[s * 6 + k] = encoded(i, g, k, 6);
      }
      fields.emplace_back(ks_param(t.subset, t.parity, t.prec, t.location));
      fields.back() = h;
    }
    return fields;
  }

  // the bytes of a field, copied to the host in its own precision
  std::vector<char> field_bytes(const quda::ColorSpinorField &f)
  {
    quda::ColorSpinorParam p(f);
    p.location = QUDA_CPU_FIELD_LOCATION;
    p.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    p.create = QUDA_NULL_FIELD_CREATE;
    quda::ColorSpinorField h(p);
    h = f;
    auto data = h.data<const char *>();
    return std::vector<char>(data, data + h.Bytes());
  }

  bool odd_site(const std::array<int, 4> &g) { return (g[0] + g[1] + g[2] + g[3]) % 2 == 1; }
} // namespace

using sign_test_t = ::testing::tuple<QudaFieldLocation, QudaPrecision>;

class StaggeredSignConventionTest : public ::testing::TestWithParam<sign_test_t>
{
};

// the conversion negates the odd sites exactly and leaves the even ones as they are, both directions do the same, and
// converting twice restores the fields bit for bit
TEST_P(StaggeredSignConventionTest, convert)
{
  using namespace quda;
  const auto location = ::testing::get<0>(GetParam());
  const auto prec = ::testing::get<1>(GetParam());
  if (!is_enabled_spin(1) || !is_enabled(prec)) GTEST_SKIP();
  const Target t = {QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, prec, location};
  const int n = 3;
  auto x = encoded_fields(n, t), y = encoded_fields(n, t);
  std::vector<std::vector<char>> original;
  for (auto &f : x) original.push_back(field_bytes(f));

  auto converted
    = [](int i, const std::array<int, 4> &g, int k) { return (odd_site(g) ? -1.0 : 1.0) * encoded(i, g, k, 6); };
  staggeredMILCToQUDASignConvention(x);
  EXPECT_EQ(mismatches(x, converted), 0);
  staggeredQUDAToMILCSignConvention(y);
  for (int i = 0; i < n; i++) EXPECT_EQ(field_bytes(y[i]), field_bytes(x[i])) << "field " << i;

  staggeredQUDAToMILCSignConvention(x);
  for (int i = 0; i < n; i++) EXPECT_EQ(field_bytes(x[i]), original[i]) << "field " << i;
  staggeredMILCToQUDASignConvention(y);
  for (int i = 0; i < n; i++) EXPECT_EQ(field_bytes(y[i]), original[i]) << "field " << i;
}

// QUDA's staggered vector files record the sign convention they were saved in, in each record's XML, identically in
// both backends; without one they record none
TEST(VectorSignConventionTest, records)
{
  using namespace quda;
  if (!is_enabled_spin(1)) GTEST_SKIP();
  const Target t = {QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, QUDA_DOUBLE_PRECISION, QUDA_CPU_FIELD_LOCATION};
  const int n = 2;
  const auto v = encoded_fields(n, t);
  const std::vector<std::optional<StaggeredSignConvention>> conventions
    = {std::nullopt, StaggeredSignConvention::QUDA, StaggeredSignConvention::MILC};

  for (auto &convention : conventions) {
    const std::string name = convention ? get_staggered_sign_convention_str(*convention) : "none";
    SCOPED_TRACE("sign convention " + name);
    const std::string element
      = convention ? "<staggeredSignConvention>" + name + "</staggeredSignConvention>" : std::string();
    std::vector<std::string> files;
    for (auto format : ks_formats()) {
      files.push_back("sign_" + name + "_" + get_file_format_str(format) + ".cs");
      auto p = file_io_param(format, QUDA_FILE_IO_PARALLEL);
      p.staggered_sign_convention = convention;
      VectorIO(files.back(), p).save({v.begin(), v.end()});
    }

    if (comm_rank() == 0) {
      for (auto &file : files) {
        SCOPED_TRACE(file);
        const auto s = saved_strings(file);
        ASSERT_EQ(s.user_record.size(), static_cast<size_t>(n));
        for (int i = 0; i < n; i++)
          EXPECT_EQ(s.user_record[i],
                    xml_head + "<qudaStaggeredColorSpinorField><version>1.0</version><info><subset>full</subset>"
                      + "<parity>full</parity><nColor>3</nColor><nSpin>1</nSpin>" + element
                      + "</info></qudaStaggeredColorSpinorField>")
            << "record " << i;
      }
      if (files.size() > 1) expect_same_saved_file(files[0], files[1], false);
    }

    for (auto &file : files) {
      for (auto format : ks_formats()) {
        SCOPED_TRACE("loading " + file + " with " + get_file_format_str(format));
        auto u = zero_fields(n, t);
        VectorIO(file, file_io_param(format, QUDA_FILE_IO_PARALLEL)).load(u);
        EXPECT_EQ(mismatches(u, [](int i, const std::array<int, 4> &g, int k) { return encoded(i, g, k, 6); }), 0);
      }
    }
    remove_files(files);
  }
}

namespace
{
  using cplx = std::complex<double>;
  using coords_t = std::array<int, 4>;

  const std::vector<KSSourceType> deterministic_ks_sources
    = {KSSourceType::POINT,          KSSourceType::CORNER_WALL,     KSSourceType::CORNER_WALL_X,
       KSSourceType::CORNER_WALL_Y,  KSSourceType::CORNER_WALL_XY,  KSSourceType::CORNER_WALL_Z,
       KSSourceType::CORNER_WALL_ZX, KSSourceType::CORNER_WALL_YZ,  KSSourceType::CORNER_WALL_XYZ,
       KSSourceType::EVEN_WALL,      KSSourceType::EVENANDODD_WALL, KSSourceType::EVENMINUSODD_WALL};

  // the time slices of the tests: the first, an interior one, either side of the boundary between ranks in t (only
  // the last local slice when t isn't split), the last one, and every slice
  std::vector<int> test_slices()
  {
    const int T = local_dims()[3], N = global_dims()[3];
    std::vector<int> t = {0, 3, T - 1};
    if (T < N) t.push_back(T);
    t.push_back(N - 1);
    t.push_back(ks_source_all_t_slices);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
  }

  bool on_slice(const coords_t &g, int t0) { return t0 == ks_source_all_t_slices || g[3] == t0; }

  bool in_corner_subset(const coords_t &g) { return g[0] % 2 == 0 && g[1] % 2 == 0 && g[2] % 2 == 0; }

  // exp(i 2 pi sum_i p_i (g_i - o_i) / N_i), computed otherwise than MILC's order of operations
  cplx momentum_factor(const KSSourceParam &p, const coords_t &g)
  {
    const auto N = global_dims();
    double f = 0.0;
    for (int i = 0; i < 3; i++) f += static_cast<double>(p.momentum[i] * (g[i] - p.origin[i])) / N[i];
    return std::polar(1.0, 2.0 * M_PI * f);
  }

  // MILC's deterministic complex-field sources on their slice (generic/quark_source.c)
  double definition(KSSourceType type, const coords_t &origin, const coords_t &g)
  {
    const bool even = (g[0] + g[1] + g[2] + g[3]) % 2 == 0;
    const std::vector<std::pair<KSSourceType, int>> corner_walls
      = {{KSSourceType::CORNER_WALL, 0},    {KSSourceType::CORNER_WALL_X, 1},  {KSSourceType::CORNER_WALL_Y, 2},
         {KSSourceType::CORNER_WALL_XY, 3}, {KSSourceType::CORNER_WALL_Z, 4},  {KSSourceType::CORNER_WALL_ZX, 5},
         {KSSourceType::CORNER_WALL_YZ, 6}, {KSSourceType::CORNER_WALL_XYZ, 7}};
    for (auto &[t, r0] : corner_walls)
      if (t == type) return g[0] % 2 + 2 * (g[1] % 2) + 4 * (g[2] % 2) == r0 ? 1.0 : 0.0;
    switch (type) {
    case KSSourceType::POINT: return g == origin ? 1.0 : 0.0;
    case KSSourceType::EVEN_WALL: return even ? 1.0 : 0.0;
    case KSSourceType::EVENANDODD_WALL: return 1.0;
    case KSSourceType::EVENMINUSODD_WALL: return even ? 1.0 : -1.0;
    default: ADD_FAILURE() << get_ks_source_type_str(type) << " is not deterministic";
    }
    return 0.0;
  }

  struct Comparison {
    double max_difference = 0.0;     // on the support
    int nonzero_off_support = 0;     // reals
    double max_norm_deviation = 0.0; // of the 3-color vectors on the support from 1
  };

  /**
     @brief Compare host fields with expected(field, host index, g, color) on the sites where on_support(g), and count
     their nonzero reals elsewhere, on all ranks
   */
  Comparison compare(const std::vector<quda::ColorSpinorField> &fields,
                     const std::function<cplx(int, size_t, const coords_t &, int)> &expected,
                     const std::function<bool(const coords_t &)> &on_support)
  {
    using namespace quda;
    const auto X = local_dims();
    Comparison r;
    for (auto i = 0u; i < fields.size(); i++) {
      ColorSpinorParam p(fields[i]);
      p.setPrecision(QUDA_DOUBLE_PRECISION);
      p.create = QUDA_NULL_FIELD_CREATE;
      ColorSpinorField h(p);
      h = fields[i];
      auto data = h.data<const double *>();
      for (size_t s = 0; s < h.Volume(); s++) {
        const auto g = host_coords(s, X, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
        const bool support = on_support(g);
        double norm2 = 0.0;
        for (int c = 0; c < 3; c++) {
          const cplx v(data[s * 6 + 2 * c], data[s * 6 + 2 * c + 1]);
          norm2 += std::norm(v);
          if (support)
            r.max_difference = std::max(r.max_difference, std::abs(v - expected(i, s, g, c)));
          else
            r.nonzero_off_support += (v.real() != 0.0) + (v.imag() != 0.0);
        }
        if (support) r.max_norm_deviation = std::max(r.max_norm_deviation, std::abs(norm2 - 1.0));
      }
    }
    comm_allreduce_max(r.max_difference);
    comm_allreduce_int(r.nonzero_off_support);
    comm_allreduce_max(r.max_norm_deviation);
    return r;
  }

  std::vector<quda::ColorSpinorField> host_sources(int n, QudaPrecision prec)
  {
    return zero_fields(n, {QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, prec, QUDA_CPU_FIELD_LOCATION});
  }

  std::string ks_source_str(const KSSourceParam &p)
  {
    return std::string(get_ks_source_type_str(p.type)) + " t0 " + std::to_string(p.t0) + " origin " + int_list(p.origin)
      + "momentum " + std::to_string(p.momentum[0]) + " " + std::to_string(p.momentum[1]) + " "
      + std::to_string(p.momentum[2]) + " subset " + get_ks_source_subset_str(p.subset);
  }

  /**
     The random walls as constructKSSources draws them, from per-site generators seeded here from the seed and each
     site's global lexicographic index, so that the values are those of any rank layout: per site, one complex Gaussian
     for random_complex_wall, three per source for random_color_wall, each normalized
   */
  class ReferenceWalls
  {
    std::vector<host_rng_t> rng;

    cplx gaussian(size_t h)
    {
      std::normal_distribution<double> re {0.0, 1.0}, im {0.0, 1.0};
      const double r = re(rng[h]);
      return {r, im(rng[h])};
    }

  public:
    explicit ReferenceWalls(uint64_t seed)
    {
      const auto X = local_dims();
      const auto N = global_dims();
      rng.resize(static_cast<size_t>(X[0]) * X[1] * X[2] * X[3]);
      for (size_t h = 0; h < rng.size(); h++) {
        const auto g = host_coords(h, X, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
        const uint64_t index = ((static_cast<uint64_t>(g[3]) * N[2] + g[2]) * N[1] + g[1]) * N[0] + g[0];
        rng[h] = host_rng_t(host_source_rand_seed(seed, index));
      }
    }

    // the next draw of n sources of the type on slice t0, unmasked and without a momentum: [source][site][color]
    std::vector<std::vector<std::array<cplx, 3>>> draw(KSSourceType type, int t0, int n)
    {
      const auto X = local_dims();
      std::vector<std::vector<std::array<cplx, 3>>> v(n, std::vector<std::array<cplx, 3>>(rng.size()));
      for (size_t h = 0; h < rng.size(); h++) {
        if (!on_slice(host_coords(h, X, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY), t0)) continue;
        if (type == KSSourceType::RANDOM_COMPLEX_WALL) {
          const auto z = gaussian(h);
          const double x = 1.0 / std::sqrt(z.real() * z.real() + z.imag() * z.imag());
          for (int c = 0; c < n; c++) v[c][h][c] = {x * z.real(), x * z.imag()};
        } else {
          for (int i = 0; i < n; i++) {
            for (auto &c : v[i][h]) c = gaussian(h);
            double magsq = 0.0;
            for (auto &c : v[i][h]) magsq += c.real() * c.real() + c.imag() * c.imag();
            const double x = 1.0 / std::sqrt(magsq);
            for (auto &c : v[i][h]) c = {x * c.real(), x * c.imag()};
          }
        }
      }
      return v;
    }
  };

  std::vector<char> host_rand_bytes()
  {
    auto p = reinterpret_cast<const char *>(host_rand.data());
    return std::vector<char>(p, p + host_rand.size() * sizeof(host_rng_t));
  }
} // namespace

class KSSourceBuildTest : public ::testing::TestWithParam<KSSourceType>
{
};

// each of MILC's deterministic sources equals its definition, computed here from global coordinates, on time slices on
// either side of the boundary between ranks and on every slice, with and without a momentum from a nonzero origin, in
// double and single precision; it is exactly zero off its slice
TEST_P(KSSourceBuildTest, deterministic)
{
  const auto type = GetParam();
  if (!quda::is_enabled_spin(1)) GTEST_SKIP();
  for (auto prec : {QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION}) {
    for (int t : test_slices()) {
      if (type == KSSourceType::POINT && t == ks_source_all_t_slices) continue;
      for (bool momentum : {false, true}) {
        KSSourceParam p;
        p.type = type;
        p.t0 = type == KSSourceType::POINT ? 0 : t;
        p.origin = {1, 2, 3, type == KSSourceType::POINT ? t : 0};
        if (momentum) p.momentum = {1, -2, 3};
        SCOPED_TRACE(ks_source_str(p) + " " + get_prec_str(prec));
        auto f = host_sources(3, prec);
        constructKSSources(f, p);
        const auto r = compare(
          f,
          [&](int i, size_t, const coords_t &g, int c) {
            return c == i ? definition(type, p.origin, g) * momentum_factor(p, g) : cplx(0.0);
          },
          [&](const coords_t &g) { return on_slice(g, type == KSSourceType::POINT ? t : p.t0); });
        EXPECT_LE(r.max_difference, prec == QUDA_DOUBLE_PRECISION ? 1e-13 : 1e-6);
        EXPECT_EQ(r.nonzero_off_support, 0);
      }
    }
  }
}

class KSRandomSourceBuildTest : public ::testing::TestWithParam<KSSourceType>
{
};

// the random walls: unit-norm color vectors (or unit-modulus values) on their slice and subset, zero elsewhere, with
// the values of per-site generators seeded from the global site, so the same on every rank layout; a second build
// continues the streams, a new seed starts others, source c doesn't depend on how many are built, and host_rand is
// untouched
TEST_P(KSRandomSourceBuildTest, random)
{
  const auto type = GetParam();
  if (!quda::is_enabled_spin(1)) GTEST_SKIP();
  const int n = 3;
  const auto host_rand_before = host_rand_bytes();
  const int T = local_dims()[3], N = global_dims()[3];

  for (auto prec : {QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION}) {
    for (int t : {0, T < N ? T : T - 1, ks_source_all_t_slices}) {
      for (auto subset : {KSSourceSubset::FULL, KSSourceSubset::CORNER}) {
        for (bool momentum : {false, true}) {
          KSSourceParam p;
          p.type = type;
          p.t0 = t;
          p.subset = subset;
          p.origin = {1, 2, 3, 0};
          if (momentum) p.momentum = {1, -2, 3};
          SCOPED_TRACE(ks_source_str(p) + " " + get_prec_str(prec));
          const double tol = prec == QUDA_DOUBLE_PRECISION ? 1e-13 : 1e-6;
          auto support = [&](const coords_t &g) {
            return on_slice(g, t) && (subset == KSSourceSubset::FULL || in_corner_subset(g));
          };

          initSourceRand(p.seed, local_dims());
          ReferenceWalls reference(p.seed);
          for (int build = 0; build < 2; build++) {
            SCOPED_TRACE("build " + std::to_string(build));
            const auto v = reference.draw(type, t, n);
            auto f = host_sources(n, prec);
            constructKSSources(f, p);
            const auto r = compare(
              f, [&](int i, size_t h, const coords_t &g, int c) { return v[i][h][c] * momentum_factor(p, g); }, support);
            EXPECT_LE(r.max_difference, momentum || prec == QUDA_SINGLE_PRECISION ? tol : 0.0);
            EXPECT_EQ(r.nonzero_off_support, 0);
            EXPECT_LE(r.max_norm_deviation, prec == QUDA_DOUBLE_PRECISION ? 1e-14 : 1e-6);
          }

          // source 0 of a single build is source 0 of three
          initSourceRand(p.seed, local_dims());
          const auto v = ReferenceWalls(p.seed).draw(type, t, n);
          auto f = host_sources(1, prec);
          constructKSSources(f, p);
          const auto r = compare(
            f, [&](int, size_t h, const coords_t &g, int c) { return v[0][h][c] * momentum_factor(p, g); }, support);
          EXPECT_LE(r.max_difference, tol) << "a single source";
          EXPECT_EQ(r.nonzero_off_support, 0) << "a single source";
        }
      }
    }
  }

  // a new seed reinitializes the generator, without initSourceRand; a second build gives new values
  KSSourceParam p;
  p.type = type;
  p.t0 = ks_source_all_t_slices;
  p.seed = 99;
  ReferenceWalls reference(p.seed);
  for (int build = 0; build < 2; build++) {
    const auto v = reference.draw(type, p.t0, n);
    auto f = host_sources(n, QUDA_DOUBLE_PRECISION);
    constructKSSources(f, p);
    const auto r = compare(
      f, [&](int i, size_t h, const coords_t &, int c) { return v[i][h][c]; }, [](const coords_t &) { return true; });
    EXPECT_EQ(r.max_difference, 0.0) << "seed 99, build " << build;
  }

  EXPECT_EQ(host_rand_bytes(), host_rand_before) << "host_rand was used";
}

int main(int argc, char **argv)
{
  quda_test test("Propagator IO Test", argc, argv);
  test.init();
  return test.execute();
}

using ::testing::Combine;
using ::testing::Values;

INSTANTIATE_TEST_SUITE_P(KS, HypercubeTest, Values(QUDA_FILE_IO_PARALLEL, QUDA_FILE_IO_SERIAL),
                         [](testing::TestParamInfo<QudaFileIOMode> param) {
                           return std::string(get_io_mode_str(param.param));
                         });

INSTANTIATE_TEST_SUITE_P(KS, KSPropagatorLoadTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), Values(1, 3), Values(false, true)),
                         [](testing::TestParamInfo<prop_test_t> param) {
                           std::string name = get_prec_str(::testing::get<0>(param.param));
                           name += "_ncolor" + std::to_string(::testing::get<1>(param.param));
                           return name + (::testing::get<2>(param.param) ? "_timeslice" : "_full");
                         });

INSTANTIATE_TEST_SUITE_P(KS, KSSourceLoadTest,
                         Values(SourceKind::COLOR, SourceKind::NO_COLOR, SourceKind::OLD_STYLE, SourceKind::PAIRS,
                                SourceKind::COMPLEX, SourceKind::COMPLEX_SLICE),
                         [](testing::TestParamInfo<SourceKind> param) {
                           return std::string(source_kind_str(param.param));
                         });

INSTANTIATE_TEST_SUITE_P(KS, KSSourceBuildTest, ::testing::ValuesIn(deterministic_ks_sources),
                         [](testing::TestParamInfo<KSSourceType> param) {
                           std::string name = get_ks_source_type_str(param.param);
                           std::replace(name.begin(), name.end(), '-', '_');
                           return name;
                         });

INSTANTIATE_TEST_SUITE_P(KS, KSRandomSourceBuildTest,
                         Values(KSSourceType::RANDOM_COMPLEX_WALL, KSSourceType::RANDOM_COLOR_WALL),
                         [](testing::TestParamInfo<KSSourceType> param) {
                           std::string name = get_ks_source_type_str(param.param);
                           std::replace(name.begin(), name.end(), '-', '_');
                           return name;
                         });

INSTANTIATE_TEST_SUITE_P(KS, StaggeredSignConventionTest,
                         Combine(Values(QUDA_CPU_FIELD_LOCATION, QUDA_CUDA_FIELD_LOCATION),
                                 Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION)),
                         [](testing::TestParamInfo<sign_test_t> param) {
                           std::string name
                             = ::testing::get<0>(param.param) == QUDA_CPU_FIELD_LOCATION ? "host" : "device";
                           return name + "_" + get_prec_str(::testing::get<1>(param.param));
                         });
