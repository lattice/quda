#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <qmp.h>
#include <qio.h>
#include <quda.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/host_field.h>
#include <io/layout_hyper.h>
#include <io/qio_field.h>

#include <string>

using namespace quda;

static QIO_Layout layout;

static int qio_serpar(QudaFileIOMode io_mode) { return io_mode == QUDA_FILE_IO_SERIAL ? QIO_SERIAL : QIO_PARALLEL; }
static int lattice_size[4];
int quda_this_node;

std::ostream &operator<<(std::ostream &out, const QIO_Layout &layout)
{
  out << "node_number = " << layout.node_number << std::endl;
  out << "node_index = " << layout.node_index << std::endl;
  out << "get_coords = " << layout.get_coords << std::endl;
  out << "num_sites = " << layout.num_sites << std::endl;
  out << "latdim = " << layout.latdim << std::endl;
  out << "latsize = {";
  for (int d = 0; d < layout.latdim; d++) out << layout.latsize[d] << (d < layout.latdim - 1 ? ", " : "}");
  out << std::endl;
  out << "volume = " << layout.volume << std::endl;
  out << "sites_on_node = " << layout.sites_on_node << std::endl;
  out << "this_node = " << layout.this_node << std::endl;
  out << "number_of_nodes = " << layout.number_of_nodes << std::endl;
  return out;
}

// QIO calls the I/O node map with only a node number, so the grid of the open file lives here
static std::array<int, 4> partition_grid;

// The I/O node of a QMP node: the node at the lowest corner of its partition
static int partition_io_node(const int node)
{
  const int *dims = QMP_get_logical_dimensions();
  int coords[4];
  QMP_get_logical_coordinates_from2(coords, node);
  for (int d = 0; d < 4; d++) coords[d] -= coords[d] % (dims[d] / partition_grid[d]);
  return QMP_get_node_number_from(coords);
}

// The file system to pass to QIO for a read or write: none with a zero grid, or in serial mode for a single file
// (where QIO ignores the map), else one whose I/O node map follows the grid while this object exists.  QIO reads and
// writes partfiles through the I/O nodes in either mode.  The QMP topology must already be declared.
class PartitionFilesystem
{
  QIO_Filesystem fs = {};
  bool used = false;

public:
  PartitionFilesystem(QudaFileIOMode io_mode, const std::array<int, 4> &grid, bool partfile = false)
  {
    const int *qmp_dims = QMP_get_logical_dimensions();
    if (!quda::io::check_partition_grid(partfile ? QUDA_FILE_IO_PARALLEL : io_mode, grid,
                                        {qmp_dims[0], qmp_dims[1], qmp_dims[2], qmp_dims[3]}))
      return;
    partition_grid = grid;
    fs.my_io_node = partition_io_node;
    used = true;
  }

  ~PartitionFilesystem()
  {
    if (used) partition_grid = {};
  }

  PartitionFilesystem(const PartitionFilesystem &) = delete;
  PartitionFilesystem &operator=(const PartitionFilesystem &) = delete;

  QIO_Filesystem *get() { return used ? &fs : nullptr; }
};

// Whether QIO will look for a partfile when reading a file: it reads <filename> if there is one
static bool partfile_name(const char *filename)
{
  int missing = 0;
  if (QMP_get_node_number() == 0) missing = access(filename, F_OK) != 0;
  QMP_broadcast(&missing, sizeof(missing));
  return missing;
}

// QIO's put and get callbacks, with arg pointing to a quda::io::HostField callback: the buffer holds the items of the
// site at the QUDA host index, in native byte order and the file precision
static void put_site(char *buf, size_t index, int, void *arg)
{
  (*static_cast<const quda::io::SiteUnpack *>(arg))(0, index, buf);
}

static void get_site(char *buf, size_t index, int, void *arg)
{
  (*static_cast<const quda::io::SitePack *>(arg))(0, index, buf);
}

QIO_Reader *open_test_input(const char *filename, int volfmt, int serpar, QIO_Filesystem *fs = nullptr)
{
  QIO_Iflag iflag;

  iflag.serpar = serpar;
  iflag.volfmt = volfmt;

  /* Create the file XML */
  QIO_String *xml_file_in = QIO_string_create();

  /* Open the file for reading */
  QIO_Reader *infile = QIO_open_read(xml_file_in, filename, &layout, fs, &iflag);

  if (infile == NULL) {
    printfQuda("%s(%d): QIO_open_read returns NULL.\n", __func__, quda_this_node);
    QIO_string_destroy(xml_file_in);
    return NULL;
  }

  printfQuda("%s: QIO_open_read done.\n", __func__);
  printfQuda("%s: User file info is \"%s\"\n", __func__, QIO_string_ptr(xml_file_in));

  QIO_string_destroy(xml_file_in);
  return infile;
}

QIO_Writer *open_test_output(const char *filename, const char *xml_write_file, int volfmt, int serpar, int ildgstyle,
                             const char *ildg_lfn, QIO_Filesystem *fs)
{
  QIO_Oflag oflag;

  oflag.serpar = serpar;
  oflag.ildgstyle = ildgstyle;
  // QIO writes the ildg-data-lfn record only if an LFN is given
  oflag.ildgLFN = nullptr;
  if (ildg_lfn) {
    oflag.ildgLFN = QIO_string_create();
    QIO_string_set(oflag.ildgLFN, ildg_lfn);
  }
  oflag.mode = QIO_TRUNC;

  /* Create the file XML */
  QIO_String *xml_file_out = QIO_string_create();
  QIO_string_set(xml_file_out, xml_write_file);

  /* Open the file for reading */
  QIO_Writer *outfile = QIO_open_write(xml_file_out, filename, volfmt, &layout, fs, &oflag);

  if (oflag.ildgLFN) QIO_string_destroy(oflag.ildgLFN);
  if (outfile == NULL) {
    printfQuda("%s(%d): QIO_open_write returns NULL.\n", __func__, quda_this_node);
    QIO_string_destroy(xml_file_out);
    return NULL;
  }

  printfQuda("%s: QIO_open_write done.\n", __func__);
  printfQuda("%s: User file info is \"%s\"\n", __func__, QIO_string_ptr(xml_file_out));

  QIO_string_destroy(xml_file_out);
  return outfile;
}

// the box of a hypercube record, as quda::scidac::hypercube_str gives it
static std::string hypercube_str(QIO_RecordInfo *rec_info)
{
  quda::io::Hypercube box;
  for (int d = 0; d < 4 && d < QIO_get_hyper_spacetime(rec_info); d++) {
    box.lower[d] = QIO_get_hyperlower(rec_info)[d];
    box.upper[d] = QIO_get_hyperupper(rec_info)[d];
  }
  return quda::scidac::hypercube_str(box);
}

// reads the next record into a host field; a hypercube record is an error unless allowed, when it fills only its box
int read_field(QIO_Reader *infile, const quda::io::HostField &field, int nSpin, int nColor, int len,
               QudaPrecision *file_prec_out = nullptr, std::string *record_xml_out = nullptr,
               const char *filename = nullptr, int record = 0, bool allow_hypercube = false)
{
  const int count = field.items();
  // Get the QIO record and string
  char dummy[100] = "";
  QIO_RecordInfo *rec_info = QIO_create_record_info(0, NULL, NULL, 0, dummy, dummy, 0, 0, 0, 0);
  QIO_String *xml_record_in = QIO_string_create();

  int status = QIO_read_record_info(infile, rec_info, xml_record_in);
  int prec = *QIO_get_precision(rec_info);

  // Check if the read was successful or not.
  logQuda(QUDA_VERBOSE, "%s: QIO_read_record_info returns status %d\n", __func__, status);
  if (status != QIO_SUCCESS) { errorQuda("get_prec failed\n"); }

  if (QIO_get_recordtype(rec_info) == QIO_HYPER && !allow_hypercube)
    errorQuda("Binary record %d of %s is a hypercube record (sites %s), which only staggered source files hold", record,
              filename ? filename : "the file", hypercube_str(rec_info).c_str());

  // Query components of record
  int in_nSpin = QIO_get_spins(rec_info);
  int in_nColor = QIO_get_colors(rec_info);
  int in_count = QIO_get_datacount(rec_info);   // 4 for gauge fields, nVec for packs of vectors
  int in_typesize = QIO_get_typesize(rec_info); // size of data at each site in bytes
  QudaPrecision file_prec = (prec == 70) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;

  // Various checks
  // Note: we exclude gauge fields from this b/c QUDA originally saved gauge fields as
  // nSpin == 1, nColor == 9, while it's supposed to be (0,3).
  // Further, even if the nSpin and nColor don't agree, what really matters is the
  // total typesize check.
  if (len != 18) {
    if (!quda::scidac::spins_match(QIO_get_datatype(rec_info), in_nSpin, nSpin))
      warningQuda("QIO_get_spins %d does not match expected number of spins %d", in_nSpin, nSpin);

    if (in_nColor != nColor)
      warningQuda("QIO_get_colors %d does not match expected number of colors %d", in_nColor, nColor);
  }

  if (in_count != count) errorQuda("QIO_get_datacount %d does not match expected number of fields %d", in_count, count);

  if (in_typesize != file_prec * len)
    errorQuda("QIO_get_typesize %d does not match expected datasize %d", in_typesize, file_prec * len);

  // Print the XML string.
  // The len != 18 is a WAR for this line segfaulting on some Chroma configs.
  // Tracked on github via #936
  if (len != 18 && QIO_string_length(xml_record_in) > 0)
    logQuda(QUDA_VERBOSE, "QIO string: %s\n", QIO_string_ptr(xml_record_in));

  // Get total size. Could probably check the filesize better, but tbd.
  size_t rec_size = file_prec * count * len;

  /* Read the field record and convert to cpu precision*/
  auto unpack = field.unpacker(file_prec);
  status = QIO_read(infile, rec_info, xml_record_in, put_site, rec_size, file_prec, &unpack);

  if (file_prec_out) *file_prec_out = file_prec;
  if (record_xml_out && QIO_string_ptr(xml_record_in)) *record_xml_out = QIO_string_ptr(xml_record_in);

  QIO_string_destroy(xml_record_in);
  QIO_destroy_record_info(rec_info);

  // QIO compares the checksums on its master I/O node only, after reading the whole record
  QMP_broadcast(&status, sizeof(status));
  logQuda(QUDA_VERBOSE, "%s: QIO_read returns status %d\n", __func__, status);
  return status;
}

int read_su3_field(QIO_Reader *infile, const quda::io::HostField &field, QudaPrecision *file_prec,
                   std::string *record_xml, const char *filename)
{
  return read_field(infile, field, 1, 9, 18, file_prec, record_xml, filename);
}

// The USQCD gauge field record XML: <usqcdInfo> with the average plaquette and link trace.  This uses QIO's own
// encoder rather than quda::scidac::usqcd_info_xml, so comparing the output of the two backends checks the latter.
static std::string encode_usqcd_info(double plaq, double linktr)
{
  char plaq_str[32], linktr_str[32];
  snprintf(plaq_str, sizeof(plaq_str), "%.16e", plaq);
  snprintf(linktr_str, sizeof(linktr_str), "%.16e", linktr);
  // QIO's XML parser does not handle nested tags of the same name, so QUDA's own record XML (which has an <info>
  // tag) cannot be embedded in the <info> field
  char info_str[] = "QUDA";

  QIO_USQCDLatticeInfo *info = QIO_create_usqcd_lattice_info(plaq_str, linktr_str, info_str);
  QIO_String *xml = QIO_string_create();
  QIO_encode_usqcd_lattice_info(xml, info);
  std::string result = QIO_string_ptr(xml);
  QIO_string_destroy(xml);
  QIO_destroy_usqcd_lattice_info(info);
  return result;
}

static bool decode_usqcd_info(const std::string &record_xml, double &plaq, double &linktr)
{
  if (record_xml.find("usqcdInfo") == std::string::npos) return false;

  QIO_String *xml = QIO_string_create();
  QIO_string_set(xml, record_xml.c_str());
  auto info = std::make_unique<QIO_USQCDLatticeInfo>(); // too large for the stack
  QIO_decode_usqcd_lattice_info(info.get(), xml);
  QIO_string_destroy(xml);

  if (!QIO_defined_plaq(info.get()) || !QIO_defined_linktr(info.get())) return false;
  plaq = strtod(QIO_get_plaq(info.get()), nullptr);
  linktr = strtod(QIO_get_linktr(info.get()), nullptr);
  return true;
}

void set_layout(const int *X, QudaSiteSubset subset = QUDA_FULL_SITE_SUBSET)
{
  // QIO numbers the ranks of QMP's global communicator
  if (static_cast<int>(comm_size()) != QMP_get_number_of_nodes())
    errorQuda("The external QIO backend cannot be used while the communicator is split (%lu of %d ranks)", comm_size(),
              QMP_get_number_of_nodes());

  /* Lattice dimensions */
  size_t lattice_dim = 4; // assume the comms topology is 4-d
  size_t lattice_volume = 1;
  for (int d = 0; d < 4; d++) {
    lattice_size[d] = comm_dim(d) * X[d];
    lattice_volume *= (size_t)lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if (quda_setup_layout(lattice_size, lattice_dim, QMP_get_number_of_nodes(), subset == QUDA_PARITY_SITE_SUBSET) != 0) {
    errorQuda("Setup layout failed\n");
  }
  printfQuda("%s layout set for %d nodes\n", __func__, QMP_get_number_of_nodes());

  /* Build the layout structure */
#ifdef QIO_HAS_EXTENDED_LAYOUT
  // members for smaller lattices are not used here
  layout.node_number = NULL;
  layout.node_index = NULL;
  layout.get_coords = NULL;
  layout.num_sites = NULL;
  // members using the extended layout for large lattice support
  layout.node_number_ext = quda_node_number_ext;
  layout.node_index_ext = quda_node_index_ext;
  layout.get_coords_ext = quda_get_coords_ext;
  layout.num_sites_ext = quda_num_sites_ext;
  layout.arg = NULL;
  layout.sites_on_node = quda_num_sites_ext(quda_this_node, nullptr);
#else
  // legacy API
  layout.node_number = quda_node_number;
  layout.node_index = quda_node_index;
  layout.get_coords = quda_get_coords;
  layout.num_sites = quda_num_sites;
  layout.sites_on_node = quda_num_sites(quda_this_node);
#endif // QIO_HAS_EXTENDED_LAYOUT
  layout.latsize = lattice_size;
  layout.latdim = lattice_dim;
  layout.volume = lattice_volume;
  layout.this_node = quda_this_node;
  layout.number_of_nodes = QMP_get_number_of_nodes();
}

void read_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X,
                      QudaFileIOMode io_mode, const std::array<int, 4> &io_partition_grid, quda::scidac::GaugeInfo &info)
{
  quda_this_node = QMP_get_node_number();

  set_layout(X);
  PartitionFilesystem fs(io_mode, io_partition_grid, partfile_name(filename));

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, qio_serpar(io_mode), fs.get());
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the su3 field record */
  printfQuda("%s: reading su3 field\n", __func__);
  fflush(stdout);
  std::string record_xml;
  int status
    = read_su3_field(infile, quda::io::HostField(gauge, 4, precision, 18), &info.file_prec, &record_xml, filename);
  // on a checksum mismatch QIO has read the whole record, including the checksum, so the field is complete
  if (status != QIO_SUCCESS && status != QIO_CHECKSUM_MISMATCH) { errorQuda("read_su3_field failed %d\n", status); }
  info.sum = {QIO_get_reader_last_checksuma(infile), QIO_get_reader_last_checksumb(infile)};

  // the ILDG style and the user record XML are only known on the QIO master I/O node
  double usqcd_info[3] = {0.0, 0.0, 0.0}; // found, plaq, linktr
  int ildg = 0;
  if (quda_this_node == 0) {
    ildg = QIO_get_ildgstyle(infile) == QIO_ILDGLAT;
    if (decode_usqcd_info(record_xml, usqcd_info[1], usqcd_info[2])) usqcd_info[0] = 1.0;
  }
  QMP_broadcast(&ildg, sizeof(ildg));
  QMP_broadcast(usqcd_info, sizeof(usqcd_info));
  info.ildg = ildg;
  // as in QIO, ILDG files written by other codes have no checksum
  int native = QIO_get_reader_format(infile) == QIO_SCIDAC_NATIVE;
  QMP_broadcast(&native, sizeof(native));
  info.has_checksum = native;
  info.checksum_matches = native && status == QIO_SUCCESS;
  info.has_usqcd_info = usqcd_info[0] != 0.0;
  info.plaq = usqcd_info[1];
  info.linktr = usqcd_info[2];

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n", __func__);
}

std::vector<quda::scidac::FieldChecksum> read_spinor_field(const char *filename,
                                                           const std::vector<quda::io::HostField> &fields, const int *X,
                                                           QudaSiteSubset subset, int nColor, int nSpin, bool milc,
                                                           QudaFileIOMode io_mode,
                                                           const std::array<int, 4> &io_partition_grid)
{
  quda_this_node = QMP_get_node_number();

  set_layout(X, subset);
  PartitionFilesystem fs(io_mode, io_partition_grid, partfile_name(filename));

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, qio_serpar(io_mode), fs.get());
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the spinor field records */
  printfQuda("%s: reading %lu records\n", __func__, fields.size());
  fflush(stdout);
  const int len = 2 * nSpin * nColor;
  std::vector<quda::scidac::FieldChecksum> sums;
  for (auto k = 0u; k < fields.size(); k++) {
    int status = read_field(infile, fields[k], milc ? 0 : nSpin, nColor, len, nullptr, nullptr, filename, k);
    // on a checksum mismatch QIO has read the whole record, including the checksum, so the vectors are complete
    if (status != QIO_SUCCESS && status != QIO_CHECKSUM_MISMATCH) {
      errorQuda("read_spinor_fields failed %d\n", status);
    }
    quda::scidac::FieldChecksum sum;
    sum.sum = {QIO_get_reader_last_checksuma(infile), QIO_get_reader_last_checksumb(infile)};
    sum.has_checksum = true;
    sum.checksum_matches = status == QIO_SUCCESS;
    sums.push_back(sum);
  }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n", __func__);
  return sums;
}

// Write a record with the given metadata, converting to its precision
static int write_record(QIO_Writer *outfile, const quda::io::HostField &field, const quda::scidac::FieldMetadata &m)
{
  // presently assumes 4-d
  const int nDim = 4;
  int lower[nDim] = {0, 0, 0, 0};
  int upper[nDim] = {lattice_size[0], lattice_size[1], lattice_size[2], lattice_size[3]};
  if (m.hypercube) {
    for (int d = 0; d < nDim; d++) {
      lower[d] = m.hypercube->lower[d];
      upper[d] = m.hypercube->upper[d];
    }
  }
  char precision[] = {m.prec == QUDA_DOUBLE_PRECISION ? 'D' : 'F', '\0'};
  QIO_RecordInfo *rec_info = QIO_create_record_info(m.hypercube ? QIO_HYPER : QIO_FIELD, lower, upper, nDim,
                                                    const_cast<char *>(m.datatype.c_str()), precision, m.colors,
                                                    m.spins, m.typesize, m.datacount);

  QIO_String *xml_record_out = QIO_string_create();
  QIO_string_set(xml_record_out, m.user_xml.c_str());

  auto pack = field.packer(m.prec);
  int status = QIO_write(outfile, rec_info, xml_record_out, get_site, static_cast<size_t>(m.typesize) * m.datacount,
                         m.prec, &pack);
  logQuda(QUDA_VERBOSE, "%s: QIO_write returns status %d\n", __func__, status);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_record_out);
  return status != QIO_SUCCESS;
}

quda::io::Checksum write_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X,
                                     bool partfile, QudaFileIOMode io_mode, const std::array<int, 4> &io_partition_grid,
                                     const quda::scidac::GaugeInfo &info)
{
  quda_this_node = QMP_get_node_number();

  set_layout(X);
  PartitionFilesystem fs(io_mode, io_partition_grid, partfile);

  if (precision != QUDA_DOUBLE_PRECISION && precision != QUDA_SINGLE_PRECISION)
    errorQuda("Error, file_prec=%d not supported", precision);
  auto records = quda::scidac::gauge_records(precision, info);
  if (info.has_usqcd_info) records.record.user_xml = encode_usqcd_info(info.plaq, info.linktr);

  /* Open the test file for writing */
  const char *lfn = info.ildg && !info.ildg_lfn.empty() ? info.ildg_lfn.c_str() : nullptr;
  QIO_Writer *outfile
    = open_test_output(filename, records.user_file_xml.c_str(), partfile ? QIO_PARTFILE : QIO_SINGLEFILE,
                       qio_serpar(io_mode), info.ildg ? QIO_ILDGLAT : QIO_ILDGNO, lfn, fs.get());
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Write the gauge field record */
  printfQuda("%s: writing the gauge field\n", __func__);
  fflush(stdout);
  if (write_record(outfile, quda::io::HostField(gauge, 4, precision, 18), records.record)) {
    errorQuda("write_gauge_field failed\n");
  }

  // QIO silently falls back to a plain SciDAC file if the record cannot be written as ILDG, recording this on the
  // master I/O node
  if (info.ildg) {
    int ildg = outfile->ildgstyle == QIO_ILDGLAT;
    QMP_broadcast(&ildg, sizeof(ildg));
    if (!ildg) errorQuda("QIO did not write %s in ILDG format", filename);
  }
  quda::io::Checksum sum = {QIO_get_writer_last_checksuma(outfile), QIO_get_writer_last_checksumb(outfile)};

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n", __func__);
  return sum;
}

std::vector<quda::io::Checksum> write_spinor_field(const char *filename, const quda::scidac::VectorRecords &records,
                                                   const std::vector<quda::io::HostField> &fields, const int *X,
                                                   QudaSiteSubset subset, bool partfile, QudaFileIOMode io_mode,
                                                   const std::array<int, 4> &io_partition_grid)
{
  quda_this_node = QMP_get_node_number();

  set_layout(X, subset);
  PartitionFilesystem fs(io_mode, io_partition_grid, partfile);

  /* Open the test file for writing */
  QIO_Writer *outfile
    = open_test_output(filename, records.user_file_xml.c_str(), partfile ? QIO_PARTFILE : QIO_SINGLEFILE,
                       qio_serpar(io_mode), QIO_ILDGNO, nullptr, fs.get());
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Write the spinor field records */
  printfQuda("%s: writing %lu records\n", __func__, fields.size());
  fflush(stdout);
  std::vector<quda::io::Checksum> sums;
  for (auto k = 0u; k < fields.size(); k++) {
    if (write_record(outfile, fields[k], records.records[k])) { errorQuda("write_spinor_fields failed\n"); }
    sums.push_back({QIO_get_writer_last_checksuma(outfile), QIO_get_writer_last_checksumb(outfile)});
  }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n", __func__);
  return sums;
}

std::vector<quda::scidac::FieldChecksum> read_ks_fields(const char *filename,
                                                        const std::vector<quda::io::HostField> &fields, const int *X,
                                                        QudaFileIOMode io_mode,
                                                        const std::array<int, 4> &io_partition_grid)
{
  quda_this_node = QMP_get_node_number();

  set_layout(X);
  PartitionFilesystem fs(io_mode, io_partition_grid, partfile_name(filename));

  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, qio_serpar(io_mode), fs.get());
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  printfQuda("%s: reading %lu records\n", __func__, fields.size());
  fflush(stdout);
  std::vector<quda::scidac::FieldChecksum> sums;
  for (auto k = 0u; k < fields.size(); k++) {
    // color vectors or complex fields, without spins
    const int len = fields[k].item_bytes(QUDA_SINGLE_PRECISION) / QUDA_SINGLE_PRECISION;
    int status = read_field(infile, fields[k], 0, len == 6 ? 3 : 0, len, nullptr, nullptr, filename, k, true);
    // on a checksum mismatch QIO has read the whole record, including the checksum, so the field is complete
    if (status != QIO_SUCCESS && status != QIO_CHECKSUM_MISMATCH) errorQuda("read_ks_fields failed %d\n", status);
    quda::scidac::FieldChecksum sum;
    sum.sum = {QIO_get_reader_last_checksuma(infile), QIO_get_reader_last_checksumb(infile)};
    sum.has_checksum = true;
    sum.checksum_matches = status == QIO_SUCCESS;
    sums.push_back(sum);
  }

  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n", __func__);
  return sums;
}
