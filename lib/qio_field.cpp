#include <iostream>
#include <qmp.h>
#include <qio.h>
#include <quda.h>
#include <util_quda.h>
#include <layout_hyper.h>

#include <string>

using namespace quda;

static QIO_Layout layout;
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

static int vlen;

// for matrix fields this order implies [color][color][complex]
// for vector fields this order implies [spin][color][complex]
// templatized version to allow for precision conversion
template <typename oFloat, typename iFloat> void vput(char *s1, size_t index, int count, void *s2)
{
  oFloat **field = (oFloat **)s2;
  iFloat *src = (iFloat *)s1;

  // For the site specified by "index", move an array of "count" data
  // from the read buffer to an array of fields

  for (int i = 0; i < count; i++) {
    oFloat *dest = field[i] + vlen * index;
    for (int j = 0; j < vlen; j++) dest[j] = src[i * vlen + j];
  }
}

// for vector fields this order implies [spin][color][complex]
// templatized version of vget_M to allow for precision conversion
template <typename oFloat, typename iFloat> void vget(char *s1, size_t index, int count, void *s2)
{
  iFloat **field = (iFloat **)s2;
  oFloat *dest = (oFloat *)s1;

  /* For the site specified by "index", move an array of "count" data
     from the array of fields to the write buffer */
  for (int i = 0; i < count; i++, dest += vlen) {
    iFloat *src = field[i] + vlen * index;
    for (int j = 0; j < vlen; j++) dest[j] = src[j];
  }
}

QIO_Reader *open_test_input(const char *filename, int volfmt, int serpar)
{
  QIO_Iflag iflag;

  iflag.serpar = serpar;
  iflag.volfmt = volfmt;

  /* Create the file XML */
  QIO_String *xml_file_in = QIO_string_create();

  /* Open the file for reading */
  QIO_Reader *infile = QIO_open_read(xml_file_in, filename, &layout, NULL, &iflag);

  if (infile == NULL) {
    printfQuda("%s(%d): QIO_open_read returns NULL.\n", __func__, quda_this_node);
    QIO_string_destroy(xml_file_in);
    return NULL;
  }

  printfQuda("%s: QIO_open_read done.\n",__func__);
  printfQuda("%s: User file info is \"%s\"\n", __func__, QIO_string_ptr(xml_file_in));

  QIO_string_destroy(xml_file_in);
  return infile;
}

QIO_Writer *open_test_output(const char *filename, int volfmt, int serpar, int ildgstyle)
{
  char xml_write_file[] = "Dummy user file XML";
  QIO_Filesystem filesys;
  QIO_Oflag oflag;

  oflag.serpar = serpar;
  oflag.ildgstyle = ildgstyle;
  oflag.ildgLFN = QIO_string_create();
  QIO_string_set(oflag.ildgLFN,"monkey");
  oflag.mode = QIO_TRUNC;

  filesys.my_io_node = 0;
  filesys.master_io_node = 0;

  /* Create the file XML */
  QIO_String *xml_file_out = QIO_string_create();
  QIO_string_set(xml_file_out,xml_write_file);

  /* Open the file for reading */
  QIO_Writer *outfile = QIO_open_write(xml_file_out, filename, volfmt, &layout, &filesys, &oflag);

  QIO_string_destroy(oflag.ildgLFN);
  if (outfile == NULL) {
    printfQuda("%s(%d): QIO_open_write returns NULL.\n", __func__, quda_this_node);
    QIO_string_destroy(xml_file_out);
    return NULL;
  }

  printfQuda("%s: QIO_open_write done.\n",__func__);
  printfQuda("%s: User file info is \"%s\"\n", __func__, QIO_string_ptr(xml_file_out));

  QIO_string_destroy(xml_file_out);
  return outfile;
}

int read_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec, QudaSiteSubset, QudaParity,
               int nSpin, int nColor, int len)
{
  // Get the QIO record and string
  char dummy[100] = "";
  QIO_RecordInfo *rec_info = QIO_create_record_info(0, NULL, NULL, 0, dummy, dummy, 0, 0, 0, 0);
  QIO_String *xml_record_in = QIO_string_create();

  int status = QIO_read_record_info(infile, rec_info, xml_record_in);
  int prec = *QIO_get_precision(rec_info);

  // Check if the read was successful or not.
  printfQuda("%s: QIO_read_record_data returns status %d\n", __func__, status);
  if (status != QIO_SUCCESS)  { errorQuda("get_prec failed\n"); }

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
    if (in_nSpin != nSpin) warningQuda("QIO_get_spins %d does not match expected number of colors %d", in_nSpin, nSpin);

    if (in_nColor != nColor)
      warningQuda("QIO_get_colors %d does not match expected number of spins %d", in_nColor, nColor);
  }

  if (in_count != count) errorQuda("QIO_get_datacount %d does not match expected number of fields %d", in_count, count);

  if (in_typesize != file_prec * len)
    errorQuda("QIO_get_typesize %d does not match expected datasize %d", in_typesize, file_prec * len);

  // Print the XML string.
  // The len != 18 is a WAR for this line segfaulting on some Chroma configs.
  // Tracked on github via #936
  if (len != 18 && QIO_string_length(xml_record_in) > 0) printfQuda("QIO string: %s\n", QIO_string_ptr(xml_record_in));

  // Get total size. Could probably check the filesize better, but tbd.
  size_t rec_size = file_prec * count * len;

  vlen = len;

  /* Read the field record and convert to cpu precision*/
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, rec_info, xml_record_in, vput<double, double>, rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, rec_info, xml_record_in, vput<double, float>, rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, rec_info, xml_record_in, vput<float, double>, rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, rec_info, xml_record_in, vput<float, float>, rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  }

  QIO_string_destroy(xml_record_in);
  QIO_destroy_record_info(rec_info);
  printfQuda("%s: QIO_read_record_data returns status %d\n", __func__, status);
  if (status != QIO_SUCCESS) return 1;
  return 0;
}

int read_su3_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec)
{
  return read_field(infile, count, field_in, cpu_prec, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 1, 9, 18);
}

void set_layout(const int *X, QudaSiteSubset subset = QUDA_FULL_SITE_SUBSET)
{
  /* Lattice dimensions */
  size_t lattice_dim = 4; // assume the comms topology is 4-d
  size_t lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = comm_dim(d)*X[d];
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
  layout.latsize         = lattice_size;
  layout.latdim          = lattice_dim;
  layout.volume          = lattice_volume;
  layout.this_node = quda_this_node;
  layout.number_of_nodes = QMP_get_number_of_nodes();
}

void read_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X, int, char *[])
{
  quda_this_node = QMP_get_node_number();

  set_layout(X);

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL);
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the su3 field record */
  printfQuda("%s: reading su3 field\n",__func__); fflush(stdout);
  int status = read_su3_field(infile, 4, gauge, precision);
  if (status) { errorQuda("read_su3_field failed %d\n", status); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",__func__);
}

void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                       QudaParity parity, int nColor, int nSpin, int Nvec, int, char *[])
{
  quda_this_node = QMP_get_node_number();

  set_layout(X, subset);

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL);
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the spinor field record */
  printfQuda("%s: reading %d vector fields\n", __func__, Nvec); fflush(stdout);
  int status = read_field(infile, Nvec, V, precision, subset, parity, nSpin, nColor, 2 * nSpin * nColor);
  if (status) { errorQuda("read_spinor_fields failed %d\n", status); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",__func__);
}

int write_field(QIO_Writer *outfile, int count, const void *field_out[], QudaPrecision file_prec, QudaPrecision cpu_prec,
                QudaSiteSubset subset, QudaParity parity, int nSpin, int nColor, int len, const char *type)
{
  // Prepare a string.
  std::string xml_record = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><quda";
  switch (len) {
  case 6: xml_record += "StaggeredColorSpinorField>"; break; // SU(3) staggered
  case 18: xml_record += "GaugeFieldFile>"; break;           // SU(3) gauge field
  case 24: xml_record += "WilsonColorSpinorField>"; break;   // SU(3) Wilson
  default: xml_record += "MGColorSpinorField>"; break;       // MG color spinor vector
  }
  xml_record += "<version>BETA</version>";
  xml_record += "<type>" + std::string(type) + "</type><info>";

  // if parity+even, it's a half-x-dim even only vector
  // if parity+odd, it's a half-x-dim odd only vector
  // if full+even, it's a full vector with only even sites filled, odd are zero
  // if full+odd, it's a full vector with only odd sites filled, even are zero
  // if full+full, it's a full vector with all sites filled (either a full ColorSpinorField or a GaugeField)

  if (subset == QUDA_PARITY_SITE_SUBSET) {
    xml_record += "<subset>parity</subset>";
  } else {
    xml_record += "<subset>full</subset>";
  }
  if (parity == QUDA_EVEN_PARITY) {
    xml_record += "<parity>even</parity>";
  } else if (parity == QUDA_ODD_PARITY) {
    xml_record += "<parity>odd</parity>";
  } else {
    xml_record += "<parity>full</parity>";
  } // abuse/hack

  // A lot of this is redundant of the record info, but eh.
  xml_record += "<nColor>" + std::to_string(nColor) + "</nColor>";
  xml_record += "<nSpin>" + std::to_string(nSpin) + "</nSpin>";
  xml_record += "</info></quda";
  switch (len) {
  case 6: xml_record += "StaggeredColorSpinorField>"; break; // SU(3) staggered
  case 18: xml_record += "GaugeFieldFile>"; break;           // SU(3) gauge field
  case 24: xml_record += "WilsonColorSpinorField>"; break;   // SU(3) Wilson
  default: xml_record += "MGColorSpinorField>";              // MG color spinor vector
  }

  int status;

  // Create the record info for the field
  if (file_prec != QUDA_DOUBLE_PRECISION && file_prec != QUDA_SINGLE_PRECISION)
    errorQuda("Error, file_prec=%d not supported", file_prec);

  const char *precision = (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F";

  // presently assumes 4-d
  const int nDim = 4;
  int lower[nDim] = {0, 0, 0, 0};
  int upper[nDim] = {lattice_size[0], lattice_size[1], lattice_size[2], lattice_size[3]};

  QIO_RecordInfo *rec_info = QIO_create_record_info(QIO_FIELD, lower, upper, nDim, const_cast<char *>(type),
                                                    const_cast<char *>(precision), nColor, nSpin, file_prec * len, count);

  // Create the record XML for the field
  QIO_String *xml_record_out = QIO_string_create();
  QIO_string_set(xml_record_out, xml_record.c_str());

  /* Write the field record converting to desired file precision*/
  vlen = len;
  size_t rec_size = file_prec*count*len;
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status
        = QIO_write(outfile, rec_info, xml_record_out, vget<double, double>, rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status
        = QIO_write(outfile, rec_info, xml_record_out, vget<double, float>, rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status
        = QIO_write(outfile, rec_info, xml_record_out, vget<float, double>, rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status
        = QIO_write(outfile, rec_info, xml_record_out, vget<float, float>, rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  }

  printfQuda("%s: QIO_write_record_data returns status %d\n", __func__, status);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_record_out);

  if (status != QIO_SUCCESS) return 1;
  return 0;
}

int write_su3_field(QIO_Writer *outfile, int count, const void *field_out[],
    QudaPrecision file_prec, QudaPrecision cpu_prec, const char* type)
{
  return write_field(outfile, count, field_out, file_prec, cpu_prec, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 0, 3,
                     18, type);
}

void write_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X, int, char *[])
{
  quda_this_node = QMP_get_node_number();

  set_layout(X);

  QudaPrecision file_prec = precision;

  char type[128];
  sprintf(type, "QUDA_%sNc%d_GaugeField", (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F", 3);

  /* Open the test file for writing */
  QIO_Writer *outfile = open_test_output(filename, QIO_SINGLEFILE, QIO_PARALLEL, QIO_ILDGNO);
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Write the gauge field record */
  printfQuda("%s: writing the gauge field\n", __func__); fflush(stdout);
  int status = write_su3_field(outfile, 4, (const void**)gauge, precision, precision, type);
  if (status) { errorQuda("write_gauge_field failed %d\n", status); }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n", __func__);
}

void write_spinor_field(const char *filename, const void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                        QudaParity parity, int nColor, int nSpin, int Nvec, int, char *[])
{
  quda_this_node = QMP_get_node_number();

  set_layout(X, subset);

  QudaPrecision file_prec = precision;

  char type[128];
  sprintf(type, "QUDA_%sNs%dNc%d_ColorSpinorField", (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F", nSpin, nColor);

  /* Open the test file for reading */
  QIO_Writer *outfile = open_test_output(filename, QIO_SINGLEFILE, QIO_PARALLEL, QIO_ILDGNO);
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the spinor field record */
  printfQuda("%s: writing %d vector fields\n", __func__, Nvec); fflush(stdout);
  int status
    = write_field(outfile, Nvec, V, precision, precision, subset, parity, nSpin, nColor, 2 * nSpin * nColor, type);
  if (status) { errorQuda("write_spinor_fields failed %d\n", status); }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n",__func__);
}
