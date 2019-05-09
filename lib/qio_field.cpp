#include <qio.h>
#include <qio_util.h>
#include <quda.h>
#include <util_quda.h>

QIO_Layout layout;
int lattice_dim;
int lattice_size[4];
int this_node;

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
    printfQuda("%s(%d): QIO_open_read returns NULL.\n", __func__, this_node);
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
  if (outfile == NULL) {
    printfQuda("%s(%d): QIO_open_write returns NULL.\n", __func__, this_node);
    return NULL;
  }

  printfQuda("%s: QIO_open_write done.\n",__func__);
  printfQuda("%s: User file info is \"%s\"\n", __func__, QIO_string_ptr(xml_file_out));

  QIO_string_destroy(xml_file_out);
  return outfile;
}

/* get QIO record precision */
QudaPrecision get_prec(QIO_Reader *infile)
{
  char dummy[100] = "";
  QIO_RecordInfo *rec_info = QIO_create_record_info(0, NULL, NULL, 0, dummy, dummy, 0, 0, 0, 0);
  QIO_String *xml_file = QIO_string_create();
  int status = QIO_read_record_info(infile, rec_info, xml_file);
  int prec = *QIO_get_precision(rec_info);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_file);

  printfQuda("%s: QIO_read_record_data returns status %d\n", __func__, status);
  if (status != QIO_SUCCESS)  { errorQuda("get_prec failed\n"); }

  return (prec == 70) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
}

template <int len>
int read_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec)
{
  QIO_String *xml_record_in;
  QIO_RecordInfo rec_info;
  int status;

  /* Query the precision */
  QudaPrecision file_prec = get_prec(infile);
  size_t rec_size = file_prec*count*len;

  /* Create the record XML */
  xml_record_in = QIO_string_create();

  /* Read the field record and convert to cpu precision*/
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<double,double,len>,
			rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<double,float,len>,
			rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<float,double,len>,
			rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<float,float,len>,
			rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  }

  QIO_string_destroy(xml_record_in);
  printfQuda("%s: QIO_read_record_data returns status %d\n", __func__, status);
  if (status != QIO_SUCCESS) return 1;
  return 0;
}

int read_su3_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec)
{
  return read_field<18>(infile, count, field_in, cpu_prec);
}

void set_layout(const int *X)
{
  /* Lattice dimensions */
  lattice_dim = 4;
  int lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = comm_dim(d)*X[d];
    lattice_volume *= lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if (setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0) { errorQuda("Setup layout failed\n"); }
  printfQuda("%s layout set for %d nodes\n", __func__, QMP_get_number_of_nodes());
  int sites_on_node = num_sites(this_node);

  /* Build the layout structure */
  layout.node_number     = node_number;
  layout.node_index      = node_index;
  layout.get_coords      = get_coords;
  layout.num_sites       = num_sites;
  layout.latsize         = lattice_size;
  layout.latdim          = lattice_dim;
  layout.volume          = lattice_volume;
  layout.sites_on_node   = sites_on_node;
  layout.this_node       = this_node;
  layout.number_of_nodes = QMP_get_number_of_nodes();
}

void read_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X, int argc, char *argv[])
{
  this_node = mynode();

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

// count is the number of vectors
// Ninternal is the size of the "inner struct" (24 for Wilson spinor)
int read_field(QIO_Reader *infile, int Ninternal, int count, void *field_in[], QudaPrecision cpu_prec)
{
  int status = 0;
  switch (Ninternal) {
  case 24:
    status = read_field<24>(infile, count, field_in, cpu_prec);
    break;
  case 96:
    status = read_field<96>(infile, count, field_in, cpu_prec);
    break;
  case 128:
    status = read_field<128>(infile, count, field_in, cpu_prec);
    break;
  default:
    errorQuda("Undefined %d", Ninternal);
  }
  return status;
}

void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X,
		       int nColor, int nSpin, int Nvec, int argc, char *argv[])
{
  this_node = mynode();

  set_layout(X);

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL);
  if (infile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the spinor field record */
  printfQuda("%s: reading %d vector fields\n", __func__, Nvec); fflush(stdout);
  int status = read_field(infile, 2*nSpin*nColor, Nvec, V, precision);
  if (status) { errorQuda("read_spinor_fields failed %d\n", status); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",__func__);
}

template <int len>
int write_field(QIO_Writer *outfile, int count, void *field_out[], QudaPrecision file_prec,
		QudaPrecision cpu_prec, int nSpin, int nColor, const char *type)
{
  char xml_write_field[] = "Dummy user record XML for SU(N) field";
  int status;

  // Create the record info for the field
  if (file_prec != QUDA_DOUBLE_PRECISION && file_prec != QUDA_SINGLE_PRECISION)
    errorQuda("Error, file_prec=%d not supported", file_prec);

  const char *precision = (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F";

  // presently assumes 4-d
  const int nDim = 4;
  int lower[nDim] = {0, 0, 0, 0};
  int upper[nDim] = {lattice_size[0], lattice_size[1], lattice_size[2], lattice_size[3]};

  QIO_RecordInfo *rec_info = QIO_create_record_info(QIO_FIELD, lower, upper, nDim, const_cast<char*>(type),
                                                    const_cast<char*>(precision), nColor, nSpin, file_prec*len, count);

  // Create the record XML for the field
  QIO_String *xml_record_out = QIO_string_create();
  QIO_string_set(xml_record_out,xml_write_field);

  /* Write the field record converting to desired file precision*/
  size_t rec_size = file_prec*count*len;
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_write(outfile, rec_info, xml_record_out, vgetM<double,double,len>,
			 rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status = QIO_write(outfile, rec_info, xml_record_out, vgetM<double,float,len>,
			 rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_write(outfile, rec_info, xml_record_out, vgetM<float,double,len>,
			 rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status = QIO_write(outfile, rec_info, xml_record_out, vgetM<float,float,len>,
			 rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  }

  printfQuda("%s: QIO_write_record_data returns status %d\n", __func__, status);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_record_out);

  if (status != QIO_SUCCESS) return 1;
  return 0;
}

int write_su3_field(QIO_Writer *outfile, int count, void *field_out[],
    QudaPrecision file_prec, QudaPrecision cpu_prec, const char* type)
{
  return write_field<18>(outfile, count, field_out, file_prec, cpu_prec, 1, 9, type); 
}

void write_gauge_field(const char *filename, void* gauge[], QudaPrecision precision, const int *X, int argc, char* argv[])
{
  this_node = mynode();

  set_layout(X);

  QudaPrecision file_prec = precision;

  char type[128];
  sprintf(type, "QUDA_%sNc%d_GaugeField", (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F", 3);

  /* Open the test file for writing */
  QIO_Writer *outfile = open_test_output(filename, QIO_SINGLEFILE, QIO_PARALLEL, QIO_ILDGNO);
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Write the gauge field record */
  printfQuda("%s: writing the gauge field\n", __func__); fflush(stdout);
  int status = write_su3_field(outfile, 4, gauge, precision, precision, type);
  if (status) { errorQuda("write_gauge_field failed %d\n", status); }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n", __func__);
}


// count is the number of vectors
// Ninternal is the size of the "inner struct" (24 for Wilson spinor)
int write_field(QIO_Writer *outfile, int Ninternal, int count, void *field_out[],
		QudaPrecision file_prec, QudaPrecision cpu_prec,
		int nSpin, int nColor, const char *type)
{
  int status = 0;
  switch (Ninternal) {
  case 24:
    status = write_field<24>(outfile, count, field_out, file_prec, cpu_prec, nSpin, nColor, type);
    break;
  case 96:
    status = write_field<96>(outfile, count, field_out, file_prec, cpu_prec, nSpin, nColor, type);
    break;
  case 128:
    status = write_field<128>(outfile, count, field_out, file_prec, cpu_prec, nSpin, nColor, type);
    break;
  default:
    errorQuda("Undefined %d", Ninternal);
  }
  return status;
}

void write_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X,
                        int nColor, int nSpin, int Nvec, int argc, char *argv[])
{
  this_node = mynode();

  set_layout(X);

  QudaPrecision file_prec = precision;

  char type[128];
  sprintf(type, "QUDA_%sNs%dNc%d_ColorSpinorField", (file_prec == QUDA_DOUBLE_PRECISION) ? "D" : "F", nSpin, nColor);

  /* Open the test file for reading */
  QIO_Writer *outfile = open_test_output(filename, QIO_SINGLEFILE, QIO_PARALLEL, QIO_ILDGNO);
  if (outfile == NULL) { errorQuda("Open file failed\n"); }

  /* Read the spinor field record */
  printfQuda("%s: writing %d vector fields\n", __func__, Nvec); fflush(stdout);
  int status = write_field(outfile, 2*nSpin*nColor, Nvec, V, precision, precision, nSpin, nColor, type);
  if (status) { errorQuda("write_spinor_fields failed %d\n", status); }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n",__func__);
}
