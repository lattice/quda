#include <qio.h>
#include <qio_util.h>
#include <quda.h>
#include <util_quda.h>

QIO_Layout layout;
int lattice_dim;
int lattice_size[4];
int this_node;

QIO_Reader *open_test_input(const char *filename, int volfmt, int serpar,
			    char *myname){
  QIO_String *xml_file_in;
  QIO_Reader *infile;
  QIO_Iflag iflag;

  iflag.serpar = serpar;
  iflag.volfmt = volfmt;

  /* Create the file XML */
  xml_file_in = QIO_string_create();

  /* Open the file for reading */
  infile = QIO_open_read(xml_file_in, filename, &layout, NULL, &iflag);
  if(infile == NULL){
    printfQuda("%s(%d): QIO_open_read returns NULL.\n",myname,this_node);
    return NULL;
  }

  printfQuda("%s: QIO_open_read done.\n",myname);
  printfQuda("%s: User file info is \"%s\"\n", myname, QIO_string_ptr(xml_file_in));

  QIO_string_destroy(xml_file_in);
  return infile;
}

QIO_Writer *open_test_output(const char *filename, int volfmt, int serpar,
			     int ildgstyle, char *myname){
  QIO_String *xml_file_out;
  char xml_write_file[] = "Dummy user file XML";
  QIO_Writer *outfile;
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
  xml_file_out = QIO_string_create();
  QIO_string_set(xml_file_out,xml_write_file);

  /* Open the file for reading */
  outfile = QIO_open_write(xml_file_out, filename, volfmt, &layout, 
			   &filesys, &oflag);
  if(outfile == NULL){
    printfQuda("%s(%d): QIO_open_write returns NULL.\n",myname,this_node);
    return NULL;
  }

  printfQuda("%s: QIO_open_write done.\n",myname);
  printfQuda("%s: User file info is \"%s\"\n", myname, QIO_string_ptr(xml_file_out));

  QIO_string_destroy(xml_file_out);
  return outfile;
}

/* get QIO record precision */
QudaPrecision get_prec(QIO_Reader *infile) {
  const char* myname="get_prec";
  char dummy[100] = "";
  QIO_RecordInfo *rec_info = QIO_create_record_info(0, NULL, NULL, 0, dummy, dummy, 0, 0, 0, 0);
  QIO_String *xml_file = QIO_string_create();
  int status = QIO_read_record_info(infile, rec_info, xml_file);
  int prec = *QIO_get_precision(rec_info);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_file);

  printfQuda("%s: QIO_read_record_data returns status %d\n", myname, status);
  if (status != QIO_SUCCESS)  { printfQuda("get_prec failed\n"); exit(0); }

  return (prec == 70) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
}

template <int len>
int read_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec, char *myname)
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

  printfQuda("%s: QIO_read_record_data returns status %d\n", myname, status);
  if (status != QIO_SUCCESS) return 1;
  return 0;
}

template <int len>
int write_field(QIO_Writer *infile, int count, void *field_out[], QudaPrecision file_prec, QudaPrecision cpu_prec, char *myname)
{
  QIO_String *xml_record_out;
  QIO_RecordInfo rec_info;
  int status;
  
  /* Query the precision */
  size_t rec_size = file_prec*count*len;

  /* Create the record XML */
  xml_record_out = QIO_string_create();

  /* Read the field record and convert to cpu precision*/
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_write(infile, &rec_info, xml_record_out, vgetM<double,double,len>, 
			 rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status = QIO_write(infile, &rec_info, xml_record_out, vgetM<double,float,len>, 
			 rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_write(infile, &rec_info, xml_record_out, vgetM<float,double,len>, 
			 rec_size, QUDA_DOUBLE_PRECISION, field_out);
    } else {
      status = QIO_write(infile, &rec_info, xml_record_out, vgetM<float,float,len>, 
			 rec_size, QUDA_SINGLE_PRECISION, field_out);
    }
  }

  printfQuda("%s: QIO_write_record_data returns status %d\n", myname, status);
  if (status != QIO_SUCCESS) return 1;
  return 0;
}

int read_su3_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec, char *myname)
{
  return read_field<18>(infile, count, field_in, cpu_prec, myname);
}

void read_gauge_field(const char *filename, void *gauge[], QudaPrecision precision, const int *X, int argc, char *argv[]) {
  char myname[] = "qio_load";

  this_node = mynode();

  /* Lattice dimensions */
  lattice_dim = 4;
  int lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = comm_dim(d)*X[d];
    lattice_volume *= lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if(setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0)
    { printfQuda("Setup layout failed\n"); exit(0); }
  printfQuda("%s layout set for %d nodes\n", myname, QMP_get_number_of_nodes());
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

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL, myname);
  if(infile == NULL) { printfQuda("Open file failed\n"); exit(0); }

  /* Read the su3 field record */
  printfQuda("%s: reading su3 field\n",myname); fflush(stdout);
  int status = read_su3_field(infile, 4, gauge, precision, myname);
  if(status) { printfQuda("read_su3_field failed %d\n", status); exit(0); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",myname);  
    
}

// count is the number of vectors
// Ninternal is the size of the "inner struct" (24 for Wilson spinor)
int read_field(QIO_Reader *infile, int Ninternal, int count, void *field_in[], QudaPrecision cpu_prec, char *myname)
{
  int status = 0;
  switch (Ninternal) {
  case 24:
    status = read_field<24>(infile, count, field_in, cpu_prec, myname);
    break;
  default:
    errorQuda("Undefined %d", Ninternal);
  }    

  return status;
}

// count is the number of vectors
// Ninternal is the size of the "inner struct" (24 for Wilson spinor)
int write_field(QIO_Writer *outfile, int Ninternal, int count, void *field_out[], 
		QudaPrecision file_prec, QudaPrecision cpu_prec, char *myname)
{
  int status = 0;
  switch (Ninternal) {
  case 24:
    status = write_field<24>(outfile, count, field_out, file_prec, cpu_prec, myname);
    break;
  default:
    errorQuda("Undefined %d", Ninternal);
  }    

  return status;
}

void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, 
		       int nColor, int nSpin, int Nvec, int argc, char *argv[]) {
  char myname[] = "qio_load";

  this_node = mynode();

  /* Lattice dimensions */
  lattice_dim = 4;
  int lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = comm_dim(d)*X[d];
    lattice_volume *= lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if(setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0)
    { printfQuda("Setup layout failed\n"); exit(0); }
  printfQuda("%s layout set for %d nodes\n", myname, QMP_get_number_of_nodes());
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

  /* Open the test file for reading */
  QIO_Reader *infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL, myname);
  if(infile == NULL) { printfQuda("Open file failed\n"); exit(0); }

  /* Read the spinor field record */
  printfQuda("%s: reading %d vector fields\n", myname, Nvec); fflush(stdout);
  int status = read_field(infile, 2*nSpin*nColor, Nvec, V, precision, myname);
  if(status) { printfQuda("read_spinor_fields failed %d\n", status); exit(0); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",myname);  
    
}

void write_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, 
		       int nColor, int nSpin, int Nvec, int argc, char *argv[]) {
  char myname[] = "qio_save";

  this_node = mynode();

  /* Lattice dimensions */
  lattice_dim = 4;
  int lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = comm_dim(d)*X[d];
    lattice_volume *= lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if(setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0)
    { printfQuda("Setup layout failed\n"); exit(0); }
  printfQuda("%s layout set for %d nodes\n", myname, QMP_get_number_of_nodes());
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

  /* Open the test file for reading */
  QIO_Writer *outfile = open_test_output(filename, QIO_SINGLEFILE, QIO_PARALLEL, QIO_ILDGNO, myname);
  if(outfile == NULL) { printfQuda("Open file failed\n"); exit(0); }

  /* Read the spinor field record */
  printfQuda("%s: writing %d vector fields\n", myname, Nvec); fflush(stdout);
  int status = write_field(outfile, 2*nSpin*nColor, Nvec, V, precision, precision, myname);
  if(status) { printfQuda("write_spinor_fields failed %d\n", status); exit(0); }

  /* Close the file */
  QIO_close_write(outfile);
  printfQuda("%s: Closed file for writing\n",myname);  
    
}
