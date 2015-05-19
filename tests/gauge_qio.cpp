#include <qio.h>
#include <qio_util.h>
#include <quda.h>
#include <util_quda.h>

QIO_Layout layout;
int lattice_dim;
int lattice_size[4];
int this_node;

QIO_Reader *open_test_input(char *filename, int volfmt, int serpar,
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
    printf("%s(%d): QIO_open_read returns NULL.\n",myname,this_node);
    return NULL;
  }

  printfQuda("%s: QIO_open_read done.\n",myname,this_node);
  printfQuda("%s: User file info is \"%s\"\n", myname, QIO_string_ptr(xml_file_in));

  QIO_string_destroy(xml_file_in);
  return infile;
}

/* get QIO record precision */
QudaPrecision get_prec(QIO_Reader *infile) {
  QIO_RecordInfo *rec_info = QIO_create_record_info(0, NULL, NULL, 0, "", "", 0, 0, 0, 0);
  QIO_String *xml_file = QIO_string_create();
  int status = QIO_read_record_info(infile, rec_info, xml_file);
  int prec = *QIO_get_precision(rec_info);
  QIO_destroy_record_info(rec_info);
  QIO_string_destroy(xml_file);

  return (prec == 70) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
}

int read_su3_field(QIO_Reader *infile, int count, void *field_in[], QudaPrecision cpu_prec, char *myname)
{
  QIO_String *xml_record_in;
  QIO_RecordInfo rec_info;
  int status;
  
  /* Query the precision */
  QudaPrecision file_prec = get_prec(infile);
  size_t rec_size = file_prec*count*18;

  /* Create the record XML */
  xml_record_in = QIO_string_create();

  /* Read the field record and convert to cpu precision*/
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<double,double,18>, 
			rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<double,float,18>, 
			rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  } else {
    if (file_prec == QUDA_DOUBLE_PRECISION) {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<float,double,18>, 
			rec_size, QUDA_DOUBLE_PRECISION, field_in);
    } else {
      status = QIO_read(infile, &rec_info, xml_record_in, vputM<float,float,18>, 
			rec_size, QUDA_SINGLE_PRECISION, field_in);
    }
  }

  printfQuda("%s: QIO_read_record_data returns status %d\n", myname, status);
  if(status != QIO_SUCCESS)return 1;
  return 0;
}

void read_gauge_field(char *filename, void *gauge[], QudaPrecision precision, int *X, int argc, char *argv[]) {
  QIO_Reader *infile;
  int status;
  int sites_on_node = 0;
  QMP_thread_level_t provided;
  char myname[] = "qio_load";

  this_node = mynode();

  /* Lattice dimensions */
  lattice_dim = 4;
  int lattice_volume = 1;
  for (int d=0; d<4; d++) {
    lattice_size[d] = QMP_get_logical_dimensions()[d]*X[d];
    lattice_volume *= lattice_size[d];
  }

  /* Set the mapping of coordinates to nodes */
  if(setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0)
    { printfQuda("Setup layout failed\n"); exit(0); }
  printfQuda("%s layout set for %d nodes\n", myname, QMP_get_number_of_nodes());
  sites_on_node = num_sites(this_node);

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
  infile = open_test_input(filename, QIO_UNKNOWN, QIO_PARALLEL, myname);
  if(infile == NULL) { printf("Open file failed\n"); exit(0); }

  /* Read the su3 field record */
  printfQuda("%s: reading su3 field\n",myname); fflush(stdout);
  status = read_su3_field(infile, 4, gauge, precision, myname);
  if(status) { printf("read_su3_field failed %d\n", status); exit(0); }

  /* Close the file */
  QIO_close_read(infile);
  printfQuda("%s: Closed file for reading\n",myname);  
    
}
