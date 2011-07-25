#include <qio.h>
#include <qio_util.h>

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

  printf("%s(%d): QIO_open_read done.\n",myname,this_node);
  printf("%s(%d): User file info is \"%s\"\n",myname,this_node,
	 QIO_string_ptr(xml_file_in));

  QIO_string_destroy(xml_file_in);
  return infile;
}

int read_su3_field(QIO_Reader *infile, int count, 
		    suN_matrix *field_in[], char *myname)
{
  QIO_String *xml_record_in;
  QIO_RecordInfo rec_info;
  int status;
  
  /* Create the record XML */
  xml_record_in = QIO_string_create();

  /* Read the field record */
  status = QIO_read(infile, &rec_info, xml_record_in, 
		    vput_M, sizeof(suN_matrix)*count, sizeof(float), field_in);
  printf("%s(%d): QIO_read_record_data returns status %d\n",
	 myname,this_node,status);
  if(status != QIO_SUCCESS)return 1;
  return 0;
}

void readGaugeField(char *filename, float *gauge[], int *X, int argc, char *argv[]) {
  QIO_Reader *infile;
  int status;
  int sites_on_node = 0;
  QMP_thread_level_t provided;
  char myname[] = "qio_load";

  /* Start message passing */
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &provided);
  this_node = mynode();
  printf("%s(%d) QMP_init_msg_passing done\n",myname,this_node);

  /* Lattice dimensions */
  lattice_dim = 4;
  lattice_size[0] = X[0];
  lattice_size[1] = X[1];
  lattice_size[2] = X[2];
  lattice_size[3] = X[3];

  /* Set the mapping of coordinates to nodes */
  if(setup_layout(lattice_size, 4, QMP_get_number_of_nodes())!=0)
    { printf("Setup layout failed\n"); exit(0); }
  printf("%s(%d) layout set for %d nodes\n",myname,this_node,
	 QMP_get_number_of_nodes());
  sites_on_node = num_sites(this_node);

  /* Build the layout structure */
  layout.node_number     = node_number;
  layout.node_index      = node_index;
  layout.get_coords      = get_coords;
  layout.num_sites       = num_sites;
  layout.latsize         = lattice_size;
  layout.latdim          = lattice_dim;
  layout.volume          = X[0]*X[1]*X[2]*X[3];
  layout.sites_on_node   = sites_on_node;
  layout.this_node       = this_node;
  layout.number_of_nodes = QMP_get_number_of_nodes();

  /* Open the test file for reading */
  infile = open_test_input(filename, QIO_UNKNOWN, QIO_SERIAL, myname);
  if(infile == NULL) { printf("Open file failed\n"); exit(0); }

  /* Read the su3 field record */
  printf("%s(%d) reading su3 field\n",myname,this_node); fflush(stdout);
  status = read_su3_field(infile, 4, (suN_matrix **)gauge, myname);
  if(status) { printf("read_su3_field failed %d\n", status); exit(0); }

  /* Close the file */
  QIO_close_read(infile);
  printf("%s(%d): Closed file for reading\n",myname,this_node);  
    
  /* Shut down QMP */
  QMP_finalize_msg_passing();
}
