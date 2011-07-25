#ifndef QIO_TEST_H
#define QIO_TEST_H

#include <qmp.h>
#include <qio.h>
#define mynode QMP_get_node_number
#ifdef MAIN
#define EXTERN
#else
#define EXTERN extern
#endif

EXTERN QIO_Layout layout;
EXTERN int lattice_dim;
EXTERN int lattice_size[4];

/* layout_hyper */
int setup_layout(int len[], int nd, int numnodes);
int node_number(const int x[]);
int node_index(const int x[]);
void get_coords(int x[], int node, int index);
int num_sites(int node);
EXTERN int this_node;

#define NCLR 3

typedef struct
{
  float re;
  float im;
} complex;

typedef struct { complex e[NCLR][NCLR]; } suN_matrix;

/* get and put */
void vput_R(char *buf, size_t index, int count, void *qfin);
void vget_R(char *buf, size_t index, int count, void *qfin);
void vput_M(char *buf, size_t index, int count, void *qfin);
void vget_M(char *buf, size_t index, int count, void *qfin);
void vput_r(char *buf, size_t index, int count, void *qfin);
void vget_r(char *buf, size_t index, int count, void *qfin);

int vcreate_R(float *field_out[],int count);
int vcreate_M(suN_matrix *field[] , int count);
void vdestroy_R(float *field[], int count);
void vdestroy_M(suN_matrix *field[], int count);
void vset_R(float *field[],int count);
void vset_M(suN_matrix *field[], int count);
float vcompare_R(float *fielda[], float *fieldb[], int count);
float vcompare_M(suN_matrix *fielda[], suN_matrix *fieldb[], int count);
float vcompare_r(float arraya[], float arrayb[], int count);

int qio_test(int output_volfmt, int output_serpar, int ildgstyle, 
	     int input_volfmt, int input_serpar, int argc, char *argv[]);

int qio_host_test(QIO_Filesystem *fs, int argc, char *argv[]);

#endif /* QIO_TEST_H */

