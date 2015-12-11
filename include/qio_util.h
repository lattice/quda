#ifndef QIO_TEST_H
#define QIO_TEST_H

#ifdef HAVE_QIO

#include <qmp.h>
#include <qio.h>
#define mynode QMP_get_node_number

extern QIO_Layout layout;
extern int lattice_dim;
extern int lattice_size[4];

#include <layout_hyper.h>

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

// for vector fields this order implies [spin][color][complex]
// templatized version of vput_M to allow for precision conversion
template <typename oFloat, typename iFloat, int len>
void vputM(char *s1, size_t index, int count, void *s2)
{
  oFloat **field = (oFloat **)s2;
  iFloat *src = (iFloat *)s1;

  //For the site specified by "index", move an array of "count" data
  //from the read buffer to an array of fields

  for (int i=0;i<count;i++)
    {
      oFloat *dest = field[i] + len*index;
      for (int j=0; j<len; j++) dest[j] = src[i*len+j];
    }
}

// for vector fields this order implies [spin][color][complex]
// templatized version of vget_M to allow for precision conversion
template <typename oFloat, typename iFloat, int len>
void vgetM(char *s1, size_t index, int count, void *s2)
{
  iFloat **field = (iFloat **)s2;
  oFloat *dest = (oFloat *)s1;

/* For the site specified by "index", move an array of "count" data
   from the array of fields to the write buffer */
  for (int i=0; i<count; i++, dest+=len)
    {
      iFloat *src = field[i] + len*index;
      for (int j=0; j<len; j++) dest[j] = src[j];
    }
}


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

#endif // HAVE_QIO


#endif /* QIO_TEST_H */
