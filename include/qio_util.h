#pragma once

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
} qio_complex;

typedef struct { qio_complex e[NCLR][NCLR]; } suN_matrix;

// for matrix fields this order implies [color][color][complex]
// for vector fields this order implies [spin][color][complex]
// templatized version to allow for precision conversion
template <typename oFloat, typename iFloat, int len>
void vput(char *s1, size_t index, int count, void *s2)
{
  oFloat **field = (oFloat **)s2;
  iFloat *src = (iFloat *)s1;

  //For the site specified by "index", move an array of "count" data
  //from the read buffer to an array of fields

  for (int i=0;i<count;i++) {
    oFloat *dest = field[i] + len*index;
    for (int j=0; j<len; j++) dest[j] = src[i*len+j];
  }
}

// for vector fields this order implies [spin][color][complex]
// templatized version of vget_M to allow for precision conversion
template <typename oFloat, typename iFloat, int len>
void vget(char *s1, size_t index, int count, void *s2)
{
  iFloat **field = (iFloat **)s2;
  oFloat *dest = (oFloat *)s1;

/* For the site specified by "index", move an array of "count" data
   from the array of fields to the write buffer */
  for (int i=0; i<count; i++, dest+=len) {
    iFloat *src = field[i] + len*index;
    for (int j=0; j<len; j++) dest[j] = src[j];
  }
}

int vcreate_M(suN_matrix *field[] , int count);
void vdestroy_M(suN_matrix *field[], int count);
float vcompare_M(suN_matrix *fielda[], suN_matrix *fieldb[], int count);

#endif // HAVE_QIO
