#include <blas_reference.h>
#include <stdio.h>
#ifdef MPI_COMMS
#include "mpicomm.h"
#elif QMP_COMMS
#include <qmp.h>
#endif

template <typename Float>
inline void aXpY(Float a, Float *x, Float *y, int len)
{
  for(int i=0; i < len; i++){ y[i] += a*x[i]; }
}

void axpy(double a, void *x, void *y, int len, QudaPrecision precision) { 
  if( precision == QUDA_DOUBLE_PRECISION ) aXpY(a, (double *)x, (double *)y, len);
  else aXpY((float)a, (float *)x, (float *)y, len);
}

// performs the operation x[i] *= a
template <typename Float>
inline void aX(Float a, Float *x, int len) {
  for (int i=0; i<len; i++) x[i] *= a;
}

void ax(double a, void *x, int len, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) aX(a, (double*)x, len);
  else aX((float)a, (float*)x, len);
}

// performs the operation y[i] -= x[i] (minus x plus y)
template <typename Float>
inline void mXpY(Float *x, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] -= x[i];
}

void mxpy(void* x, void* y, int len, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) mXpY((double*)x, (double*)y, len);
  else mXpY((float*)x, (float*)y, len);
}


// returns the square of the L2 norm of the vector
template <typename Float>
inline double norm2(Float *v, int len) {
  double sum=0.0;
  for (int i=0; i<len; i++) sum += v[i]*v[i];
#ifdef MPI_COMMS
  comm_allreduce(&sum);
#elif QMP_COMMS
  QMP_sum_double(&sum);
#endif
  return sum;
}

double norm_2(void *v, int len, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) return norm2((double*)v, len);
  else return norm2((float*)v, len);
}


/*


// sets all elements of the destination vector to zero
void zero(float* a, int cnt) {
    for (int i = 0; i < cnt; i++)
        a[i] = 0;
}

// copy one spinor to the other
void copy(float* a, float *b, int len) {
  for (int i = 0; i < len; i++) a[i] = b[i];
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpby(float a, float *x, float b, float *y, int len) {
    for (int i=0; i<len; i++) y[i] = a*x[i] + b*y[i];
}

// performs the operation y[i] = a*x[i] + y[i]
void axpy(float a, float *x, float *y, int len) {
    for (int i=0; i<len; i++) y[i] += a*x[i];
}


// returns the real part of the dot product of 2 complex valued vectors
float reDotProduct(float *v1, float *v2, int len) {

  float dot=0.0;
  for (int i=0; i<len; i++) {
    dot += v1[i]*v2[i];
  }

  return dot;
}

// returns the imaginary part of the dot product of 2 complex valued vectors
float imDotProduct(float *v1, float *v2, int len) {

  float dot=0.0;
  for (int i=0; i<len; i+=2) {
    dot += v1[i]*v2[i+1] - v1[i+1]*v2[i];
  }

  return dot;
}

// returns the square of the L2 norm of the vector
double normD(float *v, int len) {

  double sum=0.0;
  for (int i=0; i<len; i++) {
    sum += v[i]*v[i];
  }

  return sum;
}

// returns the real part of the dot product of 2 complex valued vectors
double reDotProductD(float *v1, float *v2, int len) {

  double dot=0.0;
  for (int i=0; i<len; i++) {
    dot += v1[i]*v2[i];
  }

  return dot;
}

// returns the imaginary part of the dot product of 2 complex valued vectors
double imDotProductD(float *v1, float *v2, int len) {

  double dot=0.0;
  for (int i=0; i<len; i+=2) {
    dot += v1[i]*v2[i+1] - v1[i+1]*v2[i];
  }

  return dot;
}
*/
