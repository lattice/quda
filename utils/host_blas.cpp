#include <host_utils.h>
#include <stdio.h>
#include <comm_quda.h>

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
  comm_allreduce(&sum);
  return sum;
}

double norm_2(void *v, int len, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) return norm2((double*)v, len);
  else return norm2((float*)v, len);
}

// performs the operation y[i] = x[i] + a*y[i]
template <typename Float>
static inline void xpay(Float *x, Float a, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] = x[i] + a*y[i];
}

void xpay(void *x, double a, void *y, int length, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) xpay((double*)x, a, (double*)y, length);
  else xpay((float*)x, (float)a, (float*)y, length);
}

void cxpay(void *x, double _Complex a, void *y, int length, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    xpay((double _Complex *)x, (double _Complex)a, (double _Complex *)y, length / 2);
  } else {
    xpay((float _Complex *)x, (float _Complex)a, (float _Complex *)y, length / 2);
  }
}

// CPU-style BLAS routines for staggered
void cpu_axy(QudaPrecision prec, double a, void *x, void *y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double *dst = (double *)y;
    double *src = (double *)x;
    for (int i = 0; i < size; i++) { dst[i] = a * src[i]; }
  } else { // QUDA_SINGLE_PRECISION
    float *dst = (float *)y;
    float *src = (float *)x;
    for (int i = 0; i < size; i++) { dst[i] = a * src[i]; }
  }
}

void cpu_xpy(QudaPrecision prec, void *x, void *y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double *dst = (double *)y;
    double *src = (double *)x;
    for (int i = 0; i < size; i++) { dst[i] += src[i]; }
  } else { // QUDA_SINGLE_PRECISION
    float *dst = (float *)y;
    float *src = (float *)x;
    for (int i = 0; i < size; i++) { dst[i] += src[i]; }
  }
}
