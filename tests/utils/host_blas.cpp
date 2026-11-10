#include <stdio.h>
#include <complex.h>
#include <comm_quda.h>

#include "host_utils.h"

template <typename real> inline void aXpY(real a, const real *x, real *y, int len)
{
  for (int i = 0; i < len; i++) { y[i] += a * x[i]; }
}

void axpy(double a, const void *x, void *y, int len, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    aXpY(a, (double *)x, (double *)y, len);
  else
    aXpY((float)a, (float *)x, (float *)y, len);
}

void caxpy(double _Complex a, void *x, void *y, int len, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    aXpY((double _Complex)a, (double _Complex *)x, (double _Complex *)y, len / 2);
  else
    aXpY((float _Complex)a, (float _Complex *)x, (float _Complex *)y, len / 2);
}

// performs the operation x[i] *= a
template <typename real> inline void aX(real a, real *x, int len)
{
  for (int i = 0; i < len; i++) x[i] *= a;
}

void ax(double a, void *x, int len, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    aX(a, (double *)x, len);
  else
    aX((float)a, (float *)x, len);
}

void cax(double _Complex a, void *x, int len, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    aX((double _Complex)a, (double _Complex *)x, len / 2);
  else {
    aX((float _Complex)a, (float _Complex *)x, len / 2);
  }
}

// performs the operation y[i] -= x[i] (minus x plus y)
template <typename real> inline void mXpY(real *x, real *y, int len)
{
  for (int i = 0; i < len; i++) y[i] -= x[i];
}

void mxpy(void *x, void *y, int len, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    mXpY((double *)x, (double *)y, len);
  else
    mXpY((float *)x, (float *)y, len);
}

// returns the square of the L2 norm of the vector
template <typename real> inline double norm2(real *v, int len, bool global)
{
  double sum = 0.0;
  for (int i = 0; i < len; i++) sum += v[i] * v[i];
  if (global) quda::comm_allreduce_sum(sum);
  return sum;
}

double norm_2(void *v, int len, QudaPrecision precision, bool global)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    return norm2((double *)v, len, global);
  else
    return norm2((float *)v, len, global);
}

// performs the operation y[i] = x[i] + a*y[i]
template <typename real> static inline void xpay(const real *x, real a, real *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = x[i] + a * y[i];
}

void xpay(const void *x, double a, void *y, int length, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    xpay((const double *)x, a, (double *)y, length);
  else
    xpay((const float *)x, (float)a, (float *)y, length);
}

void cxpay(const void *x, std::complex<double> a, void *y, int length, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    xpay(reinterpret_cast<const std::complex<double> *>(x), a, reinterpret_cast<std::complex<double> *>(y), length / 2);
  } else {
    xpay(reinterpret_cast<const std::complex<float> *>(x), std::complex<float>(a),
         reinterpret_cast<std::complex<float> *>(y), length / 2);
  }
}

// CPU-style BLAS routines for staggered
void cpu_axy(QudaPrecision prec, double a, const void *x, void *y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double *dst = (double *)y;
    const double *src = (const double *)x;
    for (int i = 0; i < size; i++) { dst[i] = a * src[i]; }
  } else { // QUDA_SINGLE_PRECISION
    float *dst = (float *)y;
    const float *src = (const float *)x;
    for (int i = 0; i < size; i++) { dst[i] = a * src[i]; }
  }
}

void cpu_xpy(QudaPrecision prec, const void *x, void *y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double *dst = (double *)y;
    const double *src = (const double *)x;
    for (int i = 0; i < size; i++) { dst[i] += src[i]; }
  } else { // QUDA_SINGLE_PRECISION
    float *dst = (float *)y;
    const float *src = (const float *)x;
    for (int i = 0; i < size; i++) { dst[i] += src[i]; }
  }
}
