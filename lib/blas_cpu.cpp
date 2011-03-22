#include <color_spinor_field.h>
#include <blas_quda.h>
#include <face_quda.h>

void copyCpu(cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  a.copy(b);
}

template <typename Float>
void axpby(const Float &a, const Float *x, const Float &b, Float *y, const int N) {
  for (int i=0; i<N; i++) y[i] = a*x[i] + b*y[i];
}

void axpbyCpu(const double &a, const cpuColorSpinorField &x, 
	      const double &b, cpuColorSpinorField &y) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(a, (double*)x.v, b, (double*)y.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby((float)a, (float*)x.v, (float)b, (float*)y.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

void xpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(1.0, (double*)x.v, 1.0, (double*)y.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby(1.0f, (float*)x.v, 1.0f, (float*)y.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

void axpyCpu(const double &a, const cpuColorSpinorField &x, 
	     cpuColorSpinorField &y) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(a, (double*)x.v, 1.0, (double*)y.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby((float)a, (float*)x.v, 1.0f, (float*)y.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

void xpayCpu(const cpuColorSpinorField &x, const double &a, 
	     cpuColorSpinorField &y) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(1.0, (double*)x.v, a, (double*)y.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby(1.0f, (float*)x.v, (float)a, (float*)y.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

void mxpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(-1.0, (double*)x.v, 1.0, (double*)y.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby(-1.0f, (float*)x.v, 1.0f, (float*)y.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

void axCpu(const double &a, cpuColorSpinorField &x) {
  if (x.precision == QUDA_DOUBLE_PRECISION)
    axpby(0.0, (double*)x.v, a, (double*)x.v, x.length);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    axpby(0.0f, (float*)x.v, (float)a, (float*)x.v, x.length);
  else
    errorQuda("Precision type %d not implemented", x.precision);
}

template <typename Float>
void caxpby(const std::complex<Float> &a, const std::complex<Float> *x,
	    const std::complex<Float> &b, std::complex<Float> *y, int N) {

  for (int i=0; i<N; i++) {
    y[i] = a*x[i] + b*y[i];
  }

}

void caxpyCpu(const Complex &a, const cpuColorSpinorField &x,
	      cpuColorSpinorField &y) {

  if ( x.precision == QUDA_DOUBLE_PRECISION)
    caxpby(a, (Complex*)x.v, Complex(1.0), 
	   (Complex*)y.v, x.length/2);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    caxpby((std::complex<float>)a, (std::complex<float>*)x.v, std::complex<float>(1.0), 
	   (std::complex<float>*)y.v, x.length/2);
  else 
    errorQuda("Precision type %d not implemented", x.precision);
}

void caxpbyCpu(const Complex &a, const cpuColorSpinorField &x,
	       const Complex &b, cpuColorSpinorField &y) {

  if ( x.precision == QUDA_DOUBLE_PRECISION)
    caxpby(a, (Complex*)x.v, b, (Complex*)y.v, x.length/2);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    caxpby((std::complex<float>)a, (std::complex<float>*)x.v, (std::complex<float>)b, 
	  (std::complex<float>*)y.v, x.length/2);
  else 
    errorQuda("Precision type %d not implemented", x.precision);
}

template <typename Float>
void caxpbypcz(const std::complex<Float> &a, const std::complex<Float> *x,
	      const std::complex<Float> &b, const std::complex<Float> *y, 
	      const std::complex<Float> &c, std::complex<Float> *z, int N) {

  for (int i=0; i<N; i++) {
    z[i] = a*x[i] + b*y[i] + c*z[i];
  }

}

void cxpaypbzCpu(const cpuColorSpinorField &x, const Complex &a, 
		 const cpuColorSpinorField &y, const Complex &b,
		 cpuColorSpinorField &z) {

  if (x.precision == QUDA_DOUBLE_PRECISION)
    caxpbypcz(Complex(1, 0), (Complex*)x.v, a, (Complex*)y.v, 
	     b, (Complex*)z.v, x.length/2);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    caxpbypcz(std::complex<float>(1, 0), (std::complex<float>*)x.v, (std::complex<float>)a, (std::complex<float>*)y.v, 
	     (std::complex<float>)b, (std::complex<float>*)z.v, x.length/2);
  else 
    errorQuda("Precision type %d not implemented", x.precision);
}

void axpyBzpcxCpu(const double &a, cpuColorSpinorField& x, cpuColorSpinorField& y, 
		  const double &b, const cpuColorSpinorField& z, const double &c) {
  axpyCpu(a, x, y);
  axpbyCpu(b, z, c, x);
}

// performs the operations: {y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]}
void axpyZpbxCpu(const double &a, cpuColorSpinorField &x, cpuColorSpinorField &y, 
		 const cpuColorSpinorField &z, const double &b) {
  axpyCpu(a, x, y);
  xpayCpu(z, b, x);
}

// performs the operation z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
void caxpbypzYmbwCpu(const Complex &a, const cpuColorSpinorField &x, const Complex &b, 
		     cpuColorSpinorField &y, cpuColorSpinorField &z, const cpuColorSpinorField &w) {

  if (x.precision == QUDA_DOUBLE_PRECISION)
    caxpbypcz(a, (Complex*)x.v, b, (Complex*)y.v, 
	      Complex(1, 0), (Complex*)z.v, x.length/2);
  else if (x.precision == QUDA_SINGLE_PRECISION)
    caxpbypcz((std::complex<float>)a, (std::complex<float>*)x.v, 
	      (std::complex<float>)b, (std::complex<float>*)y.v, 
	      (std::complex<float>)(1.0f), (std::complex<float>*)z.v, x.length/2);
  else 
    errorQuda("Precision type %d not implemented", x.precision);

  caxpyCpu(-b, w, y);
}

template <typename Float>
double norm(const Float *a, const int N) {
  double norm2 = 0;
  for (int i=0; i<N; i++) norm2 += a[i]*a[i];
  return norm2;
}

double normCpu(const cpuColorSpinorField &a) {
  double norm2 = 0.0;
  if (a.precision == QUDA_DOUBLE_PRECISION)
    norm2 = norm((double*)a.v, a.length);
  else if (a.precision == QUDA_SINGLE_PRECISION)
    norm2 = norm((float*)a.v, a.length);
  else
    errorQuda("Precision type %d not implemented", a.precision);
  reduceDouble(norm2);
  return norm2;
}

double axpyNormCpu(const double &a, const cpuColorSpinorField &x, 
		   cpuColorSpinorField &y) {
  axpyCpu(a, x, y);
  return normCpu(y);
}

template <typename Float>
double reDotProduct(const Float *a, const Float *b, const int N) {
  double dot = 0;
  for (int i=0; i<N; i++) dot += a[i]*b[i];
  return dot;
}

double reDotProductCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  double dot = 0.0;
  if (a.precision == QUDA_DOUBLE_PRECISION)
    dot = reDotProduct((double*)a.v, (double*)b.v, a.length);
  else if (a.precision == QUDA_SINGLE_PRECISION)
    dot = reDotProduct((float*)a.v, (float*)b.v, a.length);
  else
    errorQuda("Precision type %d not implemented", a.precision);
  reduceDouble(dot);
  return dot;
}

// First performs the operation y[i] = x[i] - y[i]
// Second returns the norm of y
double xmyNormCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y) {
  xpayCpu(x, -1, y);
  return normCpu(y);
}

template <typename Float>
Complex cDotProduct(const std::complex<Float> *a, const std::complex<Float> *b, const int N) {
  Complex dot = 0;
  for (int i=0; i<N; i++) dot += conj(a[i])*b[i];
  return dot;
}

Complex cDotProductCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  Complex dot = 0.0;
  if (a.precision == QUDA_DOUBLE_PRECISION)
    dot = cDotProduct((Complex*)a.v, (Complex*)b.v, a.length/2);
  else if (a.precision == QUDA_SINGLE_PRECISION)
    dot = cDotProduct((std::complex<float>*)a.v, (std::complex<float>*)b.v, a.length/2);
  else
    errorQuda("Precision type %d not implemented", a.precision);
  reduceDoubleArray((double*)&dot, 2);
  return dot;
}

// First performs the operation y = x + a*y
// Second returns complex dot product (z,y)
Complex xpaycDotzyCpu(const cpuColorSpinorField &x, const double &a, 
		      cpuColorSpinorField &y, const cpuColorSpinorField &z) {
  xpayCpu(x, a, y);
  return cDotProductCpu(z,y);
}

double3 cDotProductNormACpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  Complex dot = cDotProductCpu(a, b);
  double norm = normCpu(a);
  return make_double3(real(dot), imag(dot), norm);
}

double3 cDotProductNormBCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  Complex dot = cDotProductCpu(a, b);
  double norm = normCpu(b);
  return make_double3(real(dot), imag(dot), norm);
}

// This convoluted kernel does the following: z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
double3 caxpbypzYmbwcDotProductUYNormYCpu(const Complex &a, const cpuColorSpinorField &x, 
					  const Complex &b, cpuColorSpinorField &y, 
					  cpuColorSpinorField &z, const cpuColorSpinorField &w, 
					  const cpuColorSpinorField &u) {

  caxpbypzYmbwCpu(a, x, b, y, z, w);
  return cDotProductNormBCpu(u, y);
}
