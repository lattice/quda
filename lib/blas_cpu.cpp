#include <color_spinor_field.h>
#include <blas_quda.h>
#include <face_quda.h>

template <typename Float>
void axpby(const Float &a, const Float *x, const Float &b, Float *y, const int N) {
  for (int i=0; i<N; i++) y[i] = a*x[i] + b*y[i];
}

void axpbyCpu(const double &a, const cpuColorSpinorField &x, 
	      const double &b, cpuColorSpinorField &y) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(a, (double*)x.V(), b, (double*)y.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby((float)a, (float*)x.V(), (float)b, (float*)y.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

void xpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(1.0, (double*)x.V(), 1.0, (double*)y.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby(1.0f, (float*)x.V(), 1.0f, (float*)y.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

void axpyCpu(const double &a, const cpuColorSpinorField &x, 
	     cpuColorSpinorField &y) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(a, (double*)x.V(), 1.0, (double*)y.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby((float)a, (float*)x.V(), 1.0f, (float*)y.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

void xpayCpu(const cpuColorSpinorField &x, const double &a, 
	     cpuColorSpinorField &y) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(1.0, (double*)x.V(), a, (double*)y.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby(1.0f, (float*)x.V(), (float)a, (float*)y.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

void mxpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(-1.0, (double*)x.V(), 1.0, (double*)y.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby(-1.0f, (float*)x.V(), 1.0f, (float*)y.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

void axCpu(const double &a, cpuColorSpinorField &x) {
  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    axpby(0.0, (double*)x.V(), a, (double*)x.V(), x.Length());
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    axpby(0.0f, (float*)x.V(), (float)a, (float*)x.V(), x.Length());
  else
    errorQuda("Precision type %d not implemented", x.Precision());
}

template <typename Float>
void caxpby(const std::complex<Float> &a, const std::complex<Float> *x,
	    const std::complex<Float> &b, std::complex<Float> *y, int N) {

  for (int i=0; i<N; i++) {
    y[i] = a*x[i] + b*y[i];
  }

}

void caxpyCpu(const quda::Complex &a, const cpuColorSpinorField &x,
	      cpuColorSpinorField &y) {

  if ( x.Precision() == QUDA_DOUBLE_PRECISION)
    caxpby(a, (quda::Complex*)x.V(), quda::Complex(1.0), 
	   (quda::Complex*)y.V(), x.Length()/2);
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    caxpby((std::complex<float>)a, (std::complex<float>*)x.V(), std::complex<float>(1.0), 
	   (std::complex<float>*)y.V(), x.Length()/2);
  else 
    errorQuda("Precision type %d not implemented", x.Precision());
}

void caxpbyCpu(const quda::Complex &a, const cpuColorSpinorField &x,
	       const quda::Complex &b, cpuColorSpinorField &y) {

  if ( x.Precision() == QUDA_DOUBLE_PRECISION)
    caxpby(a, (quda::Complex*)x.V(), b, (quda::Complex*)y.V(), x.Length()/2);
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    caxpby((std::complex<float>)a, (std::complex<float>*)x.V(), (std::complex<float>)b, 
	  (std::complex<float>*)y.V(), x.Length()/2);
  else 
    errorQuda("Precision type %d not implemented", x.Precision());
}

template <typename Float>
void caxpbypcz(const std::complex<Float> &a, const std::complex<Float> *x,
	      const std::complex<Float> &b, const std::complex<Float> *y, 
	      const std::complex<Float> &c, std::complex<Float> *z, int N) {

  for (int i=0; i<N; i++) {
    z[i] = a*x[i] + b*y[i] + c*z[i];
  }

}

void cxpaypbzCpu(const cpuColorSpinorField &x, const quda::Complex &a, 
		 const cpuColorSpinorField &y, const quda::Complex &b,
		 cpuColorSpinorField &z) {

  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    caxpbypcz(quda::Complex(1, 0), (quda::Complex*)x.V(), a, (quda::Complex*)y.V(), 
	     b, (quda::Complex*)z.V(), x.Length()/2);
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    caxpbypcz(std::complex<float>(1, 0), (std::complex<float>*)x.V(), (std::complex<float>)a, (std::complex<float>*)y.V(), 
	     (std::complex<float>)b, (std::complex<float>*)z.V(), x.Length()/2);
  else 
    errorQuda("Precision type %d not implemented", x.Precision());
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
void caxpbypzYmbwCpu(const quda::Complex &a, const cpuColorSpinorField &x, const quda::Complex &b, 
		     cpuColorSpinorField &y, cpuColorSpinorField &z, const cpuColorSpinorField &w) {

  if (x.Precision() == QUDA_DOUBLE_PRECISION)
    caxpbypcz(a, (quda::Complex*)x.V(), b, (quda::Complex*)y.V(), 
	      quda::Complex(1, 0), (quda::Complex*)z.V(), x.Length()/2);
  else if (x.Precision() == QUDA_SINGLE_PRECISION)
    caxpbypcz((std::complex<float>)a, (std::complex<float>*)x.V(), 
	      (std::complex<float>)b, (std::complex<float>*)y.V(), 
	      (std::complex<float>)(1.0f), (std::complex<float>*)z.V(), x.Length()/2);
  else 
    errorQuda("Precision type %d not implemented", x.Precision());

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
  if (a.Precision() == QUDA_DOUBLE_PRECISION)
    norm2 = norm((double*)a.V(), a.Length());
  else if (a.Precision() == QUDA_SINGLE_PRECISION)
    norm2 = norm((float*)a.V(), a.Length());
  else
    errorQuda("Precision type %d not implemented", a.Precision());
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
  if (a.Precision() == QUDA_DOUBLE_PRECISION)
    dot = reDotProduct((double*)a.V(), (double*)b.V(), a.Length());
  else if (a.Precision() == QUDA_SINGLE_PRECISION)
    dot = reDotProduct((float*)a.V(), (float*)b.V(), a.Length());
  else
    errorQuda("Precision type %d not implemented", a.Precision());
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
quda::Complex cDotProduct(const std::complex<Float> *a, const std::complex<Float> *b, const int N) {
  quda::Complex dot = 0;
  for (int i=0; i<N; i++) dot += conj(a[i])*b[i];
  return dot;
}

quda::Complex cDotProductCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  quda::Complex dot = 0.0;
  if (a.Precision() == QUDA_DOUBLE_PRECISION)
    dot = cDotProduct((quda::Complex*)a.V(), (quda::Complex*)b.V(), a.Length()/2);
  else if (a.Precision() == QUDA_SINGLE_PRECISION)
    dot = cDotProduct((std::complex<float>*)a.V(), (std::complex<float>*)b.V(), a.Length()/2);
  else
    errorQuda("Precision type %d not implemented", a.Precision());
  reduceDoubleArray((double*)&dot, 2);
  return dot;
}

// First performs the operation y = x + a*y
// Second returns complex dot product (z,y)
quda::Complex xpaycDotzyCpu(const cpuColorSpinorField &x, const double &a, 
		      cpuColorSpinorField &y, const cpuColorSpinorField &z) {
  xpayCpu(x, a, y);
  return cDotProductCpu(z,y);
}

double3 cDotProductNormACpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  quda::Complex dot = cDotProductCpu(a, b);
  double norm = normCpu(a);
  return make_double3(real(dot), imag(dot), norm);
}

double3 cDotProductNormBCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b) {
  quda::Complex dot = cDotProductCpu(a, b);
  double norm = normCpu(b);
  return make_double3(real(dot), imag(dot), norm);
}

// This convoluted kernel does the following: z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
double3 caxpbypzYmbwcDotProductUYNormYCpu(const quda::Complex &a, const cpuColorSpinorField &x, 
					  const quda::Complex &b, cpuColorSpinorField &y, 
					  cpuColorSpinorField &z, const cpuColorSpinorField &w, 
					  const cpuColorSpinorField &u) {

  caxpbypzYmbwCpu(a, x, b, y, z, w);
  return cDotProductNormBCpu(u, y);
}

void cabxpyAxCpu(const double &a, const quda::Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y) {
  axCpu(a, x);
  caxpyCpu(b, x, y);
}

double caxpyNormCpu(const quda::Complex &a, cpuColorSpinorField &x, 
		    cpuColorSpinorField &y) {
  caxpyCpu(a, x, y);
  return norm2(y);
}

double caxpyXmazNormXCpu(const quda::Complex &a, cpuColorSpinorField &x, 
			 cpuColorSpinorField &y, cpuColorSpinorField &z) {
  caxpyCpu(a, x, y);
  caxpyCpu(-a, z, x);
  return norm2(x);
}

void caxpyXmazCpu(const quda::Complex &a, cpuColorSpinorField &x, 
		    cpuColorSpinorField &y, cpuColorSpinorField &z) {
  caxpyCpu(a, x, y);
  caxpyCpu(-a, z, x);
}

double cabxpyAxNormCpu(const double &a, const quda::Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y) {
  axCpu(a, x);
  caxpyCpu(b, x, y);
  return norm2(y);
}

void caxpbypzCpu(const quda::Complex &a, cpuColorSpinorField &x, const quda::Complex &b, cpuColorSpinorField &y, 
		 cpuColorSpinorField &z) {
  caxpyCpu(a, x, z);
  caxpyCpu(b, y, z);
}

void caxpbypczpwCpu(const quda::Complex &a, cpuColorSpinorField &x, const quda::Complex &b, cpuColorSpinorField &y, 
		    const quda::Complex &c, cpuColorSpinorField &z, cpuColorSpinorField &w) {
  caxpyCpu(a, x, w);
  caxpyCpu(b, y, w);
  caxpyCpu(c, z, w);

}

quda::Complex caxpyDotzyCpu(const quda::Complex &a, cpuColorSpinorField &x, cpuColorSpinorField &y,
		      cpuColorSpinorField &z) {
  caxpyCpu(a, x, y);
  return cDotProductCpu(z, y);
}
