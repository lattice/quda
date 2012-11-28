#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset

#include <float_vector.h>

#include <tune_quda.h>
#include <typeinfo>

#include <quda_internal.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <face_quda.h> // this is where the MPI / QMP depdendent code is

#define checkSpinor(a, b)						\
  {									\
    if (a.Precision() != b.Precision())					\
      errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision()); \
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %d %d", a.Length(), b.Length());	\
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

namespace quda {

#include <texture.h>

  unsigned long long blas_flops;
  unsigned long long blas_bytes;

  void zeroCuda(cudaColorSpinorField &a) { a.zero(); }

  // blasTuning = 1 turns off error checking
  static QudaTune blasTuning = QUDA_TUNE_NO;
  static QudaVerbosity verbosity = QUDA_SILENT;
  static cudaStream_t *blasStream;

  static struct {
    int x[QUDA_MAX_DIM];
    int stride;
  } blasConstants;

  void initReduce();
  void endReduce();

  void initBlas()
  { 
    blasStream = &streams[Nstream-1];
    initReduce();
  }
  
  void endBlas(void)
  {
    endReduce();
  }
    
  void setBlasTuning(QudaTune tune, QudaVerbosity verbose)
  {
    blasTuning = tune;
    verbosity = verbose;
  }

  QudaTune getBlasTuning() { return blasTuning; }
  QudaVerbosity getBlasVerbosity() { return verbosity; }
  cudaStream_t* getBlasStream() { return blasStream; }

#include <blas_core.h>

  /**
     Functor to perform the operation y = a*x + b*y
  */
  template <typename Float2, typename FloatN>
  struct axpby {
    const Float2 a;
    const Float2 b;
    axpby(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { y = a.x*x + b.x*y; }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 3; } //! flops per element
  };

  void axpbyCuda(const double &a, cudaColorSpinorField &x, const double &b, cudaColorSpinorField &y) {
    blasCuda<axpby,0,1,0,0>(make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0),
			    x, y, x, x);
  }

  /**
     Functor to perform the operation y += x
  */
  template <typename Float2, typename FloatN>
  struct xpy {
    xpy(const Float2 &a, const Float2 &b, const Float2 &c) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { y += x ; }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 1; } //! flops per element
  };

  void xpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    blasCuda<xpy,0,1,0,0>(make_double2(1.0, 0.0), make_double2(1.0, 0.0), make_double2(0.0, 0.0), 
			  x, y, x, x);
  }

  /**
     Functor to perform the operation y += a*x
  */
  template <typename Float2, typename FloatN>
  struct axpy {
    const Float2 a;
    axpy(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { y = a.x*x + y; }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 2; } //! flops per element
  };

  void axpyCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y) {
    blasCuda<axpy,0,1,0,0>(make_double2(a, 0.0), make_double2(1.0, 0.0), make_double2(0.0, 0.0), 
			   x, y, x, x);
  }

  /**
     Functor to perform the operation y = x + a*y
  */
  template <typename Float2, typename FloatN>
  struct xpay {
    const Float2 a;
    xpay(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { y = x + a.x*y; }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 2; } //! flops per element
  };

  void xpayCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y) {
    blasCuda<xpay,0,1,0,0>(make_double2(a,0.0), make_double2(0.0, 0.0), make_double2(0.0, 0.0),
			   x, y, x, x);
  }

  /**
     Functor to perform the operation y -= x;
  */
  template <typename Float2, typename FloatN>
  struct mxpy {
    mxpy(const Float2 &a, const Float2 &b, const Float2 &c) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { y -= x; }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 1; } //! flops per element
  };

  void mxpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    blasCuda<mxpy,0,1,0,0>(make_double2(1.0, 0.0), make_double2(1.0, 0.0), 
			   make_double2(0.0, 0.0), x, y, x, x);
  }

  /**
     Functor to perform the operation x *= a
  */
  template <typename Float2, typename FloatN>
  struct ax {
    const Float2 a;
    ax(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
    __device__ void operator()(FloatN &x, const FloatN &y, const FloatN &z, const FloatN &w) { x *= a.x; }
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 1; } //! flops per element
  };

  void axCuda(const double &a, cudaColorSpinorField &x) {
    blasCuda<ax,1,0,0,0>(make_double2(a, 0.0), make_double2(0.0, 0.0), 
			 make_double2(0.0, 0.0), x, x, x, x);
  }

  /**
     Functor to perform the operation y += a * x  (complex-valued)
  */

  __device__ void caxpy_(const float2 &a, const float4 &x, float4 &y) {
    y.x += a.x*x.x; y.x -= a.y*x.y;
    y.y += a.y*x.x; y.y += a.x*x.y;
    y.z += a.x*x.z; y.z -= a.y*x.w;
    y.w += a.y*x.z; y.w += a.x*x.w;
  }

  __device__ void caxpy_(const float2 &a, const float2 &x, float2 &y) {
    y.x += a.x*x.x; y.x -= a.y*x.y;
    y.y += a.y*x.x; y.y += a.x*x.y;
  }

  __device__ void caxpy_(const double2 &a, const double2 &x, double2 &y) {
    y.x += a.x*x.x; y.x -= a.y*x.y;
    y.y += a.y*x.x; y.y += a.x*x.y;
  }

  template <typename Float2, typename FloatN>
  struct caxpy {
    const Float2 a;
    caxpy(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { caxpy_(a, x, y); }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 4; } //! flops per element
  };

  void caxpyCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y) {
    blasCuda<caxpy,0,1,0,0>(make_double2(real(a),imag(a)), make_double2(0.0, 0.0), 
			    make_double2(0.0, 0.0), x, y, x, x);
  }

  /**
     Functor to perform the operation y = a*x + b*y  (complex-valued)
  */

  __device__ void caxpby_(const float2 &a, const float4 &x, const float2 &b, float4 &y)					
  { float4 yy;								
    yy.x = a.x*x.x; yy.x -= a.y*x.y; yy.x += b.x*y.x; yy.x -= b.y*y.y;	
    yy.y = a.y*x.x; yy.y += a.x*x.y; yy.y += b.y*y.x; yy.y += b.x*y.y;	
    yy.z = a.x*x.z; yy.z -= a.y*x.w; yy.z += b.x*y.z; yy.z -= b.y*y.w;	
    yy.w = a.y*x.z; yy.w += a.x*x.w; yy.w += b.y*y.z; yy.w += b.x*y.w;	
    y = yy; }

  __device__ void caxpby_(const float2 &a, const float2 &x, const float2 &b, float2 &y)
  { float2 yy;								
    yy.x = a.x*x.x; yy.x -= a.y*x.y; yy.x += b.x*y.x; yy.x -= b.y*y.y;	
    yy.y = a.y*x.x; yy.y += a.x*x.y; yy.y += b.y*y.x; yy.y += b.x*y.y;	
    y = yy; }

  __device__ void caxpby_(const double2 &a, const double2 &x, const double2 &b, double2 &y)				 
  { double2 yy;								
    yy.x = a.x*x.x; yy.x -= a.y*x.y; yy.x += b.x*y.x; yy.x -= b.y*y.y;	
    yy.y = a.y*x.x; yy.y += a.x*x.y; yy.y += b.y*y.x; yy.y += b.x*y.y;	
    y = yy; }

  template <typename Float2, typename FloatN>
  struct caxpby {
    const Float2 a;
    const Float2 b;
    caxpby(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) { caxpby_(a, x, b, y); }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 7; } //! flops per element
  };

  void caxpbyCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, cudaColorSpinorField &y) {
    blasCuda<caxpby,0,1,0,0>(make_double2(a.real(),a.imag()), make_double2(b.real(), b.imag()), 
			     make_double2(0.0, 0.0), x, y, x, x);
  }

  /**
     Functor to performs the operation z[i] = x[i] + a*y[i] + b*z[i]
  */

  __device__ void cxpaypbz_(const float4 &x, const float2 &a, const float4 &y, const float2 &b, float4 &z) {
    float4 zz;
    zz.x = x.x + a.x*y.x; zz.x -= a.y*y.y; zz.x += b.x*z.x; zz.x -= b.y*z.y;
    zz.y = x.y + a.y*y.x; zz.y += a.x*y.y; zz.y += b.y*z.x; zz.y += b.x*z.y;
    zz.z = x.z + a.x*y.z; zz.z -= a.y*y.w; zz.z += b.x*z.z; zz.z -= b.y*z.w;
    zz.w = x.w + a.y*y.z; zz.w += a.x*y.w; zz.w += b.y*z.z; zz.w += b.x*z.w;
    z = zz;
  }

  __device__ void cxpaypbz_(const float2 &x, const float2 &a, const float2 &y, const float2 &b, float2 &z) {
    float2 zz;
    zz.x = x.x + a.x*y.x; zz.x -= a.y*y.y; zz.x += b.x*z.x; zz.x -= b.y*z.y;
    zz.y = x.y + a.y*y.x; zz.y += a.x*y.y; zz.y += b.y*z.x; zz.y += b.x*z.y;
    z = zz;
  }

  __device__ void cxpaypbz_(const double2 &x, const double2 &a, const double2 &y, const double2 &b, double2 &z) {
    double2 zz;
    zz.x = x.x + a.x*y.x; zz.x -= a.y*y.y; zz.x += b.x*z.x; zz.x -= b.y*z.y;
    zz.y = x.y + a.y*y.x; zz.y += a.x*y.y; zz.y += b.y*z.x; zz.y += b.x*z.y;
    z = zz;
  }

  template <typename Float2, typename FloatN>
  struct cxpaypbz {
    const Float2 a;
    const Float2 b;
    cxpaypbz(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(const FloatN &x, const FloatN &y, FloatN &z, FloatN &w) 
    { cxpaypbz_(x, a, y, b, z); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 8; } //! flops per element
  };

  void cxpaypbzCuda(cudaColorSpinorField &x, const Complex &a, cudaColorSpinorField &y, 
		    const Complex &b, cudaColorSpinorField &z) {
    blasCuda<cxpaypbz,0,0,1,0>(make_double2(a.real(),a.imag()), make_double2(b.real(), b.imag()), 
			       make_double2(0.0, 0.0), x, y, z, z);
  }

  /**
     Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
  */
  template <typename Float2, typename FloatN>
  struct axpyBzpcx {
    const Float2 a;
    const Float2 b;
    const Float2 c;
    axpyBzpcx(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b), c(c) { ; }
    __device__ void operator()(FloatN &x, FloatN &y, const FloatN &z, const FloatN &w)
    { y += a.x*x; x = b.x*z + c.x*x; }
    static int streams() { return 5; } //! total number of input and output streams
    static int flops() { return 10; } //! flops per element
  };

  void axpyBzpcxCuda(const double &a, cudaColorSpinorField& x, cudaColorSpinorField& y, const double &b, 
		     cudaColorSpinorField& z, const double &c) {
    blasCuda<axpyBzpcx,1,1,0,0>(make_double2(a,0.0), make_double2(b,0.0), make_double2(c,0.0), 
				x, y, z, x);
  }

  /**
     Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]
  */
  template <typename Float2, typename FloatN>
  struct axpyZpbx {
    const Float2 a;
    const Float2 b;
    axpyZpbx(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(FloatN &x, FloatN &y, const FloatN &z, const FloatN &w)
    { y += a.x*x; x = z + b.x*x; }
    static int streams() { return 5; } //! total number of input and output streams
    static int flops() { return 8; } //! flops per element
  };

  void axpyZpbxCuda(const double &a, cudaColorSpinorField& x, cudaColorSpinorField& y,
		    cudaColorSpinorField& z, const double &b) {
    // swap arguments around 
    blasCuda<axpyZpbx,1,1,0,0>(make_double2(a,0.0), make_double2(b,0.0), make_double2(0.0,0.0),
			       x, y, z, x);
  }

  /**
     Functor performing the operations z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
  */
  template <typename Float2, typename FloatN>
  struct caxpbypzYmbw {
    const Float2 a;
    const Float2 b;
    caxpbypzYmbw(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(const FloatN &x, FloatN &y, FloatN &z, const FloatN &w)
    { caxpy_(a, x, z); caxpy_(b, y, z); caxpy_(-b, w, y); }

    static int streams() { return 6; } //! total number of input and output streams
    static int flops() { return 12; } //! flops per element
  };

  void caxpbypzYmbwCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, 
			cudaColorSpinorField &y, cudaColorSpinorField &z, cudaColorSpinorField &w) {
    blasCuda<caxpbypzYmbw,0,1,1,0>(make_double2(a.real(),a.imag()), make_double2(b.real(), b.imag()), 
				   make_double2(0.0,0.0), x, y, z, w);
  }

  /**
     Functor performing the operation y[i] += a*b*x[i], x[i] *= a
  */
  template <typename Float2, typename FloatN>
  struct cabxpyAx {
    const Float2 a;
    const Float2 b;
    cabxpyAx(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) 
    { x *= a.x; caxpy_(b, x, y); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 5; } //! flops per element
  };

  void cabxpyAxCuda(const double &a, const Complex &b, 
		    cudaColorSpinorField &x, cudaColorSpinorField &y) {
    // swap arguments around 
    blasCuda<cabxpyAx,1,1,0,0>(make_double2(a,0.0), make_double2(b.real(),b.imag()), 
			       make_double2(0.0,0.0), x, y, x, x);
  }

  /**
     Functor performing the operation z[i] = a*x[i] + b*y[i] + z[i]
  */
  template <typename Float2, typename FloatN>
  struct caxpbypz {
    const Float2 a;
    const Float2 b;
    caxpbypz(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
    __device__ void operator()(const FloatN &x, const FloatN &y, FloatN &z, const FloatN &w) 
    { caxpy_(a, x, z); caxpy_(b, y, z); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 5; } //! flops per element
  };

  void caxpbypzCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, 
		    cudaColorSpinorField &y, cudaColorSpinorField &z) {
    blasCuda<caxpbypz,0,0,1,0>(make_double2(a.real(),a.imag()), make_double2(b.real(),b.imag()), 
			       make_double2(0.0,0.0), x, y, z, z);
  }

  /**
     Functor Performing the operation w[i] = a*x[i] + b*y[i] + c*z[i] + w[i]
  */
  template <typename Float2, typename FloatN>
  struct caxpbypczpw {
    const Float2 a;
    const Float2 b;
    const Float2 c;
    caxpbypczpw(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b), c(c) { ; }
    __device__ void operator()(const FloatN &x, const FloatN &y, const FloatN &z, FloatN &w) 
    { caxpy_(a, x, w); caxpy_(b, y, w); caxpy_(c, z, w); }

    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 5; } //! flops per element
  };

  void caxpbypczpwCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, 
		       cudaColorSpinorField &y, const Complex &c, cudaColorSpinorField &z, 
		       cudaColorSpinorField &w) {
    blasCuda<caxpbypczpw,0,0,0,1>(make_double2(a.real(),a.imag()), make_double2(b.real(),b.imag()), 
				  make_double2(c.real(), c.imag()), x, y, z, w);
  }

  /**
     double caxpyXmazCuda(c a, V x, V y, V z){}
   
     First performs the operation y[i] = a*x[i] + y[i]
     Second performs the operator x[i] -= a*z[i]
  */
  template <typename Float2, typename FloatN>
  struct caxpyxmaz {
    Float2 a;
    caxpyxmaz(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
    __device__ void operator()(FloatN &x, FloatN &y, const FloatN &z, const FloatN &w) 
    { caxpy_(a, x, y); x-= a.x*z; }
    static int streams() { return 5; } //! total number of input and output streams
    static int flops() { return 8; } //! flops per element
  };

  void caxpyXmazCuda(const Complex &a, cudaColorSpinorField &x, 
		     cudaColorSpinorField &y, cudaColorSpinorField &z) {
    blasCuda<caxpyxmaz,1,1,0,0>(make_double2(a.real(), a.imag()), make_double2(0.0, 0.0), 
				make_double2(0.0, 0.0), x, y, z, x);
  }

} // namespace quda
