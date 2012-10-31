#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset

#include <float_vector.h>
#include <texture.h>

#include <tune_quda.h>
#include <typeinfo>

#include <quda_internal.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <face_quda.h> // this is where the MPI / QMP depdendent code is

#define REDUCE_MAX_BLOCKS 65536

#if (__COMPUTE_CAPABILITY__ >= 130)
#define QudaSumFloat double
#define QudaSumFloat2 double2
#define QudaSumFloat3 double3
#else
#define QudaSumFloat doublesingle
#define QudaSumFloat2 doublesingle2
#define QudaSumFloat3 doublesingle3
#include <double_single.h>
#endif

namespace quda {

  // These are used for reduction kernels
  static QudaSumFloat *d_reduce=0;
  static QudaSumFloat *h_reduce=0;
  static QudaSumFloat *hd_reduce=0;
  static cudaEvent_t reduceEnd;

  unsigned long long blas_flops;
  unsigned long long blas_bytes;

  void zeroCuda(cudaColorSpinorField &a) { a.zero(); }

#define checkSpinor(a, b)						\
  {									\
    if (a.Precision() != b.Precision())					\
      errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision()); \
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %d %d", a.Length(), b.Length());	\
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

  // For kernels with precision conversion built in
#define checkSpinorLength(a, b)						\
  {									\
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %d %d", a.Length(), b.Length());	\
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

  // blasTuning = 1 turns off error checking
  static QudaTune blasTuning = QUDA_TUNE_NO;
  static QudaVerbosity verbosity = QUDA_SILENT;

  static cudaStream_t *blasStream;

  static struct {
    int x[QUDA_MAX_DIM];
    int stride;
  } blasConstants;

  void initBlas()
  { 
    // reduction buffer size
    size_t bytes = 3*REDUCE_MAX_BLOCKS*sizeof(QudaSumFloat);

    if (!d_reduce) {
      d_reduce = (QudaSumFloat *) device_malloc(bytes);
    }
    
    // these arrays are acutally oversized currently (only needs to be QudaSumFloat3)
    
    // if the device supports host-mapped memory then use a host-mapped array for the reduction
    if (!h_reduce) {
      if(deviceProp.canMapHostMemory) {
	h_reduce = (QudaSumFloat *) mapped_malloc(bytes);	
	cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0); // set the matching device pointer
      } else {
	h_reduce = (QudaSumFloat *) pinned_malloc(bytes);
	hd_reduce = d_reduce;
      }
      memset(h_reduce, 0, bytes); // added to ensure that valgrind doesn't report h_reduce is unitialised
    }
    
    blasStream = &streams[Nstream-1];
    cudaEventCreateWithFlags(&reduceEnd, cudaEventDisableTiming);
    
    checkCudaError();
  }
  
  
  void endBlas(void)
  {
    if (d_reduce) {
      device_free(d_reduce);
      d_reduce = 0;
    }
    if (h_reduce) {
      host_free(h_reduce);
      h_reduce = 0;
    }
    hd_reduce = 0;
    
    cudaEventDestroy(reduceEnd);
  }
  
  void setBlasTuning(QudaTune tune, QudaVerbosity verbose)
  {
    blasTuning = tune;
    verbosity = verbose;
  }

  // FIXME: this should be queried from the device
#if (__COMPUTE_CAPACITY__ < 200)
#define MAX_BLOCK 512
#else
#define MAX_BLOCK 1024
#endif

  template <typename FloatN, int N, typename Output, typename Input>
  __global__ void copyKernel(Output Y, Input X, int length) {
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;

    while (i < length) {
      FloatN x[N];
      X.load(x, i);
      Y.save(x, i);
      i += gridSize;
    }
  }

  template <typename FloatN, int N, typename Output, typename Input>
  class CopyCuda : public Tunable {

  private:
    Input &X;
    Output &Y;
    const int length;

    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    virtual bool advanceSharedBytes(TuneParam &param) const
    {
      TuneParam next(param);
      advanceBlockDim(next); // to get next blockDim
      int nthreads = next.block.x * next.block.y * next.block.z;
      param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
	sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
      return false;
    }

  public:
    CopyCuda(Output &Y, Input &X, int length) : X(X), Y(Y), length(length) { ; }
    virtual ~CopyCuda() { ; }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << blasConstants.x[0] << "x";
      vol << blasConstants.x[1] << "x";
      vol << blasConstants.x[2] << "x";
      vol << blasConstants.x[3];
      aux << "stride=" << blasConstants.stride << ",out_prec=" << Y.Precision() << ",in_prec=" << X.Precision();
      return TuneKey(vol.str(), "copyKernel", aux.str());
    }  

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, blasTuning, verbosity);
      copyKernel<FloatN, N><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(Y, X, length);
    }

    void preTune() { ; } // no need to save state for copy kernels
    void postTune() { ; } // no need to restore state for copy kernels

    long long flops() const { return 0; }
    long long bytes() const { 
      const int Ninternal = (sizeof(FloatN)/sizeof(((FloatN*)0)->x))*N;
      size_t bytes = (X.Precision() + Y.Precision())*Ninternal;
      if (X.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
      if (Y.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
      return bytes*length; 
    }
  };


  void copyCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src) {
    if (&src == &dst) return; // aliasing fields
    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n", src.Nspin());

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      copyCuda(dst.Even(), src.Even());
      copyCuda(dst.Odd(), src.Odd());
      return;
    }

    checkSpinorLength(dst, src);

    for (int d=0; d<QUDA_MAX_DIM; d++) blasConstants.x[d] = src.X()[d];
    blasConstants.stride = src.Stride();

    // For a given dst precision, there are two non-trivial possibilities for the
    // src precision.

    blas_bytes += src.RealLength()*((int)src.Precision() + (int)dst.Precision());

    Tunable *copy = 0;

    if (dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION) {
      if (src.Nspin() == 4){
	SpinorTexture<float4, float4, float4, 6, 0> src_tex(src);
	Spinor<float4, float2, double2, 6> dst_spinor(dst);
	copy = new CopyCuda<float4, 6, Spinor<float4, float2, double2, 6>, 
			    SpinorTexture<float4, float4, float4, 6, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //src.Nspin() == 1
	SpinorTexture<float2, float2, float2, 3, 0> src_tex(src);
	Spinor<float2, float2, double2, 3> dst_spinor(dst);
	copy = new CopyCuda<float2, 3, Spinor<float2, float2, double2, 3>,
			    SpinorTexture<float2, float2, float2, 3, 0> >
	  (dst_spinor, src_tex, src.Stride());
  }

} else if (dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION) {
      if (src.Nspin() == 4){
	SpinorTexture<float4, float2, double2, 6, 0> src_tex(src);
	Spinor<float4, float4, float4, 6> dst_spinor(dst);
	copy = new CopyCuda<float4, 6, Spinor<float4, float4, float4, 6>,
			    SpinorTexture<float4, float2, double2, 6, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //src.Nspin() ==1
	SpinorTexture<float2, float2, double2, 3, 0> src_tex(src);
	Spinor<float2, float2, float2, 3> dst_spinor(dst);
	copy = new CopyCuda<float2, 3, Spinor<float2, float2, float2, 3>,
			    SpinorTexture<float2, float2, double2, 3, 0> >
	(dst_spinor, src_tex, src.Stride());
}
    } else if (dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_HALF_PRECISION) {
      blas_bytes += src.Volume()*sizeof(float);
      if (src.Nspin() == 4){      
	SpinorTexture<float4, float4, short4, 6, 0> src_tex(src);
	Spinor<float4, float4, float4, 6> dst_spinor(dst);
	copy = new CopyCuda<float4, 6, Spinor<float4, float4, float4, 6>,
			    SpinorTexture<float4, float4, short4, 6, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //nSpin== 1;
	SpinorTexture<float2, float2, short2, 3, 0> src_tex(src);
	Spinor<float2, float2, float2, 3> dst_spinor(dst);
	copy = new CopyCuda<float2, 3, Spinor<float2, float2, float2, 3>,
			    SpinorTexture<float2, float2, short2, 3, 0> >
	  (dst_spinor, src_tex, src.Stride());
  }
} else if (dst.Precision() == QUDA_HALF_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION) {
      blas_bytes += dst.Volume()*sizeof(float);
      if (src.Nspin() == 4){
	SpinorTexture<float4, float4, float4, 6, 0> src_tex(src);
	Spinor<float4, float4, short4, 6> dst_spinor(dst);
	copy = new CopyCuda<float4, 6, Spinor<float4, float4, short4, 6>,
			    SpinorTexture<float4, float4, float4, 6, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //nSpin == 1
	SpinorTexture<float2, float2, float2, 3, 0> src_tex(src);
	Spinor<float2, float2, short2, 3> dst_spinor(dst);
	copy = new CopyCuda<float2, 3, Spinor<float2, float2, short2, 3>,
			    SpinorTexture<float2, float2, float2, 3, 0> >
	(dst_spinor, src_tex, src.Stride());
}
    } else if (dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_HALF_PRECISION) {
      blas_bytes += src.Volume()*sizeof(float);
      if (src.Nspin() == 4){
	SpinorTexture<double2, float4, short4, 12, 0> src_tex(src);
	Spinor<double2, double2, double2, 12> dst_spinor(dst);
	copy = new CopyCuda<double2, 12, Spinor<double2, double2, double2, 12>,
			    SpinorTexture<double2, float4, short4, 12, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //nSpin == 1
	SpinorTexture<double2, float2, short2, 3, 0> src_tex(src);
	Spinor<double2, double2, double2, 3> dst_spinor(dst);
	copy = new CopyCuda<double2, 3, Spinor<double2, double2, double2, 3>,
			    SpinorTexture<double2, float2, short2, 3, 0> >
	  (dst_spinor, src_tex, src.Stride());
  }
} else if (dst.Precision() == QUDA_HALF_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION) {
      blas_bytes += dst.Volume()*sizeof(float);
      if (src.Nspin() == 4){
	SpinorTexture<double2, double2, double2, 12, 0> src_tex(src);
	Spinor<double2, double4, short4, 12> dst_spinor(dst);
	copy = new CopyCuda<double2, 12, Spinor<double2, double4, short4, 12>,
			    SpinorTexture<double2, double2, double2, 12, 0> >
	  (dst_spinor, src_tex, src.Stride());
    } else { //nSpin == 1
	SpinorTexture<double2, double2, double2, 3, 0> src_tex(src);
	Spinor<double2, double2, short2, 3> dst_spinor(dst);
	copy = new CopyCuda<double2, 3, Spinor<double2, double2, short2, 3>,
			    SpinorTexture<double2, double2, double2, 3, 0> >
	(dst_spinor, src_tex, src.Stride());
}
    }
  
    if (dst.Precision() != src.Precision()) {
      copy->apply(*blasStream);
      delete copy;
    } else {
      cudaMemcpy(dst.V(), src.V(), dst.Bytes(), cudaMemcpyDeviceToDevice);
      if (dst.Precision() == QUDA_HALF_PRECISION) {
	cudaMemcpy(dst.Norm(), src.Norm(), dst.NormBytes(), cudaMemcpyDeviceToDevice);
	blas_bytes += 2*dst.RealLength()*sizeof(float);
      }
    }

    checkCudaError();
  }

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

  /**
     Base class from which all reduction functors should derive.
  */
  template <typename ReduceType, typename Float2, typename FloatN>
  struct ReduceFunctor {
    
    //! pre-computation routine called before the "M-loop"
    virtual __device__ void pre() { ; }
    
    //! where the reduction is usually computed and any auxiliary operations
    virtual __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, 
				       FloatN &z, FloatN &w, FloatN &v) = 0;

    //! post-computation routine called after the "M-loop"
    virtual __device__ void post(ReduceType &sum) { ; }

  };

  /**
     Return the L2 norm of x
  */
  __device__ double norm2_(const double2 &a) { return a.x*a.x + a.y*a.y; }
  __device__ float norm2_(const float2 &a) { return a.x*a.x + a.y*a.y; }
  __device__ float norm2_(const float4 &a) { return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; }

  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct Norm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct Norm2 {
#endif
    Norm2(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z,FloatN  &w, FloatN &v) { sum += norm2_(x); }
    static int streams() { return 1; } //! total number of input and output streams
    static int flops() { return 2; } //! flops per element
  };

#include <reduce_core.h>

  double normCuda(const cudaColorSpinorField &x) {
    cudaColorSpinorField &y = (cudaColorSpinorField&)x; // FIXME
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,Norm2,0,0,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), y, y, y, y, y);
  }

  /**
     Return the real dot product of x and y
  */
  __device__ double dot_(const double2 &a, const double2 &b) { return a.x*b.x + a.y*b.y; }
  __device__ float dot_(const float2 &a, const float2 &b) { return a.x*b.x + a.y*b.y; }
  __device__ float dot_(const float4 &a, const float4 &b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct Dot : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct Dot {
#endif
    Dot(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { sum += dot_(x,y); }
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 2; } //! flops per element
  };

  double reDotProductCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,Dot,0,0,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     First performs the operation y[i] = a*x[i]
     Return the norm of y
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct axpyNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct axpyNorm2 {
#endif
    Float2 a;
    axpyNorm2(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { 
      y += a.x*x; sum += norm2_(y); }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 4; } //! flops per element
  };

  double axpyNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,axpyNorm2,0,1,0,false>
      (make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     First performs the operation y[i] = x[i] - y[i]
     Second returns the norm of y
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct xmyNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct xmyNorm2 {
#endif
    xmyNorm2(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { 
      y = x - y; sum += norm2_(y); }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 3; } //! flops per element
  };

  double xmyNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,xmyNorm2,0,1,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     First performs the operation y[i] = a*x[i] + y[i] (complex-valued)
     Second returns the norm of y
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct caxpyNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct caxpyNorm2 {
#endif
    Float2 a;
    caxpyNorm2(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { 
      caxpy_(a, x, y); sum += norm2_(y); }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 6; } //! flops per element
  };

  double caxpyNormCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,caxpyNorm2,0,1,0,false>
      (make_double2(a.real(), a.imag()), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     double caxpyXmayNormCuda(float a, float *x, float *y, n){}
   
     First performs the operation y[i] = a*x[i] + y[i]
     Second performs the operator x[i] -= a*z[i]
     Third returns the norm of x
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct caxpyxmaznormx : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct caxpyxmaznormx {
#endif
    Float2 a;
    caxpyxmaznormx(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { caxpy_(a, x, y); x-= a.x*z; sum += norm2_(x); }
    static int streams() { return 5; } //! total number of input and output streams
    static int flops() { return 10; } //! flops per element
  };

  double caxpyXmazNormXCuda(const Complex &a, cudaColorSpinorField &x, 
			    cudaColorSpinorField &y, cudaColorSpinorField &z) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,caxpyxmaznormx,1,1,0,false>
      (make_double2(a.real(), a.imag()), make_double2(0.0, 0.0), x, y, z, x, x);
  }

  /**
     double cabxpyAxNormCuda(float a, complex b, float *x, float *y, n){}
   
     First performs the operation y[i] += a*b*x[i]
     Second performs x[i] *= a
     Third returns the norm of x
  */
    template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
    struct cabxpyaxnorm : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
    struct cabxpyaxnorm {
#endif
    Float2 a;
    Float2 b;
    cabxpyaxnorm(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { x *= a.x; caxpy_(b, x, y); sum += norm2_(y); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 10; } //! flops per element
  };

  double cabxpyAxNormCuda(const double &a, const Complex &b, 
			  cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double,QudaSumFloat,QudaSumFloat,cabxpyaxnorm,1,1,0,false>
      (make_double2(a, 0.0), make_double2(b.real(), b.imag()), x, y, x, x, x);
  }

  /**
     Returns complex-valued dot product of x and y
  */
  __device__ double2 cdot_(const double2 &a, const double2 &b) 
  { return make_double2(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x); }
  __device__ double2 cdot_(const float2 &a, const float2 &b) 
  { return make_double2(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x); }
  __device__ double2 cdot_(const float4 &a, const float4 &b) 
  { return make_double2(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w, a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z); }

  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct Cdot : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct Cdot {
#endif
    Cdot(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { sum += cdot_(x,y); }
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 4; } //! flops per element
  };

  Complex cDotProductCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    double2 cdot = reduceCuda<double2,QudaSumFloat2,QudaSumFloat,Cdot,0,0,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    return Complex(cdot.x, cdot.y);
  }

  /**
     double2 xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int n) {}
   
     First performs the operation y = x + a*y
     Second returns cdot product (z,y)
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct xpaycdotzy : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct xpaycdotzy {
#endif
    Float2 a;
    xpaycdotzy(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { y = x + a.x*y; sum += cdot_(z,y); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 6; } //! flops per element
  };

  Complex xpaycDotzyCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y, cudaColorSpinorField &z) {
    double2 cdot = reduceCuda<double2,QudaSumFloat2,QudaSumFloat,xpaycdotzy,0,1,0,false>
      (make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, z, x, x);
    return Complex(cdot.x, cdot.y);
  }

  /**
     double caxpyDotzyCuda(float a, float *x, float *y, float *z, n){}
   
     First performs the operation y[i] = a*x[i] + y[i]
     Second returns the dot product (z,y)
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct caxpydotzy : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct caxpydotzy {
#endif
    Float2 a;
    caxpydotzy(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { caxpy_(a, x, y); sum += cdot_(z,y); }
    static int streams() { return 4; } //! total number of input and output streams
    static int flops() { return 8; } //! flops per element
  };

  Complex caxpyDotzyCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y,
			 cudaColorSpinorField &z) {
    double2 cdot = reduceCuda<double2,QudaSumFloat2,QudaSumFloat,caxpydotzy,0,1,0,false>
      (make_double2(a.real(), a.imag()), make_double2(0.0, 0.0), x, y, z, x, x);
    return Complex(cdot.x, cdot.y);
  }

  /**
     First returns the dot product (x,y)
     Returns the norm of x
  */
  __device__ double3 cdotNormA_(const double2 &a, const double2 &b) 
  { return make_double3(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x, a.x*a.x + a.y*a.y); }
  __device__ double3 cdotNormA_(const float2 &a, const float2 &b) 
  { return make_double3(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x, a.x*a.x + a.y*a.y); }
  __device__ double3 cdotNormA_(const float4 &a, const float4 &b) 
  { return make_double3(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w, 
			a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z,
			a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w); }

  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct CdotNormA : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct CdotNormA {
#endif
    CdotNormA(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { sum += cdotNormA_(x,y); }
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 6; } //! flops per element
  };

  double3 cDotProductNormACuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double3,QudaSumFloat3,QudaSumFloat,CdotNormA,0,0,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     First returns the dot product (x,y)
     Returns the norm of y
  */
  __device__ double3 cdotNormB_(const double2 &a, const double2 &b) 
  { return make_double3(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x, b.x*b.x + b.y*b.y); }
  __device__ double3 cdotNormB_(const float2 &a, const float2 &b) 
  { return make_double3(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x, b.x*b.x + b.y*b.y); }
  __device__ double3 cdotNormB_(const float4 &a, const float4 &b) 
  { return make_double3(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w, a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z,
			b.x*b.x + b.y*b.y + b.z*b.z + b.w*b.w); }

  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct CdotNormB : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct CdotNormB {
#endif
    CdotNormB(const Float2 &a, const Float2 &b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { sum += cdotNormB_(x,y); }
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 6; } //! flops per element
  };

  double3 cDotProductNormBCuda(cudaColorSpinorField &x, cudaColorSpinorField &y) {
    return reduceCuda<double3,QudaSumFloat3,QudaSumFloat,CdotNormB,0,0,0,false>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
  }

  /**
     This convoluted kernel does the following: 
     z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct caxpbypzYmbwcDotProductUYNormY : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct caxpbypzYmbwcDotProductUYNormY {
#endif
    Float2 a;
    Float2 b;
    caxpbypzYmbwcDotProductUYNormY(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { caxpy_(a, x, z); caxpy_(b, y, z); caxpy_(-b, w, y); sum += cdotNormB_(v,y); }
    static int streams() { return 7; } //! total number of input and output streams
    static int flops() { return 18; } //! flops per element
  };

  double3 caxpbypzYmbwcDotProductUYNormYCuda(const Complex &a, cudaColorSpinorField &x, 
					     const Complex &b, cudaColorSpinorField &y,
					     cudaColorSpinorField &z, cudaColorSpinorField &w,
					     cudaColorSpinorField &u) {
    return reduceCuda<double3,QudaSumFloat3,QudaSumFloat,caxpbypzYmbwcDotProductUYNormY,0,1,1,false>
      (make_double2(a.real(), a.imag()), make_double2(b.real(), b.imag()), x, y, z, w, u);
  }


  /**
     Specialized kernel for the modified CG norm computation for
     computing beta.  Computes y = y + a*x and returns norm(y) and
     dot(y, delta(y)) where delta(y) is the difference between the
     input and out y vector.
  */
  template <typename ReduceType, typename Float2, typename FloatN>
#if (__COMPUTE_CAPABILITY__ >= 200)
  struct axpyCGNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
#else
  struct axpyCGNorm2 {
#endif
    Float2 a;
    axpyCGNorm2(const Float2 &a, const Float2 &b) : a(a) { ; }
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { 
      FloatN y_new = y + a.x*x;
      sum.x += norm2_(y_new); 
      sum.y += dot_(y_new, y_new-y);
      y = y_new;
    }
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 6; } //! flops per real element
  };

  Complex axpyCGNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y) {
    double2 cg_norm = reduceCuda<double2,QudaSumFloat2,QudaSumFloat,axpyCGNorm2,0,1,0,false>
      (make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    return Complex(cg_norm.x, cg_norm.y);
  }

#if (__COMPUTE_CAPABILITY__ >= 200)

  /**
     This kernel returns (x, x) and (r,r) and also returns the so-called
     heavy quark norm as used by MILC: 1 / N * \sum_i (r, r)_i / (x, x)_i, where
     i is site index and N is the number of sites.
     
     When this kernel is launched, we must enforce that the parameter M
     in the launcher corresponds to the number of FloatN fields used to
     represent the spinor, e.g., M=6 for Wilson and M=3 for staggered.
     This is only the case for half-precision kernels by default.  To
     enable this, the siteUnroll template parameter must be set true
     when reduceCuda is instantiated.
  */
  template <typename ReduceType, typename Float2, typename FloatN>
  struct HeavyQuarkResidualNorm : public ReduceFunctor<ReduceType, Float2, FloatN> {
    Float2 a;
    Float2 b;
    ReduceType aux;
    HeavyQuarkResidualNorm(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
    
    __device__ void pre() { aux.x = 0; aux.y = 0; }
    
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { aux.x += norm2_(x); aux.y += norm2_(y); }
    
    //! sum the solution and residual norms, and compute the heavy-quark norm
    __device__ void post(ReduceType &sum) 
    { 
      sum.x += aux.x; sum.y += aux.y; sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : 1.0; 
    }
    
    static int streams() { return 2; } //! total number of input and output streams
    static int flops() { return 4; } //! undercounts since it excludes the per-site division
  };
  
  double3 HeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &r) {
    double3 rtn = reduceCuda<double3,QudaSumFloat3,QudaSumFloat,HeavyQuarkResidualNorm,0,0,0,true>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, r, r, r, r);
#ifdef MULTI_GPU
    rtn.z /= (x.Volume()*comm_size());
#else
    rtn.z /= x.Volume();
#endif
    return rtn;
  }
  
  /**
     Variant of the HeavyQuarkResidualNorm kernel: this takes three
     arguments, the first two are summed together to form the
     solution, with the third being the residual vector.  This removes
     the need an additional xpy call in the solvers, impriving
     performance.
  */
  template <typename ReduceType, typename Float2, typename FloatN>
  struct xpyHeavyQuarkResidualNorm : public ReduceFunctor<ReduceType, Float2, FloatN> {
    Float2 a;
    Float2 b;
    ReduceType aux;
    xpyHeavyQuarkResidualNorm(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
    
    __device__ void pre() { aux.x = 0; aux.y = 0; }
    
    __device__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) 
    { aux.x += norm2_(x + y); aux.y += norm2_(z); }
    
    //! sum the solution and residual norms, and compute the heavy-quark norm
    __device__ void post(ReduceType &sum) 
    { 
      sum.x += aux.x; sum.y += aux.y; sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : 1.0; 
    }
    
    static int streams() { return 3; } //! total number of input and output streams
    static int flops() { return 5; }
  };
  
  double3 xpyHeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &y,
					cudaColorSpinorField &r) {
    double3 rtn = reduceCuda<double3,QudaSumFloat3,QudaSumFloat,xpyHeavyQuarkResidualNorm,0,0,0,true>
      (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, r, r, r);
#ifdef MULTI_GPU
    rtn.z /= (x.Volume()*comm_size());
#else
    rtn.z /= x.Volume();
#endif
    return rtn;
  }

#else
  
  double3 HeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &r) {
    errorQuda("Not supported on pre-Fermi architectures");
    return make_double3(0.0,0.0,0.0);
  }

  double3 xpyHeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &y,
					cudaColorSpinorField &r) {
    errorQuda("Not supported on pre-Fermi architectures");
    return make_double3(0.0,0.0,0.0);
  }

#endif
  
} // namespace quda
