#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>

//#define QUAD_SUM
#ifdef QUAD_SUM
#include <dbldbl.h>
#endif

#include <cub_helper.cuh>

template<typename> struct ScalarType { };
template<> struct ScalarType<double> { typedef double type; };
template<> struct ScalarType<double2> { typedef double type; };
template<> struct ScalarType<double3> { typedef double type; };
template<> struct ScalarType<double4> { typedef double type; };

template<typename> struct Vec2Type { };
template<> struct Vec2Type<double> { typedef double2 type; };

#ifdef QUAD_SUM
#define QudaSumFloat doubledouble
#define QudaSumFloat2 doubledouble2
#define QudaSumFloat3 doubledouble3
template<> struct ScalarType<doubledouble> { typedef doubledouble type; };
template<> struct ScalarType<doubledouble2> { typedef doubledouble type; };
template<> struct ScalarType<doubledouble3> { typedef doubledouble type; };
template<> struct ScalarType<doubledouble4> { typedef doubledouble type; };
template<> struct Vec2Type<doubledouble> { typedef doubledouble2 type; };
#else
#define QudaSumFloat double
#define QudaSumFloat2 double2
#define QudaSumFloat3 double3
#define QudaSumFloat4 double4
#endif


void checkSpinor(const ColorSpinorField &a, const ColorSpinorField &b) {
  if (a.Precision() != b.Precision())
    errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision());
  if (a.Length() != b.Length())
    errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
  if (a.Stride() != b.Stride())
    errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());
}

void checkLength(const ColorSpinorField &a, ColorSpinorField &b) {									\
  if (a.Length() != b.Length())
    errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
  if (a.Stride() != b.Stride())
    errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());
}

static struct {
  const char *vol_str;
  const char *aux_str;
  char aux_tmp[quda::TuneKey::aux_n];
} blasStrings;

// These are used for reduction kernels
static QudaSumFloat *d_reduce=0;
static QudaSumFloat *h_reduce=0;
static QudaSumFloat *hd_reduce=0;
static cudaEvent_t reduceEnd;
static bool fast_reduce_enabled = false;

namespace quda {
  namespace blas {

    cudaStream_t* getStream();

    void* getDeviceReduceBuffer() { return d_reduce; }
    void* getMappedHostReduceBuffer() { return hd_reduce; }
    void* getHostReduceBuffer() { return h_reduce; }
    cudaEvent_t* getReduceEvent() { return &reduceEnd; }
    bool getFastReduce() { return fast_reduce_enabled; }

    void initReduce()
    {
      /* we have these different reductions to cater for:

	 - regular reductions (reduce_quda.cu) where are reducing to a
           single vector type (max length 4 presently), with possibly
           parity dimension, and a grid-stride loop with max number of
           blocks = 2 x SM count
A.S. edit: extended to 16 for CA solvers
	 - multi-reductions where we are reducing to a matrix of size
	   of size MAX_MULTI_BLAS_N^2 of vectors (max length 4), with
	   possible parity dimension, and a grid-stride loop with
	   maximum number of blocks = 2 x SM count
      */

      const int reduce_size = 16 * sizeof(QudaSumFloat); //A.S. extended from 4 to 16 for CA solvers
      const int max_reduce_blocks = 2*deviceProp.multiProcessorCount;

      const int max_reduce = 2 * max_reduce_blocks * reduce_size;
      const int max_multi_reduce = 2 * MAX_MULTI_BLAS_N * MAX_MULTI_BLAS_N * max_reduce_blocks * 4 * sizeof(QudaSumFloat);

      // reduction buffer size
      size_t bytes = max_reduce > max_multi_reduce ? max_reduce : max_multi_reduce;

      if (!d_reduce) d_reduce = (QudaSumFloat *) device_malloc(bytes);

      // these arrays are actually oversized currently (only needs to be QudaSumFloat3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
	// only use zero copy reductions when using 64-bit
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
	if(deviceProp.canMapHostMemory) {
	  h_reduce = (QudaSumFloat *) mapped_malloc(bytes);
	  cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0); // set the matching device pointer
	} else
#endif
	  {
	    h_reduce = (QudaSumFloat *) pinned_malloc(bytes);
	    hd_reduce = d_reduce;
	  }
	memset(h_reduce, 0, bytes); // added to ensure that valgrind doesn't report h_reduce is unitialised
      }

      cudaEventCreateWithFlags(&reduceEnd, cudaEventDisableTiming);

      // enable fast reductions with CPU spin waiting as opposed to using CUDA events
      char *fast_reduce_env = getenv("QUDA_ENABLE_FAST_REDUCE");
      if (fast_reduce_env && strcmp(fast_reduce_env,"1") == 0) {
        warningQuda("Experimental fast reductions enabled");
        fast_reduce_enabled = true;
      }

      checkCudaError();
    }

    void endReduce(void)
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

    namespace reduce {

#include <texture.h>
#include <reduce_core.cuh>
#include <reduce_core.h>
#include <reduce_mixed_core.h>
#include <exp_reduce_core.h>

    } // namespace reduce

    /**
       Base class from which all reduction functors should derive.
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct ReduceFunctor {

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y,
							   FloatN &z, FloatN &w, FloatN &v) = 0;

      //! post-computation routine called after the "M-loop"
      virtual __device__ __host__ void post(ReduceType &sum) { ; }

    };

    /**
       Return the L1 norm of x
    */
    template<typename ReduceType> __device__ __host__ ReduceType norm1_(const double2 &a) {
      return (ReduceType)fabs(a.x) + (ReduceType)fabs(a.y);
    }

    template<typename ReduceType> __device__ __host__ ReduceType norm1_(const float2 &a) {
      return (ReduceType)fabs(a.x) + (ReduceType)fabs(a.y);
    }

    template<typename ReduceType> __device__ __host__ ReduceType norm1_(const float4 &a) {
      return (ReduceType)fabs(a.x) + (ReduceType)fabs(a.y) + (ReduceType)fabs(a.z) + (ReduceType)fabs(a.w);
    }

    template <typename ReduceType, typename Float2, typename FloatN>
    struct Norm1 : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Norm1(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z,FloatN  &w, FloatN &v)
      { sum += norm1_<ReduceType>(x); }
      static int streams() { return 1; } //! total number of input and output streams
      static int flops() { return 2; } //! flops per element
    };

    double norm1(const ColorSpinorField &x) {
      ColorSpinorField &y = const_cast<ColorSpinorField&>(x); // FIXME
      return reduce::reduceCuda<double,QudaSumFloat,Norm1,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), y, y, y, y, y);
    }

    /**
       Return the L2 norm of x
    */
    template<typename ReduceType> __device__ __host__ void norm2_(ReduceType &sum, const double2 &a) {
      sum += (ReduceType)a.x*(ReduceType)a.x;
      sum += (ReduceType)a.y*(ReduceType)a.y;
    }

    template<typename ReduceType> __device__ __host__ void norm2_(ReduceType &sum, const float2 &a) {
      sum += (ReduceType)a.x*(ReduceType)a.x;
      sum += (ReduceType)a.y*(ReduceType)a.y;
    }

    template<typename ReduceType> __device__ __host__ void norm2_(ReduceType &sum, const float4 &a) {
      sum += (ReduceType)a.x*(ReduceType)a.x;
      sum += (ReduceType)a.y*(ReduceType)a.y;
      sum += (ReduceType)a.z*(ReduceType)a.z;
      sum += (ReduceType)a.w*(ReduceType)a.w;
    }


    template <typename ReduceType, typename Float2, typename FloatN>
      struct Norm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Norm2(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z,FloatN  &w, FloatN &v)
      { norm2_<ReduceType>(sum,x); }
      static int streams() { return 1; } //! total number of input and output streams
      static int flops() { return 2; } //! flops per element
    };

    double norm2(const ColorSpinorField &x) {
      ColorSpinorField &y = const_cast<ColorSpinorField&>(x);
      return reduce::reduceCuda<double,QudaSumFloat,Norm2,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), y, y, y, y, y);
    }


    /**
       Return the real dot product of x and y
    */
    template<typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const double2 &a, const double2 &b) {
      sum += (ReduceType)a.x*(ReduceType)b.x;
      sum += (ReduceType)a.y*(ReduceType)b.y;
    }

    template<typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const float2 &a, const float2 &b) {
      sum += (ReduceType)a.x*(ReduceType)b.x;
      sum += (ReduceType)a.y*(ReduceType)b.y;
    }

    template<typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const float4 &a, const float4 &b) {
      sum += (ReduceType)a.x*(ReduceType)b.x;
      sum += (ReduceType)a.y*(ReduceType)b.y;
      sum += (ReduceType)a.z*(ReduceType)b.z;
      sum += (ReduceType)a.w*(ReduceType)b.w;
    }

   template <typename ReduceType, typename Float2, typename FloatN>
    struct Dot : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Dot(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
     { dot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 2; } //! flops per element
    };

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y) {
      return reduce::reduceCuda<double,QudaSumFloat,Dot,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }


    /**
       First performs the operation z[i] = a*x[i] + b*y[i]
       Return the norm of y
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct axpbyzNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      axpbyzNorm2(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	z = a.x*x + b.x*y; norm2_<ReduceType>(sum,z); }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y,
                      ColorSpinorField &z) {
      return reduce::reduceCuda<double,QudaSumFloat,axpbyzNorm2,0,0,1,0,0,false>
	(make_double2(a, 0.0), make_double2(b, 0.0), x, y, z, x, x);
    }


    /**
       First performs the operation y[i] += a*x[i]
       Return real dot product (x,y)
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct AxpyReDot : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      AxpyReDot(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	y += a.x*x; dot_<ReduceType>(sum,x,y); }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y) {
      return reduce::reduceCuda<double,QudaSumFloat,AxpyReDot,0,1,0,0,0,false>
	(make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }


    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */
    __device__ __host__ void Caxpy_(const double2 &a, const double2 &x, double2 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
    }
    __device__ __host__ void Caxpy_(const float2 &a, const float2 &x, float2 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
    }
    __device__ __host__ void Caxpy_(const float2 &a, const float4 &x, float4 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
      y.z += a.x*x.z; y.z -= a.y*x.w;
      y.w += a.y*x.z; y.w += a.x*x.w;
    }

    /**
       First performs the operation y[i] = a*x[i] + y[i] (complex-valued)
       Second returns the norm of y
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct caxpyNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      caxpyNorm2(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	Caxpy_(a, x, y); norm2_<ReduceType>(sum,y); }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 6; } //! flops per element
    };

    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y) {
      return reduce::reduceCuda<double,QudaSumFloat,caxpyNorm2,0,1,0,0,0,false>
	(make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, x, x, x);
    }


    /**
       double caxpyXmayNormCuda(float a, float *x, float *y, n){}
       First performs the operation y[i] = a*x[i] + y[i]
       Second performs the operator x[i] -= a*z[i]
       Third returns the norm of x
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct caxpyxmaznormx : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      caxpyxmaznormx(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { Caxpy_(a, x, y); Caxpy_(-a,z,x); norm2_<ReduceType>(sum,x); }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 10; } //! flops per element
    };

    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x,
			  ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double,QudaSumFloat,caxpyxmaznormx,1,1,0,0,0,false>
	(make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, z, x, x);
    }


    /**
       double cabxpyzAxNorm(float a, complex b, float *x, float *y, float *z){}
       First performs the operation z[i] = y[i] + a*b*x[i]
       Second performs x[i] *= a
       Third returns the norm of x
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct cabxpyzaxnorm : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      cabxpyzaxnorm(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { x *= a.x; Caxpy_(b, x, y); z = y; norm2_<ReduceType>(sum,z); }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 10; } //! flops per element
    };


    double cabxpyzAxNorm(double a, const Complex &b,
			ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double,QudaSumFloat,cabxpyzaxnorm,1,0,1,0,0,false>
	(make_double2(a, 0.0), make_double2(REAL(b), IMAG(b)), x, y, z, x, x);
    }


    /**
       Returns complex-valued dot product of x and y
    */
    template<typename ReduceType>
    __device__ __host__ void cdot_(ReduceType &sum, const double2 &a, const double2 &b) {
      typedef typename ScalarType<ReduceType>::type scalar;
      sum.x += (scalar)a.x*(scalar)b.x;
      sum.x += (scalar)a.y*(scalar)b.y;
      sum.y += (scalar)a.x*(scalar)b.y;
      sum.y -= (scalar)a.y*(scalar)b.x;
    }

    template<typename ReduceType>
    __device__ __host__ void cdot_(ReduceType &sum, const float2 &a, const float2 &b) {
      typedef typename ScalarType<ReduceType>::type scalar;
      sum.x += (scalar)a.x*(scalar)b.x;
      sum.x += (scalar)a.y*(scalar)b.y;
      sum.y += (scalar)a.x*(scalar)b.y;
      sum.y -= (scalar)a.y*(scalar)b.x;
    }

    template<typename ReduceType>
    __device__ __host__ void cdot_(ReduceType &sum, const float4 &a, const float4 &b) {
      typedef typename ScalarType<ReduceType>::type scalar;
      sum.x += (scalar)a.x*(scalar)b.x;
      sum.x += (scalar)a.y*(scalar)b.y;
      sum.x += (scalar)a.z*(scalar)b.z;
      sum.x += (scalar)a.w*(scalar)b.w;
      sum.y += (scalar)a.x*(scalar)b.y;
      sum.y -= (scalar)a.y*(scalar)b.x;
      sum.y += (scalar)a.z*(scalar)b.w;
      sum.y -= (scalar)a.w*(scalar)b.z;
    }

    template <typename ReduceType, typename Float2, typename FloatN>
    struct Cdot : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Cdot(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { cdot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    Complex cDotProduct(ColorSpinorField &x, ColorSpinorField &y) {
      double2 cdot = reduce::reduceCuda<double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
      return Complex(cdot.x, cdot.y);
    }


    /**
       double caxpyDotzyCuda(float a, float *x, float *y, float *z, n){}
       First performs the operation y[i] = a*x[i] + y[i]
       Second returns the dot product (z,y)
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct caxpydotzy : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      caxpydotzy(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { Caxpy_(a, x, y); cdot_<ReduceType>(sum,z,y); }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 8; } //! flops per element
    };

    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      double2 cdot = reduce::reduceCuda<double2,QudaSumFloat2,caxpydotzy,0,1,0,0,0,false>
	(make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, z, x, x);
      return Complex(cdot.x, cdot.y);
    }


    /**
       First returns the dot product (x,y)
       Returns the norm of x
    */
    template<typename ReduceType, typename InputType>
    __device__ __host__ void cdotNormA_(ReduceType &sum, const InputType &a, const InputType &b) {
      typedef typename ScalarType<ReduceType>::type scalar;
      typedef typename Vec2Type<scalar>::type vec2;
      cdot_<ReduceType>(sum,a,b);
      norm2_<scalar>(sum.z,a);
    }

    /**
       First returns the dot product (x,y)
       Returns the norm of y
    */
    template<typename ReduceType, typename InputType>
    __device__ __host__ void cdotNormB_(ReduceType &sum, const InputType &a, const InputType &b) {
      typedef typename ScalarType<ReduceType>::type scalar;
      typedef typename Vec2Type<scalar>::type vec2;
      cdot_<ReduceType>(sum,a,b);
      norm2_<scalar>(sum.z,b);
    }

    template <typename ReduceType, typename Float2, typename FloatN>
    struct CdotNormA : public ReduceFunctor<ReduceType, Float2, FloatN> {
      CdotNormA(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { cdotNormA_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 6; } //! flops per element
    };

    double3 cDotProductNormA(ColorSpinorField &x, ColorSpinorField &y) {
      return reduce::reduceCuda<double3,QudaSumFloat3,CdotNormA,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }


    /**
       This convoluted kernel does the following:
       z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct caxpbypzYmbwcDotProductUYNormY_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      caxpbypzYmbwcDotProductUYNormY_(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { Caxpy_(a, x, z); Caxpy_(b, y, z); Caxpy_(-b, w, y); cdotNormB_<ReduceType>(sum,v,y); }
      static int streams() { return 7; } //! total number of input and output streams
      static int flops() { return 18; } //! flops per element
    };

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x,
					   const Complex &b, ColorSpinorField &y,
					   ColorSpinorField &z, ColorSpinorField &w,
					   ColorSpinorField &u) {
      if (x.Precision() != z.Precision()) {
	return reduce::mixed::reduceCuda<double3,QudaSumFloat3,caxpbypzYmbwcDotProductUYNormY_,0,1,1,0,0,false>
	  (make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), x, y, z, w, u);
      } else {
	return reduce::reduceCuda<double3,QudaSumFloat3,caxpbypzYmbwcDotProductUYNormY_,0,1,1,0,0,false>
	  (make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), x, y, z, w, u);
      }
    }


    /**
       Specialized kernel for the modified CG norm computation for
       computing beta.  Computes y = y + a*x and returns norm(y) and
       dot(y, delta(y)) where delta(y) is the difference between the
       input and out y vector.
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct axpyCGNorm2 : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      axpyCGNorm2(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	typedef typename ScalarType<ReduceType>::type scalar;
	FloatN z_new = z + a.x*x;
	norm2_<scalar>(sum.x,z_new);
	dot_<scalar>(sum.y,z_new,z_new-z);
	z = z_new;
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 6; } //! flops per real element
    };

    Complex axpyCGNorm(double a, ColorSpinorField &x, ColorSpinorField &y) {
      // swizzle since mixed is on z
      double2 cg_norm ;
      if (x.Precision() != y.Precision()) {
	cg_norm = reduce::mixed::reduceCuda<double2,QudaSumFloat2,axpyCGNorm2,0,0,1,0,0,false>
	  (make_double2(a, 0.0), make_double2(0.0, 0.0), x, x, y, x, x);
      } else {
	cg_norm = reduce::reduceCuda<double2,QudaSumFloat2,axpyCGNorm2,0,0,1,0,0,false>
	  (make_double2(a, 0.0), make_double2(0.0, 0.0), x, x, y, x, x);
      }
      return Complex(cg_norm.x, cg_norm.y);
    }


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
    struct HeavyQuarkResidualNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      typedef typename scalar<ReduceType>::type real;
      Float2 a;
      Float2 b;
      ReduceType aux;
      HeavyQuarkResidualNorm_(const Float2 &a, const Float2 &b) : a(a), b(b), aux{ } { ; }

      __device__ __host__ void pre() { aux.x = 0; aux.y = 0; }

      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	norm2_<real>(aux.x,x); norm2_<real>(aux.y,y);
      }

      //! sum the solution and residual norms, and compute the heavy-quark norm
      __device__ __host__ void post(ReduceType &sum)
      {
	sum.x += aux.x; sum.y += aux.y; sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : static_cast<real>(1.0);
      }

      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! undercounts since it excludes the per-site division
    };

    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r) {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = reduce::reduceCuda<double3,QudaSumFloat3,HeavyQuarkResidualNorm_,0,0,0,0,0,true>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, r, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
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
    struct xpyHeavyQuarkResidualNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
	typedef typename scalar<ReduceType>::type real;
      Float2 a;
      Float2 b;
      ReduceType aux;
      xpyHeavyQuarkResidualNorm_(const Float2 &a, const Float2 &b) : a(a), b(b), aux{ } { ; }

      __device__ __host__ void pre() { aux.x = 0; aux.y = 0; }

      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	norm2_<real>(aux.x,x + y); norm2_<real>(aux.y,z);
      }

      //! sum the solution and residual norms, and compute the heavy-quark norm
      __device__ __host__ void post(ReduceType &sum)
      {
	sum.x += aux.x; sum.y += aux.y; sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : static_cast<real>(1.0);
      }

      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 5; }
    };

    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y,
				      ColorSpinorField &r) {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = reduce::reduceCuda<double3,QudaSumFloat3,xpyHeavyQuarkResidualNorm_,0,0,0,0,0,true>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    /**
       double3 tripleCGReduction(V x, V y, V z){}
       First performs the operation norm2(x)
       Second performs the operatio norm2(y)
       Third performs the operation dotPropduct(y,z)
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct tripleCGReduction_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      tripleCGReduction_(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
	typedef typename ScalarType<ReduceType>::type scalar;
	norm2_<scalar>(sum.x,x); norm2_<scalar>(sum.y,y); dot_<scalar>(sum.z,y,z);
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 6; } //! flops per element
    };

    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double3,QudaSumFloat3,tripleCGReduction_,0,0,0,0,0,false>
	(make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, x);
    }

    /**
       double4 quadrupleCGReduction(V x, V y, V z){}
       First performs the operation norm2(x)
       Second performs the operatio norm2(y)
       Third performs the operation dotPropduct(y,z)
       Fourth performs the operation norm(z)
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct quadrupleCGReduction_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      quadrupleCGReduction_(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
        typedef typename ScalarType<ReduceType>::type scalar;
        norm2_<scalar>(sum.x,x); norm2_<scalar>(sum.y,y); dot_<scalar>(sum.z,y,z); norm2_<scalar>(sum.w,w);
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 8; } //! flops per element
    };

    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double4,QudaSumFloat4,quadrupleCGReduction_,0,0,0,0,0,false>
        (make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, x);
    }

    /**
       double quadrupleCG3InitNorm(d a, d b, V x, V y, V z, V w, V v){}
        z = x;
        w = y;
        x += a*y;
        y -= a*v;
        norm2(y);
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct quadrupleCG3InitNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      quadrupleCG3InitNorm_(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
        z = x;
        w = y;
        x += a.x*y;
        y -= a.x*v;
        norm2_<ReduceType>(sum,y);
      }
      static int streams() { return 6; } //! total number of input and output streams
      static int flops() { return 6; } //! flops per element check if it's right
    };

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v) {
      return reduce::reduceCuda<double,QudaSumFloat,quadrupleCG3InitNorm_,1,1,1,1,0,false>
	(make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, z, w, v);
    }


    /**
       double quadrupleCG3UpdateNorm(d gamma, d rho, V x, V y, V z, V w, V v){}
        tmpx = x;
        tmpy = y;
        x = b*(x + a*y) + (1-b)*z;
        y = b*(y + a*v) + (1-b)*w;
        z = tmpx;
        w = tmpy;
        norm2(y);
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct quadrupleCG3UpdateNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a,b;
      quadrupleCG3UpdateNorm_(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      FloatN tmpx{}, tmpy{};
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
        tmpx = x;
        tmpy = y;
        x = b.x*(x + a.x*y) + b.y*z;
        y = b.x*(y - a.x*v) + b.y*w;
        z = tmpx;
        w = tmpy;
        norm2_<ReduceType>(sum,y);
      }
      static int streams() { return 7; } //! total number of input and output streams
      static int flops() { return 16; } //! flops per element check if it's right
    };

    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v) {
      return reduce::reduceCuda<double,QudaSumFloat,quadrupleCG3UpdateNorm_,1,1,1,1,0,false>
	(make_double2(a, 0.0), make_double2(b, 1.-b), x, y, z, w, v);
    }

    /**
       void doubleCG3InitNorm(d a, V x, V y, V z){}
        y = x;
        x -= a*z;
        norm2(x);
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct doubleCG3InitNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a;
      doubleCG3InitNorm_(const Float2 &a, const Float2 &b) : a(a) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
        y = x;
        x -= a.x*z;
        norm2_<ReduceType>(sum,x);
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 5; } //! flops per element
    };

    double doubleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double,QudaSumFloat,doubleCG3InitNorm_,1,1,0,0,0,false>
        (make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, z, z, z);
    }

    /**
       void doubleCG3UpdateNorm(d a, d b, V x, V y, V z){}
        tmp = x;
        x = b*(x-a*z) + (1-b)*y;
        y = tmp;
        norm2(x);
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct doubleCG3UpdateNorm_ : public ReduceFunctor<ReduceType, Float2, FloatN> {
      Float2 a, b;
      doubleCG3UpdateNorm_(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      FloatN tmp{};
      __device__ __host__ void operator()(ReduceType &sum,FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) {
        tmp = x;
        x = b.x*(x-a.x*z) + b.y*y;
        y = tmp;
        norm2_<ReduceType>(sum,x);
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 9; } //! flops per element
    };

    double doubleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return reduce::reduceCuda<double,QudaSumFloat,doubleCG3UpdateNorm_,1,1,0,0,0,false>
        (make_double2(a, 0.0), make_double2(b, 1.0-b), x, y, z, z, z);
    }


/*
    Reduction routines for a number of pipelined methods
*/


    template<typename ReduceType>
    __device__ __host__ void hp_axpby_reduce(ReduceType &sum, const double &a, const double &b, double2 &x, double2 &p, double2 &u, double2 &r, double2 &s, double2 &m, double2 &q, double2 &w, double2 &n, double2 &z){

#if defined( __CUDA_ARCH__)

      //the first component
      //z = n + b*z;
      z.x  = __fma_rn (b, z.x, n.x);
      //s = w + b*s;
      s.x  = __fma_rn (b, s.x, w.x);
      //q = m + b*q;
      //q.x  = __fma_rn (b, q.x, m.x);
      q.x = s.x;
      //p = u/r + b*p;
      p.x  = __fma_rn (b, p.x, u.x);

      //x = x + a*p;
      x.x  = __fma_rn (a, p.x, x.x);
      //r = r - a*s;
      r.x  = __fma_rn (-a, s.x, r.x);
      //u = u - a*q;
      //u.x  = __fma_rn (-a, q.x, u.x);
      u.x  = r.x;
      //w = w - a*z;
      w.x  = __fma_rn (-a, z.x, w.x);

      //the second component:
      //z = n + b*z;
      z.y  = __fma_rn (b, z.y, n.y);
      //s = w + b*s;
      s.y  = __fma_rn (b, s.y, w.y);
      //q = m + b*q;
      //q.y  = __fma_rn (b, q.y, m.y);
      q.y = s.y;
      //p = u/r + b*p;
      p.y  = __fma_rn (b, p.y, u.y);

      //x = x + a*p;
      x.y  = __fma_rn (a, p.y, x.y);
      //r = r - a*s;
      r.y  = __fma_rn (-a, s.y, r.y);
      //u = u - a*q;
      //u.y  = __fma_rn (-a, q.y, u.y);
      u.y  = r.y;
      //w = w - a*z;
      w.y  = __fma_rn (-a, z.y, w.y);

      double sum_0_x = (double) sum.x;
      double sum_0_y = (double) sum.y;
      double sum_0_z = (double) sum.z;

      sum_0_x  = __fma_rn (r.x, u.x, sum_0_x);
      sum_0_x  = __fma_rn (r.y, u.y, sum_0_x);

      sum_0_y  = __fma_rn (w.x, u.x, sum_0_y);
      sum_0_y  = __fma_rn (w.y, u.y, sum_0_y);

      sum_0_z  = __fma_rn (r.x, r.x, sum_0_z);
      sum_0_z  = __fma_rn (r.y, r.y, sum_0_z);

      sum.x = sum_0_x;
      sum.y = sum_0_y;
      sum.z = sum_0_z;
      sum.w = 0.0;
#else
//cpu code
#endif

      return;
    }


    template<typename ReduceType>
    __device__ __host__  void hp_axpby_reduce(ReduceType &sum, const double &a, const double &b, float2 &x, float2 &p, float2 &u, float2 &r, float2 &s, float2 &m, float2 &q, float2 &w, float2 &n, float2 &z){

#if defined( __CUDA_ARCH__)
      double s_, p_, z_, r_x, r_y, x_, w_x, w_y, q_, u_x, u_y, m_, n_;
      //the first component
      //z = n + b*z;
      z_ = z.x, n_ = n.x;
      z_ = __fma_rn (b, z_, n_);
      z.x= z_;
      //s = w + b*s;
      s_ = s.x, w_x = w.x;
      s_ = __fma_rn (b, s_, w_x);
      s.x= s_;
      //q = m + b*q;
      //q_ = q.x, m_ = m.x;
      //q_ = __fma_rn (b, q_, m_);
      //q.x = q_;
      q.x = s.x;
      //p = u/r + b*p;
      p_ = p.x, u_x = u.x;
      p_ = __fma_rn (b, p_, u_x);
      p.x = p_;

      //x = x + a*p;
      x_ = x.x;
      x_ = __fma_rn (a, p_, x_);
      x.x = x_;
      //r = r - a*s;
      r_x = r.x;
      r_x = __fma_rn (-a, s_, r_x);
      r.x = r_x;
      //u = u - a*q;
      //u_x = __fma_rn (-a, q_, u_x);
      //u.x = u_x;
      u_x = r_x;
      u.x = r.x;
      //w = w - a*z;
      w_x = __fma_rn (-a, z_, w_x);
      w.x = w_x;

      //the second component:
      //z = n + b*z;
      z_ = z.y, n_ = n.y;
      z_ = __fma_rn (b, z_, n_);
      z.y= z_;
      //s = w + b*s;
      s_ = s.y, w_y = w.y;
      s_ = __fma_rn (b, s_, w_y);
      s.y= s_;
      //q = m + b*q;
      //q_ = q.y, m_ = m.y;
      //q_ = __fma_rn (b, q_, m_);
      //q.y = q_;
      q.y = s.y;
      //p = u/r + b*p;
      p_ = p.y, u_y = u.y;
      p_ = __fma_rn (b, p_, u_x);
      p.y = p_;

      //x = x + a*p;
      x_ = x.y;
      x_ = __fma_rn (a, p_, x_);
      x.y = x_;
      //r = r - a*s;
      r_y = r.y;
      r_y = __fma_rn (-a, s_, r_y);
      r.y = r_y;
      //u = u - a*q;
      //u_y = __fma_rn (-a, q_, u_y);
      //u.y = u_y;
      u_y = r_y;
      u.y  = r.y;
      //w = w - a*z;
      w_y = __fma_rn (-a, z_, w_y);
      w.y = w_y;

      double sum_0_x = (double) sum.x;
      double sum_0_y = (double) sum.y;
      double sum_0_z = (double) sum.z;

      sum_0_x  = __fma_rn (r_x, u_x, sum_0_x);
      sum_0_x  = __fma_rn (r_y, u_y, sum_0_x);

      sum_0_y  = __fma_rn (w_x, u_x, sum_0_y);
      sum_0_y  = __fma_rn (w_y, u_y, sum_0_y);

      sum_0_z  = __fma_rn (r_x, r_x, sum_0_z);
      sum_0_z  = __fma_rn (r_y, r_y, sum_0_z);

      sum.x = sum_0_x;
      sum.y = sum_0_y;
      sum.z = sum_0_z;
      sum.w = 0.0;
#else
//cpu code
#endif

      return;
    }

    template<typename ReduceType>
    __device__ __host__ void hp_axpby_reduce(ReduceType &sum, const double &a, const double &b, float4 &x, float4 &p, float4 &u, float4 &r, float4 &s, float4 &m, float4 &q, float4 &w, float4 &n, float4 &z){

#if defined( __CUDA_ARCH__)
      double s_, p_, z_, r_x, r_y, r_z, r_w, x_, w_x, w_y, w_z, w_w, q_, u_x, u_y, u_z, u_w, m_, n_;
      //the first component
      //z = n + b*z;
      z_ = z.x, n_ = n.x;
      z_ = __fma_rn (b, z_, n_);
      z.x= z_;
      //s = w + b*s;
      s_ = s.x, w_x = w.x;
      s_ = __fma_rn (b, s_, w_x);
      s.x= s_;
      //q = m + b*q;
      //q_ = q.x, m_ = m.x;
      //q_ = __fma_rn (b, q_, m_);
      //q.x = q_;
      q.x = s.x;
      //p = u/r + b*p;
      p_ = p.x, u_x = u.x;
      p_ = __fma_rn (b, p_, u_x);
      p.x = p_;

      //x = x + a*p;
      x_ = x.x;
      x_ = __fma_rn (a, p_, x_);
      x.x = x_;
      //r = r - a*s;
      r_x = r.x;
      r_x = __fma_rn (-a, s_, r_x);
      r.x = r_x;
      //u = u - a*q;
      //u_x = __fma_rn (-a, q_, u_x);
      //u.x = u_x;
      u_x = r_x;
      u.x = r.x;
      //w = w - a*z;
      w_x = __fma_rn (-a, z_, w_x);
      w.x = w_x;

      //the second component:
      //z = n + b*z;
      z_ = z.y, n_ = n.y;
      z_ = __fma_rn (b, z_, n_);
      z.y= z_;
      //s = w + b*s;
      s_ = s.y, w_y = w.y;
      s_ = __fma_rn (b, s_, w_y);
      s.y= s_;
      //q = m + b*q;
      //q_ = q.y, m_ = m.y;
      //q_ = __fma_rn (b, q_, m_);
      //q.y = q_;
      q.y = s.y;
      //p = u/r + b*p;
      p_ = p.y, u_y = u.y;
      p_ = __fma_rn (b, p_, u_x);
      p.y = p_;

      //x = x + a*p;
      x_ = x.y;
      x_ = __fma_rn (a, p_, x_);
      x.y = x_;
      //r = r - a*s;
      r_y = r.y;
      r_y = __fma_rn (-a, s_, r_y);
      r.y = r_y;
      //u = u - a*q;
      //u_y = __fma_rn (-a, q_, u_y);
      //u.y = u_y;
      u_y = r_y;
      u.y  = r.y;
      //w = w - a*z;
      w_y = __fma_rn (-a, z_, w_y);
      w.y = w_y;

      //the third component:
      //z = n + b*z;
      z_ = z.z, n_ = n.z;
      z_ = __fma_rn (b, z_, n_);
      z.z= z_;
      //s = w + b*s;
      s_ = s.z, w_z = w.z;
      s_ = __fma_rn (b, s_, w_z);
      s.z= s_;
      //q = m + b*q;
      //q_ = q.z, m_ = m.z;
      //q_ = __fma_rn (b, q_, m_);
      //q.z = q_;
      q.z = s.z;
      //p = u/r + b*p;
      p_ = p.z, u_z = u.z;
      p_ = __fma_rn (b, p_, u_z);
      p.z = p_;

      //x = x + a*p;
      x_ = x.z;
      x_ = __fma_rn (a, p_, x_);
      x.z = x_;
      //r = r - a*s;
      r_z = r.z;
      r_z = __fma_rn (-a, s_, r_z);
      r.z = r_z;
      //u = u - a*q;
      //u_z = __fma_rn (-a, q_, u_z);
      //u.z = u_z;
      u_z = r_z;
      u.z  = r.z;
      //w = w - a*z;
      w_z = __fma_rn (-a, z_, w_z);
      w.z = w_z;

      //the fourth component:
      //z = n + b*z;
      z_ = z.w, n_ = n.w;
      z_ = __fma_rn (b, z_, n_);
      z.w= z_;
      //s = w + b*s;
      s_ = s.w, w_w = w.w;
      s_ = __fma_rn (b, s_, w_w);
      s.w= s_;
      //q = m + b*q;
      //q_ = q.w, m_ = m.w;
      //q_ = __fma_rn (b, q_, m_);
      //q.w = q_;
      q.w = s.w;
      //p = u/r + b*p;
      p_ = p.w, u_w = u.w;
      p_ = __fma_rn (b, p_, u_w);
      p.w = p_;

      //x = x + a*p;
      x_ = x.w;
      x_ = __fma_rn (a, p_, x_);
      x.w = x_;
      //r = r - a*s;
      r_w = r.w;
      r_w = __fma_rn (-a, s_, r_w);
      r.w = r_w;
      //u = u - a*q;
      //u_w = __fma_rn (-a, q_, u_w);
      //u.w = u_w;
      u_w = r_w;
      u.w  = r.w;
      //w = w - a*z;
      w_w = __fma_rn (-a, z_, w_w);
      w.w = w_w;



      double sum_0_x = (double) sum.x;
      double sum_0_y = (double) sum.y;
      double sum_0_z = (double) sum.z;

      sum_0_x  = __fma_rn (r_x, u_x, sum_0_x);
      sum_0_x  = __fma_rn (r_y, u_y, sum_0_x);
      sum_0_x  = __fma_rn (r_z, u_z, sum_0_x);
      sum_0_x  = __fma_rn (r_w, u_w, sum_0_x);

      sum_0_y  = __fma_rn (w_x, u_x, sum_0_y);
      sum_0_y  = __fma_rn (w_y, u_y, sum_0_y);
      sum_0_y  = __fma_rn (w_z, u_z, sum_0_y);
      sum_0_y  = __fma_rn (w_w, u_w, sum_0_y);

      sum_0_z  = __fma_rn (r_x, r_x, sum_0_z);
      sum_0_z  = __fma_rn (r_y, r_y, sum_0_z);
      sum_0_z  = __fma_rn (r_z, r_z, sum_0_z);
      sum_0_z  = __fma_rn (r_w, r_w, sum_0_z);

      sum.x = sum_0_x;
      sum.y = sum_0_y;
      sum.z = sum_0_z;
      sum.w = 0.0;
#else
//cpu code
#endif

      return;
    }

///for the preconditioned system:
//     y = x + a*y;
//     z = z + b*y;


    __device__ __host__ void hp_xpaybz(const double &a, const double2 &x, double2 &y, const double &b, double2 &z){

      _fma2(a, y, x);
      _fma3(b, y, z);

    }

    __device__ __host__ void hp_xpaybz(const double &a, const float2 &x, float2 &y, const double &b, float2 &z){

      double2 x_ = make_FloatN(x);
      double2 y_ = make_FloatN(y);
      double2 z_ = make_FloatN(z);

      _fma2(a, y_, x_);
      _fma3(b, y_, z_);

      y = make_FloatN(y_);
      z = make_FloatN(z_);

    }


    __device__ __host__ void hp_xpaybz(const double &a, const float4 &x, float4 &y, const double &b, float4 &z){

      double4 x_ = make_FloatN(x);
      double4 y_ = make_FloatN(y);
      double4 z_ = make_FloatN(z);

      _fma2(a, y_, x_);
      _fma3(b, y_, z_);

      y = make_FloatN(y_);
      z = make_FloatN(z_);

    }


//q = m + a*q;
//u = u + b*q;

//(r,u)

//s = w + a*s;
//r = r + b*s;

//z = n + a*z;
//w = w + b*z;

// (r, u);
// (w, u);
// norm2_(r);

    __device__ __host__ void dot_fma(double &sum, const double2 &a, const double2 &b) {
#if defined( __CUDA_ARCH__)
      sum  = __fma_rn (a.x, b.x, sum);
      sum  = __fma_rn (a.y, b.y, sum);
#else
      sum += a.x*b.x;
      sum += a.y*b.y;
#endif
    }

    __device__ __host__ void dot_fma(double &sum, const double4 &a, const double4 &b) {
#if defined( __CUDA_ARCH__)
      sum  = __fma_rn (a.x, b.x, sum);
      sum  = __fma_rn (a.y, b.y, sum);
      sum  = __fma_rn (a.z, b.z, sum);
      sum  = __fma_rn (a.w, b.w, sum);
#else
      sum += a.x*b.x;
      sum += a.y*b.y;
      sum += a.z*b.z;
      sum += a.w*b.w;
#endif
    }


    template<typename ReduceType>
    __device__ __host__ void hp_xpaybz_combo_reduce(ReduceType &sum, const double &a, const double2 &m, double2 &q, const double &b, double2 &u, double2 &w, double2 &s, double2 &r, const double2 &n, double2 &z){

      _fma2(a, q, m);
      _fma3(b, q, u);

      double& sum_w = static_cast<double&> (sum.w);

      dot_fma (sum_w, r, u);

      _fma2(a, s, w);
      _fma3(b, s, r);

      double& sum_x = static_cast<double&> (sum.x);
      dot_fma ( sum_x, r, u);
      double& sum_z = static_cast<double&> (sum.z);
      dot_fma (sum_z, r, r);

      _fma2(a, z, n);
      _fma3(b, z, w);
      double& sum_y = static_cast<double&> (sum.y);
      dot_fma (sum_y, w, u);
    }

    template<typename ReduceType>
    __device__ __host__ void hp_xpaybz_combo_reduce(ReduceType &sum, const double &a, const float2 &m, float2 &q, const double &b, float2 &u, float2 &w, float2 &s, float2 &r, const float2 &n, float2 &z){

      double2 x_ = make_FloatN(q);
      double2 y_ = make_FloatN(m);
      double2 u_ = make_FloatN(u);
      double2 w_ = make_FloatN(w);

      _fma2(a, x_, y_);
      _fma3(b, x_, u_);

      q = make_FloatN(x_);
      u = make_FloatN(u_);

      x_ = make_FloatN(s);
      y_ = make_FloatN(r);

      double& sum_w = static_cast<double&> (sum.w);
      dot_fma (sum_w, y_, u_);

      _fma2(a, x_, w_);
      _fma3(b, x_, y_);

      s = make_FloatN(x_);
      r = make_FloatN(y_);

      double& sum_x = static_cast<double&> (sum.x);
      dot_fma ( sum_x, y_, u_);
      double& sum_z = static_cast<double&> (sum.z);
      dot_fma (sum_z, y_, y_);

      x_ = make_FloatN(n);
      y_ = make_FloatN(z);

      _fma2(a, y_, x_);
      _fma3(b, y_, w_);

      z = make_FloatN(y_);
      w = make_FloatN(w_);

      double& sum_y = static_cast<double&> (sum.y);
      dot_fma (sum_y, w_, u_);

    }


    template<typename ReduceType>
    __device__ __host__ void hp_xpaybz_combo_reduce(ReduceType &sum, const double &a, const float4 &m, float4 &q, const double &b, float4 &u, float4 &w, float4 &s, float4 &r, const float4 &n, float4 &z){

      double4 x_ = make_FloatN(q);
      double4 y_ = make_FloatN(m);
      double4 u_ = make_FloatN(u);
      double4 w_ = make_FloatN(w);

      _fma2(a, x_, y_);
      _fma3(b, x_, u_);

      q = make_FloatN(x_);
      u = make_FloatN(u_);

      x_ = make_FloatN(s);
      y_ = make_FloatN(r);

      double& sum_w = static_cast<double&> (sum.w);
      dot_fma (sum_w, y_, u_);

      _fma2(a, x_, w_);
      _fma3(b, x_, y_);

      s = make_FloatN(x_);
      r = make_FloatN(y_);

      double& sum_x = static_cast<double&> (sum.x);
      dot_fma ( sum_x, y_, u_);
      double& sum_z = static_cast<double&> (sum.z);
      dot_fma (sum_z, y_, y_);

      x_ = make_FloatN(n);
      y_ = make_FloatN(z);

      _fma2(a, y_, x_);
      _fma3(b, y_, w_);

      z = make_FloatN(y_);
      w = make_FloatN(w_);

      double& sum_y = static_cast<double&> (sum.y);
      dot_fma (sum_y, w_, u_);

    }



//////////////////////////////////////////////////////////////////////////////////////////
//#define ERROR_CONTROL

    template <int Nreduce, typename ReduceType, typename Float2, typename FloatN>
    struct ReduceFunctorExp {

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x, FloatN &p, FloatN &u,FloatN &r,
						  FloatN &s, FloatN &m, FloatN &q, FloatN &w, FloatN &n, FloatN &z) = 0;

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x1, FloatN &r1, FloatN &w1,FloatN &q1,
						  FloatN &d1, FloatN &h1, FloatN &z1, FloatN &p1, FloatN &u1, FloatN &g1,
                                                  FloatN &x2, FloatN &r2, FloatN &w2,FloatN &q2, FloatN &d2, FloatN &h2,
                                                  FloatN &z2, FloatN &p2, FloatN &u2, FloatN &g2) = 0;

      //! post-computation routine called after the "M-loop"
      virtual __device__ __host__ void post(ReduceType sum[Nreduce]) { ; }

    };
    /**
       This convoluted kernel does the following:
//Gr1
       x += a*p,
       p = u + a*p,
       r -= a*s,
//Gr2
       u -= a*q,
       q = m + b*q,
//Gr3
       s = w + b*s,
       w -= a*z,
       z = n + b*z,
//Gr4
       norm = (u,u),
       rdot = (w,u),
       rdot = (r,u),
    */
    template <int Nreduce, typename ReduceType, typename Float2, typename FloatN>
    struct pipePCGRRMergedOp_ : public ReduceFunctorExp<Nreduce, ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      pipePCGRRMergedOp_(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x, FloatN &p, FloatN &u, FloatN &r, FloatN &s, FloatN &m, FloatN &q, FloatN &w, FloatN &n, FloatN &z) {

	typedef typename ScalarType<ReduceType>::type scalar;
#ifndef ERROR_CONTROL
         norm2_<scalar> (sum[1].x, p);
         norm2_<scalar> (sum[1].y, s);
         norm2_<scalar> (sum[1].z, q);
         norm2_<scalar> (sum[1].w, z);

         z = n + b.x*z;
         q = m + b.x*q;
         s = w + b.x*s;
         p = u + b.x*p;

         norm2_<scalar> (sum[1].x, p);
         norm2_<scalar> (sum[1].y, s);
         norm2_<scalar> (sum[1].z, q);
         norm2_<scalar> (sum[1].w, z);

         norm2_<scalar> (sum[2].x, x);
         norm2_<scalar> (sum[2].y, u);
         norm2_<scalar> (sum[2].z, w);
         norm2_<scalar> (sum[2].w, m);

         x = x + a.x*p;
         u = u - a.x*q;

         sum[0].w = 0.0;

         r = r - a.x*s;
         w = w - a.x*z;

         dot_<scalar>   (sum[0].x, r, u);
         dot_<scalar>   (sum[0].y, w, u);
         norm2_<scalar> (sum[0].z, r);
#else
         double a_ = a.x;
         double b_ = b.x;

         hp_axpby_reduce<ReduceType>(sum[0], a_, b_, x, p, u, r, s, m, q, w, n, z);
#endif
    }

     __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x1, FloatN &r1, FloatN &w1,FloatN &q1,
						  FloatN &d1, FloatN &h1, FloatN &z1, FloatN &p1, FloatN &u1, FloatN &g1,
                                                  FloatN &x2, FloatN &r2, FloatN &w2,FloatN &q2, FloatN &d2, FloatN &h2,
                                                  FloatN &z2, FloatN &p2, FloatN &u2, FloatN &g2) {}

      static int streams() { return 18; } //! total number of input and output streams
      static int flops() { return (16+6); } //! flops per real element
    };

    void pipePCGRRMergedOp(double4 *buffer, const int buffer_size, ColorSpinorField &x, const double &a, ColorSpinorField &p, ColorSpinorField &u, ColorSpinorField &r, ColorSpinorField &s,
                                ColorSpinorField &m, const double &b, ColorSpinorField &q,
			        ColorSpinorField &w, ColorSpinorField &n, ColorSpinorField &z) {
      if (x.Precision() != p.Precision()) {
         errorQuda("\nMixed blas is not implemented.\n");
      }
      if(buffer_size != 3) errorQuda("Incorrect buffer size. \n");

      reduce::reduceCudaExp<3, double4, QudaSumFloat4,pipePCGRRMergedOp_,1,1,1,1,1,0,1,1,0,1,false>
	  (buffer, make_double2(a, 0.0), make_double2(b, 0.0), x, p, u, r, s, m, q, w, n, z);
      return;
    }

    template <int Nreduce, typename ReduceType, typename Float2, typename FloatN>
    struct pipePCGRRFletcherReevesMergedOp_ : public ReduceFunctorExp<Nreduce, ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      pipePCGRRFletcherReevesMergedOp_(const Float2 &a, const Float2 &b) : a(a), b(b) { ; }
      __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x, FloatN &p, FloatN &u, FloatN &r, FloatN &s, FloatN &m, FloatN &q, FloatN &w, FloatN &n, FloatN &z) {

	 typedef typename ScalarType<ReduceType>::type scalar;
#ifndef ERROR_CONTROL

         if(Nreduce == 3) {
           norm2_<scalar> (sum[2].x, x);
           norm2_<scalar> (sum[2].y, u);
           norm2_<scalar> (sum[2].z, w);
           norm2_<scalar> (sum[2].w, m);
         }

//         z = n + b.x*z;
         _fma2(b.x, z, n);
//         q = m + b.x*q;
         _fma2(b.x, q, m);
//         s = w + b.x*s;
         _fma2(b.x, s, w);
//         p = u + b.x*p;
         _fma2(b.x, p, u);

//         x = x + a.x*p;
         _fma3(+a.x, p, x);
//         u = u - a.x*q;
         _fma3(-a.x, q, u);

         dot_<scalar>   (sum[0].w, r, u);

//         r = r - a.x*s;
         _fma3(-a.x, s, r);
//         w = w - a.x*z;
         _fma3(-a.x, z, w);

         dot_<scalar>   (sum[0].x, r, u);
         dot_<scalar>   (sum[0].y, w, u);
         dot_<scalar>   (sum[0].z, r, r);

         if(Nreduce == 3) {
           norm2_<scalar> (sum[1].x, p);
           norm2_<scalar> (sum[1].y, s);
           norm2_<scalar> (sum[1].z, q);
           norm2_<scalar> (sum[1].w, z);
         }

#else
         if(Nreduce == 3) {
           norm2_<scalar> (sum[2].x, x);
           norm2_<scalar> (sum[2].y, u);
           norm2_<scalar> (sum[2].z, w);
           norm2_<scalar> (sum[2].w, m);
         }

         //p = u + b.x*p;
         //x = x + a.x*p;
         hp_xpaybz(b.x, u, p, a.x, x);

         //q = m + b.x*q;
         //u = u - a.x*q;
         //<r,u>
         //s = w + b.x*s;
         //r = r - a.x*s;
         //z = n + b.x*z;
         //w = w - a.x*z;
         //<r,u>
         //<w,u>
         //<r,r>
         hp_xpaybz_combo_reduce<ReduceType>(sum[0], b.x, m, q, -a.x, u, w, s, r, n, z);

         if(Nreduce == 3) {
           norm2_<scalar> (sum[1].x, p);
           norm2_<scalar> (sum[1].y, s);
           norm2_<scalar> (sum[1].z, q);
           norm2_<scalar> (sum[1].w, z);
         }
#endif
      }

      __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x1, FloatN &r1, FloatN &w1,FloatN &q1,
						  FloatN &d1, FloatN &h1, FloatN &z1, FloatN &p1, FloatN &u1, FloatN &g1,
                                                  FloatN &x2, FloatN &r2, FloatN &w2,FloatN &q2, FloatN &d2, FloatN &h2,
                                                  FloatN &z2, FloatN &p2, FloatN &u2, FloatN &g2) {}
      static int streams() { return 18; } //! total number of input and output streams
      static int flops() { return (16+6); } //! flops per real element
    };


    void pipePCGRRFletcherReevesMergedOp(double4 *buffer, const int buffer_size,  ColorSpinorField &x, const double &a, ColorSpinorField &p, ColorSpinorField &u,
                                ColorSpinorField &r, ColorSpinorField &s,
                                ColorSpinorField &m, const double &b, ColorSpinorField &q,
			        ColorSpinorField &w, ColorSpinorField &n, ColorSpinorField &z) {
      if (x.Precision() != p.Precision()) {
         errorQuda("\nMixed blas is not implemented.\n");
      }
#if 0
      if( buffer_size == 3 ) {
         reduce::reduceCudaExp<3, double4, QudaSumFloat4,pipePCGRRFletcherReevesMergedOp_,1,1,1,1,1,0,1,1,0,1,false>
	  (buffer, make_double2(a, 0.0), make_double2(b, 0.0), x, p, u, r, s, m, q, w, n, z);
      } else if ( buffer_size == 1 ) {
         reduce::reduceCudaExp<1, double4, QudaSumFloat4,pipePCGRRFletcherReevesMergedOp_,1,1,1,1,1,0,1,1,0,1,false>
          (buffer, make_double2(a, 0.0), make_double2(b, 0.0), x, p, u, r, s, m, q, w, n, z);
      } else {
         errorQuda("Buffer size is not implemented. \n");
      }
#endif
      return;
    }

    template <int Nreduce, typename ReduceType, typename Float2, typename FloatN>
    struct pipe2PCGMergedOp_ : public ReduceFunctorExp<Nreduce, ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      Float2 c;

      Float2 a2;
      Float2 b2;
      Float2 c2;

      pipe2PCGMergedOp_(const Float2 &a, const Float2 &b, const Float2 &c, const Float2 &a2, const Float2 &b2, const Float2 &c2) : a(a), b(b), c(c), a2(a2), b2(b2), c2(c2) { ; }
      __device__ __host__ void operator()(ReduceType sum[Nreduce],FloatN &x1, FloatN &r1, FloatN &w1,FloatN &q1,FloatN &d1, FloatN &h1, FloatN &z1, FloatN &p1, FloatN &u1, FloatN &g1,
                                          FloatN &x2, FloatN &r2, FloatN &w2,FloatN &q2, FloatN &d2, FloatN &h2,FloatN &z2, FloatN &p2, FloatN &u2, FloatN &g2) {
	 typedef typename ScalarType<ReduceType>::type scalar;

         x2 = a.x*x1 + b.x*z1 + c.x*x2;
         r2 = a.x*r1 - b.x*w1 + c.x*r2;
         w2 = a.x*w1 - b.x*q1 + c.x*w2;
         q2 = a.x*q1 - b.x*d1 + c.x*q2;
         d2 = a.x*d1 - b.x*h1 + c.x*d2;
         z2 = a.x*z1 - b.x*p1 + c.x*z2;
         p2 = a.x*p1 - b.x*u1 + c.x*p2;
         u2 = a.x*u1 - b.x*g1 + c.x*u2;

         x1 = a2.x*x2 + b2.x*z2 + c2.x*x1;
         r1 = a2.x*r2 - b2.x*w2 + c2.x*r1;
         w1 = a2.x*w2 - b2.x*q2 + c2.x*w1;
         q1 = a2.x*q2 - b2.x*d2 + c2.x*q1;
         z1 = a2.x*z2 - b2.x*p2 + c2.x*z1;
         p1 = a2.x*p2 - b2.x*d2 + c2.x*p1;

         dot_<scalar> (sum[0].x, z1, w1);//l0
         dot_<scalar> (sum[0].y, z1, q1);//l1
         dot_<scalar> (sum[0].w, z1, w2);//l2
         dot_<scalar> (sum[0].z, p1, q1);//l3

         dot_<scalar> (sum[1].x, p1, w2);//l4
         dot_<scalar> (sum[1].y, z2, w2);//l5
         dot_<scalar> (sum[1].w, z1, r1);//l6
         dot_<scalar> (sum[1].z, z1, r2);//l7

         dot_<scalar>   (sum[2].x, z2, r2);//l8
         norm2_<scalar> (sum[2].y, z1);//l9

         sum[2].w = 0.0;
         sum[2].z = 0.0;

      }

      __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x, FloatN &p, FloatN &u,FloatN &r,
						  FloatN &s, FloatN &m, FloatN &q, FloatN &w, FloatN &n, FloatN &z) {}

      static int streams() { return 18; } //! total number of input and output streams
      static int flops() { return (16+6); } //! flops per real element
    };

    void pipe2PCGMergedOp(double4 *buffer, const double &a, const double &b, const double &c, const double &a2, const double &b2, const double &c2,
                                ColorSpinorField &x1, ColorSpinorField &r1, ColorSpinorField &w1,
                                ColorSpinorField &q1, ColorSpinorField &d1, ColorSpinorField &h1, ColorSpinorField &z1,
                                ColorSpinorField &p1, ColorSpinorField &u1, ColorSpinorField &g1,
                                ColorSpinorField &x2, ColorSpinorField &r2, ColorSpinorField &w2,
                                ColorSpinorField &q2, ColorSpinorField &d2, ColorSpinorField &h2, ColorSpinorField &z2,
			        ColorSpinorField &p2, ColorSpinorField &u2, ColorSpinorField &g2) {

      if (x1.Precision() != p1.Precision()) {
         errorQuda("\nMixed blas is not implemented.\n");
      }
#if 0
      reduce::reduceComponentwiseCudaExp<3, double4, QudaSumFloat4,pipe2PCGMergedOp_,1,1,1,1,1,0,1,1,1,0,false>
	  (buffer, make_double2(a, 0.0), make_double2(b, 0.0), make_double2(c, 0.0), make_double2(a2, 0.0), make_double2(b2, 0.0), make_double2(c2, 0.0),  x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2);
#endif
       return;
    }

    template <int Nreduce, typename ReduceType, typename Float2, typename FloatN>
    struct pipe2CGMergedOp_ : public ReduceFunctorExp<Nreduce, ReduceType, Float2, FloatN> {
      Float2 a;
      Float2 b;
      Float2 c;

      Float2 a2;
      Float2 b2;
      Float2 c2;

      pipe2CGMergedOp_(const Float2 &a, const Float2 &b, const Float2 &c, const Float2 &a2, const Float2 &b2, const Float2 &c2) : a(a), b(b), c(c), a2(a2), b2(b2), c2(c2) { ; }
      __device__ __host__ void operator()(ReduceType sum[Nreduce],FloatN &x1, FloatN &r1, FloatN &w1,FloatN &q1,FloatN &d1, FloatN &h1, FloatN &z1, FloatN &p1, FloatN &u1, FloatN &g1,
                                          FloatN &x2, FloatN &r2, FloatN &w2,FloatN &q2, FloatN &d2, FloatN &h2,FloatN &z2, FloatN &p2, FloatN &u2, FloatN &g2) {
	 typedef typename ScalarType<ReduceType>::type scalar;

         x2 = a.x*x1 + b.x*r1 + c.x*x2;
         r2 = a.x*r1 - b.x*w1 + c.x*r2;
         w2 = a.x*w1 - b.x*q1 + c.x*w2;
         q2 = a.x*q1 - b.x*d1 + c.x*q2;
         d2 = a.x*d1 - b.x*h1 + c.x*d2;

         x1 = a2.x*x2 + b2.x*r2 + c2.x*x1;
         r1 = a2.x*r2 - b2.x*w2 + c2.x*r1;
         w1 = a2.x*w2 - b2.x*q2 + c2.x*w1;
         q1 = a2.x*q2 - b2.x*d2 + c2.x*q1;

         dot_<scalar> (sum[0].x, r1, w1);//l0
         dot_<scalar> (sum[0].y, r1, q1);//l1
         dot_<scalar> (sum[0].w, r1, w2);//l2
         dot_<scalar> (sum[0].z, w1, q1);//l3

         dot_<scalar> (sum[1].x, w1, w2);//l4
         dot_<scalar> (sum[1].y, r2, w2);//l5
         norm2_<scalar> (sum[1].w, r1);
         dot_<scalar> (sum[1].z, r1, r2);//l7
         norm2_<scalar> (sum[2].x, r2);


         sum[2].y = sum[1].w;
         sum[2].w = 0.0;
         sum[2].z = 0.0;
      }

      __device__ __host__ void operator()(ReduceType sum[Nreduce], FloatN &x, FloatN &p, FloatN &u,FloatN &r,
						  FloatN &s, FloatN &m, FloatN &q, FloatN &w, FloatN &n, FloatN &z) {}

      static int streams() { return 18; } //! total number of input and output streams
      static int flops() { return (16+6); } //! flops per real element
    };

    void pipe2CGMergedOp(double4 *buffer, const double &a, const double &b, const double &c, const double &a2, const double &b2, const double &c2,
                                ColorSpinorField &x1, ColorSpinorField &r1, ColorSpinorField &w1,
                                ColorSpinorField &q1, ColorSpinorField &d1, ColorSpinorField &h1, ColorSpinorField &z1,
                                ColorSpinorField &p1, ColorSpinorField &u1, ColorSpinorField &g1,
                                ColorSpinorField &x2, ColorSpinorField &r2, ColorSpinorField &w2,
                                ColorSpinorField &q2, ColorSpinorField &d2, ColorSpinorField &h2, ColorSpinorField &z2,
			        ColorSpinorField &p2, ColorSpinorField &u2, ColorSpinorField &g2) {

      if (x1.Precision() != p1.Precision()) {
         errorQuda("\nMixed blas is not implemented.\n");
      }
#if 0
      reduce::reduceComponentwiseCudaExp<3, double4, QudaSumFloat4,pipe2CGMergedOp_,1,1,1,1,1,0,0,0,0,0,false>
	  (buffer, make_double2(a, 0.0), make_double2(b, 0.0), make_double2(c, 0.0), make_double2(a2, 0.0), make_double2(b2, 0.0), make_double2(c2, 0.0),  x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2);
#endif
       return;
    }

  } // namespace blas

} // namespace quda
