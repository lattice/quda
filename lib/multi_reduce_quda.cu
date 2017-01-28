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

template<typename> struct Vec2Type { };
template<> struct Vec2Type<double> { typedef double2 type; };

#ifdef QUAD_SUM
#define QudaSumFloat doubledouble
#define QudaSumFloat2 doubledouble2
#define QudaSumFloat3 doubledouble3
template<> struct ScalarType<doubledouble> { typedef doubledouble type; };
template<> struct ScalarType<doubledouble2> { typedef doubledouble type; };
template<> struct ScalarType<doubledouble3> { typedef doubledouble type; };
template<> struct Vec2Type<doubledouble> { typedef doubledouble2 type; };
#else
#define QudaSumFloat double
#define QudaSumFloat2 double2
#define QudaSumFloat3 double3
#endif


#define REDUCE_MAX_BLOCKS 65536

static void checkSpinor(const ColorSpinorField &a, const ColorSpinorField &b) {
  if (a.Precision() != b.Precision())
    errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision());
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

namespace quda {
  namespace blas {

    cudaStream_t* getStream();
    cudaEvent_t* getReduceEvent();

    namespace reduce {

      namespace multi {
#include <texture.h>
      }

#include <multi_reduce_core.cuh>
#include <multi_reduce_core.h>

    } // namespace reduce

    /**
       Base class from which all reduction functors should derive.
    */
    template <typename ReduceType, typename Float2, typename FloatN>
    struct MultiReduceFunctor {

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y,
							   FloatN &z, FloatN &w, FloatN &v) = 0;

      //! post-computation routine called after the "M-loop"
      virtual __device__ __host__ void post(ReduceType &sum) { ; }

    };


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
    struct Dot : public MultiReduceFunctor<ReduceType, Float2, FloatN> {
      Dot(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
     { dot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 2; } //! flops per element
    };

    void reDotProduct(double* result, std::vector<cudaColorSpinorField*>& x, std::vector<cudaColorSpinorField*>& y){
#ifndef SSTEP
    errorQuda("S-step code not built\n");
#else
    switch(x.size()){
      case 1:
        reduce::multiReduceCuda<1,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 2:
        reduce::multiReduceCuda<2,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 3:
        reduce::multiReduceCuda<3,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 4:
        reduce::multiReduceCuda<4,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 5:
        reduce::multiReduceCuda<5,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 6:
        reduce::multiReduceCuda<6,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 7:
        reduce::multiReduceCuda<7,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 8:
        reduce::multiReduceCuda<8,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 9:
        reduce::multiReduceCuda<9,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 10:
        reduce::multiReduceCuda<10,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 11:
        reduce::multiReduceCuda<11,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 12:
        reduce::multiReduceCuda<12,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 13:
        reduce::multiReduceCuda<13,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 14:
        reduce::multiReduceCuda<14,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 15:
        reduce::multiReduceCuda<15,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 16:
        reduce::multiReduceCuda<16,double,QudaSumFloat,Dot,0,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      default:
        errorQuda("Unsupported vector size");
        break;
    }
#endif // SSTEP
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
    struct Cdot : public MultiReduceFunctor<ReduceType, Float2, FloatN> {
      Cdot(const Float2 &a, const Float2 &b) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      { cdot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };


    void cDotProduct(Complex* result, std::vector<cudaColorSpinorField*>& x, std::vector<cudaColorSpinorField*>& y){
      double2* cdot = new double2[x.size()];

      switch(x.size()){
      case 1:
        reduce::multiReduceCuda<1,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 2:
        reduce::multiReduceCuda<2,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 3:
        reduce::multiReduceCuda<3,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 4:
        reduce::multiReduceCuda<4,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 5:
        reduce::multiReduceCuda<5,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 6:
        reduce::multiReduceCuda<6,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 7:
        reduce::multiReduceCuda<7,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      case 8:
        reduce::multiReduceCuda<8,double2,QudaSumFloat2,Cdot,0,0,0,0,0,false>
	  (cdot, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
        break;
      default:
        errorQuda("Unsupported vector size\n");
        break;
      }

      for (unsigned int i=0; i<x.size(); ++i) result[i] = Complex(cdot[i].x,cdot[i].y);
      delete[] cdot;
    }

   } // namespace blas

} // namespace quda
