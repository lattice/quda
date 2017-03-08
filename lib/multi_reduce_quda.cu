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
    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct MultiReduceFunctor {

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y,
							   FloatN &z, FloatN &w, const int i, const int j) = 0;

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

   template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct Dot : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      Dot(const reduce::coeff_array<Complex> &a, const reduce::coeff_array<Complex> &b, const reduce::coeff_array<Complex> &c, int NYW) : NYW(NYW) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
     { dot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 2; } //! flops per element
    };

    void reDotProduct(double* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
#ifndef SSTEP
    errorQuda("S-step code not built\n");
#else
    switch(x.size()){
      case 1:
        reduce::multiReduceCuda<1,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 2:
        reduce::multiReduceCuda<2,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 3:
        reduce::multiReduceCuda<3,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 4:
        reduce::multiReduceCuda<4,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 5:
        reduce::multiReduceCuda<5,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 6:
        reduce::multiReduceCuda<6,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 7:
        reduce::multiReduceCuda<7,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 8:
        reduce::multiReduceCuda<8,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 9:
        reduce::multiReduceCuda<9,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 10:
        reduce::multiReduceCuda<10,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 11:
        reduce::multiReduceCuda<11,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 12:
        reduce::multiReduceCuda<12,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 13:
        reduce::multiReduceCuda<13,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 14:
        reduce::multiReduceCuda<14,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 15:
        reduce::multiReduceCuda<15,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
        break;
      case 16:
        reduce::multiReduceCuda<16,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x);
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

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct Cdot : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      Cdot(const reduce::coeff_array<Complex> &a, const reduce::coeff_array<Complex> &b, const reduce::coeff_array<Complex> &c, int NYW) : NYW(NYW) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      { cdot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };


    void cDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){

      printfQuda("ESW Entered cDotProduct\n");
      // if (y.size() > 16), we need to split and recurse in y. This requires awkward memory remapping.
      // 16 is because of the MAX_MULTI_BLAS in multi_reduce_core.cuh, along with the switch statement up to 15.
      if (y.size() > 4) // using 4 just for compile time at the moment. 
      {
        printfQuda("ESW Begin recurse y b/c y.size() = %d\n", y.size());
        // Do the recurse first.
        Complex* tmpresult = new Complex[x.size()*y.size()];
        memset(tmpresult, 0, x.size()*y.size()*sizeof(Complex));
        Complex* result0 = &tmpresult[0];
        Complex* result1 = &tmpresult[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());

        cDotProduct(result0, x, y0);
        cDotProduct(result1, x, y1);

        // Re-order the memory. 
        const int ylen0 = y.size()/2;
        const int ylen1 = y.size() - ylen0;
        const int xlen = x.size();
        int count = 0, count0 = 0, count1 = 0;
        for (int i = 0; i < xlen; i++)
        {
          for (int j = 0; j < ylen0; j++)
            result[count++] = result0[count0++];
          for (int j = 0; j < ylen1; j++)
            result[count++] = result1[count1++];
        }
        delete[] tmpresult;
        printfQuda("ESW End recurse y b/c y.size() = %d\n", y.size());

      }
      else
      {
        double2* cdot = new double2[x.size()*y.size()];

        reduce::coeff_array<Complex> a, b, c;

        printfQuda("ESW Try multiReduceCuda, x.size() = %d, y.size() = %d\n", x.size(), y.size());

        switch(x.size()){
        case 1:
          printfQuda("ESW multiReduceCuda<1>\n");
          reduce::multiReduceCuda<1,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 2:
          printfQuda("ESW multiReduceCuda<2>\n");
          reduce::multiReduceCuda<2,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 3:
          printfQuda("ESW multiReduceCuda<3>\n");
          reduce::multiReduceCuda<3,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 4:
          printfQuda("ESW multiReduceCuda<4>\n");
          reduce::multiReduceCuda<4,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        /*case 5:
          reduce::multiReduceCuda<5,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 6:
          reduce::multiReduceCuda<6,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 7:
          reduce::multiReduceCuda<7,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;
        case 8:
          reduce::multiReduceCuda<8,double2,QudaSumFloat2,Cdot,0,0,0,0,false>
  	  (cdot, a, b, c, x, y, x, x);
          break;*/
        default:
          // split the problem and recurse.
          printfQuda("ESW Begin recurse x b/c x.size() = %d\n", x.size());
          Complex* result0 = &result[0];
          Complex* result1 = &result[(x.size()/2)*y.size()];
          std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
          std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

          printfQuda("result0 is at %lu, result1 is at %lu\n", (unsigned long)(result0)/sizeof(unsigned long), (unsigned long)(result1)/sizeof(unsigned long));

          cDotProduct(result0, x0, y);
          cDotProduct(result1, x1, y);
          printfQuda("ESW End recurse x b/c x.size() = %d\n", x.size());
          break;
        }
        printfQuda("ESW Done multiReduceCuda, x.size() = %d, y.size() = %d\n", x.size(), y.size());

        // if x.size() > 4 we recursed directly into result.
        if (x.size() <= 4) for (unsigned int i=0; i<x.size()*y.size(); ++i) result[i] = Complex(cdot[i].x,cdot[i].y);
        delete[] cdot;
      }

      if (x.size() > 4)
      {
        for (unsigned int i = 0; i < x.size(); i++)
        {
          printf("%.15e + I %.15e\n", real(result[i]), imag(result[i]));
        }
      }
    }

   } // namespace blas

} // namespace quda
