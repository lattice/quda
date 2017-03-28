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

    template <int writeX, int writeY, int writeZ, int writeW>
    struct write {
      static constexpr int X = writeX;
      static constexpr int Y = writeY;
      static constexpr int Z = writeZ;
      static constexpr int W = writeW;
    };

    namespace reduce {

      namespace multi {
#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
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
       Broken at the moment---need to update reDotProduct with permuting, etc of cDotProduct below.
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
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 2:
        reduce::multiReduceCuda<2,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 3:
        reduce::multiReduceCuda<3,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 4:
        reduce::multiReduceCuda<4,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 5:
        reduce::multiReduceCuda<5,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 6:
        reduce::multiReduceCuda<6,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 7:
        reduce::multiReduceCuda<7,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 8:
        reduce::multiReduceCuda<8,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      /*case 9:
        reduce::multiReduceCuda<9,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 10:
        reduce::multiReduceCuda<10,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 11:
        reduce::multiReduceCuda<11,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 12:
        reduce::multiReduceCuda<12,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 13:
        reduce::multiReduceCuda<13,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 14:
        reduce::multiReduceCuda<14,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 15:
        reduce::multiReduceCuda<15,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 16:
        reduce::multiReduceCuda<16,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;*/
      default:
        errorQuda("Unsupported vector size");
        break;
    }
#endif // SSTEP
    // do a single multi-node reduction only once we have computed all local dot products
    const int Nreduce = x.size()*y.size();
    reduceDoubleArray((double*)result, Nreduce);
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

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct CdotCopy : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      CdotCopy(const reduce::coeff_array<Complex> &a, const reduce::coeff_array<Complex> &b, const reduce::coeff_array<Complex> &c, int NYW) : NYW(NYW) { ; }
      __device__ __host__ void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      { cdot_<ReduceType>(sum,x,y); if (i==j) w = y;}
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal, typename writeDiagonal,
	      template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal, typename writeOffDiagonal>
    void cDotProduct_recurse(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			     std::vector<ColorSpinorField*>&z, std::vector<ColorSpinorField*>&w, int i, int j) {

      if (y.size() > MAX_MULTI_BLAS_N) // if greater than max single-kernel size, split and recurse
      {
        // Do the recurse first.
        Complex* result0 = &result[0];
        Complex* result1 = &result[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());
        std::vector<ColorSpinorField*> w0(w.begin(), w.begin() + w.size()/2);
        std::vector<ColorSpinorField*> w1(w.begin() + w.size()/2, w.end());
        cDotProduct_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x, y0, z, w0, i, 2*j+0);
        cDotProduct_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x, y1, z, w1, i, 2*j+1);
      }
      else
      {
        double2* cdot = new double2[x.size()*y.size()];

        reduce::coeff_array<Complex> a, b, c;

        switch(x.size()){ // COMMENT HERE FOR COMPILE TIME
        case 1:
          reduce::multiReduceCuda<1,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 2
        case 2:
          reduce::multiReduceCuda<2,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 3
        case 3:
          reduce::multiReduceCuda<3,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 4
        case 4:
          reduce::multiReduceCuda<4,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 5
        case 5:
          reduce::multiReduceCuda<5,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 6
        case 6:
          reduce::multiReduceCuda<6,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 7
        case 7:
          reduce::multiReduceCuda<7,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 8
        case 8:
          reduce::multiReduceCuda<8,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 9
	case 9:
          reduce::multiReduceCuda<9,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 10
        case 10:
          reduce::multiReduceCuda<10,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 11
        case 11:
          reduce::multiReduceCuda<11,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 12
        case 12:
          reduce::multiReduceCuda<12,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 13
        case 13:
          reduce::multiReduceCuda<13,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 14
        case 14:
          reduce::multiReduceCuda<14,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 15
        case 15:
          reduce::multiReduceCuda<15,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#if MAX_MULTI_BLAS_N >= 16
        case 16:
          reduce::multiReduceCuda<16,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i, j);
          break;
#endif //16
#endif //15
#endif //14
#endif //13
#endif //12
#endif //11
#endif //10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
        default:
          // split the problem and recurse. Splitting in x requires
          // memory reshuffling (unless y = 1).
          // Use a few temporary variables. 

          Complex* tmpmajor = new Complex[x.size()*y.size()];
          Complex* result0 = &tmpmajor[0];
          Complex* result1 = &tmpmajor[(x.size()/2)*y.size()];
          std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
          std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());
          std::vector<ColorSpinorField*> z0(z.begin(), z.begin() + z.size()/2);
          std::vector<ColorSpinorField*> z1(z.begin() + z.size()/2, z.end());

          cDotProduct_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x0, y, z0, w, 2*i+0, j);
          cDotProduct_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x1, y, z1, w, 2*i+1, j);

          const unsigned int xlen0 = x.size()/2;
          const unsigned int xlen1 = x.size() - xlen0;
          const unsigned int ylen = y.size();

          // Copy back into result.
          int count = 0, count0 = 0, count1 = 0;
          for (unsigned int i = 0; i < ylen; i++)
          {
            for (unsigned int j = 0; j < xlen0; j++)
              result[count++] = result0[count0++];
            for (unsigned int j = 0; j < xlen1; j++)
              result[count++] = result1[count1++];
          }

          delete[] tmpmajor;
          break;
        }

	// we are at the leaf of the binary tree (e.g., we ran the kernel): perform the row-to-column-major transpose here.
        if (x.size() <= MAX_MULTI_BLAS_N)
        {
          const unsigned int xlen = x.size();
          const unsigned int ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++)
              result[i*xlen+j] = Complex(cdot[j*ylen + i].x, cdot[j*ylen+i].y);
        }
        delete[] cdot;
      }
    }

    void cDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++)
        result_tmp[i] = 0.0;

      // cDotProduct_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      cDotProduct_recurse<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> >(result_tmp, x, y, x, y, 0, 0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce);

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++)
          result[j*ylen+i] = result_tmp[i*xlen + j];

      delete[] result_tmp;
    }

    // takes the outer product of inner products between and y and copies y into z
    void cDotProductCopy(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			 std::vector<ColorSpinorField*>&z){

#if 0
      if (y.size() != z.size()) errorQuda("Cannot copy input y of size %lu into z of size %lu\n", y.size(), z.size());

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++)
        result_tmp[i] = 0.0;

      // When recursing, only the diagonal tiles will do the copy, the rest just do the outer product
      cDotProduct_recurse<CdotCopy,write<0,0,0,1>,Cdot,write<0,0,0,0> >(result_tmp, x, y, x, z, 0, 0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce);

      // Switch from col-major to row-major. 
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++)
          result[j*ylen+i] = result_tmp[i*xlen + j];

      delete[] result_tmp;
#else
      errorQuda("cDotProductCopy not enabled");
#endif
    }

   } // namespace blas

} // namespace quda
