#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>
#include <uint_to_char.h>

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

// work around for Fermi
#if (__COMPUTE_CAPABILITY__ < 300)
#undef MAX_MULTI_BLAS_N
#define MAX_MULTI_BLAS_N 2
#endif

static void checkSpinor(const ColorSpinorField &a, const ColorSpinorField &b) {
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

  // hooks into tune.cpp variables for policy tuning
  typedef std::map<TuneKey, TuneParam> map;
  const map& getTuneCache();

  void disableProfileCount();
  void enableProfileCount();

  void setPolicyTuning(bool);

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
      __device__ __host__ inline void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      { cdot_<ReduceType>(sum,x,y); }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct CdotCopy : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      CdotCopy(const reduce::coeff_array<Complex> &a, const reduce::coeff_array<Complex> &b, const reduce::coeff_array<Complex> &c, int NYW) : NYW(NYW) { ; }
      __device__ __host__ inline void operator()(ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      { cdot_<ReduceType>(sum,x,y); if (i==j) w = y;}
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; } //! flops per element
    };

    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal, typename writeDiagonal,
	      template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal, typename writeOffDiagonal>
    void multiReduce_recurse(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			     std::vector<ColorSpinorField*>&z, std::vector<ColorSpinorField*>&w, int i_idx, int j_idx, bool hermitian, unsigned int tile_size) {

      if (y.size() > tile_size) // if greater than max single-kernel size, split and recurse
      {
        // Do the recurse first.
        Complex* result0 = &result[0];
        Complex* result1 = &result[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());
        std::vector<ColorSpinorField*> w0(w.begin(), w.begin() + w.size()/2);
        std::vector<ColorSpinorField*> w1(w.begin() + w.size()/2, w.end());
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x, y0, z, w0, i_idx, 2*j_idx+0, hermitian, tile_size);
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x, y1, z, w1, i_idx, 2*j_idx+1, hermitian, tile_size);
      }
      else
      {
        double2* cdot = new double2[x.size()*y.size()];

	// if at bottom of recursion, return if on lower left
	if (x.size() <= tile_size && hermitian) {
	  if (j_idx < i_idx) { return; }
	}

        reduce::coeff_array<Complex> a, b, c;

	if (x.size() <= tile_size) {
        switch(x.size()){ // COMMENT HERE FOR COMPILE TIME
        case 1:
          reduce::multiReduceCuda<1,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 2
        case 2:
          reduce::multiReduceCuda<2,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 3
        case 3:
          reduce::multiReduceCuda<3,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 4
        case 4:
          reduce::multiReduceCuda<4,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 5
        case 5:
          reduce::multiReduceCuda<5,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 6
        case 6:
          reduce::multiReduceCuda<6,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 7
        case 7:
          reduce::multiReduceCuda<7,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 8
        case 8:
          reduce::multiReduceCuda<8,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 9
	case 9:
          reduce::multiReduceCuda<9,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 10
        case 10:
          reduce::multiReduceCuda<10,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 11
        case 11:
          reduce::multiReduceCuda<11,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 12
        case 12:
          reduce::multiReduceCuda<12,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 13
        case 13:
          reduce::multiReduceCuda<13,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 14
        case 14:
          reduce::multiReduceCuda<14,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 15
        case 15:
          reduce::multiReduceCuda<15,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
          break;
#if MAX_MULTI_BLAS_N >= 16
        case 16:
          reduce::multiReduceCuda<16,double2,QudaSumFloat2,ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal,false>
	    (cdot, a, b, c, x, y, z, w, i_idx, j_idx );
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
	}
	} else {
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

          multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x0, y, z0, w, 2*i_idx+0, j_idx, hermitian, tile_size);
          multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x1, y, z1, w, 2*i_idx+1, j_idx, hermitian, tile_size);

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
        }

	// we are at the leaf of the binary tree (e.g., we ran the kernel): perform the row-to-column-major transpose here.
        if (x.size() <= tile_size)
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


    template <template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal,
	      typename writeDiagonal,
	      template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal,
	      typename writeOffDiagonal>
    class TileSizeTune : public Tunable {
      typedef std::vector<ColorSpinorField*> vec;
      Complex *result;
      vec &x, &y, &z, &w;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      unsigned int max_tile_size;

    public:
      TileSizeTune(Complex *result, vec &x, vec &y, vec &z, vec &w, bool hermitian, bool Anorm = false)
	: result(result), x(x), y(y), z(z), w(w), hermitian(hermitian), Anorm(Anorm), max_tile_size(1)
      {
      	strcpy(aux, "policy,");
      	strcat(aux, x[0]->AuxString());
      	strcat(aux, ",");
      	strcat(aux, y[0]->AuxString());
        if (hermitian) strcat(aux, ",hermitian");
        if (Anorm) strcat(aux, ",Anorm");
	strcat(aux,",n=");
	char size[8];
	u64toa(size, x.size());
	strcat(aux,size);
	strcat(aux,",m=");
	u64toa(size, y.size());
	strcat(aux,size);

      	// before we do policy tuning we must ensure the kernel
      	// constituents have been tuned since we can't do nested tuning
      	// FIXME this will break if the kernels are destructive - which they aren't here
	if (getTuning() && getTuneCache().find(tuneKey()) == getTuneCache().end()) {
	  disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

	  if ( x.size()==1 || y.size()==1 ) { // 1-d reduction

	    max_tile_size = std::min(MAX_MULTI_BLAS_N, (int)std::max(x.size(), y.size()));

	    // Make sure constituents are tuned.
	    for ( unsigned int tile_size=1; tile_size <= max_tile_size; tile_size++) {
	      multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
		(result, x, y, z, w, 0, 0, hermitian, tile_size);
	    }

	  } else { // 2-d reduction

	    // max_tile_size should be set to the largest power of 2 less than
	    // MAX_MULTI_BLAS_N, since we have a requirement that the
	    // tile size is a power of 2.
	    unsigned int max_count = 0;
	    unsigned int tile_size_tmp = MAX_MULTI_BLAS_N;
	    while (tile_size_tmp != 1) { tile_size_tmp = tile_size_tmp >> 1; max_count++; }
	    tile_size_tmp = 1;
	    for (unsigned int i = 0; i < max_count; i++) { tile_size_tmp = tile_size_tmp << 1; }
	    max_tile_size = tile_size_tmp;

	    // Make sure constituents are tuned.
	    for ( unsigned int tile_size=1; tile_size <= max_tile_size && tile_size <= x.size() &&
		    (tile_size <= y.size() || y.size()==1) ; tile_size*=2) {
	      multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
		(result, x, y, z, w, 0, 0, hermitian, tile_size);
	    }

	  }

      	  enableProfileCount();
      	  setPolicyTuning(true);
      	}
      }

      virtual ~TileSizeTune() { setPolicyTuning(false); }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        // tp.aux.x is where the tile size is stored. "tp" is the tuning struct.
        // it contains blocksize, grid size, etc. Since we're only tuning
        // a policy, we don't care about those sizes. That's why we only
        // tune "aux.x", which is the tile size. 
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
          (result, x, y, z, w, 0, 0, hermitian, tp.aux.x);
      }

      // aux.x is the tile size
      bool advanceAux(TuneParam &param) const
      {

	if ( x.size()==1 || y.size()==1 ) { // 1-d reduction

	  param.aux.x++;
	  if ( (unsigned int)param.aux.x <= max_tile_size ) {
	    return true;
	  } else {
	    param.aux.x = 1;
	    return false;
	  }

	} else { // 2-d reduction

	  param.aux.x *= 2; // only tune powers of two (FIXME)

	  if ( (unsigned int)param.aux.x <= max_tile_size && param.aux.x <= (int)x.size() &&
	       param.aux.x <= (int)y.size() ) {
	    return true;
	  } else {
	    param.aux.x = 1; // reset to the beginning (which we'd need for multi-dimensional tuning)
	    return false;
	  }

	}
      }

      bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const  {
      	Tunable::initTuneParam(param);
      	param.aux.x = 1; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
      }

      void defaultTuneParam(TuneParam &param) const  {
      	Tunable::defaultTuneParam(param); // default is max tile size
        // max_tile_size is MAX_MULTI_BLAS_N rounded down to the nearest power of 2.
      	param.aux.x = max_tile_size; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
      }

      TuneKey tuneKey() const {
        return TuneKey(x[0]->VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return 0; } // FIXME
      long long bytes() const { return 0; } // FIXME

      void preTune() { } // FIXME - use write to determine what needs to be saved
      void postTune() { } // FIXME - use write to determine what needs to be saved
    };

    void cDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      // cDotProduct_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, false);
      tile.apply(0);

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

    void hDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true, false); // last false is b/c L2 norm
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce); // FIXME - could optimize this for Hermiticity as well

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = j; i < ylen; i++) {
          result[j*ylen+i] = result_tmp[i*xlen + j];
          result[i*ylen+j] = conj(result_tmp[i*xlen + j]);
	}

      delete[] result_tmp;
    }

    // for (p, Ap) norms in CG which are Hermitian. 
    void hDotProduct_Anorm(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block A-norm dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true, true); // last true is b/c A norm
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce); // FIXME - could optimize this for Hermiticity as well

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = j; i < ylen; i++) {
          result[j*ylen+i] = result_tmp[i*xlen + j];
          result[i*ylen+j] = conj(result_tmp[i*xlen + j]);
  }

      delete[] result_tmp;
    }

    // takes the outer product of inner products between and y and copies y into z
    void cDotProductCopy(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			 std::vector<ColorSpinorField*>&z){

#if 0
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (y.size() != z.size()) errorQuda("Cannot copy input y of size %lu into z of size %lu\n", y.size(), z.size());

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      // When recursing, only the diagonal tiles will do the copy, the rest just do the outer product
      TileSizeTune<CdotCopy,write<0,0,0,1>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true);
      tile.apply(0);

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
