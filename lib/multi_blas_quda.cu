#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset



#include <tune_quda.h>
#include <typeinfo>

#include <quda_internal.h>
#include <float_vector.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <face_quda.h> // this is where the MPI / QMP depdendent code is

#define checkSpinor(a, b)						\
  {									\
    if (a.Precision() != b.Precision())					\
      errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision()); \
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length()); \
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

#define checkLength(a, b)						\
  {									\
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length()); \
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

namespace quda {

  namespace blas {

    namespace multi {
#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>
    }

    cudaStream_t* getStream();

    static struct {
      const char *vol_str;
      const char *aux_str;
      char aux_tmp[TuneKey::aux_n];
    } blasStrings;

#include <multi_blas_core.cuh>
#include <multi_blas_core.h>
#include <multi_blas_mixed_core.h>


    template <int NXZ, typename Float2, typename FloatN>
    struct MultiBlasFunctor {

      //! pre-computation routine before the main loop
      virtual __device__ __host__ void init() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j) = 0;
    };


    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */

    __device__ __host__ inline void _caxpy(const float2 &a, const float4 &x, float4 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
      y.z += a.x*x.z; y.z -= a.y*x.w;
      y.w += a.y*x.z; y.w += a.x*x.w;
    }

    __device__ __host__ inline void _caxpy(const float2 &a, const float2 &x, float2 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
    }

    __device__ __host__ inline void _caxpy(const double2 &a, const double2 &x, double2 &y) {
      y.x += a.x*x.x; y.x -= a.y*x.y;
      y.y += a.y*x.x; y.y += a.x*x.y;
    }

    template<int NXZ, typename Float2, typename FloatN>
    struct multicaxpy_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      const int NYW;
      // ignore parameter arrays since we place them in constant memory
      multicaxpy_(const coeff_array<Complex> &a, const coeff_array<Complex> &b,
		  const coeff_array<Complex> &c, int NYW) : NYW(NYW) { }

      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
	Float2 *a = reinterpret_cast<Float2*>(Amatrix_d); // fetch coefficient matrix from constant memory
	_caxpy(a[MAX_MULTI_BLAS_N*j+i], x, y);
#else
	Float2 *a = reinterpret_cast<Float2*>(Amatrix_h);
	_caxpy(a[NYW*j+i], x, y);
#endif
      }

      int streams() { return 2*NYW + NXZ*NYW; } //! total number of input and output streams
      int flops() { return 4*NXZ*NYW; } //! flops per real element
    };

    void caxpy(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {

      // mark true since we will copy the "a" matrix into constant memory
      coeff_array<Complex> a(a_, true), b, c;

      switch (x.size()) {
      case 1:
	multiblasCuda<1,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 2:
	multiblasCuda<2,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 3:
	multiblasCuda<3,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 4:
	multiblasCuda<4,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 5:
	multiblasCuda<5,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 6:
	multiblasCuda<6,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 7:
	multiblasCuda<7,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 8:
	multiblasCuda<8,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 9:
	multiblasCuda<9,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 10:
	multiblasCuda<10,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 11:
	multiblasCuda<11,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 12:
	multiblasCuda<12,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 13:
	multiblasCuda<13,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 14:
	multiblasCuda<14,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 15:
	multiblasCuda<15,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      case 16:
	multiblasCuda<16,multicaxpy_,0,1,0,0>(a, b, c, x, y, x, y);
        break;
      default:
	// split the problem in half and recurse
	const Complex *a0 = &a_[0];
	const Complex *a1 = &a_[x.size()*y.size()/2];

	std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
	std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

	caxpy(a0, x0, y);
	caxpy(a1, x1, y);
      }
    }

    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy(a, x.Components(), y.Components()); }


    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template<int NXZ, typename Float2, typename FloatN>
    struct multi_axpyBzpcx_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      real a[MAX_MULTI_BLAS_N], b[MAX_MULTI_BLAS_N], c[MAX_MULTI_BLAS_N];

      multi_axpyBzpcx_(const coeff_array<double> &a, const coeff_array<double> &b, const coeff_array<double> &c, int NYW) : NYW(NYW){
	// copy arguments into the functor
	for (int i=0; i<NYW; i++) { this->a[i] = a.data[i]; this->b[i] = b.data[i]; this->c[i] = c.data[i]; }
      }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
	y += a[i] * w;
	w = b[i] * x + c[i] * w;
      }
      int streams() { return 4*NYW + NXZ; } //! total number of input and output streams
      int flops() { return 5*NXZ*NYW; } //! flops per real element
    };

    void axpyBzpcx(const double *a_, std::vector<ColorSpinorField*> &x_, std::vector<ColorSpinorField*> &y_,
		   const double *b_, ColorSpinorField &z_, const double *c_) {

      if (y_.size() <= MAX_MULTI_BLAS_N) {
	// swizzle order since we are writing to x_ and y_, but the
	// multi-blas only allow writing to y and w, and moreover the
	// block width of y and w must match, and x and z must match.
	std::vector<ColorSpinorField*> &y = y_;
	std::vector<ColorSpinorField*> &w = x_;

	// wrap a container around the third solo vector
	std::vector<ColorSpinorField*> x;
	x.push_back(&z_);

	// we will curry the parameter arrays into the functor
	coeff_array<double> a(a_,false), b(b_,false), c(c_,false);

	if (x[0]->Precision() != y[0]->Precision() ) {
	  mixed::multiblasCuda<1,multi_axpyBzpcx_,0,1,0,1>(a, b, c, x, y, x, w);
	} else {
	  multiblasCuda<1,multi_axpyBzpcx_,0,1,0,1>(a, b, c, x, y, x, w);
	}
      } else {
	// split the problem in half and recurse
	const double *a0 = &a_[0];
	const double *b0 = &b_[0];
	const double *c0 = &c_[0];

	std::vector<ColorSpinorField*> x0(x_.begin(), x_.begin() + x_.size()/2);
	std::vector<ColorSpinorField*> y0(y_.begin(), y_.begin() + y_.size()/2);

	axpyBzpcx(a0, x0, y0, b0, z_, c0);

	const double *a1 = &a_[y_.size()/2];
	const double *b1 = &b_[y_.size()/2];
	const double *c1 = &c_[y_.size()/2];

	std::vector<ColorSpinorField*> x1(x_.begin() + x_.size()/2, x_.end());
	std::vector<ColorSpinorField*> y1(y_.begin() + y_.size()/2, y_.end());

	axpyBzpcx(a1, x1, y1, b1, z_, c1);
      }
    }
    
    
    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template<int NXZ, typename Float2, typename FloatN>
    struct multi_caxpyBxpz_ : public MultiBlasFunctor<NXZ, Float2, FloatN>
    {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      Float2 a[MAX_MULTI_BLAS_N], b[MAX_MULTI_BLAS_N], c[MAX_MULTI_BLAS_N];

      multi_caxpyBxpz_(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) : NYW(NYW)
      {
        // copy arguments into the functor
        for (int i=0; i<NXZ; i++)
        {
          this->a[i] = make_Float2<Float2>(a.data[i]);
          this->b[i] = make_Float2<Float2>(b.data[i]);
        }
      }
      
      // i loops over NYW, j loops over NXZ
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        _caxpy(a[j], x, y); _caxpy(b[j], x, w); // b/c we swizzled z into w.
      }
      int streams() { return 4*NYW + NXZ; } //! total number of input and output streams
      int flops() { return 8*NXZ*NYW; } //! flops per real element
    };

    void caxpyBxpz(const Complex *a_, std::vector<ColorSpinorField*> &x_, ColorSpinorField &y_,
		   const Complex *b_, ColorSpinorField &z_)
    {

      const int xsize = x_.size();
      if (xsize <= 10) // only swizzle if we have to. BiCGstab-10 is as far as I ever want to push.
      {
        // swizzle order since we are writing to y_ and z_, but the
        // multi-blas only allow writing to y and w, and moreover the
        // block width of y and w must match, and x and z must match.
        // Also, wrap a container around them.
        std::vector<ColorSpinorField*> y;
        y.push_back(&y_);
        std::vector<ColorSpinorField*> w;
        w.push_back(&z_);

        // we're reading from x
        std::vector<ColorSpinorField*> &x = x_;

        // we will carry the parameter arrays into the functor
        coeff_array<Complex> a(a_,false), b(b_,false), c; 

        if (x[0]->Precision() != y[0]->Precision() )
        {
          switch(xsize)
          {
            case 1:
              mixed::multiblasCuda<1,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 2:
              mixed::multiblasCuda<2,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 3:
              mixed::multiblasCuda<3,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 4:
              mixed::multiblasCuda<4,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 5:
              mixed::multiblasCuda<5,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 6:
              mixed::multiblasCuda<6,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 7:
              mixed::multiblasCuda<7,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 8:
              mixed::multiblasCuda<8,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 9:
              mixed::multiblasCuda<9,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 10:
              mixed::multiblasCuda<10,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            default:
              // we can't hit the default, it ends up in the else below.
              break;
          }
        }
        else
        {
          switch(xsize)
          {
            case 1:
              multiblasCuda<1,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 2:
              multiblasCuda<2,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 3:
              multiblasCuda<3,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 4:
              multiblasCuda<4,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 5:
              multiblasCuda<5,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 6:
              multiblasCuda<6,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 7:
              multiblasCuda<7,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 8:
              multiblasCuda<8,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 9:
              multiblasCuda<9,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            case 10:
              multiblasCuda<10,multi_caxpyBxpz_,0,1,0,1>(a, b, c, x, y, x, w);
              break;
            default:
              // we can't hit the default, it ends up in the else below.
              break;
          } 
        }
      }
      else
      {
        // split the problem in half and recurse
        const Complex *a0 = &a_[0];
        const Complex *b0 = &b_[0];

        std::vector<ColorSpinorField*> x0(x_.begin(), x_.begin() + x_.size()/2);

        caxpyBxpz(a0, x0, y_, b0, z_);

        const Complex *a1 = &a_[x_.size()/2];
        const Complex *b1 = &b_[x_.size()/2];

        std::vector<ColorSpinorField*> x1(x_.begin() + x_.size()/2, x_.end());

        caxpyBxpz(a1, x1, y_, b1, z_);
      }
    }


  } // namespace blas

} // namespace quda
