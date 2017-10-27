#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset
#include <typeinfo>

#include <tune_quda.h>
#include <quda_internal.h>
#include <float_vector.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

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

    template <int writeX, int writeY, int writeZ, int writeW>
    struct write {
      static constexpr int X = writeX;
      static constexpr int Y = writeY;
      static constexpr int Z = writeZ;
      static constexpr int W = writeW;
    };

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
		  const coeff_array<Complex> &c, int NYW) : NYW(NYW)
      { }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
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


    void caxpy_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
          int i_idx ,int j_idx, int upper) {

      if (y.size() > MAX_MULTI_BLAS_N) // if greater than max single-kernel size, recurse.
      {
        // We need to split up 'a' carefully since it's row-major.
        Complex* tmpmajor = new Complex[x.size()*y.size()];
        Complex* tmpmajor0 = &tmpmajor[0];
        Complex* tmpmajor1 = &tmpmajor[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());

        const unsigned int xlen = x.size();
        const unsigned int ylen0 = y.size()/2;
        const unsigned int ylen1 = y.size() - y.size()/2;
        
        int count = 0, count0 = 0, count1 = 0;
        for (unsigned int i = 0; i < xlen; i++)
        {
          for (unsigned int j = 0; j < ylen0; j++)
            tmpmajor0[count0++] = a_[count++];
          for (unsigned int j = 0; j < ylen1; j++)
            tmpmajor1[count1++] = a_[count++];
        }

        caxpy_recurse(tmpmajor0, x, y0, i_idx, 2*j_idx+0, upper);
        caxpy_recurse(tmpmajor1, x, y1, i_idx, 2*j_idx+1, upper);

        delete[] tmpmajor;
      }
      else
      {
        // if at the bottom of recursion,
        // return if on lower left for upper triangular,
        // return if on upper right for lower triangular. 
        if (x.size() <= MAX_MULTI_BLAS_N) {
          if (upper == 1 && j_idx < i_idx) { return; } 
          if (upper == -1 && j_idx > i_idx) { return; }
        }

        // mark true since we will copy the "a" matrix into constant memory
        coeff_array<Complex> a(a_, true), b, c;

        if (x[0]->Precision() == y[0]->Precision())
        {
          switch (x.size()) {
            case 1:
              multiblasCuda<1,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 2
            case 2:
              multiblasCuda<2,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 3
            case 3:
              multiblasCuda<3,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 4
            case 4:
              multiblasCuda<4,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 5
            case 5:
              multiblasCuda<5,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 6
            case 6:
              multiblasCuda<6,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 7
            case 7:
              multiblasCuda<7,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 8
            case 8:
              multiblasCuda<8,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 9
            case 9:
              multiblasCuda<9,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 10
            case 10:
              multiblasCuda<10,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 11
            case 11:
              multiblasCuda<11,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 12
            case 12:
              multiblasCuda<12,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 13
            case 13:
              multiblasCuda<13,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 14
            case 14:
              multiblasCuda<14,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 15
            case 15:
              multiblasCuda<15,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 16
            case 16:
              multiblasCuda<16,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpy_recurse(a0, x0, y, 2*i_idx+0, j_idx, upper);
              caxpy_recurse(a1, x1, y, 2*i_idx+1, j_idx, upper);
              break;
          }
        }
        else // precisions don't agree.
        {
          switch (x.size()) {
            case 1:
              mixed::multiblasCuda<1,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 2
            case 2:
              mixed::multiblasCuda<2,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 3
            case 3:
              mixed::multiblasCuda<3,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 4
            case 4:
              mixed::multiblasCuda<4,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 5
            case 5:
              mixed::multiblasCuda<5,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 6
            case 6:
              mixed::multiblasCuda<6,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 7
            case 7:
              mixed::multiblasCuda<7,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 8
            case 8:
              mixed::multiblasCuda<8,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 9
            case 9:
              mixed::multiblasCuda<9,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 10
            case 10:
              mixed::multiblasCuda<10,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 11
            case 11:
              mixed::multiblasCuda<11,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 12
            case 12:
              mixed::multiblasCuda<12,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 13
            case 13:
              mixed::multiblasCuda<13,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 14
            case 14:
              mixed::multiblasCuda<14,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 15
            case 15:
              mixed::multiblasCuda<15,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #if MAX_MULTI_BLAS_N >= 16
            case 16:
              mixed::multiblasCuda<16,multicaxpy_,write<0,1,0,0> >(a, b, c, x, y, x, y);
              break;
  #endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpy_recurse(a0, x0, y, 2*i_idx+0, j_idx, upper);
              caxpy_recurse(a1, x1, y, 2*i_idx+1, j_idx, upper);
              break;
          }
        }
      } // end if (y.size() > MAX_MULTI_BLAS_N)
    }

    void caxpy(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion. 
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      caxpy_recurse(a_, x, y, 0, 0, 0);
    }

    void caxpy_U(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion. 
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles. 
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_U with non-square 'a' has not yet been implemented. Use block caxpy instead.\n");
        return; 
      }
      caxpy_recurse(a_, x, y, 0, 0, 1);
    }

    void caxpy_L(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion. 
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles. 
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_L with non-square 'a' has not yet been implemented. Use block caxpy instead.\n");
        return; 
      }
      caxpy_recurse(a_, x, y, 0, 0, -1);
    }


    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy(a, x.Components(), y.Components()); }

    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_U(a, x.Components(), y.Components()); }

    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_L(a, x.Components(), y.Components()); }

    /**
       Functor to perform the operation z = a * x + y  (complex-valued)
    */
    template<int NXZ, typename Float2, typename FloatN>
    struct multicaxpyz_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      const int NYW;
      // ignore parameter arrays since we place them in constant memory
      multicaxpyz_(const coeff_array<Complex> &a, const coeff_array<Complex> &b,
      const coeff_array<Complex> &c, int NYW) : NYW(NYW)
      { }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
        Float2 *a = reinterpret_cast<Float2*>(Amatrix_d); // fetch coefficient matrix from constant memory
        if (j==0) w = y;
        _caxpy(a[MAX_MULTI_BLAS_N*j+i], x, w);
#else
        Float2 *a = reinterpret_cast<Float2*>(Amatrix_h);
        if (j==0) w = y;
        _caxpy(a[NYW*j+i], x, w);
#endif
      }

      int streams() { return 2*NYW + NXZ*NYW; } //! total number of input and output streams
      int flops() { return 4*NXZ*NYW; } //! flops per real element
    };

    void caxpyz_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z, int i, int j, int pass, int upper) {

      if (y.size() > MAX_MULTI_BLAS_N) // if greater than max single-kernel size, recurse.
      {
        // We need to split up 'a' carefully since it's row-major.
        Complex* tmpmajor = new Complex[x.size()*y.size()];
        Complex* tmpmajor0 = &tmpmajor[0];
        Complex* tmpmajor1 = &tmpmajor[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());

        std::vector<ColorSpinorField*> z0(z.begin(), z.begin() + z.size()/2);
        std::vector<ColorSpinorField*> z1(z.begin() + z.size()/2, z.end());

        const unsigned int xlen = x.size();
        const unsigned int ylen0 = y.size()/2;
        const unsigned int ylen1 = y.size() - y.size()/2;
        
        int count = 0, count0 = 0, count1 = 0;
        for (unsigned int i_ = 0; i_ < xlen; i_++)
        {
          for (unsigned int j = 0; j < ylen0; j++)
            tmpmajor0[count0++] = a_[count++];
          for (unsigned int j = 0; j < ylen1; j++)
            tmpmajor1[count1++] = a_[count++];
        }

        caxpyz_recurse(tmpmajor0, x, y0, z0, i, 2*j+0, pass, upper);
        caxpyz_recurse(tmpmajor1, x, y1, z1, i, 2*j+1, pass, upper);

        delete[] tmpmajor;
      }
      else
      {
      	// if at bottom of recursion check where we are
      	if (x.size() <= MAX_MULTI_BLAS_N) {
      	  if (pass==1) {
      	    if (i!=j)
            {
              if (upper == 1 && j < i) { return; } // upper right, don't need to update lower left.
              if (upper == -1 && i < j) { return; } // lower left, don't need to update upper right.
              caxpy(a_, x, z); return;  // off diagonal
            }
      	    return;
      	  } else {
      	    if (i!=j) return; // We're on the first pass, so we only want to update the diagonal.
      	  }
      	}

        // mark true since we will copy the "a" matrix into constant memory
        coeff_array<Complex> a(a_, true), b, c;

        if (x[0]->Precision() == y[0]->Precision())
        {
          switch (x.size()) {
            case 1:
              multiblasCuda<1,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 2
            case 2:
              multiblasCuda<2,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 3
            case 3:
              multiblasCuda<3,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 4
            case 4:
              multiblasCuda<4,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 5
            case 5:
              multiblasCuda<5,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 6
            case 6:
              multiblasCuda<6,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 7
            case 7:
              multiblasCuda<7,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 8
            case 8:
              multiblasCuda<8,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 9
            case 9:
              multiblasCuda<9,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 10
            case 10:
              multiblasCuda<10,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 11
            case 11:
              multiblasCuda<11,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 12
            case 12:
              multiblasCuda<12,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 13
            case 13:
              multiblasCuda<13,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 14
            case 14:
              multiblasCuda<14,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 15
            case 15:
              multiblasCuda<15,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 16
            case 16:
              multiblasCuda<16,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpyz_recurse(a0, x0, y, z, 2*i+0, j, pass, upper);
              caxpyz_recurse(a1, x1, y, z, 2*i+1, j, pass, upper); // b/c we don't want to re-zero z.
              break;
          }
        }
        else // precisions don't agree.
        {
          switch (x.size()) {
            case 1:
              mixed::multiblasCuda<1,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 2
            case 2:
              mixed::multiblasCuda<2,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 3
            case 3:
              mixed::multiblasCuda<3,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 4
            case 4:
              mixed::multiblasCuda<4,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 5
            case 5:
              mixed::multiblasCuda<5,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 6
            case 6:
              mixed::multiblasCuda<6,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 7
            case 7:
              mixed::multiblasCuda<7,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 8
            case 8:
              mixed::multiblasCuda<8,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 9
            case 9:
              mixed::multiblasCuda<9,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 10
            case 10:
              mixed::multiblasCuda<10,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 11
            case 11:
              mixed::multiblasCuda<11,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 12
            case 12:
              mixed::multiblasCuda<12,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 13
            case 13:
              mixed::multiblasCuda<13,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 14
            case 14:
              mixed::multiblasCuda<14,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 15
            case 15:
              mixed::multiblasCuda<15,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #if MAX_MULTI_BLAS_N >= 16
            case 16:
              mixed::multiblasCuda<16,multicaxpyz_,write<0,0,0,1> >(a, b, c, x, y, x, z);
              break;
  #endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpyz_recurse(a0, x0, y, z, 2*i+0, j, pass, upper);
              caxpyz_recurse(a1, x1, y, z, 2*i+1, j, pass, upper);
              break;
          }
        }
      } // end if (y.size() > MAX_MULTI_BLAS_N)
    }

    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, 0);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, 0);
    }

    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, 1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, 1);
    }

    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, -1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, -1);
    }


    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz_U(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz_L(a, x.Components(), y.Components(), z.Components());
    }

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
      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
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
	  mixed::multiblasCuda<1,multi_axpyBzpcx_,write<0,1,0,1> >(a, b, c, x, y, x, w);
	} else {
	  multiblasCuda<1,multi_axpyBzpcx_,write<0,1,0,1> >(a, b, c, x, y, x, w);
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

      multi_caxpyBxpz_(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) : NYW(NYW)
      { }
      
      // i loops over NYW, j loops over NXZ
      __device__ __host__  inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
	Float2 *a = reinterpret_cast<Float2*>(Amatrix_d); // fetch coefficient matrix from constant memory
	Float2 *b = reinterpret_cast<Float2*>(Bmatrix_d); // fetch coefficient matrix from constant memory
        _caxpy(a[MAX_MULTI_BLAS_N*j], x, y); _caxpy(b[MAX_MULTI_BLAS_N*j], x, w); // b/c we swizzled z into w.
#else
	Float2 *a = reinterpret_cast<Float2*>(Amatrix_h);
	Float2 *b = reinterpret_cast<Float2*>(Bmatrix_h);
        _caxpy(a[j], x, y); _caxpy(b[j], x, w); // b/c we swizzled z into w.
#endif
      }
      int streams() { return 4*NYW + NXZ; } //! total number of input and output streams
      int flops() { return 8*NXZ*NYW; } //! flops per real element
    };

    void caxpyBxpz(const Complex *a_, std::vector<ColorSpinorField*> &x_, ColorSpinorField &y_,
		   const Complex *b_, ColorSpinorField &z_)
    {

      const int xsize = x_.size();
      if (xsize <= MAX_MULTI_BLAS_N) // only swizzle if we have to. 
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

        // put a and b into constant space
        coeff_array<Complex> a(a_,true), b(b_,true), c;

        if (x[0]->Precision() != y[0]->Precision() )
        {
          switch(xsize)
          {
            case 1:
              mixed::multiblasCuda<1,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 2
            case 2:
              mixed::multiblasCuda<2,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 3
            case 3:
              mixed::multiblasCuda<3,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 4
            case 4:
              mixed::multiblasCuda<4,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 5
            case 5:
              mixed::multiblasCuda<5,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 6
            case 6:
              mixed::multiblasCuda<6,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 7
            case 7:
              mixed::multiblasCuda<7,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 8
            case 8:
              mixed::multiblasCuda<8,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 9
            case 9:
              mixed::multiblasCuda<9,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 10
            case 10:
              mixed::multiblasCuda<10,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 11
            case 11:
              mixed::multiblasCuda<11,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 12
            case 12:
              mixed::multiblasCuda<12,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 13
            case 13:
              mixed::multiblasCuda<13,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 14
            case 14:
              mixed::multiblasCuda<14,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 15
            case 15:
              mixed::multiblasCuda<15,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 16
            case 16:
              mixed::multiblasCuda<16,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#endif // 16
#endif // 15
#endif // 14
#endif // 13
#endif // 12
#endif // 11
#endif // 10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
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
              multiblasCuda<1,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 2
            case 2:
              multiblasCuda<2,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 3
            case 3:
              multiblasCuda<3,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 4
            case 4:
              multiblasCuda<4,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 5
            case 5:
              multiblasCuda<5,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 6
            case 6:
              multiblasCuda<6,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 7
            case 7:
              multiblasCuda<7,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 8
            case 8:
              multiblasCuda<8,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 9
            case 9:
              multiblasCuda<9,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 10
            case 10:
              multiblasCuda<10,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 11
            case 11:
              multiblasCuda<11,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 12
            case 12:
              multiblasCuda<12,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 13
            case 13:
              multiblasCuda<13,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 14
            case 14:
              multiblasCuda<14,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 15
            case 15:
              multiblasCuda<15,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#if MAX_MULTI_BLAS_N >= 16
            case 16:
              multiblasCuda<16,multi_caxpyBxpz_,write<0,1,0,1> >(a, b, c, x, y, x, w);
              break;
#endif // 16
#endif // 15
#endif // 14
#endif // 13
#endif // 12
#endif // 11
#endif // 10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
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
