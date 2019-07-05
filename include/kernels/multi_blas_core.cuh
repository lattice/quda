#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>

#if CUDA_VERSION < 9000
#define CONSTANT_ARG
#endif

namespace quda
{

  namespace blas
  {

#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>

    // storage for matrix coefficients
#define MAX_MATRIX_SIZE 8192
#define MAX_ARG_SIZE 4096
    static __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

    static signed char *Amatrix_h;
    static signed char *Bmatrix_h;
    static signed char *Cmatrix_h;

#ifdef CONSTANT_ARG
    // as a performance work around we put the argument struct into
    // __constant__ memory to prevent the compiler from spilling
    // registers on older CUDA
    static __constant__ signed char arg_buffer[MAX_ARG_SIZE];
#endif

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <int NXZ, typename RegType> inline constexpr int max_YW_size()
    {
      // the size of the accessor doesn't change with precision just instantiate some precision
      using SpinorX = SpinorTexture<float4,short4,6>;
      using SpinorY = Spinor<float4,short4,6,1>;
      using SpinorZ = SpinorTexture<float4,short4,6>;
      using SpinorW = Spinor<float4,short4,6,1>;

      // compute the size remaining for the Y and W accessors
      constexpr int arg_size = (MAX_ARG_SIZE
                                - sizeof(int)          // NYW parameter
                                - sizeof(SpinorX[NXZ]) // SpinorX array
                                - sizeof(SpinorZ[NXZ]) // SpinorW array
                                - sizeof(int)          // functor NYW member
                                - sizeof(int) - 16     // length parameter
                                - 16)                  // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + sizeof(SpinorW));

      // this is the maximum size limit imposed by the coefficient arrays
      constexpr int coeff_size = MAX_MATRIX_SIZE / (NXZ * sizeof(RegType));

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    inline int max_YW_size(int NXZ, int precision)
    {
      // ensure NXZ is a valid size
      NXZ = std::min(NXZ, MAX_MULTI_BLAS_N);

      // the size of the accessor doesn't change with precision just instantiate some precision
      using SpinorX = SpinorTexture<float4,short4,6>;
      using SpinorY = Spinor<float4,short4,6,1>;
      using SpinorZ = SpinorTexture<float4,short4,6>;
      using SpinorW = Spinor<float4,short4,6,1>;

      // compute the size remaining for the Y and W accessors
      int arg_size = (MAX_ARG_SIZE
                      - sizeof(int)         // NYW parameter
                      - NXZ*sizeof(SpinorX) // SpinorX array
                      - NXZ*sizeof(SpinorZ) // SpinorW array
                      - sizeof(int)         // functor NYW member
                      - sizeof(int) - 16     // length parameter
                      - 16)                  // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + sizeof(SpinorW));

      int coeff_size = MAX_MATRIX_SIZE / (NXZ * precision);

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Parameter struct for generic multi-blas kernel.
       @tparam NXZ is dimension of input vectors: X,Z
       @tparam NYW is dimension of in-output vectors: Y,W
       @tparam SpinorX Type of input spinor for x argument
       @tparam SpinorY Type of input spinor for y argument
       @tparam SpinorZ Type of input spinor for z argument
       @tparam SpinorW Type of input spinor for w argument
       @tparam Functor Functor used to operate on data
    */
    template <int NXZ, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Functor>
    struct MultiBlasArg {
      static constexpr int NYW_max = max_YW_size<NXZ, typename Functor::type>();
      const int NYW;
      SpinorX X[NXZ];
      SpinorY Y[NYW_max];
      SpinorZ Z[NXZ];
      SpinorW W[NYW_max];
      Functor f;
      const int length;

      MultiBlasArg(SpinorX X[NXZ], SpinorY Y[], SpinorZ Z[NXZ], SpinorW W[], Functor f, int NYW, int length) :
          NYW(NYW),
          f(f),
          length(length)
      {
	if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i] = X[i];
          this->Z[i] = Z[i];
        }
        for (int i = 0; i < NYW; ++i) {
          this->Y[i] = Y[i];
          this->W[i] = W[i];
        }
      }
    };

    /**
       @brief Generic multi-blas kernel with four loads and up to four stores.
       @param[in,out] arg Argument struct with required meta data
       (input/output fields, functor, etc.)
    */
#ifndef CONSTANT_ARG
    template <typename FloatN, int M, int NXZ, typename Arg> __global__ void multiBlasKernel(Arg arg)
    {
#else
    template <typename FloatN, int M, int NXZ, typename Arg> __global__ void multiBlasKernel()
    {
      Arg &arg = *((Arg *)arg_buffer);
#endif

      // use i to loop over elements in kernel
      unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
      unsigned int parity = blockIdx.z;

      if (k >= arg.NYW) return;

      while (idx < arg.length) {

        FloatN x[M], y[M], z[M], w[M];
        arg.Y[k].load(y, idx, parity);
        arg.W[k].load(w, idx, parity);

#pragma unroll
        for (int l = 0; l < NXZ; l++) {
          arg.X[l].load(x, idx, parity);
          arg.Z[l].load(z, idx, parity);

#pragma unroll
          for (int j = 0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], k, l);
        }
        arg.Y[k].save(y, idx, parity);
        arg.W[k].save(w, idx, parity);

        idx += gridDim.x * blockDim.x;
      }
    }

    template <typename T> struct coeff_array {
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };

    template <int NXZ, typename Float2, typename FloatN> struct MultiBlasFunctor
    {
      typedef Float2 type;
      int NYW;
      MultiBlasFunctor(int NYW) : NYW(NYW) { }

      __device__ __host__ inline Float2 a(int i, int j) const {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Amatrix_d)[i*NYW + j];
#else
        return reinterpret_cast<Float2 *>(Amatrix_h)[i*NYW + j];
#endif
      }

      __device__ __host__ inline Float2 b(int i, int j) const {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Amatrix_d)[i*NYW + j];
#else
        return reinterpret_cast<Float2 *>(Amatrix_h)[i*NYW + j];
#endif
      }

      __device__ __host__ inline Float2 c(int i, int j) const {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Amatrix_d)[i*NYW + j];
#else
        return reinterpret_cast<Float2 *>(Amatrix_h)[i*NYW + j];
#endif
      }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
          = 0;
    };

    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */

    __device__ __host__ inline void _caxpy(const float2 &a, const float4 &x, float4 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
      y.z += a.x * x.z;
      y.z -= a.y * x.w;
      y.w += a.y * x.z;
      y.w += a.x * x.w;
    }

    __device__ __host__ inline void _caxpy(const float2 &a, const float2 &x, float2 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    __device__ __host__ inline void _caxpy(const double2 &a, const double2 &x, double2 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    template <int NXZ, typename Float2, typename FloatN>
    struct multicaxpy_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      using MultiBlasFunctor<NXZ, Float2, FloatN>::NYW;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::a;
      multicaxpy_(int NYW) : MultiBlasFunctor<NXZ, Float2, FloatN>(NYW)
      {
      }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        _caxpy(a(j,i), x, y);
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor to perform the operation z = a * x + y  (complex-valued)
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multicaxpyz_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      using MultiBlasFunctor<NXZ, Float2, FloatN>::NYW;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::a;
      multicaxpyz_(int NYW) : MultiBlasFunctor<NXZ, Float2, FloatN>(NYW)
      {
      }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        if (j == 0) w = y;
        _caxpy(a(j,i), x, w);
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multi_axpyBzpcx_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      using MultiBlasFunctor<NXZ, Float2, FloatN>::NYW;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::a;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::b;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::c;
      multi_axpyBzpcx_(int NYW) : MultiBlasFunctor<NXZ, Float2, FloatN>(NYW)
      {
      }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        y += a(0,i).x * w;
        w = b(0,i).x * x + c(0,i).x * w;
      }

      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 5 * NXZ * NYW; }   //! flops per real element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multi_caxpyBxpz_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      using MultiBlasFunctor<NXZ, Float2, FloatN>::NYW;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::a;
      using MultiBlasFunctor<NXZ, Float2, FloatN>::b;
      multi_caxpyBxpz_(int NYW) : MultiBlasFunctor<NXZ, Float2, FloatN>(NYW)
      {
      }

      // i loops over NYW, j loops over NXZ
      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        _caxpy(a(0,j), x, y);
        _caxpy(b(0,j), x, w); // b/c we swizzled z into w.
      }

      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 8 * NXZ * NYW; }   //! flops per real element
    };

  } // namespace blas

} // namespace quda
