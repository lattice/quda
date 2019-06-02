#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>

namespace quda
{

  namespace blas
  {

#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>

    // storage for matrix coefficients
#define MAX_MATRIX_SIZE 4096
    static __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

    static signed char *Amatrix_h;
    static signed char *Bmatrix_h;
    static signed char *Cmatrix_h;

#if CUDA_VERSION < 9000
    // as a performance work around we put the argument struct into
    // __constant__ memory to prevent the compiler from spilling
    // registers on older CUDA
    static __constant__ signed char arg_buffer[MAX_MATRIX_SIZE];
#endif

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
      const int NYW;
      SpinorX X[NXZ];
      SpinorY Y[MAX_MULTI_BLAS_N];
      SpinorZ Z[NXZ];
      SpinorW W[MAX_MULTI_BLAS_N];
      Functor f;
      const int length;

      MultiBlasArg(SpinorX X[NXZ], SpinorY Y[], SpinorZ Z[NXZ], SpinorW W[], Functor f, int NYW, int length) :
          NYW(NYW),
          f(f),
          length(length)
      {
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
    template <typename FloatN, int M, int NXZ, typename Arg> __global__ void multiBlasKernel(Arg arg_)
    {
#if CUDA_VERSION >= 9000
      Arg &arg = arg_;
#else
      Arg &arg = *((Arg *)arg_buffer);
#endif

      // use i to loop over elements in kernel
      unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
      unsigned int parity = blockIdx.z;

      arg.f.init();
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
      const bool use_const;
      coeff_array() : data(nullptr), use_const(false) {}
      coeff_array(const T *data, bool use_const) : data(data), use_const(use_const) {}
    };

    template <int NXZ, typename Float2, typename FloatN> struct MultiBlasFunctor {

      //! pre-computation routine before the main loop
      virtual __device__ __host__ void init() { ; }

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
      const int NYW;
      // ignore parameter arrays since we place them in constant memory
      multicaxpy_(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
          NYW(NYW)
      {
      }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_d); // fetch coefficient matrix from constant memory
        _caxpy(a[MAX_MULTI_BLAS_N * j + i], x, y);
#else
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_h);
        _caxpy(a[NYW * j + i], x, y);
#endif
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor to perform the operation z = a * x + y  (complex-valued)
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multicaxpyz_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      const int NYW;
      // ignore parameter arrays since we place them in constant memory
      multicaxpyz_(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
          NYW(NYW)
      {
      }

      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_d); // fetch coefficient matrix from constant memory
        if (j == 0) w = y;
        _caxpy(a[MAX_MULTI_BLAS_N * j + i], x, w);
#else
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_h);
        if (j == 0) w = y;
        _caxpy(a[NYW * j + i], x, w);
#endif
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multi_axpyBzpcx_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;
      real a[MAX_MULTI_BLAS_N], b[MAX_MULTI_BLAS_N], c[MAX_MULTI_BLAS_N];

      multi_axpyBzpcx_(const coeff_array<double> &a, const coeff_array<double> &b, const coeff_array<double> &c, int NYW) :
          NYW(NYW),
          a {},
          b {},
          c {}
      {
        // copy arguments into the functor
        for (int i = 0; i < NYW; i++) {
          this->a[i] = a.data[i];
          this->b[i] = b.data[i];
          this->c[i] = c.data[i];
        }
      }
      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        y += a[i] * w;
        w = b[i] * x + c[i] * w;
      }
      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 5 * NXZ * NYW; }   //! flops per real element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <int NXZ, typename Float2, typename FloatN>
    struct multi_caxpyBxpz_ : public MultiBlasFunctor<NXZ, Float2, FloatN> {
      typedef typename scalar<Float2>::type real;
      const int NYW;

      multi_caxpyBxpz_(
          const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
          NYW(NYW)
      {
      }

      // i loops over NYW, j loops over NXZ
      __device__ __host__ inline void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
#ifdef __CUDA_ARCH__
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_d); // fetch coefficient matrix from constant memory
        Float2 *b = reinterpret_cast<Float2 *>(Bmatrix_d); // fetch coefficient matrix from constant memory
        _caxpy(a[MAX_MULTI_BLAS_N * j], x, y);
        _caxpy(b[MAX_MULTI_BLAS_N * j], x, w); // b/c we swizzled z into w.
#else
        Float2 *a = reinterpret_cast<Float2 *>(Amatrix_h);
        Float2 *b = reinterpret_cast<Float2 *>(Bmatrix_h);
        _caxpy(a[j], x, y);
        _caxpy(b[j], x, w); // b/c we swizzled z into w.
#endif
      }
      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 8 * NXZ * NYW; }   //! flops per real element
    };

  } // namespace blas

} // namespace quda
