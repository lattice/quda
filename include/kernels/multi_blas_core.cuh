#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>

#include <multi_blas_helper.cuh>
#include <float_vector.h>

#if (__COMPUTE_CAPABILITY__ >= 300 || __CUDA_ARCH__ >= 300) && !defined(QUDA_FAST_COMPILE_REDUCE)
#define WARP_SPLIT
#include <generics/shfl.h>
#endif

namespace quda
{

  namespace blas
  {

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
    struct MultiBlasArg
      : SpinorXZ<NXZ, SpinorX, SpinorZ, Functor::use_z>,
        SpinorYW<max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor>(), SpinorY, SpinorW, Functor::use_w> {
      static constexpr int NYW_max = max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor>();
      const int NYW;
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
          if (Functor::use_z) this->Z[i] = Z[i];
        }
        for (int i = 0; i < NYW; ++i) {
          this->Y[i] = Y[i];
          if (Functor::use_w) this->W[i] = W[i];
        }
      }
    };

    template <int M, int warp_split, typename FloatN> __device__ __host__ void warp_combine(FloatN x[M])
    {
#ifdef WARP_SPLIT
      constexpr int warp_size = 32;
      if (warp_split > 1) {
#pragma unroll
        for (int j = 0; j < M; j++) {
          // reduce down to the first group of column-split threads
#pragma unroll
          for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
#define WARP_CONVERGED 0xffffffff // we know warp should be converged here
            x[j] += __shfl_down_sync(WARP_CONVERGED, x[j], offset);
          }
        }
      }

#endif // WARP_SPLIT
    }

    /**
       @brief Generic multi-blas kernel with four loads and up to four stores.
       @param[in,out] arg Argument struct with required meta data
       (input/output fields, functor, etc.)
    */
    template <typename FloatN, int M, int NXZ, int warp_split, typename Arg> __global__ void multiBlasKernel(Arg arg)
    {
      // use i to loop over elements in kernel
      const int k = blockIdx.y * blockDim.y + threadIdx.y;
      const int parity = blockIdx.z;

      // partition the warp between grid points and the NXZ update
      constexpr int warp_size = 32;
      constexpr int vector_site_width = warp_size / warp_split;
      const int lane_id = threadIdx.x % warp_size;
      const int warp_id = threadIdx.x / warp_size;
      unsigned int idx
        = blockIdx.x * (blockDim.x / warp_split) + warp_id * (warp_size / warp_split) + lane_id % vector_site_width;
      const int l_idx = lane_id / vector_site_width;

      if (k >= arg.NYW) return;

      while (idx < arg.length) {

        FloatN x[M], y[M], z[M], w[M];

#pragma unroll
        for (int m = 0; m < M; m++) {
          ::quda::zero(y[m]);
          ::quda::zero(w[m]);
        }

        if (l_idx == 0 || warp_split == 1) {
          arg.Y[k].load(y, idx, parity);
          arg.W[k].load(w, idx, parity);
        }

#pragma unroll
        for (int l_ = 0; l_ < NXZ; l_ += warp_split) {
          const int l = l_ + l_idx;
          if (l < NXZ || warp_split == 1) {
            arg.X[l].load(x, idx, parity);
            arg.Z[l].load(z, idx, parity);

#pragma unroll
            for (int j = 0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], k, l);
          }
        }

        // now combine the results across the warp if needed
        if (arg.f.write.Y) warp_combine<M, warp_split>(y);
        if (arg.f.write.W) warp_combine<M, warp_split>(w);

        if (l_idx == 0 || warp_split == 1) {
          if (arg.f.write.Y) arg.Y[k].save(y, idx, parity);
          if (arg.f.write.W) arg.W[k].save(w, idx, parity);
        }

        idx += gridDim.x * blockDim.x / warp_split;
      }
    }

    template <typename T> struct coeff_array {
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };


    template <typename coeff_t_>
    struct MultiBlasFunctor {
      using coeff_t = coeff_t_;
      static constexpr bool reducer = false;
      static constexpr bool coeff_mul  = true;

      int NYW;
      MultiBlasFunctor(int NYW) : NYW(NYW) {}

      __device__ __host__ inline coeff_t a(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<coeff_t *>(Amatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<coeff_t *>(Amatrix_h)[i * NYW + j];
#endif
      }

      __device__ __host__ inline coeff_t b(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<coeff_t *>(Bmatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<coeff_t *>(Bmatrix_h)[i * NYW + j];
#endif
      }

      __device__ __host__ inline coeff_t c(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<coeff_t *>(Cmatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<coeff_t *>(Cmatrix_h)[i * NYW + j];
#endif
      }
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]
    */
    template <int NXZ, typename real>
    struct multiaxpy_ : public MultiBlasFunctor<real> {
      static constexpr write<0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiBlasFunctor<real>::NYW;
      using MultiBlasFunctor<real>::a;
      multiaxpy_(int NYW) : MultiBlasFunctor<real>(NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
        y += a(j, i) * x;
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 2 * NXZ * NYW; }         //! flops per real element
    };

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 4>::type &x, typename VectorType<T, 4>::type &y)
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

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 2>::type &x, typename VectorType<T, 2>::type &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 8>::type &x, typename VectorType<T, 8>::type &y)
    {
      _caxpy(a, x.x, y.x);
      _caxpy(a, x.y, y.y);
    }

    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */
    template <int NXZ, typename real>
    struct multicaxpy_ : public MultiBlasFunctor<complex<real>> {
      static constexpr write<0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiBlasFunctor<complex<real>>::NYW;
      using MultiBlasFunctor<complex<real>>::a;
      multicaxpy_(int NYW) : MultiBlasFunctor<complex<real>>(NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
        _caxpy(a(j, i), x, y);
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor to perform the operation z = a * x + y  (complex-valued)
    */
    template <int NXZ, typename real>
    struct multicaxpyz_ : public MultiBlasFunctor<complex<real>> {
      static constexpr write<0, 0, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      using MultiBlasFunctor<complex<real>>::NYW;
      using MultiBlasFunctor<complex<real>>::a;
      multicaxpyz_(int NYW) : MultiBlasFunctor<complex<real>>(NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
        if (j == 0) w = y;
        _caxpy(a(j, i), x, w);
      }

      int streams() { return 2 * NYW + NXZ * NYW; } //! total number of input and output streams
      int flops() { return 4 * NXZ * NYW; }         //! flops per real element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <int NXZ, typename real>
    struct multi_axpyBzpcx_ : public MultiBlasFunctor<real> {
      static constexpr write<0, 1, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      using MultiBlasFunctor<real>::NYW;
      using MultiBlasFunctor<real>::a;
      using MultiBlasFunctor<real>::b;
      using MultiBlasFunctor<real>::c;
      multi_axpyBzpcx_(int NYW) : MultiBlasFunctor<real>(NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
        y += a(0, i) * w;
        w = b(0, i) * x + c(0, i) * w;
      }

      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 5 * NXZ * NYW; }   //! flops per real element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <int NXZ, typename real>
    struct multi_caxpyBxpz_ : public MultiBlasFunctor<complex<real>> {
      static constexpr write<0, 1, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      using MultiBlasFunctor<complex<real>>::NYW;
      using MultiBlasFunctor<complex<real>>::a;
      using MultiBlasFunctor<complex<real>>::b;
      multi_caxpyBxpz_(int NYW) : MultiBlasFunctor<complex<real>>(NYW) {}

      // i loops over NYW, j loops over NXZ
      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
        _caxpy(a(0, j), x, y);
        _caxpy(b(0, j), x, w); // b/c we swizzled z into w.
      }

      int streams() { return 4 * NYW + NXZ; } //! total number of input and output streams
      int flops() { return 8 * NXZ * NYW; }   //! flops per real element
    };

  } // namespace blas

} // namespace quda
