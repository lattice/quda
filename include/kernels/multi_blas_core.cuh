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
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam N Y-field vector i/o length
       @tparam Functor Functor used to operate on data
    */
    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Functor>
    struct MultiBlasArg :
      SpinorXZ<NXZ_, store_t, N, Functor::use_z>,
      SpinorYW<max_YW_size<NXZ_, store_t, y_store_t, Functor>(), store_t, N, y_store_t, Ny, Functor::use_w> {
      static constexpr int NXZ = NXZ_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Functor>();
      const int NYW;
      Functor f;
      const int length;

      MultiBlasArg(std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                   std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
                   Functor f, int NYW, int length) :
        NYW(NYW),
        f(f),
        length(length)
      {
        if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i].set(*x[i]);
          if (Functor::use_z) this->Z[i].set(*z[i]);
        }
        for (int i = 0; i < NYW; ++i) {
          this->Y[i].set(*y[i]);
          if (Functor::use_w) this->W[i].set(*w[i]);
        }
      }
    };

    // strictly required pre-C++17 and can cause link errors otherwise
    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Functor>
    constexpr int MultiBlasArg<NXZ_, store_t, N, y_store_t, Ny, Functor>::NXZ;

    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Functor>
    constexpr int MultiBlasArg<NXZ_, store_t, N, y_store_t, Ny, Functor>::NYW_max;

    template <int warp_split, typename T> __device__ __host__ void warp_combine(T &x)
    {
#ifdef WARP_SPLIT
      constexpr int warp_size = 32;
      if (warp_split > 1) {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          // reduce down to the first group of column-split threads
#pragma unroll
          for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
#define WARP_CONVERGED 0xffffffff // we know warp should be converged here
            x[i] += __shfl_down_sync(WARP_CONVERGED, x[i], offset);
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
    template <typename real, int n, int NXZ, int warp_split, typename Arg> __global__ void multiBlasKernel(Arg arg)
    {
      // n is real numbers per thread
      using vec = vector_type<complex<real>, n/2>;
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

        vec x, y, z, w;
        if (l_idx == 0 || warp_split == 1) {
          if (arg.f.read.Y) arg.Y[k].load(y, idx, parity);
          if (arg.f.read.W) arg.W[k].load(w, idx, parity);
        } else {
          zero(y);
          zero(w);
        }

#pragma unroll
        for (int l_ = 0; l_ < NXZ; l_ += warp_split) {
          const int l = l_ + l_idx;
          if (l < NXZ || warp_split == 1) {
            if (arg.f.read.X) arg.X[l].load(x, idx, parity);
            if (arg.f.read.Z) arg.Z[l].load(z, idx, parity);

            arg.f(x, y, z, w, k, l);
          }
        }

        // now combine the results across the warp if needed
        if (arg.f.write.Y) warp_combine<warp_split>(y);
        if (arg.f.write.W) warp_combine<warp_split>(w);

        if (l_idx == 0 || warp_split == 1) {
          if (arg.f.write.Y) arg.Y[k].save(y, idx, parity);
          if (arg.f.write.W) arg.W[k].save(w, idx, parity);
        }

        idx += gridDim.x * blockDim.x / warp_split;
      }
    }

    template <typename coeff_t_, bool multi_1d_ = false>
    struct MultiBlasFunctor {
      using coeff_t = coeff_t_;
      static constexpr bool reducer = false;
      static constexpr bool coeff_mul = true;
      static constexpr bool multi_1d = multi_1d_;

      const int NXZ;
      const int NYW;
      MultiBlasFunctor(int NXZ, int NYW) : NXZ(NXZ), NYW(NYW) {}

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
    template <typename real>
    struct multiaxpy_ : public MultiBlasFunctor<real> {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      static constexpr int NXZ_max = 0;
      using MultiBlasFunctor<real>::a;
      multiaxpy_(int NXZ, int NYW) : MultiBlasFunctor<real>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) y[k] += a(j, i) * x[k];
      }

      constexpr int flops() const { return 2; }         //! flops per real element
    };

    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */
    template <typename real>
    struct multicaxpy_ : public MultiBlasFunctor<complex<real>> {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      static constexpr int NXZ_max = 0;
      using MultiBlasFunctor<complex<real>>::a;
      multicaxpy_(int NXZ, int NYW) : MultiBlasFunctor<complex<real>>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) y[k] = cmac(a(j, i), x[k], y[k]);
      }

      constexpr int flops() const { return 4; }         //! flops per real element
    };

    /**
       Functor to perform the operation w = a * x + y  (complex-valued)
    */
    template <typename real>
    struct multicaxpyz_ : public MultiBlasFunctor<complex<real>> {
      static constexpr memory_access<1, 0, 0, 1> read{ };
      static constexpr memory_access<0, 0, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      static constexpr int NXZ_max = 0;
      using MultiBlasFunctor<complex<real>>::a;
      multicaxpyz_(int NXZ, int NYW) : MultiBlasFunctor<complex<real>>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          if (j == 0) w[k] = y[k];
          w[k] = cmac(a(j, i), x[k], w[k]);
        }
      }

      constexpr int flops() const { return 4; }         //! flops per real element
    };

    /**
       Functor performing the operations: y[i] = a*w[i] + y[i]; w[i] = b*x[i] + c*w[i]
    */
    template <typename real>
    struct multi_axpyBzpcx_ : public MultiBlasFunctor<real, true> {
      static constexpr memory_access<1, 1, 0, 1> read{ };
      static constexpr memory_access<0, 1, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      static constexpr int NXZ_max = 1; // we never have NXZ > 1 for this kernel
      // this is a multi-1d functor so the coefficients are stored in the struct
      // set max 1-d size equal to max power of two
      static constexpr int N = max_N_multi_1d();
      real a[N];
      real b[N];
      real c[N];
      multi_axpyBzpcx_(int NXZ, int NYW) : MultiBlasFunctor<real, true>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          y[k] += a[i] * w[k];
          w[k] = b[i] * x[k] + c[i] * w[k];
        }
      }

      constexpr int flops() const { return 5; }   //! flops per real element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and w[i] = b*x[i] + w[i]
    */
    template <typename real>
    struct multi_caxpyBxpz_ : public MultiBlasFunctor<complex<real>, true> {
      static constexpr memory_access<1, 1, 0, 1> read{ };
      static constexpr memory_access<0, 1, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      static constexpr int NXZ_max = 0;
      static constexpr int N = max_N_multi_1d();
      complex<real> a[N];
      complex<real> b[N];
      complex<real> c[N];
      multi_caxpyBxpz_(int NXZ, int NYW) : MultiBlasFunctor<complex<real>, true>(NXZ, NYW)
      {
        for (int i = 0; i < N; i++) {
          a[i] = 0.0;
          b[i] = 0.0;
          c[i] = 0.0;
        }
      }

      // i loops over NYW, j loops over NXZ
      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &z, T &w, int i, int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          y[k] = cmac(a[j], x[k], y[k]);
          w[k] = cmac(b[j], x[k], w[k]);
        }
      }

      constexpr int flops() const { return 8; }   //! flops per real element
    };

  } // namespace blas

} // namespace quda
