#pragma once

#include <blas_helper.cuh>
#include <multi_blas_helper.cuh>
#include <array.h>
#include <constant_kernel_arg.h>
#include <kernel.h>
#include <warp_collective.h>

namespace quda
{

  namespace blas
  {

#ifndef QUDA_FAST_COMPILE_REDUCE
    constexpr bool enable_warp_split() { return false; }
#else
    constexpr bool enable_warp_split() { return true; }
#endif

    /**
       @brief Parameter struct for generic multi-blas kernel.
       @tparam warp_split_ The degree of warp splitting over the NXZ dimension
       @tparam real_ The precision of the calculation
       @tparam n_ The number of real elements per thread
       @tparam NXZ_ is dimension of input vectors: X,Z
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam N Y-field vector i/o length
       @tparam Functor_ Functor used to operate on data
    */
    template <int warp_split_, typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Functor_>
    struct MultiBlasArg : kernel_param<>,
      SpinorXZ<NXZ_, store_t, N, Functor_::use_z>,
      SpinorYW<max_YW_size<NXZ_, store_t, y_store_t, Functor_>(), store_t, N, y_store_t, Ny, Functor_::use_w> {
      using real = real_;
      using Functor = Functor_;
      static constexpr int warp_split = warp_split_;
      static constexpr int n = n_;
      static constexpr int NXZ = NXZ_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Functor>();
      Functor f;

      template <typename V>
      MultiBlasArg(V &x, V&y, V &z, V &w, Functor f, int NYW, int length) :
        kernel_param(dim3(length * warp_split, NYW, x[0].SiteSubset())),
        f(f)
      {
        if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i] = static_cast<ColorSpinorField&>(x[i]);
          if (Functor::use_z) this->Z[i] = static_cast<ColorSpinorField&>(z[i]);
        }
        for (int i = 0; i < NYW; ++i) {
          this->Y[i] = static_cast<ColorSpinorField&>(y[i]);
          if (Functor::use_w) this->W[i] = static_cast<ColorSpinorField&>(w[i]);
        }
      }
    };

    /**
       @brief Generic multi-blas kernel with four loads and up to four stores.
       @param[in,out] arg Argument struct with required meta data
       (input/output fields, functor, etc.)
    */
    template <typename Arg> struct MultiBlas_ {
      const Arg &arg;
      constexpr MultiBlas_(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline void operator()(int i, int k, int parity)
      {
        using vec = array<complex<typename Arg::real>, Arg::n/2>;

        // partition the warp between grid points and the NXZ update
        constexpr int warp_size = device::warp_size();
        constexpr int warp_split = Arg::warp_split;
        constexpr int vector_site_width = warp_size / warp_split;
        const int lane_id = i % warp_size;
        const int warp_id = i / warp_size;
        const int idx = warp_id * (warp_size / warp_split) + lane_id % vector_site_width;
        const int l_idx = lane_id / vector_site_width;

        vec x, y, z, w;
        if (l_idx == 0 || warp_split == 1) {
          if (arg.f.read.Y) arg.Y[k].load(y, idx, parity);
          if (arg.f.read.W) arg.W[k].load(w, idx, parity);
        } else {
          y = ::quda::zero<complex<typename Arg::real>, Arg::n/2>();
          w = ::quda::zero<complex<typename Arg::real>, Arg::n/2>();
        }

#pragma unroll
        for (int l_ = 0; l_ < Arg::NXZ; l_ += warp_split) {
          const int l = l_ + l_idx;
          if (l < Arg::NXZ || warp_split == 1) {
            if (arg.f.read.X) arg.X[l].load(x, idx, parity);
            if (arg.f.read.Z) arg.Z[l].load(z, idx, parity);

            arg.f(x, y, z, w, k, l);
          }
        }

        // now combine the results across the warp if needed
        if (arg.f.write.Y) y = warp_combine<warp_split>(y);
        if (arg.f.write.W) w = warp_combine<warp_split>(w);

        if (l_idx == 0 || warp_split == 1) {
          if (arg.f.write.Y) arg.Y[k].save(y, idx, parity);
          if (arg.f.write.W) arg.W[k].save(w, idx, parity);
        }
      }
    };

    template <typename coeff_t_, bool multi_1d_ = false>
    struct MultiBlasFunctor : MultiBlasParam<coeff_t_, false, multi_1d_> {
      using coeff_t = coeff_t_;
      static constexpr bool reducer = false;
      static constexpr bool coeff_mul = true;
      static constexpr bool multi_1d = multi_1d_;

      MultiBlasFunctor(int NXZ, int NYW) : MultiBlasParam<coeff_t, reducer, multi_1d>(NXZ, NYW) {}
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

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &, int i, int j) const
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

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &, int i, int j) const
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) y[k] = cmac(a(j, i), x[k], y[k]);
      }

      constexpr int flops() const { return 4; }         //! flops per real element
    };

    /**
       Functor to perform the operation w = a * x + y
    */
    template <typename real>
    struct multiaxpyz_ : public MultiBlasFunctor<real> {
      static constexpr memory_access<1, 1, 0, 0> read{ };
      static constexpr memory_access<0, 0, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      static constexpr int NXZ_max = 0;
      using MultiBlasFunctor<real>::a;
      multiaxpyz_(int NXZ, int NYW) : MultiBlasFunctor<real>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &w, int i, int j) const
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          if (j == 0) w[k] = y[k];
          w[k] = a(j, i) * x[k] + w[k];
        }
      }

      constexpr int flops() const { return 2; }         //! flops per real element
    };

    /**
       Functor to perform the operation w = a * x + y  (complex-valued)
    */
    template <typename real>
    struct multicaxpyz_ : public MultiBlasFunctor<complex<real>> {
      static constexpr memory_access<1, 1, 0, 0> read{ };
      static constexpr memory_access<0, 0, 0, 1> write{ };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      static constexpr int NXZ_max = 0;
      using MultiBlasFunctor<complex<real>>::a;
      multicaxpyz_(int NXZ, int NYW) : MultiBlasFunctor<complex<real>>(NXZ, NYW) {}

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &w, int i, int j) const
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

      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &w, int i, int) const
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
      template <typename T> __device__ __host__ inline void operator()(T &x, T &y, T &, T &w, int, int j) const
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
