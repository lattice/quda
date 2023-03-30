#pragma once

#include <convert.h>
#include <reduce_helper.h>
#include <blas_helper.cuh>
#include <multi_blas_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda
{

  namespace blas
  {

    /**
       @brief Return the batch block size used for multi reductions.
     */
    constexpr unsigned int max_n_batch_block_multi_reduce() { return 8; }

    /**
       @brief Parameter struct for generic multi-reduce blas kernel.
       @tparam real_ The precision of the calculation
       @tparam n_ The number of real elements per thread
       @tparam NXZ_ the dimension of input vectors: X,Z
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam Ny Y-field vector i/o length
       @tparam Reducer_ Functor used to operate on data
    */
    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer_>
    struct MultiReduceArg :
      public ReduceArg<array<typename Reducer_::reduce_t, NXZ_>>,
      SpinorXZ<NXZ_, store_t, N, Reducer_::use_z>,
      SpinorYW<max_YW_size<NXZ_, store_t, y_store_t, Reducer_>(), store_t, N, y_store_t, Ny, Reducer_::use_w>
    {
      using real = real_;
      using Reducer = Reducer_;
      using reduce_t = array<typename Reducer_::reduce_t, NXZ_>;
      static constexpr int n = n_;
      static constexpr int NXZ = NXZ_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Reducer>();
      static constexpr unsigned int max_n_batch_block = max_n_batch_block_multi_reduce();
      const int NYW;
      Reducer f;

      const int length_cb;
      const int nParity;

      template <typename V>
      MultiReduceArg(V &x, V &y, V &z, V &w, Reducer f, int NYW, int length, int nParity) :
        // we have NYW * nParity reductions each of length NXZ
        ReduceArg<reduce_t>(dim3(length, 1, NYW), NYW),
        NYW(NYW),
        f(f),
        length_cb(length / nParity),
        nParity(nParity)
      {
        if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i] = static_cast<ColorSpinorField&>(x[i]);
          if (Reducer::use_z) this->Z[i] = static_cast<ColorSpinorField&>(z[i]);
        }

        for (int i = 0; i < NYW; ++i) {
          this->Y[i] = static_cast<ColorSpinorField&>(y[i]);
          if (Reducer::use_w) this->W[i] = static_cast<ColorSpinorField&>(w[i]);
        }
      }
    };

    // strictly required pre-C++17 and can cause link errors otherwise
    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<real_, n_, NXZ_, store_t, N, y_store_t, Ny, Reducer>::NXZ;

    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<real_, n_, NXZ_, store_t, N, y_store_t, Ny, Reducer>::NYW_max;

    /**
       Generic multi-reduction functor with up to four loads and saves
    */
    template <typename Arg> struct MultiReduce_ : plus<typename Arg::reduce_t> {
      using reduce_t = typename Arg::reduce_t;
      using plus<reduce_t>::operator();
      static constexpr int reduce_block_dim = 1; // x_cb and parity are mapped to x dim
      using vec = array<complex<typename Arg::real>, Arg::n/2>;
      const Arg &arg;
      constexpr MultiReduce_(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      // overload comm_reduce to defer until the entire "tile" is complete
      template <typename U> static inline void comm_reduce(U &) { }

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int tid, int, int k) const
      {
        unsigned int parity = tid >= arg.length_cb ? 1 : 0;
        unsigned int i = tid - parity * arg.length_cb;

        vec x, y, z, w;
        if (arg.f.read.Y) arg.Y[k].load(y, i, parity);
        if (arg.f.read.W) arg.W[k].load(w, i, parity);

        // Each NYW owns its own thread.
        // The NXZ's are all in the same thread block,
        // so they can share the same memory.
#pragma unroll
        for (int l = 0; l < Arg::NXZ; l++) {
          if (arg.f.read.X) arg.X[l].load(x, i, parity);
          if (arg.f.read.Z) arg.Z[l].load(z, i, parity);

          arg.f(sum[l], x, y, z, w, k, l);

        }
        if (arg.f.write.Y) arg.Y[k].save(y, i, parity);
        if (arg.f.write.W) arg.W[k].save(w, i, parity);

        return sum;
      }
    };

    /**
       Base class from which all reduction functors should derive.

       @tparam NXZ The number of elements we accumulate per thread
       @tparam reduce_t The fundamental reduction type
       @tparam coeff_t The type of any coefficients we multiply by
    */
    template <typename reduce_t_, typename coeff_t_>
    struct MultiReduceFunctor : MultiBlasParam<coeff_t_, true, false> {
      using reduce_t = reduce_t_;
      using coeff_t = coeff_t_;
      static constexpr bool reducer = true;
      static constexpr bool coeff_mul  = false;
      static constexpr bool multi_1d = false;

      MultiReduceFunctor(int NXZ, int NYW) : MultiBlasParam<coeff_t, reducer, multi_1d>(NXZ, NYW) {}
    };

    /**
       Return the real dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const complex<T> &a, const complex<T> &b)
    {
      sum += static_cast<reduce_t>(a.real()) * static_cast<reduce_t>(b.real());
      sum += static_cast<reduce_t>(a.imag()) * static_cast<reduce_t>(b.imag());
    }

    template <typename reduce_t, typename real>
    struct multiDot : public MultiReduceFunctor<reduce_t, real> {
      static constexpr memory_access<1, 1> read { };
      static constexpr memory_access< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      multiDot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, real>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &, int, int) const
      {
#pragma unroll
        for (int k=0; k < x.size(); k++) dot_<reduce_t, real>(sum, x[k], y[k]);
      }

      constexpr int flops() const { return 2; }   //! flops per element
    };

    /**
       Returns complex-valued dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const complex<T> &a, const complex<T> &b)
    {
      using scalar = typename reduce_t::value_type;
      sum[0] += static_cast<scalar>(a.real()) * static_cast<scalar>(b.real());
      sum[0] += static_cast<scalar>(a.imag()) * static_cast<scalar>(b.imag());
      sum[1] += static_cast<scalar>(a.real()) * static_cast<scalar>(b.imag());
      sum[1] -= static_cast<scalar>(a.imag()) * static_cast<scalar>(b.real());
    }

    template <typename real_reduce_t, typename real>
    struct multiCdot : public MultiReduceFunctor<array<real_reduce_t, 2>, complex<real>> {
      using reduce_t = array<real_reduce_t, 2>;
      static constexpr memory_access<1, 1> read { };
      static constexpr memory_access< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      multiCdot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, complex<real>>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &, int, int) const
      {
#pragma unroll
        for (int k=0; k < x.size(); k++) cdot_<reduce_t, real>(sum, x[k], y[k]);
      }

      constexpr int flops() const { return 4; }   //! flops per element
    };

    template <typename real_reduce_t, typename real>
    struct multiCdotCopy : public MultiReduceFunctor<typename VectorType<real_reduce_t, 2>::type, complex<real>> {
      using reduce_t = typename VectorType<real_reduce_t, 2>::type;
      static constexpr memory_access<1, 1> read { };
      static constexpr memory_access<0, 0, 0, 1> write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      multiCdotCopy(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, complex<real>>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &w, int i, int j) const
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          cdot_<reduce_t, real>(sum, x[k], y[k]);
          if (i == j) w[k] = y[k];
        }
      }

      constexpr int flops() const { return 4; }   //! flops per element
    };

  } // namespace blas

} // namespace quda
