#pragma once

#include <convert.h>
#include <reduce_helper.h>
#include <blas_helper.cuh>
#include <multi_blas_helper.cuh>
#include <reduction_kernel.h>

namespace quda
{

  namespace blas
  {

    /**
       @brief Parameter struct for generic multi-blas kernel.
       @tparam NXZ is dimension of input vectors: X,Z,V
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam N Y-field vector i/o length
       @tparam Reducer Functor used to operate on data
    */
    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer_>
    struct MultiReduceArg :
      public ReduceArg<vector_type<typename Reducer_::reduce_t, NXZ_>>,
      SpinorXZ<NXZ_, store_t, N, Reducer_::use_z>,
      SpinorYW<max_YW_size<NXZ_, store_t, y_store_t, Reducer_>(), store_t, N, y_store_t, Ny, Reducer_::use_w>
    {
      using real = real_;
      using Reducer = Reducer_;
      using reduce_t = vector_type<typename Reducer_::reduce_t, NXZ_>;
      static constexpr int n = n_;
      static constexpr int NXZ = NXZ_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Reducer>();
      const int NYW;
      Reducer f;

      const int length_cb;
      const int nParity;
      dim3 threads;

      MultiReduceArg(std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                     std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
                     Reducer f, int NYW, int length, int nParity) :
        // we have NYW * nParity reductions each of length NXZ
        ReduceArg<reduce_t>(NYW),
        NYW(NYW),
        f(f),
        length_cb(length / nParity),
        nParity(nParity),
        threads(length, NYW, 1)
      {
        if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i].set(*x[i]);
          if (Reducer::use_z) this->Z[i].set(*z[i]);
        }

        for (int i = 0; i < NYW; ++i) {
          this->Y[i].set(*y[i]);
          if (Reducer::use_w) this->W[i].set(*w[i]);
        }
      }

      __device__ __host__ auto init() const { return ::quda::zero<typename Reducer_::reduce_t, NXZ>(); }
    };

    // strictly required pre-C++17 and can cause link errors otherwise
    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<real_, n_, NXZ_, store_t, N, y_store_t, Ny, Reducer>::NXZ;

    template <typename real_, int n_, int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<real_, n_, NXZ_, store_t, N, y_store_t, Ny, Reducer>::NYW_max;

    /**
       Generic multi-reduction functor with up to four loads and saves
    */
    template <typename Arg> struct MultiReduce_ : plus<vector_type<typename Arg::Reducer::reduce_t, Arg::NXZ>> {
      using vec = vector_type<complex<typename Arg::real>, Arg::n/2>;
      using reduce_t = vector_type<typename Arg::Reducer::reduce_t, Arg::NXZ>;
      using plus<reduce_t>::operator();
      Arg &arg;
      constexpr MultiReduce_(Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int tid, int k, int)
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
    template <typename reduce_t_, typename coeff_t_> struct MultiReduceFunctor {
      using reduce_t = reduce_t_;
      using coeff_t = coeff_t_;
      static constexpr bool reducer = true;
      static constexpr bool coeff_mul  = false;
      static constexpr bool multi_1d = false;
      const int NXZ;
      const int NYW;

      MultiReduceFunctor(int NXZ, int NYW) : NXZ(NXZ), NYW(NYW) {}

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<is_device, coeff_t> a(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Amatrix_d)[i * NYW + j];
      }

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<!is_device, coeff_t> a(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Amatrix_h)[i * NYW + j];
      }

      __device__ __host__ inline coeff_t a(int i, int j) const { return a<device::is_device()>(i, j); }

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<is_device, coeff_t> b(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Bmatrix_d)[i * NYW + j];
      }

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<!is_device, coeff_t> b(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Bmatrix_h)[i * NYW + j];
      }

      __device__ __host__ inline coeff_t b(int i, int j) const { return b<device::is_device()>(i, j); }

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<is_device, coeff_t> c(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Cmatrix_d)[i * NYW + j];
      }

      template <bool is_device>
      __device__ __host__ inline std::enable_if_t<!is_device, coeff_t> c(int i, int j) const
      {
        return reinterpret_cast<coeff_t *>(Cmatrix_h)[i * NYW + j];
      }

      __device__ __host__ inline coeff_t c(int i, int j) const { return c<device::is_device()>(i, j); }
    };

    /**
       Return the real dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      sum += static_cast<reduce_t>(a.x) * static_cast<reduce_t>(b.x);
      sum += static_cast<reduce_t>(a.y) * static_cast<reduce_t>(b.y);
    }

    template <typename reduce_t, typename real>
    struct multiDot : public MultiReduceFunctor<reduce_t, real> {
      static constexpr memory_access<1, 1> read { };
      static constexpr memory_access< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      multiDot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, real>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &, int, int)
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
    __device__ __host__ void cdot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      using scalar = typename scalar<reduce_t>::type;
      sum.x += static_cast<scalar>(a.x) * static_cast<scalar>(b.x);
      sum.x += static_cast<scalar>(a.y) * static_cast<scalar>(b.y);
      sum.y += static_cast<scalar>(a.x) * static_cast<scalar>(b.y);
      sum.y -= static_cast<scalar>(a.y) * static_cast<scalar>(b.x);
    }

    template <typename real_reduce_t, typename real>
    struct multiCdot : public MultiReduceFunctor<typename VectorType<real_reduce_t, 2>::type, complex<real>> {
      using reduce_t = typename VectorType<real_reduce_t, 2>::type;
      static constexpr memory_access<1, 1> read { };
      static constexpr memory_access< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      multiCdot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, complex<real>>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &, int, int)
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

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &, T &w, int i, int j)
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
