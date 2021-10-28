#pragma once

#include <color_spinor_field_order.h>
#include <reduce_helper.h>
#include <blas_helper.cuh>
#include <multi_blas_helper.cuh>
#include <fast_intdiv.h>

namespace quda
{

  namespace blas
  {

    /**
       @brief Parameter struct for generic multi-blas kernel.
       @tparam NXZ is dimension of input vectors: X,Z,V
       @tparam NXZ is dimension of input vectors: X,Z
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam N Y-field vector i/o length
       @tparam Reducer Functor used to operate on data
    */
    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer_>
    struct MultiReduceArg :
      public ReduceArg<vector_type<typename Reducer_::reduce_t, NXZ_>>,
      SpinorXZ<NXZ_, store_t, N, Reducer_::use_z>,
      SpinorYW<max_YW_size<NXZ_, store_t, y_store_t, Reducer_>(), store_t, N, y_store_t, Ny, Reducer_::use_w>
    {
      using Reducer = Reducer_;
      static constexpr int NXZ = NXZ_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Reducer>();
      const int NYW;
      Reducer r;
      const int length;
      const int_fastdiv gridSize;

      MultiReduceArg(std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                     std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
                     Reducer r, int NYW, int length, int nParity, TuneParam &tp) :
        // we have NYW * nParity reductions each of length NXZ
        ReduceArg<vector_type<typename Reducer_::reduce_t, NXZ>>(NYW),
        NYW(NYW),
        r(r),
        length(length),
        gridSize(tp.grid.x * tp.block.x / nParity)
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
    };

    // strictly required pre-C++17 and can cause link errors otherwise
    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<NXZ_, store_t, N, y_store_t, Ny, Reducer>::NXZ;

    template <int NXZ_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer>
    constexpr int MultiReduceArg<NXZ_, store_t, N, y_store_t, Ny, Reducer>::NYW_max;

    template <int block_size, typename real, int n, int NXZ, typename Arg>
    __global__ void multiReduceKernel(Arg arg)
    {
      // n is real numbers per thread
      using vec = vector_type<complex<real>, n/2>;
      unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int idx = tid % arg.gridSize;
      unsigned int parity = tid / arg.gridSize;

      const int k = blockIdx.y * blockDim.y + threadIdx.y;

      if (k >= arg.NYW) return; // safe since k are different thread blocks

      using block_reduce_t = vector_type<typename Arg::Reducer::reduce_t, NXZ>;
      block_reduce_t sum;

      while (idx < arg.length) {

        vec x, y, z, w;
        if (arg.r.read.Y) arg.Y[k].load(y, idx, parity);
        if (arg.r.read.W) arg.W[k].load(w, idx, parity);

        // Each NYW owns its own thread.
        // The NXZ's are all in the same thread block,
        // so they can share the same memory.
#pragma unroll
        for (int l = 0; l < NXZ; l++) {
          if (arg.r.read.X) arg.X[l].load(x, idx, parity);
          if (arg.r.read.Z) arg.Z[l].load(z, idx, parity);

          arg.r(sum[l], x, y, z, w, k, l);
        }
        if (arg.r.write.Y) arg.Y[k].save(y, idx, parity);
        if (arg.r.write.W) arg.W[k].save(w, idx, parity);

        idx += arg.gridSize;
      }

      arg.template reduce<block_size>(sum, k);
    } // multiReduceKernel

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

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
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

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
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

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
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
