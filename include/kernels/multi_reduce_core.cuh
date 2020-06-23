#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>
#include <multi_blas_helper.cuh>
#include <cub_helper.cuh>

//#define WARP_MULTI_REDUCE

namespace quda
{

  namespace blas
  {

    /**
       @brief Parameter struct for generic multi-blas kernel.
       @tparam NXZ is dimension of input vectors: X,Z,V
       @tparam NYW is dimension of in-output vectors: Y,W
       @tparam SpinorX Type of input spinor for x argument
       @tparam SpinorY Type of input spinor for y argument
       @tparam SpinorZ Type of input spinor for z argument
       @tparam SpinorW Type of input spinor for w argument
       @tparam SpinorW Type of input spinor for v argument
       @tparam Reducer Functor used to operate on data
    */
    template <int NXZ, typename store_t, int N, typename y_store_t, int Ny, typename Reducer_>
    struct MultiReduceArg :
      public ReduceArg<vector_type<typename Reducer_::reduce_t, NXZ>>,
      SpinorXZ<NXZ, store_t, N, Reducer_::use_z>,
      SpinorYW<max_YW_size<NXZ, store_t, y_store_t, Reducer_>(), y_store_t, Ny, Reducer_::use_w>
    {
      using Reducer = Reducer_;
      static constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Reducer>();
      const int NYW;
      Reducer r;
      const int length;

      MultiReduceArg(std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                     std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
                     Reducer r, int NYW, int length) :
        NYW(NYW),
        r(r),
        length(length)
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

#ifdef WARP_MULTI_REDUCE
    template <typename real, int n, int NXZ, typename Arg>
#else
    template <int block_size, typename real, int n, int NXZ, typename Arg>
#endif
    __global__ void multiReduceKernel(Arg arg)
    {
      // n is real numbers per thread
      using vec = vector_type<complex<real>, n/2>;
      unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int k = blockIdx.y * blockDim.y + threadIdx.y;
      const int parity = blockIdx.z;

      if (k >= arg.NYW) return; // safe since k are different thread blocks

      using block_reduce_t = vector_type<typename Arg::Reducer::reduce_t, NXZ>;
      block_reduce_t sum;

      while (idx < arg.length) {

        vec x, y, z, w;
        arg.Y[k].load(y, idx, parity);
        arg.W[k].load(w, idx, parity);

        // Each NYW owns its own thread.
        // The NXZ's are all in the same thread block,
        // so they can share the same memory.
#pragma unroll
        for (int l = 0; l < NXZ; l++) {
          arg.X[l].load(x, idx, parity);
          arg.Z[l].load(z, idx, parity);

          arg.r.pre();
          arg.r(sum[l], x, y, z, w, k, l);
          arg.r.post(sum[l]);
        }
        if (arg.r.write.X) arg.Y[k].save(y, idx, parity);
        if (arg.r.write.X) arg.W[k].save(w, idx, parity);

        idx += gridDim.x * blockDim.x;
      }

#ifdef WARP_MULTI_REDUCE
      ::quda::warp_reduce<block_reduce_t>(arg, sum, arg.NYW * parity + k);
#else
      ::quda::reduce<block_size, block_reduce_t>(arg, sum, arg.NYW * parity + k);
#endif
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

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! post-computation routine called after the "M-loop"
      virtual __device__ __host__ void post(reduce_t &sum) { ; }
    };

    /**
       Return the real dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      sum += (reduce_t)a.x * (reduce_t)b.x;
      sum += (reduce_t)a.y * (reduce_t)b.y;
    }

    template <typename reduce_t, typename real>
    struct Dot : public MultiReduceFunctor<reduce_t, real> {
      static constexpr write< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiReduceFunctor<reduce_t, real>::NXZ;
      using MultiReduceFunctor<reduce_t, real>::NYW;
      Dot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, real>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
#pragma unroll
        for (int k=0; k < x.size(); k++) dot_<reduce_t, real>(sum, x[k], y[k]);
      }

      int streams() const { return 2; } //! total number of input and output streams
      int flops() const { return 2; }   //! flops per element
    };

    /**
       Returns complex-valued dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      typedef typename scalar<reduce_t>::type scalar;
      sum.x += (scalar)a.x * (scalar)b.x;
      sum.x += (scalar)a.y * (scalar)b.y;
      sum.y += (scalar)a.x * (scalar)b.y;
      sum.y -= (scalar)a.y * (scalar)b.x;
    }

    template <typename real_reduce_t, typename real>
    struct Cdot : public MultiReduceFunctor<complex<real_reduce_t>, complex<real>> {
      using reduce_t = complex<real_reduce_t>;
      static constexpr write< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiReduceFunctor<reduce_t, complex<real>>::NXZ;
      using MultiReduceFunctor<reduce_t, complex<real>>::NYW;
      Cdot(int NXZ, int NYW) : MultiReduceFunctor<reduce_t, complex<real>>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
#pragma unroll
        for (int k=0; k < x.size(); k++) cdot_<reduce_t, real>(sum, x[k], y[k]);
      }

      int streams() const { return 2; } //! total number of input and output streams
      int flops() const { return 4; }   //! flops per element
    };

    template <typename real_reduce_t, typename real>
    struct CdotCopy : public MultiReduceFunctor<complex<real_reduce_t>, complex<real>> {
      using reduce_t = complex<real_reduce_t>;
      static constexpr write<0, 0, 0, 1> write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      using MultiReduceFunctor<reduce_t, complex<real>>::NXZ;
      using MultiReduceFunctor<reduce_t, complex<real>>::NYW;

      CdotCopy(int NYW) : MultiReduceFunctor<reduce_t, complex<real>>(NXZ, NYW) { }

      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
#pragma unroll
        for (int k = 0; k < x.size(); k++) {
          cdot_<reduce_t, real>(sum, x[k], y[k]);
          if (i == j) w[k] = y[k];
        }
      }

      int streams() const { return 2; } //! total number of input and output streams
      int flops() const { return 4; }   //! flops per element
    };

  } // namespace blas

} // namespace quda
