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
    template <int NXZ, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer_>
    struct MultiReduceArg
      : public ReduceArg<vector_type<typename Reducer_::reduce_t, NXZ>>,
        SpinorXZ<NXZ, SpinorX, SpinorZ, Reducer_::use_z>,
        SpinorYW<max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer_>(), SpinorY, SpinorW, Reducer_::use_w> {
      using Reducer = Reducer_;
      static constexpr int NYW_max = max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer>();
      const int NYW;
      Reducer r;
      const int length;

      MultiReduceArg(SpinorX X[NXZ], SpinorY Y[], SpinorZ Z[NXZ], SpinorW W[], Reducer r, int NYW, int length) :
          NYW(NYW),
          r(r),
          length(length)
      {
        if (NYW > NYW_max) errorQuda("NYW = %d greater than maximum size of %d", NYW, NYW_max);

        for (int i = 0; i < NXZ; ++i) {
          this->X[i] = X[i];
          if (Reducer::use_z) this->Z[i] = Z[i];
        }

        for (int i = 0; i < NYW; ++i) {
          this->Y[i] = Y[i];
          if (Reducer::use_w) this->W[i] = W[i];
        }
      }
    };

#ifdef WARP_MULTI_REDUCE
    template <typename FloatN, int M, int NXZ, typename Arg>
#else
    template <int block_size, typename FloatN, int M, int NXZ, typename Arg>
#endif

    __global__ void multiReduceKernel(Arg arg)
    {
      unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
      unsigned int parity = blockIdx.z;

      if (k >= arg.NYW) return; // safe since k are different thread blocks

      using block_reduce_t = vector_type<typename Arg::Reducer::reduce_t, NXZ>;
      block_reduce_t sum;

      while (idx < arg.length) {

        FloatN x[M], y[M], z[M], w[M];

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

#pragma unroll
          for (int j = 0; j < M; j++) arg.r(sum[l], x[j], y[j], z[j], w[j], k, l);

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

    template <typename T> struct coeff_array {
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };

    /**
       Base class from which all reduction functors should derive.

       @tparam NXZ The number of elements we accumulate per thread
       @tparam reduce_t The fundamental reduction type
       @tparam coeff_t The type of any coefficients we multiply by
    */
    template <int NXZ, typename reduce_t_, typename coeff_t_> struct MultiReduceFunctor {
      using reduce_t = reduce_t_;
      using coeff_t = coeff_t_;
      static constexpr bool reducer = true;
      static constexpr bool coeff_mul  = false;
      int NYW;
      MultiReduceFunctor(int NYW) : NYW(NYW) {}

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

    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const typename VectorType<T, 4>::type &a, const typename VectorType<T, 4>::type &b)
    {
      sum += (reduce_t)a.x * (reduce_t)b.x;
      sum += (reduce_t)a.y * (reduce_t)b.y;
      sum += (reduce_t)a.z * (reduce_t)b.z;
      sum += (reduce_t)a.w * (reduce_t)b.w;
    }

    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const typename VectorType<T, 8>::type &a, const typename VectorType<T, 8>::type &b)
    {
      dot_<reduce_t, T>(sum, a.x, b.x);
      dot_<reduce_t, T>(sum, a.y, b.y);
    }

    template <int NXZ, typename reduce_t, typename real>
    struct Dot : public MultiReduceFunctor<NXZ, reduce_t, real> {
      static constexpr write< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiReduceFunctor<NXZ, reduce_t, real>::NYW;
      Dot(int NYW) :
        MultiReduceFunctor<NXZ, reduce_t, real>(NYW)
      {
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
        dot_<reduce_t, real>(sum, x, y);
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 2; }   //! flops per element
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

    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const typename VectorType<T, 4>::type &a, const typename VectorType<T, 4>::type &b)
    {
      typedef typename scalar<reduce_t>::type scalar;
      sum.x += (scalar)a.x * (scalar)b.x;
      sum.x += (scalar)a.y * (scalar)b.y;
      sum.x += (scalar)a.z * (scalar)b.z;
      sum.x += (scalar)a.w * (scalar)b.w;
      sum.y += (scalar)a.x * (scalar)b.y;
      sum.y -= (scalar)a.y * (scalar)b.x;
      sum.y += (scalar)a.z * (scalar)b.w;
      sum.y -= (scalar)a.w * (scalar)b.z;
    }

    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const typename VectorType<T, 8>::type &a, const typename VectorType<T, 8>::type &b)
    {
      cdot_(sum, a.x, b.x);
      cdot_(sum, a.y, b.y);
    }

    template <int NXZ, typename real_reduce_t, typename real>
    struct Cdot : public MultiReduceFunctor<NXZ, complex<real_reduce_t>, complex<real>> {
      using reduce_t = complex<real_reduce_t>;
      static constexpr write< > write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      using MultiReduceFunctor<NXZ, reduce_t, complex<real>>::NYW;
      Cdot(int NYW) :
        MultiReduceFunctor<NXZ, reduce_t, complex<real>>(NYW)
      {
      }
      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
        cdot_<reduce_t, real>(sum, x, y);
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    template <int NXZ, typename real_reduce_t, typename real>
    struct CdotCopy : public MultiReduceFunctor<NXZ, complex<real_reduce_t>, complex<real>> {
      using reduce_t = complex<real_reduce_t>;
      static constexpr write<0, 0, 0, 1> write { };
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      using MultiReduceFunctor<NXZ, reduce_t, complex<real>>::NYW;
      CdotCopy(int NYW) :
        MultiReduceFunctor<NXZ, reduce_t, complex<real>>(NYW)
      {
      }
      template <typename T> __device__ __host__ inline void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, const int i, const int j)
      {
        cdot_<reduce_t, real>(sum, x, y);
        if (i == j) w = y;
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

  } // namespace blas

} // namespace quda
