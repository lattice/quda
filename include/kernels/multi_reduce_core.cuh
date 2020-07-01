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
    template <int NXZ, typename ReduceType, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW,
              typename Reducer>
    struct MultiReduceArg
      : public ReduceArg<vector_type<ReduceType, NXZ>>,
        SpinorXZ<NXZ, SpinorX, SpinorZ, Reducer::use_z>,
        SpinorYW<max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer>(), SpinorY, SpinorW, Reducer::use_w> {
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
    template <typename ReduceType, typename FloatN, int M, int NXZ, typename Arg>
#else
    template <int block_size, typename ReduceType, typename FloatN, int M, int NXZ, typename Arg>
#endif

    __global__ void multiReduceKernel(Arg arg)
    {
      unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
      unsigned int parity = blockIdx.z;

      if (k >= arg.NYW) return; // safe since k are different thread blocks

      vector_type<ReduceType, NXZ> sum;

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
        arg.Y[k].save(y, idx, parity);
        arg.W[k].save(w, idx, parity);

        idx += gridDim.x * blockDim.x;
      }

#ifdef WARP_MULTI_REDUCE
      ::quda::warp_reduce<vector_type<ReduceType, NXZ>>(arg, sum, arg.NYW * parity + k);
#else
      ::quda::reduce<block_size, vector_type<ReduceType, NXZ>>(arg, sum, arg.NYW * parity + k);
#endif
    } // multiReduceKernel

    template <typename T> struct coeff_array {
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };

    /**
       Base class from which all reduction functors should derive.
    */
    template <int NXZ, typename ReduceType, typename Float2, typename FloatN> struct MultiReduceFunctor {
      typedef Float2 type;
      static constexpr bool reducer = true;
      int NYW;
      MultiReduceFunctor(int NYW) : NYW(NYW) {}

      __device__ __host__ inline Float2 a(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Amatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<Float2 *>(Amatrix_h)[i * NYW + j];
#endif
      }

      __device__ __host__ inline Float2 b(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Bmatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<Float2 *>(Bmatrix_h)[i * NYW + j];
#endif
      }

      __device__ __host__ inline Float2 c(int i, int j) const
      {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<Float2 *>(Cmatrix_d)[i * NYW + j];
#else
        return reinterpret_cast<Float2 *>(Cmatrix_h)[i * NYW + j];
#endif
      }

      //! pre-computation routine called before the "M-loop"
      virtual __device__ __host__ void pre() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ __host__ void operator()(
          ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
          = 0;

      //! post-computation routine called after the "M-loop"
      virtual __device__ __host__ void post(ReduceType &sum) { ; }
    };

    /**
       Return the real dot product of x and y
       Broken at the moment---need to update reDotProduct with permuting, etc of cDotProduct below.
    */
    template <typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const double2 &a, const double2 &b)
    {
      sum += (ReduceType)a.x * (ReduceType)b.x;
      sum += (ReduceType)a.y * (ReduceType)b.y;
    }

    template <typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const float2 &a, const float2 &b)
    {
      sum += (ReduceType)a.x * (ReduceType)b.x;
      sum += (ReduceType)a.y * (ReduceType)b.y;
    }

    template <typename ReduceType> __device__ __host__ void dot_(ReduceType &sum, const float4 &a, const float4 &b)
    {
      sum += (ReduceType)a.x * (ReduceType)b.x;
      sum += (ReduceType)a.y * (ReduceType)b.y;
      sum += (ReduceType)a.z * (ReduceType)b.z;
      sum += (ReduceType)a.w * (ReduceType)b.w;
    }

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct Dot : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      typedef typename scalar<Float2>::type real;
      using MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>::NYW;
      Dot(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
        MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>(NYW)
      {
      }
      __device__ __host__ void operator()(
          ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        dot_<ReduceType>(sum, x, y);
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 2; }   //! flops per element
    };

    /**
       Returns complex-valued dot product of x and y
    */
    template <typename ReduceType> __device__ __host__ void cdot_(ReduceType &sum, const double2 &a, const double2 &b)
    {
      typedef typename scalar<ReduceType>::type scalar;
      sum.x += (scalar)a.x * (scalar)b.x;
      sum.x += (scalar)a.y * (scalar)b.y;
      sum.y += (scalar)a.x * (scalar)b.y;
      sum.y -= (scalar)a.y * (scalar)b.x;
    }

    template <typename ReduceType> __device__ __host__ void cdot_(ReduceType &sum, const float2 &a, const float2 &b)
    {
      typedef typename scalar<ReduceType>::type scalar;
      sum.x += (scalar)a.x * (scalar)b.x;
      sum.x += (scalar)a.y * (scalar)b.y;
      sum.y += (scalar)a.x * (scalar)b.y;
      sum.y -= (scalar)a.y * (scalar)b.x;
    }

    template <typename ReduceType> __device__ __host__ void cdot_(ReduceType &sum, const float4 &a, const float4 &b)
    {
      typedef typename scalar<ReduceType>::type scalar;
      sum.x += (scalar)a.x * (scalar)b.x;
      sum.x += (scalar)a.y * (scalar)b.y;
      sum.x += (scalar)a.z * (scalar)b.z;
      sum.x += (scalar)a.w * (scalar)b.w;
      sum.y += (scalar)a.x * (scalar)b.y;
      sum.y -= (scalar)a.y * (scalar)b.x;
      sum.y += (scalar)a.z * (scalar)b.w;
      sum.y -= (scalar)a.w * (scalar)b.z;
    }

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct Cdot : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      static constexpr bool use_z = false;
      static constexpr bool use_w = false;
      typedef typename scalar<Float2>::type real;
      using MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>::NYW;
      Cdot(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
        MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>(NYW)
      {
      }
      __device__ __host__ inline void operator()(
          ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        cdot_<ReduceType>(sum, x, y);
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    template <int NXZ, typename ReduceType, typename Float2, typename FloatN>
    struct CdotCopy : public MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN> {
      static constexpr bool use_z = false;
      static constexpr bool use_w = true;
      typedef typename scalar<Float2>::type real;
      using MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>::NYW;
      CdotCopy(const coeff_array<Complex> &a, const coeff_array<Complex> &b, const coeff_array<Complex> &c, int NYW) :
        MultiReduceFunctor<NXZ, ReduceType, Float2, FloatN>(NYW)
      {
      }
      __device__ __host__ inline void operator()(
          ReduceType &sum, FloatN &x, FloatN &y, FloatN &z, FloatN &w, const int i, const int j)
      {
        cdot_<ReduceType>(sum, x, y);
        if (i == j) w = y;
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

  } // namespace blas

} // namespace quda
