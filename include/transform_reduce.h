#pragma once
#include <typeinfo>

#include <cub_helper.cuh>
#include <uint_to_char.h>
#include <tune_quda.h>

/**
   @file transform_reduce.h

   @brief QUDA reimplementation of thrust::transform_reduce as well as
   wrappers also implementing thrust::reduce.
 */

namespace quda
{

  template <typename T> struct plus {
    __device__ __host__ T operator()(T a, T b) { return a + b; }
  };

  template <typename T> struct maximum {
    __device__ __host__ T operator()(T a, T b) { return a > b ? a : b; }
  };

  template <typename T> struct minimum {
    __device__ __host__ T operator()(T a, T b) { return a < b ? a : b; }
  };

  template <typename T> struct identity {
    __device__ __host__ T operator()(T a) { return a; }
  };

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  struct TransformReduceArg : public ReduceArg<reduce_t> {
    static constexpr int block_size = 512;
    static constexpr int n_batch_max = 4;
    const T *v[n_batch_max];
    count_t n_items;
    int n_batch;
    reduce_t init;
    reduce_t result[n_batch_max];
    transformer h;
    reducer r;
    TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h, reduce_t init, reducer r) :
      n_items(n_items),
      n_batch(v.size()),
      init(init),
      h(h),
      r(r)
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      for (size_t j = 0; j < v.size(); j++) this->v[j] = v[j];
    }
  };

  template <typename Arg> void transform_reduce(Arg &arg)
  {
    using count_t = decltype(arg.n_items);
    using reduce_t = decltype(arg.init);

    for (int j = 0; j < arg.n_batch; j++) {
      auto v = arg.v[j];
      reduce_t r_ = arg.init;
      for (count_t i = 0; i < arg.n_items; i++) {
        auto v_ = arg.h(v[i]);
        r_ = arg.r(r_, v_);
      }
      arg.result[j] = r_;
    }
  }

  template <typename Arg> __launch_bounds__(Arg::block_size) __global__ void transform_reduce_kernel(Arg arg)
  {
    using count_t = decltype(arg.n_items);
    using reduce_t = decltype(arg.init);

    count_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    auto v = arg.v[j];
    reduce_t r_ = arg.init;

    while (i < arg.n_items) {
      auto v_ = arg.h(v[i]);
      r_ = arg.r(r_, v_);
      i += blockDim.x * gridDim.x;
    }

    reduce<Arg::block_size, reduce_t, false, decltype(arg.r)>(arg, r_, j);
  }

  template <typename Arg> class TransformReduce : Tunable
  {
    Arg &arg;
    QudaFieldLocation location;

    bool tuneSharedBytes() const { return false; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    int blockMin() const { return Arg::block_size; }
    unsigned int maxBlockSize(const TuneParam &param) const { return Arg::block_size; }

    bool advanceTuneParam(TuneParam &param) const
    {
      // only do autotuning if we have device fields
      if (location == QUDA_CUDA_FIELD_LOCATION)
        return Tunable::advanceTuneParam(param);
      else
        return false;
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid.y = arg.n_batch;
    }

  public:
    TransformReduce(Arg &arg, QudaFieldLocation location) : arg(arg), location(location)
    {
      strcpy(aux, "batch_size=");
      u32toa(aux + 11, arg.n_batch);
      if (location == QUDA_CPU_FIELD_LOCATION) strcat(aux, ",cpu");
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (location == QUDA_CUDA_FIELD_LOCATION) {
        transform_reduce_kernel<<<tp.grid, tp.block>>>(arg);
        qudaDeviceSynchronize();
        for (decltype(arg.n_batch) j = 0; j < arg.n_batch; j++) arg.result[j] = arg.result_h[j];
      } else {
        transform_reduce(arg);
      }
    }

    TuneKey tuneKey() const
    {
      char count[16];
      u32toa(count, arg.n_items);
      return TuneKey(count, typeid(*this).name(), aux);
    }

    long long flops() const { return 0; } // just care about bandwidth
    long long bytes() const { return arg.n_batch * arg.n_items * sizeof(*arg.v); }
  };

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Vector of results
     @param[in] v Vector of inputs
     @param[in] n_items Number of elements to be reduced in each input
     @param[in] transformer Functor that applies transform to each element
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  void transform_reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, I n_items,
                        transformer h, reduce_t init, reducer r)
  {
    if (result.size() != v.size())
      errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
    TransformReduceArg<reduce_t, T, I, transformer, reducer> arg(v, n_items, h, init, r);
    TransformReduce<decltype(arg)> reduce(arg, location);
    reduce.apply(0);
    for (size_t j = 0; j < result.size(); j++) result[j] = arg.result[j];
  }

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] transformer Functor that applies transform to each element
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  reduce_t transform_reduce(QudaFieldLocation location, const T *v, I n_items, transformer h, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, h, init, r);
    return result[0];
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  void reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, I n_items,
              reduce_t init, reducer r)
  {
    transform_reduce(location, result, v, n_items, identity<T>(), init, r);
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init Result is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename reducer>
  reduce_t reduce(QudaFieldLocation location, const T *v, I n_items, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, identity<T>(), init, r);
    return result[0];
  }
} // namespace quda
