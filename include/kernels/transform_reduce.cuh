#pragma once

#include <limits>
#include <algorithm>
#include <reduction_kernel.h>

namespace quda {

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer_>
  struct TransformReduceArg : public ReduceArg<reduce_t> {
    using reducer = reducer_;
    static constexpr int n_batch_max = 8;
    const T *v[n_batch_max];
    count_t n_items;
    int n_batch;
    reduce_t init_value;
    transformer h;
    reducer r;

    TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h, reduce_t init_value, reducer r) :
      ReduceArg<reduce_t>(dim3(n_items, v.size(), 1), v.size()),
      n_items(n_items),
      n_batch(v.size()),
      init_value(init_value),
      h(h),
      r(r)
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      if (n_items > std::numeric_limits<count_t>::max())
        errorQuda("Requested size %lu greater than max supported %lu",
                  (uint64_t)n_items, (uint64_t)std::numeric_limits<count_t>::max());
      std::copy(v.begin(), v.end(), this->v);
    }

    __device__ __host__ reduce_t init() const { return init_value; }
  };

  template <typename Arg> struct transform_reducer {
    using count_t = decltype(Arg::n_items);
    using reduce_t = decltype(Arg::init_value);

    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr transform_reducer(const Arg &arg) : arg(arg) {}

    static constexpr bool do_sum = Arg::reducer::do_sum;

    __device__ __host__ inline reduce_t operator()(reduce_t a, reduce_t b) const { return arg.r(a, b); }

    __device__ __host__ inline reduce_t operator()(reduce_t &value, count_t i, int j, int)
    {
      auto v = arg.v[j];
      auto t = arg.h(v[i]);
      return arg.r(t, value);
    }
  };

}
