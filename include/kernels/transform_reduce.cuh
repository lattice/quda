#pragma once

#include <limits>
#include <algorithm>
#include <reduction_kernel.h>

namespace quda {

  template <typename reducer_, typename T, typename count_t, typename transformer>
  struct TransformReduceArg : public ReduceArg<typename reducer_::reduce_t> {
    using reducer = reducer_;
    using reduce_t = typename reducer::reduce_t;
    static constexpr int n_batch_max = 8;
    const T *v[n_batch_max];
    count_t n_items;
    int n_batch;
    transformer h;

    TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h) :
      ReduceArg<reduce_t>(dim3(n_items, 1, v.size()), v.size()),
      n_items(n_items),
      n_batch(v.size()),
      h(h)
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      if (n_items > std::numeric_limits<count_t>::max())
        errorQuda("Requested size %lu greater than max supported %lu",
                  (uint64_t)n_items, (uint64_t)std::numeric_limits<count_t>::max());
      std::copy(v.begin(), v.end(), this->v);
    }
  };

  template <typename Arg> struct transform_reducer : Arg::reducer {
    using reduce_t = typename Arg::reduce_t;
    using Arg::reducer::operator();
    using count_t = decltype(Arg::n_items);

    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr transform_reducer(const Arg &arg) : arg(arg) {}

    __device__ __host__ inline reduce_t operator()(reduce_t &value, count_t i, int, int j)
    {
      auto v = arg.v[j];
      auto t = arg.h(v[i]);
      return operator()(t, value);
    }
  };

}
