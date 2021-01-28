#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg>
  __global__ void Reduction2D(Arg arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      idx += blockDim.x * gridDim.x;
    }

    // perform final inter-block reduction and write out result
    reduce<block_size_x, block_size_y>(arg, t, value);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg>
  __global__ void MultiReduction(Arg arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z;

    if (j >= arg.threads.y) return;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j, k);
      idx += blockDim.x * gridDim.x;
    }

    // perform final inter-block reduction and write out result
    reduce<block_size_x, block_size_y>(arg, t, value, j);
  }

}
