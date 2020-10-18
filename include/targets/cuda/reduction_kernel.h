#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer,
            template <typename> class Reducer, typename Arg>
  __global__ void Reduction2D(Arg arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);
    Reducer<reduce_t> r;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    reduce_t reduced_value = arg.init();

    while (idx < arg.threads.x) {
      auto value = t(idx, parity);
      reduced_value = r(value, reduced_value);
      idx += blockDim.x * gridDim.x;
    }

    // perform final inter-block reduction and write out result
    reduce<block_size_x, block_size_y, decltype(r)>(arg, reduced_value);
  }

}
