#pragma once

#include <reduce_helper.h>

namespace quda {

  /**
     @brief This helper function swizzles the block index through
     mapping the block index onto a matrix and tranposing it.  This is
     done to potentially increase the cache utilization.  Requires
     that the argument class has a member parameter "swizzle" which
     determines if we are swizzling and a parameter "swizzle_factor"
     which is the effective matrix dimension that we are tranposing in
     this mapping.
   */
  template <typename Arg> __device__ constexpr int virtual_block_idx(const Arg &arg)
  {
    int block_idx = blockIdx.x;
    if (arg.swizzle) {
      // the portion of the grid that is exactly divisible by the number of SMs
      const int gridp = gridDim.x - gridDim.x % arg.swizzle_factor;

      block_idx = blockIdx.x;
      if (blockIdx.x < gridp) {
        // this is the portion of the block that we are going to transpose
        const int i = blockIdx.x % arg.swizzle_factor;
        const int j = blockIdx.x / arg.swizzle_factor;

        // transpose the coordinates
        block_idx = i * (gridp / arg.swizzle_factor) + j;
      }
    }
    return block_idx;
  }

#if 0
  template <template <typename> class Transformer, typename Arg>
  void BlockReduction2D(Arg &arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    for (int block = 0; block < arg.n_block; block++) {
      for (int j = 0; j < (int)arg.threads.y; j++) {
        reduce_t value;
        for (int i = 0; i < (int)arg.threads.x; i++) value += t(block, i, j);

        t.store(value, block, j);
      }
    }
  }
#endif

  /**
     @brief Generic block reduction kernel.  Here, we ensure that each
     thread block maps exactly to a logical block to be reduced, with
     number of threads equal to the number of sites per block.  The y
     thread dimension is a trivial vectorizable dimension, though
     until we utilize a reduce-by-key algorithm, any blockDim.y > 1
     will be erroneous.

     TODO: add a Reducer class for non summation reductions
  */
  template <int block_size, template <typename> class Transformer, typename Arg>
  __global__ void BlockReductionKernel2D(Arg arg)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    const int block = block_idx(arg);
    const int i = threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (j >= arg.threads.y) return;

    reduce_t value = t(block, i, j);

    using BlockReduce = cub::BlockReduce<reduce_t, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    value = BlockReduce(temp_storage).Sum(value);

    if (i == 0) t.store(value, block, j);
  }

}
