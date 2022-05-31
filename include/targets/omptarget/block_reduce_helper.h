#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specialziations for
   warp- and block-level reductions, using the CUB library
 */

using namespace quda;

namespace quda
{

  namespace target
  {
    /**
       @brief OpenMP reduction over a group of consecutive threads smaller than omp_num_threads()
     */
    template <typename T, typename reducer_t>
    inline T any_reduce(const reducer_t &r, const T &value, const int batch, const int block_size, const bool all, const bool async)
    {
      constexpr max_nthr = device::max_block_size();
      static_assert(max_nthr*sizeof(T) <= device::max_shared_memory_size()-sizeof(target::omptarget::get_shared_cache()[0])*128, "Shared cache not large enough for tempStorage");  // FIXME arbitrary, the number is arbitrary, offset 128 below
      T *storage = (T*)&target::omptarget::get_shared_cache()[128];  // FIXME arbitrary
      const int nthr = omp_get_num_threads();
      const int tid = omp_get_thread_num();
      if(!async){ // only synchronize if we are not pipelining
        #pragma omp barrier
      }
      storage[tid] = value;
      #pragma omp barrier
      int offset = 1;
      const auto t0 = r.init();
      do{
        #pragma omp for
        for(int i=0;i<nthr;++i){  // TODO: benchmark against i+=offset+1
          const auto j = i+offset;
          const auto t = i/block_size==j/block_size ? storage[j] : t0;
          storage[i] = r(storage[i], t);
        }
        offset *= 2;
      }while(offset<block_size);
      return all ? storage[block_size*batch] : storage[tid];
    }
  }

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief CUDA specialization of warp_reduce, utilizing cub::WarpReduce
  */
  template <> struct warp_reduce<true> {

    /**
       @brief Perform a warp-wide reduction
       @param[in] value_ thread-local value to be reduced
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       warp (all = false)
       @param[in] r The reduction operation we want to apply
       @return The warp-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    __device__ inline T operator()(const T &value_, bool all, const reducer_t &r, const param_t &)
    {
      constexpr int block_size = device::warp_size();
      const int batch = omp_get_thread_num() / block_size;
      return target::any_reduce(r, value_, batch, block_size, all, false);
    }
  };

  // pre-declaration of block_reduce that we wish to specialize
  template <bool> struct block_reduce;

  /**
     @brief CUDA specialization of block_reduce, utilizing target::omptarget::get_shared_cache() and omp for
  */
  template <> struct block_reduce<true> {

    /**
       @brief Perform a block-wide reduction
       @param[in] value_ thread-local value to be reduced
       @param[in] async Whether this reduction will be performed
       asynchronously with respect to the calling threads
       @param[in] batch The batch index of the reduction
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       block (all = false)
       @param[in] r The reduction operation we want to apply
       @return The block-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    __device__ inline T operator()(const T &value_, bool async, int batch, bool all, const reducer_t &r, const param_t &)
    {
      const auto block_size = target::block_size<param_t::block_dim>();
      return target::any_reduce(r, value_, batch, block_size, all, async);
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
