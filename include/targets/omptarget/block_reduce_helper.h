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
    template <typename T>
    constexpr bool enough_shared_mem(void)
    {
      constexpr auto max_nthr = device::max_block_size();
      return max_nthr*sizeof(T) <= device::max_shared_memory_size()-sizeof(target::omptarget::get_shared_cache()[0])*128;  // FIXME arbitrary, the number is arbitrary, offset 128 below & in reduce_helper.h:/reduce
    }
    /**
       @brief OpenMP reduction over a group of consecutive threads smaller than omp_num_threads()
     */
    template <typename T, typename reducer_t>
    inline T any_reduce_impl(const reducer_t &r, const T &value_, const int batch, const int block_size, const bool all, const bool async)
    {
      static_assert(enough_shared_mem<T>(), "Shared cache not large enough for tempStorage");
      T *storage = (T*)&target::omptarget::get_shared_cache()[128];  // FIXME arbitrary
      const int tid = omp_get_thread_num();
      const auto& v0 = r.init();
#if 1
      const auto batch_begin = block_size*batch;
      const auto batch_end = batch_begin+block_size;
      auto value = value_;
      for(int offset=1;offset<block_size;offset*=2){
        if(offset>1 || !async){ // only synchronize if we are not pipelining
          #pragma omp barrier
        }
        storage[tid] = value;
        const auto j = tid+offset;
        #pragma omp barrier
        const auto& v = j<batch_end ? storage[j] : v0;
        value = r(value, v);
/*
        const auto oid = (tid-batch_begin)%(offset*2);
        if(oid==offset)
          storage[tid] = value;
        const auto j = tid+offset;
        #pragma omp barrier
        if(oid==0 && j<batch_end)
          value = r(value, storage[j]);
*/
      }
      if(all){
        if(tid==block_size*batch)
          storage[tid] = value;
        #pragma omp barrier
        value = storage[block_size*batch];
        if(!async){
          #pragma omp barrier
        }
      }
#else
      const int nthr = omp_get_num_threads();
      if(!async){ // only synchronize if we are not pipelining
        #pragma omp barrier
      }
      storage[tid] = value_;
      #pragma omp barrier
      for(int offset=1;offset<block_size;offset*=2){
        #pragma omp for
        for(int i=0;i<nthr;i+=2*offset){
          const auto j = i+offset;
          const auto batch_end = block_size*(1+i/block_size);
          const auto& u = storage[i];
          const auto& v = j<batch_end ? storage[j] : v0;
          const auto& z = r(u, v);
          storage[i] = z;
        }
      }
      const auto& value = storage[block_size*batch];
#endif
      return value;
    }
    template <typename T, typename R>
    inline T any_reduce(const R &r, const T &value_, const int batch, const int block_size, const bool all, const bool async)
    {
      if constexpr (enough_shared_mem<T>())
        return any_reduce_impl(r, value_, batch, block_size, all, async);
      else{
        using V = typename T::value_type;
        if constexpr (
            std::is_same_v<typename R::reducer_t,plus<typename R::reduce_t>> &&
            std::is_same_v<T,typename R::reduce_t> &&
            std::is_same_v<T,array<V,T::N>>){
          const constexpr plus<V> re {};
          auto value = value_;
          value[0] = any_reduce_impl(re, value_[0], batch, block_size, all, async);
QUDA_UNROLL
          for (int i = 1; i < value.size(); i++) {
            value[i] = any_reduce_impl(re, value_[i], batch, block_size, all, false);
          }
          return value;
        }else
          static_assert(sizeof(T)==0, "unimplemented reduction");  // let me fail at compile time
      }
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
