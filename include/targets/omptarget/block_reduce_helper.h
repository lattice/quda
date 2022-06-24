#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the OpenMP target specializations for
   warp- and block-level reductions
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
        constexpr auto N = T::N;
        if constexpr (
            std::is_same_v<typename R::reducer_t,plus<typename R::reduce_t>> &&
            std::is_same_v<T,typename R::reduce_t> &&
            std::is_same_v<T,array<V,N>>){
          // make sure the implementation is still compatible: ../../array.h
          constexpr auto N0 = N/2;
          constexpr auto N1 = N-N0;
          using T0 = array<V,N0>;
          using T1 = array<V,N1>;
          const constexpr plus<T0> r0 {};
          const constexpr plus<T1> r1 {};
          auto value = value_;
          // recurse to myself
          reinterpret_cast<T0&>(value[0]) = any_reduce(r0, reinterpret_cast<const T0&>(value_[0]), batch, block_size, all, async);
          // recurse with async==false
          reinterpret_cast<T1&>(value[N0]) = any_reduce(r1, reinterpret_cast<const T1&>(value_[N0]), batch, block_size, all, false);
          return value;
        }else
          static_assert(sizeof(T)==0, "unimplemented reduction");  // let me fail at compile time
      }
    }
  }

  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief OpenMP target specialization of warp_reduce
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
     @brief OpenMP target  specialization of block_reduce
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
