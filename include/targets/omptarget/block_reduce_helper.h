#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specialziations for
   warp- and block-level reductions, using the CUB library
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

using namespace quda;

namespace cub {

  // Naive implementation of cub collectives in OpenMP target.
  // FIXME: use team shared memory

  enum BlockReduceAlgorithm {BLOCK_REDUCE_WARP_REDUCTIONS};

  constexpr int get_warp_steps(int s)
  {
    // log2(s)
    return s==1?0:s==2?1:s==4?2:s==8?3:s==16?4:s==32?5:s==64?6:-1;
  }

// https://github.com/NVlabs/cub/blob/1.8.0/cub/warp/specializations/warp_reduce_smem.cuh
template <typename T, int warp_threads, int ptx_arch = __COMPUTE_CAPABILITY__>
struct WarpReduce
{
  static constexpr int warp_steps = get_warp_steps(warp_threads);
  static_assert(warp_steps>=0, "unsupported warp_threads");
  static constexpr int half_warp_threads = 1 << (warp_steps-1);
  static constexpr int warp_smem_elments = warp_threads + half_warp_threads;
  struct TempStorage
  {
    T reduce[warp_smem_elments];
  };
  TempStorage &temp_storage;
  int tid,lane_id;
  inline WarpReduce(TempStorage &temp_storage) :
    temp_storage(temp_storage),
    tid(omp_get_thread_num()),
    lane_id(tid%warp_threads)
  {}
  template <typename ReductionOp>
  inline T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {
    QUDA_UNROLL
    for(int i=0;i<warp_steps;++i){
      temp_storage.reduce[lane_id] = input;
      const int offset = 1 << i;
      // this is bad, but we don't have warp barrier
      #pragma omp barrier
      if((lane_id + offset) < valid_items)
        input = reduction_op(input, temp_storage.reduce[lane_id+offset]);
    }
    return input;
  }
  inline T Sum(T input, int valid_items = warp_threads)  // FIXME warp_num_valid < warp_threads
  {
    QUDA_UNROLL
    for(int i=0;i<warp_steps;++i){
      temp_storage.reduce[lane_id] = input;
      const int offset = 1 << i;
      // this is bad, but we don't have warp barrier
      #pragma omp barrier
      if((lane_id + offset) < valid_items)
        input += temp_storage.reduce[lane_id+offset];
    }
    return input;
  }
};

template <typename T, int warp_threads, int ptx_arch = __COMPUTE_CAPABILITY__>
struct WarpScan
{
  struct TempStorage
  {
    T v;
  };
  TempStorage &temp_storage;
  int lane_id;
  inline WarpScan(TempStorage &temp_storage) :
    temp_storage(temp_storage),
    lane_id(omp_get_thread_num()%warp_threads)
  {}
  inline T Broadcast(T input, unsigned int src_lane)
  {
    if (lane_id == src_lane)
      temp_storage.v = input;
    #pragma omp barrier
    return temp_storage.v;
  }
};

// https://github.com/NVlabs/cub/blob/1.8.0/cub/block/specializations/block_reduce_warp_reductions.cuh
template <typename T, int block_dim_x, BlockReduceAlgorithm alg = BLOCK_REDUCE_WARP_REDUCTIONS, int block_dim_y = 1, int block_dim_z = 1, int ptx_arch = __COMPUTE_CAPABILITY__>
struct BlockReduce
{
  static constexpr int block_threads = block_dim_x*block_dim_y*block_dim_z;
  static constexpr int warp_threads = device::warp_size();
  static constexpr int warps = (block_threads + warp_threads - 1) / warp_threads;
  static constexpr int logical_warp_size = std::min(block_threads, warp_threads);
  static constexpr bool even_warp_multiple = block_threads%logical_warp_size == 0;
  typedef WarpReduce<T, logical_warp_size, ptx_arch> WarpReduce_;
  struct TempStorage
  {
    typename WarpReduce_::TempStorage warp_reduce[warps];
    T warp_aggregates[warps];
  };
  TempStorage &temp_storage;
  int tid,lane_id,warp_id;
  inline BlockReduce(TempStorage &temp_storage) :
    temp_storage(temp_storage),
    tid(omp_get_thread_num()),
    lane_id(tid%logical_warp_size),
    warp_id(tid/logical_warp_size)
  {}
  template <typename ReductionOp>
  inline T Reduce(T input, ReductionOp reduction_op)
  {
    int warp_offset = warp_id * logical_warp_size;
    int warp_num_valid = (even_warp_multiple || (warp_offset + logical_warp_size <= block_threads)) ? logical_warp_size : block_threads - warp_offset;
    T warp_aggregate = WarpReduce_(temp_storage.warp_reduce[warp_id]).Reduce(input, warp_num_valid, reduction_op);

    if (lane_id == 0) temp_storage.warp_aggregates[warp_id] = warp_aggregate;
    #pragma omp barrier
    if (tid == 0) {
QUDA_UNROLL
      for(int i=1;i<warps;++i) warp_aggregate = reduction_op(warp_aggregate, temp_storage.warp_aggregates[i]);
    }
    return warp_aggregate;
  }
  inline T Sum(T input)
  {
    return Reduce(input, plus<T>());
  }
};

}

namespace quda
{

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
      using warp_reduce_t = cub::WarpReduce<T, param_t::width, __COMPUTE_CAPABILITY__>;
      using tempStorage = typename warp_reduce_t::TempStorage;
      static_assert(sizeof(tempStorage) <= sizeof(target::omptarget::shared_cache.addr[0])*1024, "Shared cache not large enough for tempStorage");  // FIXME arbitrary, 1024 is used below
      tempStorage *dummy_storage = (tempStorage*)&target::omptarget::shared_cache.addr[1024];
      warp_reduce_t warp_reduce(*dummy_storage);
      T value = reducer_t::do_sum ? warp_reduce.Sum(value_) : warp_reduce.Reduce(value_, r);

      if (all) {
        using warp_scan_t = cub::WarpScan<T, param_t::width, __COMPUTE_CAPABILITY__>;
        using tempStorage = typename warp_scan_t::TempStorage;
        static_assert(sizeof(tempStorage) <= sizeof(target::omptarget::shared_cache.addr[0])*1024, "Shared cache not large enough for tempStorage");  // FIXME arbitrary, 1024 is arbitrary
        tempStorage *dummy_storage = (tempStorage*)&target::omptarget::shared_cache.addr[2048];
        // typename warp_scan_t::TempStorage dummy_storage;
        warp_scan_t warp_scan(*dummy_storage);
        value = warp_scan.Broadcast(value, 0);
      }

      return value;
    }
  };

  // pre-declaration of block_reduce that we wish to specialize
  template <bool> struct block_reduce;

  /**
     @brief CUDA specialization of block_reduce, utilizing cub::BlockReduce
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
    __device__ inline T operator()(const T &value_, bool async, int batch, bool all, const reducer_t &r,
                                   const param_t &)
    {
      constexpr int nthr = param_t::block_size_x*param_t::block_size_y*(param_t::batch_size>param_t::block_size_z?param_t::batch_size:param_t::block_size_z);
      int tid = omp_get_thread_num();
      static_assert(nthr*sizeof(T) <= sizeof(target::omptarget::shared_cache.addr[0])*8192, "Shared cache not large enough for tempStorage");  // FIXME arbitrary, the number is arbitrary, do we have that many?
      T *storage = (T*)&target::omptarget::shared_cache.addr[128];  // FIXME arbitrary
      // #pragma omp barrier
      storage[tid] = value_;
      // #pragma omp barrier
      int offset = 1;
      constexpr int bs = param_t::batch_size;
      static_assert(nthr%bs==0, "Block size not divisible by batch_size");
      constexpr int subblock = nthr/bs;
      auto t0 = r.init();
      do{
        #pragma omp for
        for(int i=0;i<nthr;++i){
          auto j = i+offset;
          auto t = i/subblock==j/subblock ? storage[j] : t0;
          storage[i] = r(storage[i], t);
        }
        // #pragma omp barrier
        offset *= 2;
      }while(offset<subblock);
      // #pragma omp barrier
      T value = storage[tid];
      if(all)
        value = storage[subblock*(tid/subblock)];
      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
