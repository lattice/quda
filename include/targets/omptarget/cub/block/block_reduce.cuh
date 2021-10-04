#include <target_device.h>

namespace cub {

  // Naive implementation of cub collectives in OpenMP target.

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
  inline T Sum(T input)
  {
    int warp_offset = warp_id * logical_warp_size;
    int warp_num_valid = (even_warp_multiple || (warp_offset + logical_warp_size <= block_threads)) ? logical_warp_size : block_threads - warp_offset;
    T warp_aggregate = WarpReduce_(temp_storage.warp_reduce[warp_id]).Sum(input, warp_num_valid);

    if (lane_id == 0) temp_storage.warp_aggregates[warp_id] = warp_aggregate;
    #pragma omp barrier
    if (tid == 0) {
QUDA_UNROLL
      for(int i=1;i<warps;++i) warp_aggregate += temp_storage.warp_aggregates[i];
    }
    return warp_aggregate;
  }
};

}
