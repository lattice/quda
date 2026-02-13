#include <cuda/ptx>
#include <target_device.h>

namespace quda
{

  namespace ptx = cuda::ptx;

  template <int dim> struct work_steal {
    uint4 *result; // Request result (points to block __shared__).
    uint64_t *bar; // Synchronization barrier (points to block __shared__).
    int phase = 0; // Synchronization barrier phase.
    bool last_success_ = false; // result of last complete() (true => we got a next block to work on)
    dim3 next_block_idx_ {0, 0,
                          0}; // next block to work on (set by get_block_idx(), read by kernel after functor returns)

    __device__ __forceinline__ work_steal(uint4 *result_, uint64_t *bar_) : result(result_), bar(bar_)
    {
      if (target::thread_idx_linear<dim>() == 0) ptx::mbarrier_init(bar, 1);
    }

    __device__ __forceinline__ void request()
    {
      __syncthreads();
      if (target::is_thread_zero<dim>()) { // One thread per block does request + arrive (same thread for both)
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
        ptx::clusterlaunchcontrol_try_cancel(result, bar);
        ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, bar, sizeof(uint4));
      }
    }

    __device__ __forceinline__ bool complete()
    {
      // Cancellation request synchronization:
      while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, bar, phase)) { }
      phase ^= 1;

      // Cancellation request decoding:
      last_success_ = ptx::clusterlaunchcontrol_query_cancel_is_canceled(*result);
      return last_success_;
    }

    __device__ __forceinline__ dim3 get_block_idx()
    {
      dim3 block_idx;
      block_idx.x = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(*result);
      block_idx.y = dim >= 2 ? ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y<int>(*result) : 0;
      block_idx.z = dim >= 3 ? ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z<int>(*result) : 0;
      next_block_idx_ = block_idx;
      return block_idx;
    }

    constexpr bool last_success() const { return last_success_; }
    constexpr dim3 next_block_idx() const { return next_block_idx_; }

    __device__ __forceinline__ void release()
    {
      // Release read of result to the async proxy:
      ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
  };

} // namespace quda
