#pragma once
#ifdef NVSHMEM_COMMS
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace quda
{
  /**
   * @brief Helper to avoid dynamic indexing into the packBuffer
   *
   * @tparam dest 1 for local pack buffer for staging IB packing, 0 for remote buffer used over NVLink
   * @tparam Arg kernelArguments
   * @param shmemindex  dimension / direction we want to get the buffer for
   * @param arg kernelArguments
   */
  template <int dest, typename Arg> __device__ inline void *getShmemBuffer(int shmemindex, const Arg &arg)
  {
    switch (shmemindex) {
    case 0: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 0]);
    case 1: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 1]);
    case 2: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 2]);
    case 3: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 3]);
    case 4: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 4]);
    case 5: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 5]);
    case 6: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 6]);
    case 7: return static_cast<void *>(arg.packBuffer[dest * 2 * QUDA_MAX_DIM + 7]);
    default: return nullptr;
    }
  }

  /**
   * @brief Get the rank of your neighbor in idx dimdir, avoids dynamic indexing in kernel
   *
   * @tparam Arg kernelArguments
   * @param shmemindex dimension / direction we want to get the rank for
   * @param arg kernelArguments
   * @return rank of the neighbor
   */
  template <typename Arg> __device__ inline int getNeighborRank(int shmemindex, const Arg &arg)
  {
    switch (shmemindex) {
    case 0: return arg.neighbor_ranks[0];
    case 1: return arg.neighbor_ranks[1];
    case 2: return arg.neighbor_ranks[2];
    case 3: return arg.neighbor_ranks[3];
    case 4: return arg.neighbor_ranks[4];
    case 5: return arg.neighbor_ranks[5];
    case 6: return arg.neighbor_ranks[6];
    case 7: return arg.neighbor_ranks[7];
    default: return -1;
    }
  }

  /**
   * @brief helper to send the data (nvshmem_putmem) from packing to remote buffer used fer IB
   *
   * @tparam Arg kernelArguments
   * @param shmemindex dimension / direction we want to get the rank for
   * @param arg kernelArguments
   */
  template <typename Arg> __device__ inline void shmem_putbuffer(int shmemindex, const Arg &arg)
  {
    switch (shmemindex) {
    case 0:
      nvshmem_putmem_nbi(getShmemBuffer<1>(0, arg), getShmemBuffer<0>(0, arg), arg.bytes[0], arg.neighbor_ranks[0]);
      return;
    case 1:
      nvshmem_putmem_nbi(getShmemBuffer<1>(1, arg), getShmemBuffer<0>(1, arg), arg.bytes[1], arg.neighbor_ranks[1]);
      return;
    case 2:
      nvshmem_putmem_nbi(getShmemBuffer<1>(2, arg), getShmemBuffer<0>(2, arg), arg.bytes[2], arg.neighbor_ranks[2]);
      return;
    case 3:
      nvshmem_putmem_nbi(getShmemBuffer<1>(3, arg), getShmemBuffer<0>(3, arg), arg.bytes[3], arg.neighbor_ranks[3]);
      return;
    case 4:
      nvshmem_putmem_nbi(getShmemBuffer<1>(4, arg), getShmemBuffer<0>(4, arg), arg.bytes[4], arg.neighbor_ranks[4]);
      return;
    case 5:
      nvshmem_putmem_nbi(getShmemBuffer<1>(5, arg), getShmemBuffer<0>(5, arg), arg.bytes[5], arg.neighbor_ranks[5]);
      return;
    case 6:
      nvshmem_putmem_nbi(getShmemBuffer<1>(6, arg), getShmemBuffer<0>(6, arg), arg.bytes[6], arg.neighbor_ranks[6]);
      return;
    case 7:
      nvshmem_putmem_nbi(getShmemBuffer<1>(7, arg), getShmemBuffer<0>(7, arg), arg.bytes[7], arg.neighbor_ranks[7]);
      return;
    default: return;
    }
  }

  /**
   * @brief Check if for the given arg.shmem (deciding on which policy we use) and whether we
   * are in the fused pack+interior kernel or in a separate packing kernel do
   * intra-node packing (and/or) inter-node packing
   * @tparam Arg kernel arguments
   * @param dim communication dimension which we are working on
   * @param dir communication direction which we are working on
   * @param arg kernel argument structure
   * @return whether to pack for that dim/dir
   */
  template <typename Arg> __device__ inline bool do_shmempack(int dim, int dir, const Arg &arg)
  {
    const int shmemidx = 2 * dim + dir;
    const bool intranode = getShmemBuffer<1, decltype(arg)>(shmemidx, arg) == nullptr;
    const bool pack_intranode = (!arg.packkernel) != (!(arg.shmem & 1));
    const bool pack_internode = (!arg.packkernel) != (!(arg.shmem & 2));
    return (arg.shmem == 0 || (intranode && pack_intranode) || (!intranode && pack_internode));
  }

  /**
   * @brief this function ensures all packing is done, then sends the signal and
   * also the data over IB when using nvshmem
   * this currently operates on arg.sync_arr, i.e. is used for dslash
   * @tparam Arg kernel arguments
   * @param dim communication dimension which we are working on
   * @param dir communication direction which we are working on
   * @param arg kernel argument structure
   */
  template <typename Arg> __device__ inline void shmem_signal(int dim, int dir, const Arg &arg)
  {
    const int shmemidx = 2 * dim + dir;
    const bool intranode = getShmemBuffer<1, decltype(arg)>(shmemidx, arg) == nullptr;

    // depending on the chosen policy (arg.shmem value) decide and whether we
    // are in the pack kernel or in the fused interior+pack we use these to
    // figure whether we do intra-node / inter-node packing
    const bool pack_intranode = (!arg.packkernel) != (!(arg.shmem & 1));
    const bool pack_internode = (!arg.packkernel) != (!(arg.shmem & 2));

    bool amLast;
    if (!intranode && pack_internode) {

      __syncthreads(); // make sure all threads in this block arrived here

      if (quda::target::thread_idx().x == 0 && quda::target::thread_idx().y == 0 && quda::target::thread_idx().z == 0) {
        int ticket = arg.retcount_inter[shmemidx].fetch_add(1);
        // currently CST order -- want to make sure all stores are done before and for the last block we need that
        // all uses of that data are visible
        amLast = (ticket == arg.blocks_per_dir * gridDim.y * gridDim.z - 1);
      }
      if (quda::target::thread_idx().x == 0 && quda::target::thread_idx().y == 0 && quda::target::thread_idx().z == 0) {
        if (amLast) {
          // send data over IB if necessary
          if (getShmemBuffer<1, decltype(arg)>(shmemidx, arg) != nullptr) shmem_putbuffer(shmemidx, arg);
          // if we fused the inter-node packing in the interior+pack kernel we can signal here
          if (!arg.packkernel) {
            if (!(getNeighborRank(2 * dim + dir, arg) < 0))
              nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                                 getNeighborRank(2 * dim + dir, arg));
          }
          arg.retcount_inter[shmemidx].store(0); // this could probably be relaxed
        }
      }
    }
    // if we are using a separate packing kernel for inter-node traffic we
    // anyway issue the signal in the fused packing+interior part to avoid
    // potential hangs in the tune process of the fully fused kernel
    if (!intranode && !arg.packkernel && (!(arg.shmem & 2))) {
      if (quda::target::thread_idx().x == 0 && quda::target::thread_idx().y == 0 && quda::target::thread_idx().z == 0
          && quda::target::block_idx().x % arg.blocks_per_dir == 0) {
        if (!(getNeighborRank(2 * dim + dir, arg) < 0))
          nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                             getNeighborRank(2 * dim + dir, arg));
      }
    }

    // this part is handling the intranode traffic using direct load store, data
    // is already packed to the remote memory and we just need to signal
    if (intranode && pack_intranode) {
      __syncthreads(); // make sure all threads in this block arrived here
      if (quda::target::thread_idx().x == 0 && quda::target::thread_idx().y == 0 && quda::target::thread_idx().z == 0) {
        // recount has system scope
        int ticket = arg.retcount_intra[shmemidx].fetch_add(1);
        // currently CST order -- want to make sure all stores are done before (release) and check for ticket
        // acquires. For the last block we need that all uses of that data are visible
        amLast = (ticket == arg.blocks_per_dir * gridDim.y * gridDim.z - 1);
      }
      if (quda::target::thread_idx().x == 0 && quda::target::thread_idx().y == 0 && quda::target::thread_idx().z == 0) {
        if (amLast) {
          if (arg.shmem & 8) {
            if (!(getNeighborRank(2 * dim + dir, arg) < 0))
              nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                                 getNeighborRank(2 * dim + dir, arg));
          }
          arg.retcount_intra[shmemidx].store(0); // this could probably be relaxed
        }
      }
    }
  }

  /**
   * @brief helper to send the data from packing to remote buffer used fer IB and put the signal in one call
   * this currently operates on arg.sync_arr, i.e. is used for exchange ghost
   *
   * @tparam Arg kernel arguments
   * @param shmemindex  communication dimension/dir which we are working on
   * @param arg kernel argument structure
   */
  template <typename Arg> __device__ inline void shmem_putbuffer_signal(int shmemindex, Arg &arg)
  {
    switch (shmemindex) {
    case 0:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(0, arg), getShmemBuffer<0>(0, arg), arg.bytes[0], arg.sync_arr + 1,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[0]);
      return;
    case 1:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(1, arg), getShmemBuffer<0>(1, arg), arg.bytes[1], arg.sync_arr + 0,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[1]);
      return;
    case 2:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(2, arg), getShmemBuffer<0>(2, arg), arg.bytes[2], arg.sync_arr + 3,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[2]);
      return;
    case 3:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(3, arg), getShmemBuffer<0>(3, arg), arg.bytes[3], arg.sync_arr + 2,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[3]);
      return;
    case 4:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(4, arg), getShmemBuffer<0>(4, arg), arg.bytes[4], arg.sync_arr + 5,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[4]);
      return;
    case 5:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(5, arg), getShmemBuffer<0>(5, arg), arg.bytes[5], arg.sync_arr + 4,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[5]);
      return;
    case 6:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(6, arg), getShmemBuffer<0>(6, arg), arg.bytes[6], arg.sync_arr + 7,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[6]);
      return;
    case 7:
      nvshmem_putmem_signal_nbi(getShmemBuffer<1>(7, arg), getShmemBuffer<0>(7, arg), arg.bytes[7], arg.sync_arr + 6,
                                arg.counter, NVSHMEM_SIGNAL_SET, arg.neighbor_ranks[7]);
      return;
    default: return;
    }
  }

  /**
   * @brief this function ensures all packing is done, then sends the signal and
   * also the data over IB when using nvshmem
   * this currently operates on arg.sync_arr + 2 * QUDA_MAX_DIM, i.e. is used for
   * exchange ghost
   * @tparam Arg kernel arguments
   * @param dim communication dimension which we are working on
   * @param dir communication direction which we are working on
   * @param wait whether to also wait on the signal (immediately after sending it)
   * @param arg kernel argument structure
   */
  template <typename Arg> __device__ inline void shmem_signalwait(int dim, int dir, bool wait, const Arg &arg)
  {

    bool amLast = false;

    __syncthreads(); // make sure all threads in this block arrived here
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      int ticket = arg.retcount_intra[0].fetch_add(1);
      amLast = (ticket == target::grid_dim().x * target::grid_dim().y * target::grid_dim().z - 1);
    }
    auto my_tile = cg::coalesced_threads();

    // This operation will be performed by only the
    // first 32-thread tile of each block
    if (my_tile.meta_group_rank() == 0) {
      amLast = my_tile.shfl(amLast, 0);
      my_tile.sync();
    }
    if (amLast && my_tile.meta_group_rank() == 0) {

      for (int shmemidx = my_tile.thread_rank(); shmemidx < 8; shmemidx += my_tile.size()) {
        dim = shmemidx / 2;
        dir = shmemidx % 2;

        if (!(getNeighborRank(2 * dim + dir, arg) < 0)) {
          if (getShmemBuffer<1, decltype(arg)>(shmemidx, arg) != nullptr) {
            // for IB transfers we packed to a local buffer so send data over IB and put the signal
            shmem_putbuffer_signal(shmemidx, arg);
          } else {
            // for direct remote store we already packed to remote memory, no need to send, just do a signal
            nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                               getNeighborRank(2 * dim + dir, arg));
          }
        }
      }
      if (my_tile.thread_rank() == 0) arg.retcount_intra[0].store(0); // this could probably be relaxed
      my_tile.sync();

      for (int shmemidx = my_tile.thread_rank(); shmemidx < 8; shmemidx += my_tile.size()) {
        dim = shmemidx / 2;
        dir = shmemidx % 2;
        if (wait && !(getNeighborRank(2 * dim + dir, arg) < 0)) {
          nvshmem_signal_wait_until((arg.sync_arr + shmemidx), NVSHMEM_CMP_GE, arg.waitcounter);
        }
      }
    }
  }
} // namespace quda
#endif
