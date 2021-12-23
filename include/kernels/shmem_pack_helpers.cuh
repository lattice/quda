#pragma once

#ifdef NVSHMEM_COMMS
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

template <typename Arg> __device__ inline int getNeighborRank(int idx, const Arg &arg)
{
  switch (idx) {
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

template <typename Arg> __device__ inline bool do_shmempack(int dim, int dir, const Arg &arg)
{
  const int shmemidx = 2 * dim + dir;
  const bool intranode = getShmemBuffer<1, decltype(arg)>(shmemidx, arg) == nullptr;
  const bool pack_intranode = (!arg.packkernel) != (!(arg.shmem & 1));
  const bool pack_internode = (!arg.packkernel) != (!(arg.shmem & 2));
  return (arg.shmem == 0 || (intranode && pack_intranode) || (!intranode && pack_internode));
}

template <typename Arg> __device__ inline void shmem_signal(int dim, int dir, const Arg &arg)
{
  const int shmemidx = 2 * dim + dir;
  const bool intranode = getShmemBuffer<1, decltype(arg)>(shmemidx, arg) == nullptr;
  const bool pack_intranode = (!arg.packkernel) != (!(arg.shmem & 1));
  const bool pack_internode = (!arg.packkernel) != (!(arg.shmem & 2));

  bool amLast;
  if (!intranode && pack_internode) {
    __syncthreads(); // make sure all threads in this block arrived here

    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      int ticket = arg.retcount_inter[shmemidx].fetch_add(1);
      // currently CST order -- want to make sure all stores are done before and for the last block we need that
      // all uses of that data are visible
      amLast = (ticket == arg.blocks_per_dir * gridDim.y * gridDim.z - 1);
    }
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      if (amLast) {
        // send data over IB if necessary
        if (getShmemBuffer<1, decltype(arg)>(shmemidx, arg) != nullptr) shmem_putbuffer(shmemidx, arg);
        // is we are in the uber kernel signal here
        if (!arg.packkernel) {
          if (!(getNeighborRank(2 * dim + dir, arg) < 0))
            nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                               getNeighborRank(2 * dim + dir, arg));
        }
        arg.retcount_inter[shmemidx].store(0); // this could probably be relaxed
      }
    }
  }
  // if we are not in the uber kernel
  if (!intranode && !arg.packkernel && (!(arg.shmem & 2))) {
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0
        && target::block_idx().x % arg.blocks_per_dir == 0) {
      if (!(getNeighborRank(2 * dim + dir, arg) < 0))
        nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
                           getNeighborRank(2 * dim + dir, arg));
    }
  }

  if (intranode && pack_intranode) {
    __syncthreads(); // make sure all threads in this block arrived here
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      // recount has system scope
      int ticket = arg.retcount_intra[shmemidx].fetch_add(1);
      // currently CST order -- want to make sure all stores are done before (release) and check for ticket
      // acquires. For the last block we need that all uses of that data are visible
      amLast = (ticket == arg.blocks_per_dir * gridDim.y * gridDim.z - 1);
    }
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
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


template <typename Arg> __device__ inline void shmem_signal2(int dim, int dir, const Arg &arg)
{
  const int shmemidx = 2 * dim + dir;
  const bool intranode = getShmemBuffer<1, decltype(arg)>(shmemidx, arg) == nullptr;
  
  bool amLast;
  if (!intranode) {
    __syncthreads(); // make sure all threads in this block arrived here
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      int ticket = arg.retcount_inter[shmemidx].fetch_add(1);
      // currently CST order -- want to make sure all stores are done before and for the last block we need that
      // all uses of that data are visible
      amLast = (ticket == arg.blocks_per_dir * gridDim.y * gridDim.z - 1);
    }
    if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
      if (amLast) {
        // send data over IB if necessary
        if (getShmemBuffer<1, decltype(arg)>(shmemidx, arg) != nullptr) shmem_putbuffer(shmemidx, arg);
        // is we are in the uber kernel signal here

          // if (!(getNeighborRank(2 * dim + dir, arg) < 0))
          //   nvshmemx_signal_op(arg.sync_arr + 2 * dim + (1 - dir), arg.counter, NVSHMEM_SIGNAL_SET,
          //                      getNeighborRank(2 * dim + dir, arg));
        arg.retcount_inter[shmemidx].store(0); // this could probably be relaxed
      }
    }
  }
}
#endif