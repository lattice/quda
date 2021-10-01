#pragma once

#include <color_spinor_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <dslash_helper.cuh>
#include <shmem_helper.cuh>

namespace quda
{
  int *getPackComms();

  template <typename Float_, int nColor_, int nSpin_, bool spin_project_ = true,
            bool dagger_ = false, int twist_ = 0, QudaPCType pc_type_ = QUDA_4D_PC>
  struct PackArg : kernel_param<> {

    typedef Float_ Float;
    typedef typename mapper<Float>::type real;

    static constexpr int nColor = nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr bool spin_project = (nSpin == 4 && spin_project_ ? true : false);
    static constexpr bool dagger = dagger_;
    static constexpr int twist = twist_; // whether we are doing preconditioned twisted-mass or not (1 - singlet, 2 - doublet)
    static constexpr QudaPCType pc_type = pc_type_; // preconditioning type (4-d or 5-d)

    static constexpr bool spinor_direct_load = false; // false means texture load

    static constexpr bool packkernel = true;
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    const F in_pack; // field we are packing

    const int nFace;
    const int parity;         // only use this for single parity fields
    const int nParity;        // number of parities we are working on

    const DslashConstant dc; // pre-computed dslash constants for optimized indexing

    real twist_a; // preconditioned twisted-mass scaling parameter
    real twist_b; // preconditioned twisted-mass chiral twist factor
    real twist_c; // preconditioned twisted-mass flavor twist factor

    int_fastdiv work_items;
    int threadDimMapLower[4];
    int threadDimMapUpper[4];

    int_fastdiv blocks_per_dir;
    int dim_map[4];

    int sites_per_block;

    char *packBuffer[4 * QUDA_MAX_DIM];
    int neighbor_ranks[2 * QUDA_MAX_DIM];
    int bytes[2 * QUDA_MAX_DIM];
    // shmem bitfield encodes
    // 0 - no shmem
    // 1 - pack P2P
    // 2 - pack IB
    // 3 - pack P2P + IB
    // 8 - barrier part I (just the put part)
    // 16 - barrier part II (wait on shmem to complete, all directions) -- not implemented
    dslash::shmem_sync_t counter;
#ifdef NVSHMEM_COMMS
    int shmem;

    dslash::shmem_sync_t *sync_arr;
    dslash::shmem_retcount_intra_t *retcount_intra;
    dslash::shmem_retcount_inter_t *retcount_inter;
#else
    static constexpr int shmem = 0;
#endif
    PackArg(void **ghost, const ColorSpinorField &in, int nFace, int parity, int work_items, double a,
            double b, double c, unsigned int block, unsigned int grid,
#ifdef NVSHMEM_COMMS
            int shmem_) :
#else
            int) :
#endif
      kernel_param(dim3(block * grid, in.getDslashConstant().Ls, in.SiteSubset())),
      in_pack(in, nFace, nullptr, nullptr, reinterpret_cast<Float **>(ghost)),
      nFace(nFace),
      parity(parity),
      nParity(in.SiteSubset()),
      dc(in.getDslashConstant()),
      twist_a(a),
      twist_b(b),
      twist_c(c),
      work_items(work_items),
      threadDimMapLower{ },
      threadDimMapUpper{ },
      sites_per_block((work_items + grid - 1) / grid)
#ifdef NVSHMEM_COMMS
      ,
      counter(dslash::get_shmem_sync_counter()),
      shmem(shmem_),
      sync_arr(dslash::get_shmem_sync_arr()),
      retcount_intra(dslash::get_shmem_retcount_intra()),
      retcount_inter(dslash::get_shmem_retcount_inter())
#endif
    {
      for (int i = 0; i < 4 * QUDA_MAX_DIM; i++) { packBuffer[i] = static_cast<char *>(ghost[i]); }
      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) {
          neighbor_ranks[2 * dim + dir] = comm_neighbor_rank(dir, dim);
          bytes[2 * dim + dir] = in.GhostFaceBytes(dim);
        }
      }
      if (!in.isNative()) errorQuda("Unsupported field order colorspinor=%d\n", in.FieldOrder());

      int d = 0;
      int prev = -1; // previous dimension that was partitioned
      for (int i = 0; i < 4; i++) {
        if (!getPackComms()[i]) continue;
        threadDimMapLower[i] = (prev >= 0 ? threadDimMapUpper[prev] : 0);
        threadDimMapUpper[i] = threadDimMapLower[i] + 2 * nFace * dc.ghostFaceCB[i];
        prev = i;

        dim_map[d++] = i;
      }
      blocks_per_dir = grid / (2 * d);
    }
  };

  template <bool dagger, int twist, int dim, QudaPCType pc, typename Arg>
  __device__ __host__ inline void pack(const Arg &arg, int ghost_idx, int s, int parity)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, Arg::nSpin> Vector;
    constexpr int nFace = 1;

    // this means we treat 4-d preconditioned fields as 4-d fields,
    // and don't fold in any fifth dimension until after we have
    // computed the 4-d indices (saves division)
    constexpr int nDim = pc;

    // for 5-d preconditioning the face_size includes the Ls dimension
    const int face_size = nFace * arg.dc.ghostFaceCB[dim] * (pc == QUDA_5D_PC ? arg.dc.Ls : 1);

    int spinor_parity = (arg.nParity == 2) ? parity : 0;

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face

    // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
    const int face_num = (ghost_idx >= face_size) ? 1 : 0;
    ghost_idx -= face_num * face_size;

    if (face_num == 0) { // backwards

      int idx = indexFromFaceIndex<nDim, pc, dim, nFace, 0>(ghost_idx, parity, arg);
      constexpr int proj_dir = dagger ? +1 : -1;
      Vector f = arg.in_pack(idx + s * arg.dc.volume_4d_cb, spinor_parity);
      if (twist == 1) {
        f = arg.twist_a * (f + arg.twist_b * f.igamma(4));
      } else if (twist == 2) {
        Vector f1 = arg.in_pack(idx + (1 - s) * arg.dc.volume_4d_cb, spinor_parity); // load other flavor
        if (s == 0)
          f = arg.twist_a * (f + arg.twist_b * f.igamma(4) + arg.twist_c * f1);
        else
          f = arg.twist_a * (f - arg.twist_b * f.igamma(4) + arg.twist_c * f1);
      }
      if (arg.spin_project) {
        arg.in_pack.Ghost(dim, 0, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f.project(dim, proj_dir);
      } else {
        arg.in_pack.Ghost(dim, 0, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f;
      }
    } else { // forwards

      int idx = indexFromFaceIndex<nDim, pc, dim, nFace, 1>(ghost_idx, parity, arg);
      constexpr int proj_dir = dagger ? -1 : +1;
      Vector f = arg.in_pack(idx + s * arg.dc.volume_4d_cb, spinor_parity);
      if (twist == 1) {
        f = arg.twist_a * (f + arg.twist_b * f.igamma(4));
      } else if (twist == 2) {
        Vector f1 = arg.in_pack(idx + (1 - s) * arg.dc.volume_4d_cb, spinor_parity); // load other flavor
        if (s == 0)
          f = arg.twist_a * (f + arg.twist_b * f.igamma(4) + arg.twist_c * f1);
        else
          f = arg.twist_a * (f - arg.twist_b * f.igamma(4) + arg.twist_c * f1);
      }
      if (arg.spin_project) {
        arg.in_pack.Ghost(dim, 1, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f.project(dim, proj_dir);
      } else {
        arg.in_pack.Ghost(dim, 1, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f;
      }
    }
  }

  template <int dim, int nFace = 1, typename Arg>
  __device__ __host__ inline void packStaggered(const Arg &arg, int ghost_idx, int s, int parity)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, Arg::nSpin> Vector;

    int spinor_parity = (arg.nParity == 2) ? parity : 0;

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor and write spinor to face buffer

    // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
    const int face_num = (ghost_idx >= nFace * arg.dc.ghostFaceCB[dim]) ? 1 : 0;
    ghost_idx -= face_num * nFace * arg.dc.ghostFaceCB[dim];

    if (face_num == 0) { // backwards
      int idx = indexFromFaceIndexStaggered<4, QUDA_4D_PC, dim, nFace, 0>(ghost_idx, parity, arg);
      Vector f = arg.in_pack(idx + s * arg.dc.volume_4d_cb, spinor_parity);
      arg.in_pack.Ghost(dim, 0, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f;
    } else { // forwards
      int idx = indexFromFaceIndexStaggered<4, QUDA_4D_PC, dim, nFace, 1>(ghost_idx, parity, arg);
      Vector f = arg.in_pack(idx + s * arg.dc.volume_4d_cb, spinor_parity);
      arg.in_pack.Ghost(dim, 1, ghost_idx + s * arg.dc.ghostFaceCB[dim], spinor_parity) = f;
    }
  }

  template <typename Arg> struct pack_wilson {
    const Arg &arg;
    constexpr pack_wilson(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int, int s, int parity)
    {
      int local_tid = target::thread_idx().x;
      int tid = arg.sites_per_block * target::block_idx().x + local_tid;
      // this is the parity used for load/store, but we use arg.parity for index mapping
      if (arg.nParity == 1) parity = arg.parity;

      while (local_tid < arg.sites_per_block && tid < arg.work_items) {
        // determine which dimension we are packing
        int ghost_idx;
        const int dim = dimFromFaceIndex(ghost_idx, tid, arg);

        if (Arg::pc_type == QUDA_5D_PC) { // 5-d checkerboarded, include s (not ghostFaceCB since both faces)
          switch (dim) {
          case 0: pack<Arg::dagger, Arg::twist, 0, Arg::pc_type>(arg, ghost_idx + s * arg.dc.ghostFace[0], 0, parity); break;
          case 1: pack<Arg::dagger, Arg::twist, 1, Arg::pc_type>(arg, ghost_idx + s * arg.dc.ghostFace[1], 0, parity); break;
          case 2: pack<Arg::dagger, Arg::twist, 2, Arg::pc_type>(arg, ghost_idx + s * arg.dc.ghostFace[2], 0, parity); break;
          case 3: pack<Arg::dagger, Arg::twist, 3, Arg::pc_type>(arg, ghost_idx + s * arg.dc.ghostFace[3], 0, parity); break;
          }
        } else { // 4-d checkerboarding, keeping s separate (if it exists)
          switch (dim) {
          case 0: pack<Arg::dagger, Arg::twist, 0, Arg::pc_type>(arg, ghost_idx, s, parity); break;
          case 1: pack<Arg::dagger, Arg::twist, 1, Arg::pc_type>(arg, ghost_idx, s, parity); break;
          case 2: pack<Arg::dagger, Arg::twist, 2, Arg::pc_type>(arg, ghost_idx, s, parity); break;
          case 3: pack<Arg::dagger, Arg::twist, 3, Arg::pc_type>(arg, ghost_idx, s, parity); break;
          }
        }

        local_tid += target::block_dim().x;
        tid += target::block_dim().x;
      } // while tid
    }
  };

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
      if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0 && target::block_idx().x % arg.blocks_per_dir == 0) {
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
#endif

  // shmem bitfield encodes
  // 0 - no shmem
  // 1 - pack P2P (merged in interior)
  // 2 - pack IB (merged in interior)
  // 3 - pack P2P + IB (merged in interior)
  // 8 - barrier part I (packing) (merged in interior, only useful if packing) -- currently required
  // 16 - barrier part II (spin exterior) (merged in exterior) -- currently required
  // 32 - use packstream -- not used
  // 64 - use uber kernel (merge exterior)
  template <bool dagger, QudaPCType pc, typename Arg> struct packShmem {

    template <int twist> __device__ __forceinline__ void operator()(const Arg &arg, int s, int parity)
    {
      // (active_dims * 2 + dir) * blocks_per_dir + local_block_idx
      int local_block_idx = target::block_idx().x % arg.blocks_per_dir;
      int dim_dir = target::block_idx().x / arg.blocks_per_dir;
      int dir = dim_dir % 2;
      int dim;
      switch (dim_dir / 2) {
      case 0: dim = arg.dim_map[0]; break;
      case 1: dim = arg.dim_map[1]; break;
      case 2: dim = arg.dim_map[2]; break;
      case 3: dim = arg.dim_map[3]; break;
      }

      int local_tid = local_block_idx * target::block_dim().x + target::thread_idx().x;

#ifdef NVSHMEM_COMMS
      if (do_shmempack(dim, dir, arg)) {
#endif
        switch (dim) {
        case 0:
          while (local_tid < arg.dc.ghostFaceCB[0]) {
            int ghost_idx = dir * arg.dc.ghostFaceCB[0] + local_tid;
            if (pc == QUDA_5D_PC)
              pack<dagger, twist, 0, pc>(arg, ghost_idx + s * arg.dc.ghostFace[0], 0, parity);
            else
              pack<dagger, twist, 0, pc>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 1:
          while (local_tid < arg.dc.ghostFaceCB[1]) {
            int ghost_idx = dir * arg.dc.ghostFaceCB[1] + local_tid;
            if (pc == QUDA_5D_PC)
              pack<dagger, twist, 1, pc>(arg, ghost_idx + s * arg.dc.ghostFace[1], 0, parity);
            else
              pack<dagger, twist, 1, pc>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 2:
          while (local_tid < arg.dc.ghostFaceCB[2]) {
            int ghost_idx = dir * arg.dc.ghostFaceCB[2] + local_tid;
            if (pc == QUDA_5D_PC)
              pack<dagger, twist, 2, pc>(arg, ghost_idx + s * arg.dc.ghostFace[2], 0, parity);
            else
              pack<dagger, twist, 2, pc>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 3:
          while (local_tid < arg.dc.ghostFaceCB[3]) {
            int ghost_idx = dir * arg.dc.ghostFaceCB[3] + local_tid;
            if (pc == QUDA_5D_PC)
              pack<dagger, twist, 3, pc>(arg, ghost_idx + s * arg.dc.ghostFace[3], 0, parity);
            else
              pack<dagger, twist, 3, pc>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        }
#ifdef NVSHMEM_COMMS
      }
      if (arg.shmem) shmem_signal(dim, dir, arg);
#endif
    }

    __device__ __forceinline__ void operator()(const Arg &arg, int s, int parity, int twist_pack)
    {
      switch (twist_pack) {
      case 0: this->operator()<0>(arg, s, parity); break;
      case 1: this->operator()<1>(arg, s, parity); break;
      case 2: this->operator()<2>(arg, s, parity); break;
      }
    }
  };

  template <typename Arg> struct pack_wilson_shmem {
    const Arg &arg;
    constexpr pack_wilson_shmem(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int, int s, int parity)
    {
      if (arg.nParity == 1) parity = arg.parity;
      packShmem<Arg::dagger, Arg::pc_type, Arg> pack;
      pack.operator()<Arg::twist>(arg, s, parity);
    }
  };

  template <typename Arg> struct pack_staggered {
    const Arg &arg;
    constexpr pack_staggered(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int, int s, int parity)
    {
      int local_tid = target::thread_idx().x;
      int tid = arg.sites_per_block * target::block_idx().x + local_tid;
      // this is the parity used for load/store, but we use arg.parity for index mapping
      if (arg.nParity == 1) parity = arg.parity;

      while (local_tid < arg.sites_per_block && tid < arg.work_items) {
        // determine which dimension we are packing
        int ghost_idx;
        const int dim = dimFromFaceIndex(ghost_idx, tid, arg);

        if (arg.nFace == 1) {
          switch (dim) {
          case 0: packStaggered<0, 1>(arg, ghost_idx, s, parity); break;
          case 1: packStaggered<1, 1>(arg, ghost_idx, s, parity); break;
          case 2: packStaggered<2, 1>(arg, ghost_idx, s, parity); break;
          case 3: packStaggered<3, 1>(arg, ghost_idx, s, parity); break;
          }
        } else if (arg.nFace == 3) {
          switch (dim) {
          case 0: packStaggered<0, 3>(arg, ghost_idx, s, parity); break;
          case 1: packStaggered<1, 3>(arg, ghost_idx, s, parity); break;
          case 2: packStaggered<2, 3>(arg, ghost_idx, s, parity); break;
          case 3: packStaggered<3, 3>(arg, ghost_idx, s, parity); break;
          }
        }

        local_tid += target::block_dim().x;
        tid += target::block_dim().x;
      } // while tid
    }
  };

  template <bool dagger, QudaPCType pc, typename Arg> struct packStaggeredShmem {

    __device__ __forceinline__ void operator()(const Arg &arg, int s, int parity, int = 0)
    {
      // (active_dims * 2 + dir) * blocks_per_dir + local_block_idx
      int local_block_idx = target::block_idx().x % arg.blocks_per_dir;
      int dim_dir = target::block_idx().x / arg.blocks_per_dir;
      int dir = dim_dir % 2;
      int dim;
      switch (dim_dir / 2) {
      case 0: dim = arg.dim_map[0]; break;
      case 1: dim = arg.dim_map[1]; break;
      case 2: dim = arg.dim_map[2]; break;
      case 3: dim = arg.dim_map[3]; break;
      }

      int local_tid = local_block_idx * target::block_dim().x + target::thread_idx().x;

#ifdef NVSHMEM_COMMS
      if (do_shmempack(dim, dir, arg)) {
#endif
        switch (dim) {
        case 0:
          while (local_tid < arg.nFace * arg.dc.ghostFaceCB[0]) {
            int ghost_idx = dir * arg.nFace * arg.dc.ghostFaceCB[0] + local_tid;
            if (arg.nFace == 1)
              packStaggered<0, 1>(arg, ghost_idx, s, parity);
            else
              packStaggered<0, 3>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 1:
          while (local_tid < arg.nFace * arg.dc.ghostFaceCB[1]) {
            int ghost_idx = dir * arg.nFace * arg.dc.ghostFaceCB[1] + local_tid;
            if (arg.nFace == 1)
              packStaggered<1, 1>(arg, ghost_idx, s, parity);
            else
              packStaggered<1, 3>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 2:
          while (local_tid < arg.nFace * arg.dc.ghostFaceCB[2]) {
            int ghost_idx = dir * arg.nFace * arg.dc.ghostFaceCB[2] + local_tid;
            if (arg.nFace == 1)
              packStaggered<2, 1>(arg, ghost_idx, s, parity);
            else
              packStaggered<2, 3>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        case 3:
          while (local_tid < arg.nFace * arg.dc.ghostFaceCB[3]) {
            int ghost_idx = dir * arg.nFace * arg.dc.ghostFaceCB[3] + local_tid;
            if (arg.nFace == 1)
              packStaggered<3, 1>(arg, ghost_idx, s, parity);
            else
              packStaggered<3, 3>(arg, ghost_idx, s, parity);
            local_tid += arg.blocks_per_dir * target::block_dim().x;
          }
          break;
        }
#ifdef NVSHMEM_COMMS
      }
      if (arg.shmem) shmem_signal(dim, dir, arg);
#endif
    }
  };

  template <typename Arg> struct pack_staggered_shmem {
    const Arg &arg;
    constexpr pack_staggered_shmem(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int, int s, int parity)
    {
      if (arg.nParity == 1) parity = arg.parity;
      packStaggeredShmem<0, QUDA_4D_PC, Arg> pack;
      pack.operator()(arg, s, parity);
    }
  };

} // namespace quda
