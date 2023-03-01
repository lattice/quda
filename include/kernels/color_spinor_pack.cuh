#include <cassert>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <kernel.h>
#include <dslash_quda.h>
#include <dslash_shmem.h>
#include <shmem_helper.cuh>
#include <shmem_pack_helper.cuh>
#include <shared_memory_cache_helper.h>

namespace quda {

  // these helper functions return the thread coarseness, with both
  // constexpr variants (to be called from parallel regions) and
  // run-time variants (to be called from host serial code)
  template <bool is_device> constexpr int spins_per_thread(int nSpin)
  {
    if (is_device)
      return (nSpin == 1) ? 1 : 2;
    else
      return nSpin;
  }

  __host__ __device__ int spins_per_thread(int nSpin)
  {
    if (target::is_device())
      return spins_per_thread<true>(nSpin);
    else
      return spins_per_thread<false>(nSpin);
  }

  int spins_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return spins_per_thread<true>(a.Nspin());
    else
      return spins_per_thread<false>(a.Nspin());
  }

  template <bool is_device> constexpr int colors_per_thread(int nColor)
  {
    if (is_device)
      return (nColor > 128 && nColor % 8 == 0) ? 8 : (nColor > 32 && nColor % 4 == 0) ? 4 : (nColor % 2 == 0) ? 2 : 1;
    else
      return nColor;
  }

  __host__ __device__ int colors_per_thread(int nColor)
  {
    if (target::is_device())
      return colors_per_thread<true>(nColor);
    else
      return colors_per_thread<false>(nColor);
  }

  int colors_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return colors_per_thread<true>(a.Ncolor());
    else
      return colors_per_thread<false>(a.Ncolor());
  }

  template <typename store_t, typename ghost_store_t, int nSpin_, int nColor_, int nDim_, QudaFieldOrder order>
  struct PackGhostArg : kernel_param<> {
    static constexpr bool block_float = sizeof(store_t) == QUDA_SINGLE_PRECISION && isFixed<ghost_store_t>::value;

    // ensure we only compile supported block-float kernels
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nDim = nDim_;

    using real = typename mapper<store_t>::type;
    using G = typename colorspinor::GhostOrder<real, nSpin, nColor, 1, order, store_t, ghost_store_t>;
    // disable ghost to reduce arg size
    using F = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, order, store_t, ghost_store_t, true>;

    static constexpr bool is_native = order <= 8;
    static constexpr int max_n_src = 64;
    const int_fastdiv n_src;
    G out;
    F in[max_n_src];

    const int_fastdiv volumeCB;
    const int nFace;
    const int parity;
    const int_fastdiv nParity;
    const int dagger;
    const QudaPCType pc_type;
    DslashConstant dc;
    int_fastdiv work_items;
    int_fastdiv ghostFaceCB[4] = {}; /** (non-batched) ghost face dimensions */
    int ghostThreadsCB[4] = {};      /** (batched) ghost face thread count */
    int threadDimMapLower[4] = {};
    int threadDimMapUpper[4] = {};
#ifdef NVSHMEM_COMMS
    char *packBuffer[4 * QUDA_MAX_DIM];
    int neighbor_ranks[2 * QUDA_MAX_DIM];
    int bytes[2 * QUDA_MAX_DIM];

    dslash::shmem_sync_t counter;
    dslash::shmem_sync_t waitcounter;
    dslash::shmem_retcount_intra_t *retcount_intra;
    dslash::shmem_retcount_inter_t *retcount_inter;
    dslash::shmem_sync_t *sync_arr;
#endif
    int shmem = 0;

    PackGhostArg(const ColorSpinorField &a, int work_items, void **ghost, int parity, int nFace, int dagger, int shmem_,
                 cvector_ref<const ColorSpinorField> &v) :
      kernel_param(
        dim3(work_items, (a.Nspin() / spins_per_thread(a)) * (a.Ncolor() / colors_per_thread(a)), a.SiteSubset())),
      n_src(v.size() > 0 ? 1 : v.size()),
      out(a, nFace, ghost),
      volumeCB(a.VolumeCB()),
      nFace(nFace),
      parity(parity),
      nParity(a.SiteSubset()),
      dagger(dagger),
      pc_type(a.PCType()),
      dc(a.getDslashConstant()),
#ifdef NVSHMEM_COMMS
      counter((activeTuning() && !policyTuning()) ? 2 : dslash::inc_exchangeghost_shmem_sync_counter()),
      waitcounter(counter),
      retcount_intra(dslash::get_shmem_retcount_intra()),
      retcount_inter(dslash::get_shmem_retcount_inter()),
      sync_arr(dslash::get_exchangeghost_shmem_sync_arr()),
#endif
      shmem(shmem_)
    {
      int prev = -1; // previous dimension that was partitioned
      for (int i = 0; i < 4; i++) {
        if (!comm_dim_partitioned(i)) continue;
        // include fifth dimension but not batch dimension in face indices
        ghostFaceCB[i] = dc.ghostFaceCB[i] * nFace * ((nDim == 5 && v.size() == 0) ? dc.Ls : 1);
        // include fifth and batch dimensions in thread count
        ghostThreadsCB[i] = dc.ghostFaceCB[i] * nFace * (nDim == 5 ? dc.Ls : 1);
        threadDimMapLower[i] = (prev >= 0 ? threadDimMapUpper[prev] : 0);
        threadDimMapUpper[i] = threadDimMapLower[i] + 2 * ghostThreadsCB[i];
        prev = i;
      }
#ifdef NVSHMEM_COMMS
      for (int i = 0; i < 4 * QUDA_MAX_DIM; i++) { packBuffer[i] = static_cast<char *>(ghost[i]); }
      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) {
          neighbor_ranks[2 * dim + dir] = comm_dim_partitioned(dim) ? comm_neighbor_rank(dir, dim) : -1;
          bytes[2 * dim + dir] = a.GhostFaceBytes(dim);
        }
      }
#endif

      if (n_src > max_n_src) errorQuda("vector set size %d greater than max size %d", (int)n_src, max_n_src);
      if (v.size() == 0) {
        this->in[0] = a;
      } else {
        for (auto i = 0u; i < v.size(); i++) this->in[i] = v[i];
      }
    }
  };

  template <bool is_device> struct site_max {
    template <typename Arg> inline auto operator()(typename Arg::real thread_max, Arg &)
    {
      // on the host we require that both spin and color are fully thread local
      constexpr int Ms = spins_per_thread<is_device>(Arg::nSpin);
      constexpr int Mc = colors_per_thread<is_device>(Arg::nColor);
      static_assert(Ms == Arg::nSpin, "on host spins per thread must match total spins");
      static_assert(Mc == Arg::nColor, "on host colors per thread must match total colors");
      return thread_max;
    }
  };

  template <> struct site_max<true> {
    template <typename Arg> __device__ inline auto operator()(typename Arg::real thread_max, Arg &)
    {
      using real = typename Arg::real;
      constexpr int Ms = spins_per_thread<true>(Arg::nSpin);
      constexpr int Mc = colors_per_thread<true>(Arg::nColor);
      constexpr int color_spin_threads = (Arg::nSpin/Ms) * (Arg::nColor/Mc);
      auto block = target::block_dim();
      // pad the shared block size to avoid bank conflicts for native ordering
      if (Arg::is_native) block.x = ((block.x + device::warp_size() - 1) / device::warp_size()) * device::warp_size();
      block.y = color_spin_threads; // state the y block since we know it at compile time
      SharedMemoryCache<real> cache(block);
      cache.save(thread_max);
      cache.sync();
      real this_site_max = static_cast<real>(0);
#pragma unroll
      for (int sc = 0; sc < color_spin_threads; sc++) {
        auto sc_max = cache.load_y(sc);
        this_site_max = this_site_max > sc_max ? this_site_max : sc_max;
      }
      return this_site_max;
    }
  };

  template <typename Arg> __device__ __host__ inline std::enable_if_t<!Arg::block_float, typename Arg::real>
  compute_site_max(const Arg &, int, int, int, int, int)
  {
    return static_cast<typename Arg::real>(1.0); // dummy return for non-block float
  }

  /**
     Compute the max element over the spin-color components of a given site.
  */
  template <typename Arg> __device__ __host__ inline std::enable_if_t<Arg::block_float, typename Arg::real>
  compute_site_max(const Arg &arg, int src_idx, int x_cb, int spinor_parity, int spin_block, int color_block)
  {
    using real = typename Arg::real;
    const int Ms = spins_per_thread(Arg::nSpin);
    const int Mc = colors_per_thread(Arg::nColor);
    complex<real> thread_max = {0.0, 0.0};

#pragma unroll
    for (int spin_local=0; spin_local<Ms; spin_local++) {
      int s = spin_block + spin_local;
#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
        int c = color_block + color_local;
        complex<real> z = arg.in[src_idx](spinor_parity, x_cb, s, c);
        thread_max.real(std::max(thread_max.real(), std::abs(z.real())));
        thread_max.imag(std::max(thread_max.imag(), std::abs(z.imag())));
      }
    }

    return target::dispatch<site_max>(std::max(thread_max.real(), thread_max.imag()), arg);
  }

  /**
     This is distinct from the variant in index_helper.cuh in that the
     fifth dimension is included in the thread map array.  At some
     point we should replace that one with this one, which has less
     division for the 5-d operators.
  */
  template <typename Arg>
  constexpr auto dimFromFaceIndex(int &face_idx, int tid, const Arg &arg)
  {
    face_idx = tid;
    if (face_idx < arg.threadDimMapUpper[0]) {
      return 0;
    } else if (face_idx < arg.threadDimMapUpper[1]) {
      face_idx -= arg.threadDimMapLower[1];
      return 1;
    } else if (face_idx < arg.threadDimMapUpper[2]) {
      face_idx -= arg.threadDimMapLower[2];
      return 2;
    } else {
      face_idx -= arg.threadDimMapLower[3];
      return 3;
    }
  }

  /**
     @brief Determine which end of the lattice we are packing, e.g.,
     which direction: 0 = start (backwards), 1 = end (forwards)
     @param[in] dim Dimension we are working on
     @param[in,out] ghost_idx The aggregate ghost index into this
     dimension.  This will be updated stripping out the direction
     index
  */
  template <typename Arg>
  constexpr auto dirFromFaceIndex(int dim, int &ghost_idx, const Arg &arg)
  {
    int dir = (ghost_idx >= arg.ghostThreadsCB[dim]) ? 1 : 0;
    ghost_idx -= dir * arg.ghostThreadsCB[dim];
    return dir;
  }

  template <typename Arg>
  constexpr auto indexFromFaceIndex(int &src_idx, int dim, int dir, int ghost_idx, int parity, const Arg &arg)
  {
    src_idx = ghost_idx / arg.ghostFaceCB[dim]; // this must include the fifth dimension but not the batched dimension
    if (arg.nFace == 1) {
      return indexFromFaceIndex<Arg::nDim>(dim, dir, ghost_idx % arg.ghostFaceCB[dim], parity, 1, arg.pc_type, arg);
    } else {
      return indexFromFaceIndexStaggered<Arg::nDim>(dim, dir, ghost_idx % arg.ghostFaceCB[dim], parity, 3, arg.pc_type, arg);
    }
  }

  template <typename Arg> struct GhostPacker {
    const Arg &arg;
    constexpr GhostPacker(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int tid, int spin_color_block, int parity)
    {
      const int Ms = spins_per_thread(Arg::nSpin);
      const int Mc = colors_per_thread(Arg::nColor);

      if (arg.nParity == 1) parity = arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      const int spin_block = (spin_color_block / (Arg::nColor / Mc)) * Ms;
      const int color_block = (spin_color_block % (Arg::nColor / Mc)) * Mc;

      int ghost_idx;
      const int dim = dimFromFaceIndex<Arg>(ghost_idx, tid, arg);
      const int dir = dirFromFaceIndex(dim, ghost_idx, arg);

      int src_idx;
      int x_cb = indexFromFaceIndex(src_idx, dim, dir, ghost_idx, parity, arg);
      auto max = compute_site_max<Arg>(arg, src_idx, x_cb, spinor_parity, spin_block, color_block);

#pragma unroll
      for (int spin_local=0; spin_local<Ms; spin_local++) {
        int s = spin_block + spin_local;
#pragma unroll
        for (int color_local=0; color_local<Mc; color_local++) {
          int c = color_block + color_local;
          arg.out.Ghost(dim, dir, spinor_parity, ghost_idx, s, c, 0, max) = arg.in[src_idx](spinor_parity, x_cb, s, c);
        }
      }

#ifdef NVSHMEM_COMMS
      if (arg.shmem) shmem_signalwait(0, 0, (arg.shmem & 4), arg);
#endif
    }
  };

} // namespace quda
