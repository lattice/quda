#include <cassert>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <kernel.h>
#include <shared_memory_cache_helper.cuh>
#include <dslash_quda.h>

namespace quda {

  // these helper functions return the thread coarseness, with both
  // constexpr variants (to be called from parallel regions) and
  // run-time variants (to be called from host serial code)
  template <bool is_device, int nSpin> constexpr int spins_per_thread()
  {
    if (is_device)
      return (nSpin == 1) ? 1 : 2;
    else
      return nSpin;
  }

  template <int nSpin> __host__ __device__ int spins_per_thread()
  {
    if (target::is_device()) return spins_per_thread<true, nSpin>();
    else return spins_per_thread<false, nSpin>();
  }

  int spins_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return (a.Nspin() == 1) ? 1 : 2;
    else
      return a.Nspin();
  }

  template <bool is_device, int nColor> constexpr int colors_per_thread()
  {
    if (is_device)
      return (nColor % 2 == 0) ? 2 : 1;
    else
      return nColor;
  }

  template <int nColor> __host__ __device__ int colors_per_thread()
  {
    if (target::is_device()) return colors_per_thread<true, nColor>();
    else return colors_per_thread<false, nColor>();
  }

  int colors_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return (a.Ncolor() % 2 == 0) ? 2 : 1;
    else
      return a.Ncolor();
  }

  template <typename store_t, typename ghost_store_t, int nSpin_, int nColor_, int nDim_, QudaFieldOrder order>
  struct PackGhostArg : kernel_param<> {
    static constexpr bool block_float = sizeof(store_t) == QUDA_SINGLE_PRECISION && isFixed<ghost_store_t>::value;

    // ensure we only compile supported block-float kernels
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nDim = nDim_;

    using real = typename mapper<store_t>::type;
    using Field = typename colorspinor::FieldOrderCB<real,nSpin,nColor,1,order,store_t,ghost_store_t>;

    Field field;
    const int_fastdiv volumeCB;
    const int nFace;
    const int parity;
    const int_fastdiv nParity;
    const int dagger;
    const QudaPCType pc_type;
    DslashConstant dc; // pre-computed dslash constants for optimized indexing
    int_fastdiv work_items;
    int threadDimMapLower[4];
    int threadDimMapUpper[4];

    PackGhostArg(const ColorSpinorField &a, int work_items, void **ghost, int parity, int nFace, int dagger) :
      kernel_param(dim3(work_items, (a.Nspin() / spins_per_thread(a)) * (a.Ncolor() / colors_per_thread(a)), a.SiteSubset())),
      field(a, nFace, 0, ghost),
      volumeCB(a.VolumeCB()),
      nFace(nFace),
      parity(parity),
      nParity(a.SiteSubset()),
      dagger(dagger),
      pc_type(a.PCType()),
      dc(a.getDslashConstant()),
      threadDimMapLower{ },
      threadDimMapUpper{ }
    {
      int prev = -1; // previous dimension that was partitioned
      for (int i = 0; i < 4; i++) {
        if (!comm_dim_partitioned(i)) continue;
        // unlike the dslash kernels, we include the fifth dimension here
        dc.ghostFaceCB[i] *= nFace * (nDim == 5 ? dc.Ls : 1);
        threadDimMapLower[i] = (prev >= 0 ? threadDimMapUpper[prev] : 0);
        threadDimMapUpper[i] = threadDimMapLower[i] + 2 * dc.ghostFaceCB[i];
        prev = i;
      }
    }
  };

  template <bool is_device> struct site_max {
    template <typename Arg> inline auto operator()(typename Arg::real thread_max, Arg &)
    {
      // on the host we require that both spin and color are fully thread local
      constexpr int Ms = spins_per_thread<is_device, Arg::nSpin>();
      constexpr int Mc = colors_per_thread<is_device, Arg::nColor>();
      static_assert(Ms == Arg::nSpin, "on host spins per thread must match total spins");
      static_assert(Mc == Arg::nColor, "on host colors per thread must match total colors");
      return thread_max;
    }
  };

  template <> struct site_max<true> {
    template <typename Arg> __device__ inline auto operator()(typename Arg::real thread_max, Arg &)
    {
      using real = typename Arg::real;
      constexpr int Ms = spins_per_thread<true, Arg::nSpin>();
      constexpr int Mc = colors_per_thread<true, Arg::nColor>();
      constexpr int color_spin_threads = (Arg::nSpin/Ms) * (Arg::nColor/Mc);
      SharedMemoryCache<real, color_spin_threads, 2, true> cache; // 2 comes from parity
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
  compute_site_max(const Arg &, int, int, int, int)
  {
    return static_cast<typename Arg::real>(1.0); // dummy return for non-block float
  }

  /**
     Compute the max element over the spin-color components of a given site.
  */
  template <typename Arg> __device__ __host__ inline std::enable_if_t<Arg::block_float, typename Arg::real>
  compute_site_max(const Arg &arg, int x_cb, int spinor_parity, int spin_block, int color_block)
  {
    using real = typename Arg::real;
    const int Ms = spins_per_thread<Arg::nSpin>();
    const int Mc = colors_per_thread<Arg::nColor>();
    real thread_max_r = 0.0;
    real thread_max_i = 0.0;

#pragma unroll
    for (int spin_local=0; spin_local<Ms; spin_local++) {
      int s = spin_block + spin_local;
#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
        int c = color_block + color_local;
        complex<real> z = arg.field(spinor_parity, x_cb, s, c);
        thread_max_r = std::max(thread_max_r, std::abs(z.real()));
        thread_max_i = std::max(thread_max_i, std::abs(z.imag()));
      }
    }

    return target::dispatch<site_max>(std::max(thread_max_r, thread_max_i), arg);
  }

  /**
     This is distinct from the variant in index_helper.cuh in that the
     fifth dimension is included in the thread map array.  At some
     point we should replace that one with this one, which has less
     division for the 5-d operators.
  */
  template <typename Arg>
  constexpr int dimFromFaceIndex(int &face_idx, int tid, const Arg &arg)
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

  template <typename Arg>
  constexpr auto indexFromFaceIndex(int dim, int dir, int ghost_idx, int parity, const Arg &arg)
  {
    if (arg.nFace == 1) {
      return indexFromFaceIndex<Arg::nDim>(dim, dir, ghost_idx, parity, 1, arg.pc_type, arg);
    } else {
      return indexFromFaceIndexStaggered<Arg::nDim>(dim, dir, ghost_idx, parity, 3, arg.pc_type, arg);
    }
  }

  template <typename Arg> struct GhostPacker {
    const Arg &arg;
    constexpr GhostPacker(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int tid, int spin_color_block, int parity)
    {
      const int Ms = spins_per_thread<Arg::nSpin>();
      const int Mc = colors_per_thread<Arg::nColor>();

      if (arg.nParity == 1) parity = arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      const int spin_block = (spin_color_block / (Arg::nColor / Mc)) * Ms;
      const int color_block = (spin_color_block % (Arg::nColor / Mc)) * Mc;

      int ghost_idx;
      const int dim = dimFromFaceIndex<Arg>(ghost_idx, tid, arg);

      // determine which end of the lattice we are packing: 0 = start, 1 = end
      const int dir = (ghost_idx >= arg.dc.ghostFaceCB[dim]) ? 1 : 0;
      ghost_idx -= dir * arg.dc.ghostFaceCB[dim];

      int x_cb = indexFromFaceIndex(dim, dir, ghost_idx, parity, arg);
      auto max = compute_site_max<Arg>(arg, x_cb, spinor_parity, spin_block, color_block);

#pragma unroll
      for (int spin_local=0; spin_local<Ms; spin_local++) {
        int s = spin_block + spin_local;
#pragma unroll
        for (int color_local=0; color_local<Mc; color_local++) {
          int c = color_block + color_local;
          arg.field.Ghost(dim, dir, spinor_parity, ghost_idx, s, c, 0, max) = arg.field(spinor_parity, x_cb, s, c);
        }
      }
    }
  };

} // namespace quda
