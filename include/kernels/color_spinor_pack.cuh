#include <cassert>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <kernel.h>
#include <shared_memory_cache_helper.cuh>

namespace quda {

  // this is the maximum number of colors for which we support block-float format
  constexpr int max_block_float_nc = 96;

  // these helper functions return the thread coarseness, with both
  // constexpr variants (to be called from parallel regions) and
  // run-time variants (to be called from host serial code)
  template <int nSpin> constexpr int spins_per_thread()
  {
    if (device::is_device())
      return (nSpin == 1) ? 1 : 2;
    else
      return nSpin;
  }

  int spins_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return (a.Nspin() == 1) ? 1 : 2;
    else
      return a.Nspin();
  }

  template <int nColor> constexpr int colors_per_thread()
  {
    if (device::is_device())
      return (nColor % 2 == 0) ? 2 : 1;
    else
      return nColor;
  }

  int colors_per_thread(const ColorSpinorField &a)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION)
      return (a.Ncolor() % 2 == 0) ? 2 : 1;
    else
      return a.Ncolor();
  }

  template <typename store_t, typename ghost_store_t, int nSpin_, int nColor_, int nDim_, QudaFieldOrder order>
  struct PackGhostArg {
    static constexpr bool block_float_requested = sizeof(store_t) == QUDA_SINGLE_PRECISION &&
      isFixed<ghost_store_t>::value;

    // if we only have short precision for the ghost then this means we have block-float
    static constexpr bool block_float = block_float_requested && nColor_ <= max_block_float_nc;

    // ensure we only compile supported block-float kernels
    static constexpr int nColor = (block_float_requested && nColor_ > max_block_float_nc) ? max_block_float_nc : nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nDim = nDim_;

    using real = typename mapper<store_t>::type;
    using Field = typename colorspinor::FieldOrderCB<real,nSpin,nColor,1,order,store_t,ghost_store_t>;

    Field field;
    int_fastdiv X[QUDA_MAX_DIM];
    const int_fastdiv volumeCB;
    const int nFace;
    const int parity;
    const int_fastdiv nParity;
    const int dagger;
    const QudaPCType pc_type;
    int commDim[4]; // whether a given dimension is partitioned or not
    dim3 threads;

    PackGhostArg(const ColorSpinorField &a, void **ghost, int parity, int nFace, int dagger) :
      field(a, nFace, 0, ghost),
      volumeCB(a.VolumeCB()),
      nFace(nFace),
      parity(parity),
      nParity(a.SiteSubset()),
      dagger(dagger),
      pc_type(a.PCType()),
      threads(a.VolumeCB(), (a.Nspin() / spins_per_thread(a)) * (a.Ncolor() / colors_per_thread(a)), 2 * a.SiteSubset())
    {
      if (block_float_requested && nColor_ > max_block_float_nc)
        errorQuda("Block-float format not supported for Nc = %d", nColor_);

      X[0] = ((nParity == 1) ? 2 : 1) * a.X(0); // set to full lattice dimensions
      for (int d=1; d<nDim; d++) X[d] = a.X(d);
      X[4] = (nDim == 5) ? a.X(4) : 1; // set fifth dimension correctly
      for (int i=0; i<4; i++) {
	commDim[i] = comm_dim_partitioned(i);
      }
    }
  };

  /**
     Compute the max element over the spin-color components of a given site.
  */
  template <int Ms, int Mc, typename Arg>
  __device__ __host__ inline auto compute_site_max(Arg &arg, int x_cb, int spinor_parity, int spin_block, int color_block, bool active)
  {
    using real = typename Arg::real;
    real thread_max = 0.0;

    if (active) {
      const auto &rhs = arg.field;
#pragma unroll
      for (int spin_local=0; spin_local<Ms; spin_local++) {
        int s = spin_block + spin_local;
#pragma unroll
        for (int color_local=0; color_local<Mc; color_local++) {
          int c = color_block + color_local;
          complex<real> z = rhs(spinor_parity, x_cb, s, c);
          thread_max = thread_max > fabs(z.real()) ? thread_max : fabs(z.real());
          thread_max = thread_max > fabs(z.imag()) ? thread_max : fabs(z.imag());
        }
      }
    }

    real site_max = thread_max;
    if (device::is_device()) { // if on the device we need to reduce across threads in the block
      constexpr int color_spin_threads = Arg::nColor <= max_block_float_nc ? (Arg::nSpin/Ms) * (Arg::nColor/Mc) : 1;
      SharedMemoryCache<real, color_spin_threads, 2, true> cache;
      if (active) cache.save(thread_max);
      cache.sync();
      if (active) {
#pragma unroll
        for (int sc = 0; sc < color_spin_threads; sc++) {
          auto sc_max = cache.load_y(sc);
          site_max = site_max > sc_max ? site_max : sc_max;
        }
      }
    } else {
      // on the CPU we require that both spin and color are fully thread local
      // when we move to C++17 we can make this static_assert with a constexpr if/else
      assert(Ms == Arg::nSpin);
      assert(Mc == Arg::nColor);
    }

    return site_max;
  }

  template <int dim, int dir, typename Arg>
  __device__ __host__ inline void packGhost(Arg &arg, int x_cb, int parity, int spinor_parity, int spin_block, int color_block)
  {
    using real = typename Arg::real;
    constexpr int Ms = spins_per_thread<Arg::nSpin>();
    constexpr int Mc = colors_per_thread<Arg::nColor>();

    int x[5] = { };
    if (Arg::nDim == 5) getCoords5(x, x_cb, arg.X, parity, arg.pc_type);
    else getCoords(x, x_cb, arg.X, parity);

    const auto &rhs = arg.field;

    {
      real max = 1.0;
      if (Arg::block_float) {
        bool active = ( arg.commDim[dim] && ( (dir == 0 && x[dim] < arg.nFace) || (dir == 1 && x[dim] >= arg.X[dim] - arg.nFace) ) );
        max = compute_site_max<Ms, Mc, Arg>(arg, x_cb, spinor_parity, spin_block, color_block, active);
      }

      if (dir == 0 && arg.commDim[dim] && x[dim] < arg.nFace) {
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
            arg.field.Ghost(dim, 0, spinor_parity, ghostFaceIndex<0, Arg::nDim>(x, arg.X, dim, arg.nFace), s, c, 0, max)
              = rhs(spinor_parity, x_cb, s, c);
          }
	}
      }

      if (dir == 1 && arg.commDim[dim] && x[dim] >= arg.X[dim] - arg.nFace) {
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
            arg.field.Ghost(dim, 1, spinor_parity, ghostFaceIndex<1, Arg::nDim>(x, arg.X, dim, arg.nFace), s, c, 0, max)
                = rhs(spinor_parity, x_cb, s, c);
          }
	}
      }
    }
  }

  template <typename Arg> struct GhostPacker {
    Arg &arg;
    constexpr GhostPacker(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int spin_color_block, int parity_dir)
    {
      constexpr int Ms = spins_per_thread<Arg::nSpin>();
      constexpr int Mc = colors_per_thread<Arg::nColor>();

      spin_color_block %= (Arg::nSpin/Ms)*(Arg::nColor/Mc);

      const int dir = parity_dir / arg.nParity;
      const int parity = (arg.nParity == 2) ? (parity_dir % arg.nParity) : arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      const int spin_block = (spin_color_block / (Arg::nColor / Mc)) * Ms;
      const int color_block = (spin_color_block % (Arg::nColor / Mc)) * Mc;

#pragma unroll
      for (int dim=0; dim<4; dim++) {
        switch(dir) {
        case 0: // backwards pack
          switch (dim) {
          case 0: packGhost<0,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 1: packGhost<1,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 2: packGhost<2,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 3: packGhost<3,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          }
          break;
        case 1: // forwards pack
          switch (dim) {
          case 0: packGhost<0,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 1: packGhost<1,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 2: packGhost<2,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          case 3: packGhost<3,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
          }
        }
      }
    }
  };

} // namespace quda
