#include <color_spinor_field_order.h>
#include <block_reduce_helper.h>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <array.h>
#include <block_reduction_kernel.h>

namespace quda {

  using namespace quda::colorspinor;

  template <int fineColor, int coarseColor> constexpr int coarse_colors_per_thread()
  {
    // for fine grids (Nc=3) have more parallelism so can use more coarse strategy
    return fineColor != 3 ? 2 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
    //coarseColor >= 8 && coarseColor % 8 == 0 ? 8 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
  }

  constexpr int max_z_block() { return 12; }

  /** 
      Kernel argument struct
  */
  template <typename out_t, typename in_t, typename v_t, int fineSpin_, int fineColor_, int coarseSpin_,
            int coarseColor_, bool from_non_rel_>
  struct RestrictArg : kernel_param<> {
    using real = out_t;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int coarseColor = coarseColor_;
    static constexpr bool from_non_rel = from_non_rel_;

    // disable ghost to reduce arg size
    using F = FieldOrderCB<real, fineSpin, fineColor, 1, colorspinor::getNative<in_t>(fineSpin), in_t, in_t, true,
                           isFixed<in_t>::value>;
    using C
      = FieldOrderCB<real, coarseSpin, coarseColor, 1, colorspinor::getNative<out_t>(coarseSpin), out_t, out_t, true>;
    using V = FieldOrderCB<real, fineSpin, fineColor, coarseColor, colorspinor::getNative<v_t>(fineSpin), v_t>;

    static constexpr unsigned int max_n_src = MAX_MULTI_RHS;
    const int_fastdiv n_src;

    C out[max_n_src];
    F in[max_n_src];
    const V v;
    const int aggregate_size;    // number of sites that form a single aggregate
    const int_fastdiv aggregate_size_cb; // number of checkerboard sites that form a single aggregate
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field

    // enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
    static constexpr bool swizzle = true;
    int_fastdiv swizzle_factor; // for transposing blockIdx.x mapping to coarse grid coordinate

    // when setting n_vector_z we can only use the compile-time part so can't include n_src in this min (not really an issue)
    static constexpr int n_vector_z = std::min(coarseColor/coarse_colors_per_thread<fineColor, coarseColor>(), max_z_block());
    static_assert(n_vector_z > 0, "n_vector_z cannot be less than 1");

    static constexpr bool launch_bounds = false;
    dim3 grid_dim;
    dim3 block_dim;

    RestrictArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
		const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      kernel_param(dim3(in.Volume() / out.Volume(), 1, out.size() * (coarseColor / coarse_colors_per_thread<fineColor, coarseColor>()))),
      n_src(out.size()),
      v(v),
      aggregate_size(in.Volume() / out.Volume()),
      aggregate_size_cb(in.VolumeCB() / out.Volume()),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(in.SiteSubset()),
      swizzle_factor(1)
    {
      if (out.size() > max_n_src) errorQuda("vector set size %lu greater than max size %d", out.size(), max_n_src);
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }
    }
  };

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <typename Out, typename Arg>
  __device__ __host__ inline void rotateCoarseColor(Out &out, const Arg &arg, int src_idx, int parity, int x_cb, int coarse_color_block)
  {
    constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int v_parity = (arg.v.Nparity() == 2) ? parity : 0;

#pragma unroll
    for (int s = 0; s < Arg::fineSpin; s++)
#pragma unroll
      for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
	out[s*coarse_color_per_thread+coarse_color_local] = 0.0;
      }

#pragma unroll
    for (int coarse_color_local = 0; coarse_color_local < coarse_color_per_thread; coarse_color_local++) {
      int i = coarse_color_block + coarse_color_local;

      ColorSpinor<typename Arg::real, Arg::fineColor, Arg::fineSpin> in;
      arg.in[src_idx].template load<Arg::fineSpin>(in.data, spinor_parity, x_cb);

      if constexpr (Arg::fineSpin == 4 && Arg::from_non_rel) {
        in.toRel();
        in *= rsqrt(static_cast<typename Arg::real>(2.0));
      }

#pragma unroll
      for (int s = 0; s < Arg::fineSpin; s++) {
	for (int c = 0; c < Arg::fineColor; c++) {
          out[s * coarse_color_per_thread + coarse_color_local] =
            cmac(conj(arg.v(v_parity, x_cb, s, c, i)), in(s, c), out[s * coarse_color_per_thread + coarse_color_local]);
        }
      }
    }
  }

  template <typename Arg> struct Restrictor {
    static constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    using vector = array<complex<typename Arg::real>, Arg::coarseSpin*coarse_color_per_thread>;
    const Arg &arg;
    constexpr Restrictor(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(dim3 block, dim3 thread)
    {
      int x_fine_offset = thread.x;
      const int x_coarse = block.x;
      const int src_coarse_color_thread = block.z * arg.block_dim.z + thread.z;
      int src_idx = src_coarse_color_thread % arg.n_src;
      int coarse_color_thread = src_coarse_color_thread / arg.n_src;
      const int coarse_color_block = coarse_color_thread * coarse_color_per_thread;

      vector reduced{0};
      while (x_fine_offset < arg.aggregate_size) {
        // all threads with x_fine_offset greater than aggregate_size_cb are second parity
        const int parity_offset = x_fine_offset >= arg.aggregate_size_cb ? 1 : 0;
        const int x_fine_cb_offset = x_fine_offset % arg.aggregate_size_cb;
        const int parity = arg.nParity == 2 ? parity_offset : arg.parity;

        // look-up map is ordered as (coarse-block-id + fine-point-id),
        // with fine-point-id parity ordered
        const int x_fine_site_id = (x_coarse * 2 + parity) * arg.aggregate_size_cb + x_fine_cb_offset;
        const int x_fine = arg.coarse_to_fine[x_fine_site_id];
        const int x_fine_cb = x_fine - parity * arg.in[src_idx].VolumeCB();

        array<complex<typename Arg::real>, Arg::fineSpin * coarse_color_per_thread> tmp{0};

        rotateCoarseColor(tmp, arg, src_idx, parity, x_fine_cb, coarse_color_block);

        // perform any local spin coarsening
#pragma unroll
        for (int s = 0; s<Arg::fineSpin; s++) {
#pragma unroll
          for (int v = 0; v<coarse_color_per_thread; v++) {
            if (arg.spin_map(s, parity) == 0) {
              reduced[0 * coarse_color_per_thread + v] += tmp[s*coarse_color_per_thread + v];
            } else {
              reduced[1 * coarse_color_per_thread + v] += tmp[s*coarse_color_per_thread + v];
            }
          }
        }

        x_fine_offset += target::block_dim().x;
      }

      constexpr int block_dim = 1;
      reduced = BlockReduce<vector, block_dim, Arg::n_vector_z>(thread.z).Sum(reduced);

      if (target::thread_idx().x == 0) {
        const int parity_coarse = x_coarse >= arg.out[src_idx].VolumeCB() ? 1 : 0;
        const int x_coarse_cb = x_coarse - parity_coarse*arg.out[src_idx].VolumeCB();

#pragma unroll
        for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
          for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
            int v = coarse_color_thread * coarse_color_per_thread + coarse_color_local;
            arg.out[src_idx](parity_coarse, x_coarse_cb, s, v) = reduced[s*coarse_color_per_thread+coarse_color_local];
          }
        }
      }
    }
  };

}
