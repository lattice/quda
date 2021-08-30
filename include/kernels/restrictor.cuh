#include <color_spinor_field_order.h>
#include <cub_helper.cuh>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <block_reduction_kernel.h>

namespace quda {

  using namespace quda::colorspinor;

  template <int fineColor, int coarseColor> constexpr int coarse_colors_per_thread()
  {
    // for fine grids (Nc=3) have more parallelism so can use more coarse strategy
    return fineColor != 3 ? 2 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
    //coarseColor >= 8 && coarseColor % 8 == 0 ? 8 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
  }

  constexpr int max_y_block() { return 8; }

  /** 
      Kernel argument struct
  */
  template <typename Float, typename vFloat, int fineSpin_, int fineColor_,
	    int coarseSpin_, int coarseColor_, QudaFieldOrder order>
  struct RestrictArg : kernel_param<> {
    using real = Float;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int coarseColor = coarseColor_;

    FieldOrderCB<Float,coarseSpin,coarseColor,1,order> out;
    const FieldOrderCB<Float,fineSpin,fineColor,1,order> in;
    const FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order,vFloat> V;
    const int aggregate_size;    // number of sites that form a single aggregate
    const int aggregate_size_cb; // number of checkerboard sites that form a single aggregate
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field

    // enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
    static constexpr bool swizzle = true;
    int_fastdiv swizzle_factor; // for transposing blockIdx.x mapping to coarse grid coordinate

    static constexpr int n_vector_y = std::min(coarseColor/coarse_colors_per_thread<fineColor, coarseColor>(), max_y_block());
    static_assert(n_vector_y > 0, "n_vector_y cannot be less than 1");

    static constexpr bool launch_bounds = false;
    dim3 grid_dim;
    dim3 block_dim;

    RestrictArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
		const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      kernel_param(dim3(in.Volume()/out.Volume(), coarseColor/coarse_colors_per_thread<fineColor, coarseColor>(), 1)),
      out(out), in(in), V(V),
      aggregate_size(in.Volume()/out.Volume()),
      aggregate_size_cb(in.VolumeCB()/out.Volume()),
      fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      spin_map(), parity(parity), nParity(in.SiteSubset()), swizzle_factor(1)
    { }
  };

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <typename Out, typename Arg>
  __device__ __host__ inline void rotateCoarseColor(Out &out, const Arg &arg, int parity, int x_cb, int coarse_color_block)
  {
    constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int v_parity = (arg.V.Nparity() == 2) ? parity : 0;

#pragma unroll
    for (int s=0; s<Arg::fineSpin; s++)
#pragma unroll
      for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
	out[s*coarse_color_per_thread+coarse_color_local] = 0.0;
      }

#pragma unroll
    for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
      int i = coarse_color_block + coarse_color_local;
#pragma unroll
      for (int s=0; s<Arg::fineSpin; s++) {

	constexpr int color_unroll = Arg::fineColor == 3 ? 3 : 2;

	complex<typename Arg::real> partial[color_unroll];
#pragma unroll
	for (int k=0; k<color_unroll; k++) partial[k] = 0.0;

#pragma unroll
	for (int j=0; j<Arg::fineColor; j+=color_unroll) {
#pragma unroll
	  for (int k=0; k<color_unroll; k++)
	    partial[k] = cmac(conj(arg.V(v_parity, x_cb, s, j+k, i)), arg.in(spinor_parity, x_cb, s, j+k), partial[k]);
	}

#pragma unroll
	for (int k=0; k<color_unroll; k++) out[s*coarse_color_per_thread + coarse_color_local] += partial[k];
      }
    }
  }

  template <typename Arg> struct Restrictor {
    static constexpr unsigned block_size = Arg::block_size;
    static constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    using vector = vector_type<complex<typename Arg::real>, Arg::coarseSpin*coarse_color_per_thread>;
    const Arg &arg;
    constexpr Restrictor(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(dim3 block, dim3 thread)
    {
      int x_coarse = block.x;
      int x_fine_offset = thread.x;
      int coarse_color_thread = block.y * arg.block_dim.y + thread.y;

      vector reduced;
      if (x_fine_offset < arg.aggregate_size) {
        // all threads with x_fine_offset greater than aggregate_size_cb are second parity
        const int parity_offset = x_fine_offset >= arg.aggregate_size_cb ? 1 : 0;
        const int x_fine_cb_offset = x_fine_offset - parity_offset * arg.aggregate_size_cb;
        const int parity = arg.nParity == 2 ? parity_offset : arg.parity;

        // look-up map is ordered as (coarse-block-id + fine-point-id),
        // with fine-point-id parity ordered
        const int x_fine_site_id = (x_coarse * 2 + parity) * arg.aggregate_size_cb + x_fine_cb_offset;
        const int x_fine = arg.coarse_to_fine[x_fine_site_id];
        const int x_fine_cb = x_fine - parity * arg.in.VolumeCB();

        const int coarse_color_block = coarse_color_thread * coarse_color_per_thread;

        vector_type<complex<typename Arg::real>, Arg::fineSpin*coarse_color_per_thread> tmp;
        rotateCoarseColor(tmp, arg, parity, x_fine_cb, coarse_color_block);

        // perform any local spin coarsening
#pragma unroll
        for (int s=0; s<Arg::fineSpin; s++) {
#pragma unroll
          for (int v=0; v<coarse_color_per_thread; v++) {
            reduced[arg.spin_map(s,parity)*coarse_color_per_thread+v] += tmp[s*coarse_color_per_thread+v];
          }
        }
      }

      reduced = BlockReduce<vector, block_size, Arg::n_vector_y>(thread.y). template Sum<true>(reduced);

      if (x_fine_offset == 0) {
        const int parity_coarse = x_coarse >= arg.out.VolumeCB() ? 1 : 0;
        const int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();

#pragma unroll
        for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
          for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
            int v = coarse_color_thread * coarse_color_per_thread + coarse_color_local;
            arg.out(parity_coarse, x_coarse_cb, s, v) = reduced[s*coarse_color_per_thread+coarse_color_local];
          }
        }
      }
    }
  };

}
