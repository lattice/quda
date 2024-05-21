#include <color_spinor_field_order.h>
#include <color_spinor.h>
#include <multigrid_helper.cuh>
#include <kernel.h>

namespace quda {

  using namespace quda::colorspinor;
  
  template <int fineColor, int coarseColor> constexpr int fine_colors_per_thread()
  {
    return 1; // for now, all grids use 1 color per thread
  }

  /** 
      Kernel argument struct
  */
  template <typename Float, typename vFloat, int fineSpin_, int fineColor_, int coarseSpin_, int coarseColor_, bool to_non_rel_>
  struct ProlongateArg : kernel_param<> {
    using real = Float;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseColor = coarseColor_;
    static constexpr bool to_non_rel = to_non_rel_;

    // disable ghost to reduce arg size
    using F = FieldOrderCB<Float, fineSpin, fineColor, 1, colorspinor::getNative<Float>(fineSpin), Float, Float, true>;
    using C = FieldOrderCB<Float, coarseSpin, coarseColor, 1, colorspinor::getNative<Float>(coarseSpin), Float, Float, true>;
    using V = FieldOrderCB<Float, fineSpin, fineColor, coarseColor, colorspinor::getNative<vFloat>(fineSpin), vFloat, vFloat>;

    static constexpr unsigned int max_n_src = MAX_MULTI_RHS;
    const int_fastdiv n_src;
    F out[max_n_src];
    C in[max_n_src];
    const V v;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field
    
    ProlongateArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *geo_map,  const int parity) :
      kernel_param(dim3(out.VolumeCB(), out.SiteSubset() * out.size(), fineColor/fine_colors_per_thread<fineColor, coarseColor>())),
      n_src(out.size()),
      v(v),
      geo_map(geo_map),
      spin_map(),
      parity(parity),
      nParity(out.SiteSubset())
    {
      if (out.size() > max_n_src) errorQuda("vector set size %lu greater than max size %d", out.size(), max_n_src);
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }
    }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <typename Arg>
  __device__ __host__ inline void prolongate(complex<typename Arg::real> out[], const Arg &arg, int src_idx, int parity, int x_cb)
  {
    int x = parity * arg.out[src_idx].VolumeCB() + x_cb;
    int x_coarse = arg.geo_map[x];
    int parity_coarse = (x_coarse >= arg.in[src_idx].VolumeCB()) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse * arg.in[src_idx].VolumeCB();

#pragma unroll
    for (int s=0; s<Arg::fineSpin; s++) {
#pragma unroll
      for (int c=0; c<Arg::coarseColor; c++) {
        out[s*Arg::coarseColor+c] = arg.in[src_idx](parity_coarse, x_coarse_cb, arg.spin_map(s,parity), c);
      }
    }
  }

  /**
     Rotates from the coarse-color basis into the fine-color basis.  This
     is the second step of applying the prolongator.
  */
  template <typename Arg>
  __device__ __host__ inline void rotateFineColor(const Arg &arg, const complex<typename Arg::real> in[], int src_idx, int parity, int x_cb, int fine_color_block)
  {
    constexpr int fine_color_per_thread = fine_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int v_parity = (arg.v.Nparity() == 2) ? parity : 0;

    constexpr int color_unroll = 2;

    ColorSpinor<typename Arg::real, fine_color_per_thread, Arg::fineSpin> out;

#pragma unroll
    for (int s=0; s<Arg::fineSpin; s++) {
#pragma unroll
      for (int fine_color_local = 0; fine_color_local < fine_color_per_thread; fine_color_local++) {
        int i = fine_color_block + fine_color_local; // global fine color index

        complex<typename Arg::real> partial[color_unroll];
#pragma unroll
        for (int k=0; k<color_unroll; k++) partial[k] = 0.0;

#pragma unroll
        for (int j=0; j<Arg::coarseColor; j+=color_unroll) {
          // v is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
#pragma unroll
          for (int k=0; k<color_unroll; k++)
            partial[k] = cmac(arg.v(v_parity, x_cb, s, i, j + k), in[s * Arg::coarseColor + j + k], partial[k]);
        }

#pragma unroll
        for (int k = 0; k < color_unroll; k++) out(s, fine_color_local) += partial[k];
      }
    }

    if constexpr (Arg::fineSpin == 4 && Arg::to_non_rel) {
      out.toNonRel();
      out *= rsqrt(static_cast<typename Arg::real>(2.0));
    }

#pragma unroll
    for (int s = 0; s < Arg::fineSpin; s++) {
#pragma unroll
      for (int fine_color_local = 0; fine_color_local < fine_color_per_thread; fine_color_local++) {
        int i = fine_color_block + fine_color_local; // global fine color index
        arg.out[src_idx](spinor_parity, x_cb, s, i) = out(s, fine_color_local);
      }
    }
  }

  template <typename Arg> struct Prolongator
  {
    static constexpr int fine_color_per_thread = fine_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    const Arg &arg;
    constexpr Prolongator(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int src_parity, int fine_color_thread)
    {
      int src_idx = src_parity % arg.n_src;
      int parity = (arg.nParity == 2) ? (src_parity / arg.n_src) : arg.parity;
      const int fine_color_block = fine_color_thread * fine_color_per_thread;
      complex<typename Arg::real> tmp[Arg::fineSpin*Arg::coarseColor];

      prolongate(tmp, arg, src_idx, parity, x_cb);
      rotateFineColor(arg, tmp, src_idx, parity, x_cb, fine_color_block);
    }
  };

}
