#include <color_spinor_field_order.h>
#include <kernel.h>

namespace quda {

  /**
     @brief Parameter structure for driving the coarse chiral projector
   */
  template <typename store_t, int nSpin_, int nColor_>
  struct CoarseChiralProjArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;

    using F = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, QUDA_NATIVE_FIELD_ORDER>;
    
    F out[MAX_MULTI_RHS]; // output vector field
    F in[MAX_MULTI_RHS];  // input vector field
    const int nParity;    // input parity
    const int volumeCB;   // checkerboarded volume
    const int proj;       // plus or minus projector

    CoarseChiralProjArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int proj) :
      kernel_param(dim3(in.VolumeCB(), in.size(), in.SiteSubset())),
      nParity(in.SiteSubset()),
      volumeCB(in.VolumeCB()),
      proj(proj)
    {
      for (auto i = 0u; i < in.size(); i++) {
        this->in[i] = in[i];
        this->out[i] = out[i];
      }

      checkPrecision(out, in);
      checkLocation(out, in);
      if (proj != -1 && proj != 1) errorQuda("Undefined gamma projection %d", proj);
      if (in.Nspin() != 2) errorQuda("Cannot apply coarse gamma5 to nSpin=%d field", in.Nspin());
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
    }
  };

  /**
     Functor for applying the coarse chiral projector
   */
  template <typename Arg> struct CoarseChiralProj {
    using real = typename Arg::real;
    static constexpr int nColor = Arg::nColor;
    static constexpr int nSpin = Arg::nSpin;

    const Arg &arg;
    constexpr CoarseChiralProj(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      // we can expose more parallelism in the future
#pragma unroll
      for (int s_c = 0; s_c < nSpin / 2; s_c++) {
#pragma unroll
        for (int c = 0; c < nColor; c++) {
          if (arg.proj == 1) {
            arg.out[s](parity, x_cb, s_c, c) = arg.in[s](parity, x_cb, s_c, c);
            arg.out[s](parity, x_cb, s_c + nSpin / 2, c) = 0;
          } else {
            arg.out[s](parity, x_cb, s_c, c) = 0;
            arg.out[s](parity, x_cb, s_c + nSpin / 2, c) = arg.in[s](parity, x_cb, s_c + nSpin / 2, c);
          }
        }
      }
    }

  };

}
