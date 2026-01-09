#pragma once

#include "color_spinor_field_order.h"
#include "color_spinor.h"
#include "kernel.h"

namespace quda {

   /**
     @brief Parameter structure for driving splitting a 5-d field into 4-d fields
   */
  template <typename Float, int nColor_, int nSpin_>
  struct Split5DTo4DArgs : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    constexpr static int nSpin = nSpin_;

    // for nSpin != 2
    using F_coarse_grained = typename colorspinor_mapper<Float, nSpin, nColor>::type;
    // else
    using F_fine_grained = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, QUDA_NATIVE_FIELD_ORDER>;

    using F = std::conditional_t<nSpin == 2, F_fine_grained, F_coarse_grained>;

    F out[QUDA_MAX_DWF_LS]; // 4-d output vector fields
    const F in;             // 5-d input vector field
    const int volumeCB;  // checkerboarded volume

    Split5DTo4DArgs(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in) :
      kernel_param(dim3(out[0].VolumeCB(), out.size(), in.SiteSubset())),
      in(in),
      volumeCB(out[0].VolumeCB())
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
      }

      checkPrecision(out[0], in);
      checkLocation(out[0], in);
      auto nSpin = checkSpin(out[0], in);
      if (in.PCType() != QUDA_4D_PC) errorQuda("Unexpected PC type %d", in.PCType());
      if (in.Ndim() != 5) errorQuda("Unexpected nDim %d for input field", in.Ndim());
      if (out[0].Ndim() != 4) errorQuda("Unexpected nDim %d for output field", out[0].Ndim());
      if (nSpin != 4 && nSpin != 2) errorQuda("Unexpected nSpin %d", nSpin);
      if (static_cast<int>(out.size()) != in.X(4)) errorQuda("Vector length %lu does not match Ls %d", out.size(), in.X(4));
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out = %d in = %d\n", out[0].FieldOrder(), in.FieldOrder());
    }
  };

  /**
     @brief Split a 5-d fields into 4-d fields
  */
  template <typename Arg> struct Split5DTo4D {
    using real = typename Arg::real;
    static constexpr int nColor = Arg::nColor;
    static constexpr int nSpin = Arg::nSpin;

    const Arg &arg;
    constexpr Split5DTo4D(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <typename Arg_>
    __device__ __host__ inline std::enable_if_t<Arg_::nSpin != 2, void> copy_spinor(const Arg_ &arg, int x_cb, int s, int parity)
    {
      using fermion_t = ColorSpinor<real, nColor, nSpin>;

      fermion_t in = arg.in(x_cb + s * arg.volumeCB, parity);
      arg.out[s](x_cb, parity) = in;
    }

    template <typename Arg_>
    __device__ __host__ inline std::enable_if_t<Arg_::nSpin == 2, void> copy_spinor(const Arg_ &arg, int x_cb, int s, int parity)
    {
      // we can expose more parallelism in the future
#pragma unroll
      for (int s_c = 0; s_c < nSpin; s_c++) {
#pragma unroll
        for (int c = 0; c < nColor; c++) {
          arg.out[s](parity, x_cb, s_c, c) = arg.in(parity, x_cb + s * arg.volumeCB, s_c, c);
        }
      }
    }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      copy_spinor(arg, x_cb, s, parity);
    }
  };

   /**
     @brief Parameter structure for joining 4-d fields into a 5-d field
   */
  template <typename Float, int nColor_, int nSpin_>
  struct Join4DTo5DArgs : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    constexpr static int nSpin = nSpin_;

    // for nSpin != 2
    using F_coarse_grained = typename colorspinor_mapper<Float, nSpin, nColor>::type;
    // else
    using F_fine_grained = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, QUDA_NATIVE_FIELD_ORDER>;

    using F = std::conditional_t<nSpin == 2, F_fine_grained, F_coarse_grained>;

    F out;                 // 5-d output vector field
    F in[QUDA_MAX_DWF_LS]; // 4-d input vector fields
    const int volumeCB;  // checkerboarded volume

    Join4DTo5DArgs(ColorSpinorField &out, cvector_ref<const ColorSpinorField> &in) :
      kernel_param(dim3(in[0].VolumeCB(), in.size(), out.SiteSubset())),
      out(out),
      volumeCB(in[0].VolumeCB())
    {
      for (auto i = 0u; i < in.size(); i++) {
        this->in[i] = in[i];
      }

      checkPrecision(out, in[0]);
      checkLocation(out, in[0]);
      checkColor(out, in[0]);
      auto nSpin = checkSpin(out, in[0]);
      if (out.PCType() != QUDA_4D_PC) errorQuda("Unexpected PC type %d", in.PCType());
      if (out.Ndim() != 5) errorQuda("Unexpected nDim %d for input field", out.Ndim());
      if (in[0].Ndim() != 4) errorQuda("Unexpected nDim %d for output field", in[0].Ndim());
      if (nSpin != 4 && nSpin != 2) errorQuda("Unexpected nSpin %d", nSpin);
      if (static_cast<int>(in.size()) != out.X(4)) errorQuda("Vector length %lu does not match Ls %d", in.size(), out.X(4));
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out = %d in = %d\n", out.FieldOrder(), in.FieldOrder());
    }
  };

  /**
     @brief Join 4-d fields into a 5-d field
  */
  template <typename Arg> struct Join4DTo5D {
    using real = typename Arg::real;
    static constexpr int nColor = Arg::nColor;
    static constexpr int nSpin = Arg::nSpin;

    const Arg &arg;
    constexpr Join4DTo5D(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <typename Arg_>
    __device__ __host__ inline std::enable_if_t<Arg_::nSpin != 2, void> copy_spinor(const Arg_ &arg, int x_cb, int s, int parity)
    {
      using fermion_t = ColorSpinor<real, nColor, nSpin>;

      fermion_t in = arg.in[s](x_cb, parity);
      arg.out(x_cb + s * arg.volumeCB, parity) = in;
    }

    template <typename Arg_>
    __device__ __host__ inline std::enable_if_t<Arg_::nSpin == 2, void> copy_spinor(const Arg_ &arg, int x_cb, int s, int parity)
    {
      // we can expose more parallelism in the future
#pragma unroll
      for (int s_c = 0; s_c < nSpin; s_c++) {
#pragma unroll
        for (int c = 0; c < nColor; c++) {
          arg.out(parity, x_cb + s * arg.volumeCB, s_c, c) = arg.in[s](parity, x_cb, s_c, c);
        }
      }
    }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      copy_spinor(arg, x_cb, s, parity);
    }
  };

}
