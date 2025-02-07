#pragma once

#include "color_spinor_field_order.h"
#include "color_spinor.h"
#include "kernel.h"

namespace quda {

   /**
     @brief Parameter structure for driving splitting a 5-d field into 4-d fields
   */
  template <typename Float, int nColor_>
  struct Split5DTo4DArgs : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    using F = typename colorspinor_mapper<Float, 4, nColor>::type;

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
      if (nSpin != 4) errorQuda("Unexpected nSpin %d", nSpin);
      if (static_cast<int>(out.size()) != in.X(4)) errorQuda("Vector length %lu does not match Ls %d", out.size(), in.X(4));
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out = %d in = %d\n", out[0].FieldOrder(), in.FieldOrder());
    }
  };

  /**
     @brief Split a 5-d fields into 4-d fields
  */
  template <typename Arg> struct Split5DTo4D {
    using fermion_t = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    const Arg &arg;
    constexpr Split5DTo4D(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      fermion_t in = arg.in(x_cb + s * arg.volumeCB, parity);
      arg.out[s](x_cb, parity) = in;
    }
  };

   /**
     @brief Parameter structure for joining 4-d fields into a 5-d field
   */
  template <typename Float, int nColor_>
  struct Join4DTo5DArgs : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    using F = typename colorspinor_mapper<Float, 4, nColor>::type;

    F out;                       // 5-d output vector field
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
      auto nSpin = checkSpin(out, in[0]);
      if (out.PCType() != QUDA_4D_PC) errorQuda("Unexpected PC type %d", in.PCType());
      if (out.Ndim() != 5) errorQuda("Unexpected nDim %d for input field", out.Ndim());
      if (in[0].Ndim() != 4) errorQuda("Unexpected nDim %d for output field", in[0].Ndim());
      if (nSpin != 4) errorQuda("Unexpected nSpin %d", nSpin);
      if (static_cast<int>(in.size()) != out.X(4)) errorQuda("Vector length %lu does not match Ls %d", in.size(), out.X(4));
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out = %d in = %d\n", out.FieldOrder(), in.FieldOrder());
    }
  };

  /**
     @brief Join 4-d fields into a 5-d field
  */
  template <typename Arg> struct Join4DTo5D {
    using fermion_t = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    const Arg &arg;
    constexpr Join4DTo5D(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      fermion_t in = arg.in[s](x_cb, parity);
      arg.out(x_cb + s * arg.volumeCB, parity) = in;
    }
  };

}
