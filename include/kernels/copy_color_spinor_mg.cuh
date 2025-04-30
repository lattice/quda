#pragma once

#include <color_spinor_field_order.h>
#include <kernel.h>

namespace quda {

  using namespace colorspinor;

  template <int nSpin_, int nColor_, typename OutOrder, typename InOrder>
  struct CopyArg : kernel_param<> {
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    OutOrder out;
    const InOrder in;

    template <typename T1, typename T2>
    CopyArg(ColorSpinorField &out, const ColorSpinorField &in, T1 *Out, T2 *In) :
      kernel_param(dim3(in.VolumeCB(), nSpin, nColor)), out(out, 1, Out), in(in, 1, In)
    {}
  };

  template <typename Arg> struct CopySpinor_ {
    const Arg &arg;
    constexpr CopySpinor_(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int s, int c)
    {
      arg.out(0, x_cb, s, c) = arg.in(0, x_cb, s, c);
    }
  };

}
