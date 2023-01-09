#pragma once

#include <kernel_helper.h>
#include <color_spinor_field_order.h>
#include <color_spinor_field.h>

namespace quda {

  /**
      Kernel argument struct
  */
  template <class v_t_, class b_t_, bool is_device_, typename vFloat, typename vAccessor, typename bFloat, typename bAccessor, int nSpin_, int nColor_, int nVec_>
  struct BlockTransposeArg : kernel_param<> {
    using real = typename mapper<vFloat>::type;
    static constexpr bool is_device = is_device_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nVec = nVec_;

    using v_t = v_t_;
    using b_t = b_t_;

    vAccessor V;
    bAccessor B[nVec];

    BlockTransposeArg(v_t &V, cvector_ref<b_t> &B_) :
      kernel_param(dim3(V.VolumeCB(), V.SiteSubset(), nVec)),
      V(V)
    {
      for (auto i = 0u; i < B_.size(); i++) {
        B[i] = bAccessor(B_[i]);
      }
    }
  };

  template <typename Arg> struct BlockTransposeKernel {
    const Arg &arg;
    constexpr BlockTransposeKernel(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int nv)
    {
      for (int color = 0; color < Arg::nColor; color++) {
        for (int spin = 0; spin < Arg::nSpin; spin++) {
          if constexpr (std::is_const_v<typename Arg::v_t>) {
            arg.B[nv](parity, x_cb, spin, color) = arg.V(parity, x_cb, spin, color, nv);
          } else {
            arg.V(parity, x_cb, spin, color, nv) = arg.B[nv](parity, x_cb, spin, color);
          }
        }
      }
    }
  };

} // namespace quda
