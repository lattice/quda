#pragma once

#include <clover_field_order.h>
#include <kernel.h>

namespace quda {

  /** 
      Kernel argument struct
  */
  template <typename store_out_t_, typename store_in_t_, typename Out, typename In>
  struct CopyCloverArg : kernel_param<> {
    using store_out_t = store_out_t_;
    using store_in_t = store_in_t_;
    static constexpr int length = 72;
    Out out;
    const In in;
    CopyCloverArg(CloverField &out, const CloverField &in, bool inverse,
                  store_out_t *out_p, float *out_norm,
                  const store_in_t *in_p, const float *in_norm) :
      kernel_param(dim3(in.VolumeCB(), 2, 1)),
      out(out, inverse, out_p, out_norm),
      in(in, inverse, const_cast<store_in_t*>(in_p), const_cast<float*>(in_norm)) { }
  };

  /** 
      Generic clover reordering and packing
  */
  template <typename Arg> struct CloverCopy {
    const Arg &arg;
    constexpr CloverCopy(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      typename mapper<typename Arg::store_out_t>::type out[Arg::length];
      typename mapper<typename Arg::store_in_t>::type in[Arg::length];
      arg.in.load(in, x_cb, parity);
#pragma unroll
      for (int i=0; i<Arg::length; i++) out[i] = in[i];
      arg.out.save(out, x_cb, parity);
    }
  };

}
