#pragma once

#include <clover_field_order.h>
#include <kernel.h>

namespace quda {

  /** 
      Kernel argument struct
  */
  template <typename store_out_t_, typename store_in_t_, typename Out, typename In>
  struct CopyCloverArg {
    using store_out_t = store_out_t_;
    using store_in_t = store_in_t_;
    static constexpr int length = 72;
    Out out;
    const In in;
    dim3 threads;
    CopyCloverArg(const Out &out, const In &in, const CloverField &meta) :
      out(out),
      in(in),
      threads(meta.VolumeCB(), 2, 1) { }
  };

  /** 
      Generic clover reordering and packing
  */
  template <typename Arg> struct CloverCopy {
    Arg &arg;
    constexpr CloverCopy(Arg &arg) : arg(arg) {}
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
