#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_u>
  struct GaugeShiftArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    typedef typename gauge_mapper<Float,recon_u>::type Gauge;

    Gauge out;
    const Gauge in;
    int geometry;

    int S[4]; // the regular volume parameters
    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border
    int P; // change of parity

    GaugeShiftArg(GaugeField &out, const GaugeField &in, const int* dx) :
      kernel_param(dim3(in.VolumeCB(), 2, in.Geometry())),
      out(out),
      in(in),
      geometry(in.Geometry())
    {
      P = 0;
      for (int i=0; i<4; i++) {
	S[i] = dx[i];
	X[i] = out.X()[i];
	E[i] = in.X()[i];
	border[i] = (E[i] - X[i])/2;
	P += dx[i];
      }
      P = std::abs(P)%2;
    }
  };

  template <typename Arg, int dir>
  __device__ __host__ inline void GaugeShiftKernel(const Arg &arg, int idx, int parity)
  {
    using real = typename Arg::Float;
    typedef Matrix<complex<real>,Arg::nColor> Link;
    
    int x[4] = {0, 0, 0, 0};
    getCoords(x, idx, arg.X, parity);
    for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates
    int nbr_oddbit = arg.P==1 ? (parity^1) : parity;
    
    Link link = arg.in(dir, linkIndexShift(x,arg.S,arg.E), nbr_oddbit);
    arg.out(dir, idx, parity) = link;
  }
  
  template <typename Arg> struct GaugeShift
  {
    const Arg &arg;
    constexpr GaugeShift(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }    

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      if(dir>=arg.geometry) return;
      switch(dir) {
      case 0: GaugeShiftKernel<Arg,0>(arg, x_cb, parity); break;
      case 1: GaugeShiftKernel<Arg,1>(arg, x_cb, parity); break;
      case 2: GaugeShiftKernel<Arg,2>(arg, x_cb, parity); break;
      case 3: GaugeShiftKernel<Arg,3>(arg, x_cb, parity); break;
      }
    }
  };
  
}
