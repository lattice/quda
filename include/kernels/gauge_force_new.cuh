#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>

namespace quda {
  
  template <typename Float_, int nColor_, QudaReconstructType recon_in, int paths_ = 1>
  struct GaugeForceNewArg {
    // The number of paths corresponds to the gauge action type
    // 1: Wilson
    // 2: Symanzik
    // 3: Luscher-Weisz
    static constexpr int paths = paths_;
    
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    typedef typename gauge_mapper<Float,recon_in>::type Gauge;
    typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_10>::type Mom;
    
    Mom mom;
    const Gauge in;
    const double epsilon;
    Float *path_coeff;
    
    dim3 threads;
    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border

    GaugeForceNewArg(GaugeField &mom, const GaugeField &in, const double epsilon, Float *path_coeff) :
      mom(mom),
      in(in),
      epsilon(epsilon),
      path_coeff(path_coeff),
      threads(mom.VolumeCB(), 2, 4)
    {
      for (int i=0; i<4; i++) {
	X[i] = mom.X()[i];
	E[i] = in.X()[i];
	border[i] = (E[i] - X[i])/2;
      }
    }
  };


  template <typename Arg, int dir>
  __device__ __host__ inline void GaugeForceKernelNew(Arg &arg, int idx, int parity)
  {
    using real = typename Arg::Float;
    typedef Matrix<complex<real>,Arg::nColor> Link;

    int x[4] = {0, 0, 0, 0};
    getCoords(x, idx, arg.X, parity);
    for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

    Link staple, rectangle, parallelogram, Ux;
    
    if(Arg::paths == 1) {
      computeForceStaple(arg, x, arg.X, parity, dir, staple);
      staple *= (real)1.0;
    }      
    else if(Arg::paths == 2) {
      computeForceStapleRectangle(arg, x, arg.X, parity, dir, staple, rectangle);
      staple = staple * (real)1.0;
      rectangle = rectangle * (real)2.0;
      staple = staple + rectangle;
    }
    else if(Arg::paths == 3) {
      computeForceStapleRectangle(arg, x, arg.X, parity, dir, staple, rectangle);
      computeForceParallelogram(arg, x, arg.X, parity, dir, parallelogram);
      staple = staple * (real)1.0;
      rectangle = rectangle * (real)2.0;
      parallelogram = parallelogram * (real)3.0;
      
      staple = staple + rectangle;
      staple = staple + parallelogram;
    }
    
    // multiply by U(x)
    Ux = arg.in(dir, linkIndex(x,arg.E), parity);
    Ux = Ux * staple;
    
    // update mom(x)
    Link mom = arg.mom(dir, idx, parity);
    mom = mom - (real)arg.epsilon * Ux;
    makeAntiHerm(mom);
    arg.mom(dir, idx, parity) = mom;    
  }
  
  template <typename Arg> struct GaugeForceNew
  {
    Arg &arg;
    constexpr GaugeForceNew(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }    
    
    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      switch(dir) {
      case 0: GaugeForceKernelNew<Arg,0>(arg, x_cb, parity); break;
      case 1: GaugeForceKernelNew<Arg,1>(arg, x_cb, parity); break;
      case 2: GaugeForceKernelNew<Arg,2>(arg, x_cb, parity); break;
      case 3: GaugeForceKernelNew<Arg,3>(arg, x_cb, parity); break;
      }
    }
  };

}
