#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>

namespace quda
{

  template <typename real> struct TemporalDiluteArg {
    int threads; // number of active threads required
    int X[4];    // grid dimensions

    static constexpr int nSpin = 4;
    static constexpr int nColor = 3;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    
    // Create a typename F for the ColorSpinorField
    typedef typename colorspinor_mapper<real, nSpin, nColor, spin_project, spinor_direct_load>::type F;
    
    F x;   // input vector
    int t; // time slice to remain populated
    
    TemporalDiluteArg(ColorSpinorField &x, const int t) :
      threads(x.VolumeCB()),
      x(x),
      t(t)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };
  
  template <typename real, typename Arg> __global__ void computeTemporalDilute(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpin> Vector;

    Vector4 xlcl;
    Vector1 ylcl = arg.y(x_cb, parity);
    
    for (int mu = 0; mu < nSpinX; mu++) {
      for (int a = 0; a < nColor; a++) {
	if(mu == arg.alpha) xlcl.data[nColor*mu + a] = ylcl.data[a];
	else xlcl.data[nColor*mu + a] = 0.0;
      }
    }
    arg.x.save(xlcl.data, x_cb, parity);;
  }  
} // namespace quda
