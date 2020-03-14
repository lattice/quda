#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>

namespace quda
{

  template <typename real> struct SpinDiluteArg {
    int threads; // number of active threads required
    int X[4];    // grid dimensions

    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr int nColor = 3;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields (F4 for spin 4 fermion, F1 for spin 1)
    typedef typename colorspinor_mapper<real, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;
    typedef typename colorspinor_mapper<real, nSpinX, nColor, spin_project, spinor_direct_load>::type F1;

    F4 x;
    F1 y;
    int alpha;
    
    SpinDiluteArg(const ColorSpinorField &x, const ColorSpinorField &y, const int alpha) :
      threads(x.VolumeCB()),
      x(x),
      y(y),
      alpha(alpha)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename real, typename Arg> __global__ void computeSpinDilute(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpinX> Vector4;
    typedef ColorSpinor<real, nColor, nSpinY> Vector1;

    Vector4 x = arg.x(x_cb, parity);
    Vector1 y = arg.y(x_cb, parity);
    int alpha = arg.alpha;
#pragma unroll
    for (int mu = 0; mu < nSpinX; mu++) {
#pragma unroll
      for (int a = 0; a < nColor; a++) {
	if(mu == alpha) x.data[nColor*mu + a] = y.data[a];
	else x.data[nColor*mu + a] = 0.0;
      }
    }
  }  
} // namespace quda
