#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>

namespace quda
{

  template <typename Float_, int nColor_> struct EvecProjectArg {
    int threads; // number of active threads required
    //int X[4];    // grid dimensions

    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;

    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields (F4 for spin 4 fermion, F1 for spin 1)
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F1;

    F4 x;
    F1 y;
    //matrix_field<complex<real>, nSpinX> s;
    // Why can't I declare this as a complex<real>?
    //complex<real> *s;
    Float *s;
    
    EvecProjectArg(const ColorSpinorField &x, const ColorSpinorField &y, Float *s) :
      threads(x.VolumeCB()),
      x(x),
      y(y),      
      s(s)
    {
      //for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename Arg> __global__ void computeEvecProject(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    using real = typename Arg::Float;
    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpinX> Vector4;
    typedef ColorSpinor<real, nColor, nSpinY> Vector1;
    
    Vector4 x = arg.x(x_cb, parity);
    Vector1 y = arg.y(x_cb, parity);

    complex<real> result;
#pragma unroll
    for (int mu = 0; mu < nSpinX; mu++) {
      //result = 
      arg.s[x_cb + parity * arg.threads + mu + 0] = innerProduct(y, x, 0, mu).real();
      arg.s[x_cb + parity * arg.threads + mu + 1] = innerProduct(y, x, 0, mu).imag();
    }
  }
} // namespace quda
