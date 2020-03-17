#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>

namespace quda
{

  template <typename real> struct EvecProjectArg {
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
    //matrix_field<complex<real>, nSpinX> s;
    // Why can't I declare this as a complex<real>?
    complex<real> s[nSpinX];
    
    EvecProjectArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<real> *s) :
      threads(x.VolumeCB()),
      x(x),
      y(y),      
      s(s, x.VolumeCB())
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename real, typename Arg> __global__ void computeEvecProject(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpinX> Vector4;
    typedef ColorSpinor<real, nColor, nSpinY> Vector1;

    complex<real> A[nSpinX * nSpinY];
    
    Vector4 x = arg.x(x_cb, parity);
    Vector1 y = arg.y(x_cb, parity);
    
#pragma unroll
    for (int mu = 0; mu < nSpinX; mu++) {
      A[mu] = innerProduct(y, x, 0, mu);      
    }
    arg.s.save(A, x_cb, parity);
  }
} // namespace quda
