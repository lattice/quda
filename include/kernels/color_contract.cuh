#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>


namespace quda
{

  template <typename Float, int nColor_> struct ColorContractArg 
  {
    int threads; // number of active threads required
    
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x_vec;
    F y_vec;
    complex<Float> *s;
    
    ColorContractArg(const ColorSpinorField &x_vec, const ColorSpinorField &y_vec, complex<Float> *s) :
      threads(x_vec.VolumeCB()),
      x_vec(x_vec),
      y_vec(y_vec),
      s(s)
    {
    }
  };

  template <typename Float, typename Arg> __global__ void computeColorContract(Arg arg)
  {
    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;

    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<Float, nColor, nSpin> Vector;

    complex<Float> res;
    Vector x;
    Vector y;

    // Get vector data for this spacetime point
    x = arg.x_vec(idx_cb, parity);
    y = arg.y_vec(idx_cb, parity);
    
    // Compute the inner product over color
    res = colorContract(x, y, 0, 0);
    
    arg.s[idx_cb + parity*arg.threads] = res;    
  }
}
