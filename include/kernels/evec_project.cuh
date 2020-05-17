#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>


namespace quda
{

  template <typename Float, int nColor_> struct EvecProjectArg 
  {
    int threads; // number of active threads required
    
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields (F4 for spin 4 fermion, F1 for spin 1)
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;
    typedef typename colorspinor_mapper<Float, nSpinY, nColor, false, spinor_direct_load>::type F1;

    F4 x_vec;
    F1 y_vec;
    // 2x2 matrix will store the 4 spins.
    // We use it because it has a nice save method    
    matrix_field<complex<Float>, 2> s;
    
    EvecProjectArg(const ColorSpinorField &x_vec, const ColorSpinorField &y_vec, complex<Float> *s) :
      threads(x_vec.VolumeCB()),
      x_vec(x_vec),
      y_vec(y_vec),
      s(s, x_vec.VolumeCB())
    {
    }
  };

  template <typename Float, typename Arg> __global__ void computeEvecProject(Arg arg)
  {
    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    
    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<Float, nColor, nSpinX> Vector4;
    typedef ColorSpinor<Float, nColor, nSpinY> Vector1;

    Matrix<complex<Float>, 2> res;
    Vector4 x;
    Vector1 y;

    // Get vector data for this spacetime point
    x = arg.x_vec(idx_cb, parity);
    y = arg.y_vec(idx_cb, parity);
    
    // Compute the inner product over color
#pragma unroll
    for (int mu = 0; mu < 2; mu++) {
#pragma unroll
      for (int nu = 0; nu < 2; nu++) {
	res(mu,nu) = innerProduct(y, x, 0, mu*2 + nu);
      }
    } 
    arg.s.save(res, idx_cb, parity);    
  }
}
