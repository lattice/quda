#pragma once
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>
#include <index_helper.cuh>
#include <cub_helper.cuh>

namespace quda
{

  template <typename Float_, int nColor_> struct EvecProjectSumArg :
    public ReduceArg<double2>
  {
    int threads; // number of active threads required
    int X[4]; // true grid dimensions

    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields (F4 for spin 4 fermion, F1 for spin 1)
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;
    typedef typename colorspinor_mapper<Float, nSpinY, nColor, false       , spinor_direct_load>::type F1;

    F4 x_vec;
    F1 y_vec;
    
    EvecProjectSumArg(const ColorSpinorField &x_vec, const ColorSpinorField &y_vec) :
      ReduceArg<double2>(),
      threads(x_vec.VolumeCB() / x_vec.X(3)), // the thread-x dimension is only for 3-d space 
      x_vec(x_vec),
      y_vec(y_vec)
    {
      for (int dir=0; dir<4; ++dir) {
        X[dir] = x_vec.X()[dir];
      }
    }
  };

  template <int blockSize, typename Arg> __global__ void computeEvecProjectSum(Arg arg)
  {
    int t = blockIdx.z; // map t to z block index
    int xyz = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    using real = typename Arg::Float;
    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpinX> Vector4;
    typedef ColorSpinor<real, nColor, nSpinY> Vector1;

    double2 res[nSpinX];
    for (int i=0; i<nSpinX; i++) res[i] = make_double2(0.0, 0.0);

    // the while loop is restricted to the same time slice
    while (xyz < arg.threads) {

      // arg.threads is the parity timeslice volume 
      int idx_cb = t * arg.threads + xyz;

      // Get vector data for this spacetime point
      Vector4 x_vec_local = arg.x_vec(idx_cb, parity);
      Vector1 y_vec_local = arg.y_vec(idx_cb, parity);
     
      // Compute the inner product over colour
      for (int mu = 0; mu < nSpinX; mu++) {
	auto res_ = innerProduct(y_vec_local, x_vec_local, 0, mu);

	// Real data
        res[mu].x += res_.real();
        res[mu].y += res_.imag();
      }      
      xyz += blockDim.x * gridDim.x;
    }

    for (int i=0; i<nSpinX; i++) reduce2d<blockSize, 2>(arg, res[i], t * nSpinX + i);
  }

} // namespace quda
