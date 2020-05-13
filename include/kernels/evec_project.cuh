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

  template <typename Float_, int nColor_, int t_> struct EvecProjectArg :
    public ReduceArg<double2>
  {
    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    
    static constexpr int t = t_;
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields (F4 for spin 4 fermion, F1 for spin 1)
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;
    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F1;

    F4 x_vec;
    F1 y_vec;
    int t_dim_start;
    
    EvecProjectArg(const ColorSpinorField &x_vec, const ColorSpinorField &y_vec, const int t_dim_start) :
      ReduceArg<double2>(),
      threads(x_vec.VolumeCB()),
      x_vec(x_vec),
      y_vec(y_vec),
      t_dim_start(t_dim_start)
    {
      int R = 0;
      for (int dir=0; dir<4; ++dir){
        border[dir] = x_vec.R()[dir];
	E[dir] = x_vec.X()[dir];
        X[dir] = x_vec.X()[dir] - border[dir]*2;
        R += border[dir];
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
  };

  template <int blockSize, typename Arg> __global__ void computeEvecProject(Arg arg)
  {
    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;
    
    using real = typename Arg::Float;
    constexpr int nSpinX = Arg::nSpinX;
    constexpr int nSpinY = Arg::nSpinY;
    constexpr int nColor = Arg::nColor;
    constexpr int t = Arg::t;
    typedef ColorSpinor<real, nColor, nSpinX> Vector4;
    typedef ColorSpinor<real, nColor, nSpinY> Vector1;

    double2 res_local[t*nSpinX];
    for(int i=0; i<t*nSpinX; i++) {
      res_local[i].x = 0.0;
      res_local[i].y = 0.0;
    }
    
    int x[4];
    complex<double> res[nSpinX];
    Vector4 x_vec_local;
    Vector1 y_vec_local;
    
    while (idx_cb < arg.threads) {

      // Compute spacetime and local coords to query t coord
      getCoords(x, idx_cb, arg.X, parity);

      // Get vector data for this spacetime point
      x_vec_local = arg.x_vec(idx_cb, parity);
      y_vec_local = arg.y_vec(idx_cb, parity);

      // Compute the inner product over color
#pragma unroll
      for (int mu = 0; mu < nSpinX; mu++) {
	res[mu] = innerProduct(y_vec_local, x_vec_local, 0, mu);
      }

      // Place color contracted data in local array for reduction
      for(int i=arg.t_dim_start; i<arg.t_dim_start + t; i++) {
	if(x[3] == i) {
	  //#pragma unroll	  
	  for (int mu = 0; mu < nSpinX; mu++) {
	    
	    res_local[i*nSpinX + mu].x = res[mu].x; 
	    res_local[i*nSpinX + mu].y = res[mu].y;
	    
	    //res_local[i*nSpinX + mu].x = 1.0*(i+1)*(mu+1); 
	    //res_local[i*nSpinX + mu].y = 2.0*(i+1)*(mu+1);	    
	  }
	}
      }
      idx_cb += blockDim.x * gridDim.x;
    }
    for(int i=0; i<nSpinX*t; i++) reduce2d<blockSize, 2>(arg, res_local[i], i);
  }
} // namespace quda
