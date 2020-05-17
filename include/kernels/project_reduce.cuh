#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <complex_quda.h>
#include <matrix_field.h>


namespace quda
{

  template <typename Float, int t_> struct ProjectReduceArg : 
    public ReduceArg<double2>
  {
    int threads; // number of active threads required

    static constexpr int t = t_;
    static constexpr int nSpin = 4;
    
    EvecProjectArg(complex<Float> *s) :
      threads(x_vec.VolumeCB()),
      s(s, x_vec.VolumeCB())
    {
    }
  };

  template <typename Float, typename Arg> __global__ void computeProjectReduce(Arg arg)
  {
    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;
    
    constexpr int nSpin = Arg::nSpin;
    constexpr int t = Arg::t;

    double2 res_local[t*nSpinX];
    for(int i=0; i<t*nSpinX; i++) {
      res_local[i].x = 0.0;
      res_local[i].y = 0.0;
    }

    while (idx_cb < arg.threads) {
      
      // Compute spacetime and local coords to query t coord
      getCoords(x, idx_cb, arg.X, parity);
#pragma unroll
      for (int mu = 0; mu < nSpinX; mu++) {
	res_local[x[3]*nSpinX + mu].x += arg.s[x[3]*nSpinX + mu];
      }
    }
    idx_cb += blockDim.x * gridDim.x;
  }
  for(int i=0; i<nSpinX*t; i++) reduce2d<blockSize, 2>(arg, res_local, i);
}
