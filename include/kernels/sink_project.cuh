#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <su3_project.cuh>
#include <cub_helper.cuh>

namespace quda
{

  template <typename Float>
  struct SinkProjectArg : public ReduceArg<double2> {
    int threads; // number of active threads required
    int X[4];    // grid dimensions

    static constexpr int nSpinEvecs = 1;
    static constexpr int nSpinPsi = 4;
    static constexpr int nColor = 3;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    
    // Create a typename F for the ColorSpinorField (F for fermion)
    typedef typename colorspinor_mapper<Float, nSpinPsi, nColor, spin_project, spinor_direct_load>::type F;

    F evecs;
    F psi;
    complex<Float> *q;
    
    SinkProjectArg(const ColorSpinorField &evecs, const ColorSpinorField &psi, complex<Float> *q) :  ReduceArg<double2>(),
      threads(psi.VolumeCB()),
      evecs(evecs),
      psi(psi),
      q(q)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = psi.X()[dir];
    }
  };
  
  // Core routine for computing the topological charge from the field strength
  template <typename Float, typename Arg> __global__ void computeSinkProjection(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;
    
    constexpr int nSpinPsi = Arg::nSpinPsi;
    constexpr int nSpinEvecs = Arg::nSpinEvecs;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<Float, nColor, nSpinPsi> VectorPsi;
    typedef ColorSpinor<Float, nColor, nSpinEvecs> VectorEvecs;
    
    while (x_cb < arg.threads) {
      VectorEvecs evecs = arg.evecs(x_cb, parity);
      VectorPsi psi = arg.psi(x_cb, parity);
      
#pragma unroll
      for (int mu = 0; mu < nSpinPsi; mu++) {
	// Color inner product: <\evecs(x) | \phi(x)_{\mu}>
	// The Bra is conjugated
	(arg.q)[mu] += innerProduct(evecs, psi, mu);
	//reinterpret_cast<complex<Float>*>(arg.q)[mu] = innerProduct(evecs, psi, mu);
      }
      x_cb += blockDim.x * gridDim.x;
    }
    
    // perform final inter-block reduction and write out result
    //reduce2d<blockSize,2>(arg, plaq);
  }  
} // namespace quda
