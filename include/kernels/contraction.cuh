#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>

namespace quda
{
  
  template <typename Float> struct ContractionArg {
    
    int threads; // number of active threads required
    int X[4];    // grid dimensions
    
    //DMH: Hardcode Wilson types for now
    static constexpr int nSpin = 4;
    static constexpr int nColor = 3;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    
    //Create a typename F for the ColorSpinorField (F for fermion)
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;
    
    F x;
    F y;
    Float *result;
    
    ContractionArg(const ColorSpinorField &x, const ColorSpinorField &y, Float *result) :
      threads(x.VolumeCB()),
      x(x),
      y(y),
      result(result)
    {
      for (int dir=0; dir<4; dir++) X[dir] = x.X()[dir];      
    }
  };
  
  
  template <typename Float, typename Arg> __global__ void computeContraction(Arg arg)
  {
    
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    const int nSpin = arg.nSpin;
    const int nColor = arg.nColor;
    
    if (x_cb >= arg.threads) return;
    
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, nSpin> Vector;

    Vector x = arg.x(x_cb, parity);
    Vector y = arg.y(x_cb, parity);
    
    complex<Float> innerP(0.0,0.0);

    int idx = x_cb + parity*arg.threads;

    for (int mu=0; mu<nSpin; mu++) {
      for (int nu=0; nu<nSpin; nu++) {
	
	//Color inner product: <\phi(x)_{\mu} | \phi(y)_{\nu}>
	//The Bra is conjugated	
	innerP = innerProduct(x,y,mu,nu);
	
	//static_cast<complex<Float>*>(arg.result)[nSpin*nSpin*idx + mu*nSpin + nu] = innerProduct(x,y,mu,nu);
	arg.result[2*(nSpin*nSpin*idx + mu*nSpin + nu)  ] = innerP.real();
	arg.result[2*(nSpin*nSpin*idx + mu*nSpin + nu)+1] = innerP.imag();
      }
    }
  }  
} // namespace quda
