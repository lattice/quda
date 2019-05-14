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
      threads(x.Volume()/2),
      x(x),
      y(y),
      result(result)
    {
    }
  };
  
  
  template <typename Float, typename Arg> __global__ void computeContraction(Arg arg)
  {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    const int nSpin = arg.nSpin;
    const int nColor = arg.nColor;
    
    if (idx >= arg.threads) return;
    
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, nSpin> Vector;

    Vector x = arg.x(idx,parity);
    Vector y = arg.y(idx,parity);
    
    for (int i=0; i<arg.nSpin; i++)
      for (int j=0; j<arg.nSpin; j++) {
	arg.result[idx + i*nSpin + j] = innerProduct(x,y,i,j).real();
	//printf("%d %d %d %d %f\n", idx, parity, i, j, arg.result[idx]);
      }
    
  }  
} // namespace quda
