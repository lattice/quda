#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <kernels/contraction_helper.cuh>

namespace quda {
  
  template <typename Float, int nColor_> struct ColorContractArg 
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;    
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    complex<Float> *s;

    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    
    ColorContractArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<Float> *s) :
      x(x),
      y(y),
      s(s),
      threads(x.VolumeCB())
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }
  };


  template <typename Arg> struct ColorContraction {
    Arg &arg;
    constexpr ColorContraction(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      complex<real> res;
      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      // Compute the inner product over color
      res = colorContract(x, y, 0, 0);
      //printf("parity = %d, idx_cb = %d\n", parity, idx_cb);
      arg.s[x_cb + parity*arg.threads.x] = res;
      //arg.s.save(A, x_cb, parity);
    }
  };

  template <typename Float, int nColor_> struct ColorCrossArg 
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields 
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    F result;

    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    
    ColorCrossArg(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result) :
      x(x),
      y(y),
      result(result),
      threads(x.VolumeCB())
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }            
    }
  };

  template <typename Arg> struct ColorCrossCompute {
    Arg &arg;
    constexpr ColorCrossCompute(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);
      Vector result;
      
      // Get vector data for this spacetime point
      x = arg.x(x_cb, parity);
      y = arg.y(x_cb, parity);
      
      // Compute the cross product
      result = crossProduct(x, y, 0, 0);      
      arg.result(x_cb, parity) = result;
    }
  };
}
