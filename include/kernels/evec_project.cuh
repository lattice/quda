#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <kernels/contraction_helper.cuh>

namespace quda {
  
  template <typename Float, int nColor_, int red = 3>
  struct EvecProjectionArg : public ReduceArg<spinor_array>
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;    
    typedef typename colorspinor_mapper<Float, nSpinY, nColor, false, spinor_direct_load>::type F1;
    
    F4 x;
    F1 y;

    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    
    EvecProjectionArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<spinor_array>(x.X()[3]),
      x(x),
      y(y),
      // Launch xyz threads per t, t times.
      threads(x.Volume()/x.X()[3], x.X()[3])
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }
    }
    __device__ __host__ spinor_array init() const { return spinor_array(); }
  };
  
  
  template <typename Arg> struct EvecProjection : plus<spinor_array> {
    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();    
    const Arg &arg;
    constexpr EvecProjection(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      constexpr int nSpinX = Arg::nSpinX;
      constexpr int nSpinY = Arg::nSpinY;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector4 = ColorSpinor<real, nColor, nSpinX>;
      using Vector1 = ColorSpinor<real, nColor, nSpinY>;

      reduce_t result_all_channels = spinor_array();

      // Collect vector data
      int parity = 0;
      // Always use the <T> dim in this kernel
      int idx = idx_from_t_xyz<3>(t, xyz, arg.X);

      // This helper will change the value of 'parity' to the correct one, 
      // and return the checkerboard index.
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
      Vector4 x = arg.x(idx_cb, parity);
      Vector1 y = arg.y(idx_cb, parity);

      // Compute the inner product over colour
      complex<real> prod;
      for (int mu = 0; mu < nSpinX; mu++) {
        prod = innerProduct(y, x, 0, mu);
        result_all_channels[mu].x += prod.real();
        result_all_channels[mu].y += prod.imag();
      }
      
      return plus::operator()(result_all_channels, result);
    }
  };
}
