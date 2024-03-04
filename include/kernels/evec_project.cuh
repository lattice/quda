#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>

namespace quda {
  
  using spinor_array = array<double, 8>;

  template <typename Float, int nColor_>
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
      for (int i=0; i<4; i++) X[i] = x.X()[i];
    }
    __device__ __host__ spinor_array init() const { return spinor_array(); }
  };
  
 template <int reduction_dim, class T> __device__ int idx_from_t_xyz(int t, int xyz, T X[4])
  {
    int x[4];
#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != reduction_dim) {
	x[d] = xyz % X[d];
	xyz /= X[d];
      }
    }    
    x[reduction_dim] = t;    
    return (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);
  }
  
  template <typename Arg> struct EvecProjection : plus<spinor_array> {
    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();    
    static constexpr int reduce_block_dim = 1; // only doing a reduct in the x thread dimension
    
    const Arg &arg;
    constexpr EvecProjection(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    // overload comm_reduce to defer until the entire "tile" is complete
    template <typename U> static inline void comm_reduce(U &) { }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int, int t)
    {
      constexpr int nSpinX = Arg::nSpinX;
      constexpr int nSpinY = Arg::nSpinY;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector4 = ColorSpinor<real, nColor, nSpinX>;
      using Vector1 = ColorSpinor<real, nColor, nSpinY>;

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
      reduce_t result_local;
      for (int mu = 0; mu < nSpinX; mu++) {
        complex<real> prod = innerProduct(y, x, 0, mu);
        result_local[2 * mu + 0] = prod.real();
        result_local[2 * mu + 1] = prod.imag();
      }
      
      return plus::operator()(result_local, result);
    }
  };
}
