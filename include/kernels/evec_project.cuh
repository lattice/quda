#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <constant_kernel_arg.h>
#include <reduce_helper.h>
#include <reduction_kernel.h>

namespace quda {
  
  using spinor_array = array<double, 8>;

  constexpr unsigned long max_nx = 4;
  constexpr unsigned long max_ny = 4;

  template <typename Float, int nColor_>
  struct EvecProjectionArg : public ReduceArg<spinor_array>
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;

    typedef typename colorspinor_mapper<Float, nSpinX, nColor, false, false, true>::type F4;
    typedef typename colorspinor_mapper<Float, nSpinY, nColor, false, false, true>::type F1;
    
    static constexpr unsigned int max_n_batch_block = 8;
    int_fastdiv nx;
    int_fastdiv ny;
    F4 x[max_nx];
    F1 y[max_ny];

    int_fastdiv X[4]; // grid dimensions

    EvecProjectionArg(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y) :
      ReduceArg<spinor_array>(dim3(x.Volume() / x.X(3), 1, x.size() * y.size() * x.X(3)), x.size() * y.size() * x.X(3)),
      nx(x.size()),
      ny(y.size())
    {
      if (x.size() > max_nx) errorQuda("Requested vector size %lu greater than max %lu", x.size(), max_nx);
      if (y.size() > max_ny) errorQuda("Requested vector size %lu greater than max %lu", y.size(), max_ny);
      for (int i = 0; i < 4; i++) X[i] = x.X(i);
      for (auto i = 0u; i < x.size(); i++) this->x[i] = x[i];
      for (auto i = 0u; i < y.size(); i++) this->y[i] = y[i];
    }
    __device__ __host__ spinor_array init() const { return spinor_array(); }
  };
  
  template <typename Arg> struct EvecProjection : plus<spinor_array> {
    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 1; // only doing a reduce in the x thread dimension

    const Arg &arg;
    constexpr EvecProjection(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    // overload comm_reduce to defer until the entire "tile" is complete
    template <typename U> static inline void comm_reduce(U &) { }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int, int ijt)
    {
      int i = ijt % arg.nx;
      int jt = ijt / arg.nx;
      int j = jt % arg.ny;
      int t = jt / arg.ny;

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
      Vector4 x = arg.x[i](idx_cb, parity);
      Vector1 y = arg.y[j](idx_cb, parity);

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
