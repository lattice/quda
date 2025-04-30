#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <constant_kernel_arg.h>
#include <reduce_helper.h>
#include <kernel.h>

namespace quda
{

  struct baseArg : kernel_param<> {
    int_fastdiv X[4]; // grid dimensions

    baseArg(dim3 threads, const ColorSpinorField &x) : kernel_param(threads)
    {
      for (int i = 0; i < 4; i++) { X[i] = x.X()[i]; }
      if (x.SiteSubset() == 1) X[0] = 2 * X[0];
    }
  };

  template <typename Float, int nColor_> struct copy3dArg : baseArg {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F y;
    F x;
    const int slice;

    copy3dArg(ColorSpinorField &y, ColorSpinorField &x, int slice) :
      baseArg(dim3(y.VolumeCB(), y.SiteSubset(), 1), y), y(y), x(x), slice(slice)
    {
    }
  };

  template <typename Arg> struct copyTo3d {
    const Arg &arg;
    constexpr copyTo3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      // Isolate the slice of the 4D array
      int idx[4];
      getCoords(idx, x_cb, arg.X, parity);

      if (idx[3] == arg.slice) {
        using Vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;

        // Get 4D data
        Vector y = arg.y(x_cb, parity);
        // Get 3D location
        int xyz = ((idx[2] * arg.X[1] + idx[1]) * arg.X[0] + idx[0]);

        // Write to 3D
        arg.x(xyz / 2, xyz % 2) = y;
      }
    }
  };

  template <typename Arg> struct copyFrom3d {
    const Arg &arg;
    constexpr copyFrom3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int idx[4] = {};
      getCoords(idx, x_cb, arg.X, parity);

      if (idx[3] == arg.slice) {
        using Vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;

        // Get 3D location
        int xyz = ((idx[2] * arg.X[1] + idx[1]) * arg.X[0] + idx[0]);
        Vector x = arg.x(xyz / 2, xyz % 2);
        // Write to 4D
        arg.y(x_cb, parity) = x;
      }
    }
  };

  template <typename Arg> struct swap3d {
    const Arg &arg;
    constexpr swap3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int idx[4] = {};
      getCoords(idx, x_cb, arg.X, parity);

      if (idx[3] == arg.slice) {
        using Vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
        Vector x = arg.x(x_cb, parity);
        Vector y = arg.y(x_cb, parity);
        arg.x(x_cb, parity) = y;
        arg.y(x_cb, parity) = x;
      }
    }
  };

  template <typename Float, int nColor_> struct axpby3dArg : baseArg {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    static constexpr bool disable_ghost = true;

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, disable_ghost>::type F;

    static constexpr int MAX_ORTHO_DIM = 256;
    real a[MAX_ORTHO_DIM];
    const F x;
    real b[MAX_ORTHO_DIM];
    F y;

    axpby3dArg(const std::vector<double> &a, const ColorSpinorField &x, const std::vector<double> &b,
               ColorSpinorField &y) :
      baseArg(dim3(x.VolumeCB(), x.SiteSubset(), 1), x), x(x), y(y)
    {
      if (x.X(3) > MAX_ORTHO_DIM) errorQuda("Orthogonal dimension %d exceeds maximum %d", x.X(3), MAX_ORTHO_DIM);
      for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
    }
  };

  template <typename Arg> struct axpby3d {
    const Arg &arg;
    constexpr axpby3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      int idx[4];
      getCoords(idx, x_cb, arg.X, parity);

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      // Get the coeffs for this dim slice
      real a = arg.a[idx[3]];
      real b = arg.b[idx[3]];

      // Compute the axpby
      Vector out = a * x + b * y;

      // Write to y
      arg.y(x_cb, parity) = out;
    }
  };

  template <typename Float, int nColor_> struct caxpby3dArg : baseArg {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    static constexpr bool disable_ghost = true;

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, disable_ghost>::type F;

    static constexpr int MAX_ORTHO_DIM = 256;
    complex<real> a[MAX_ORTHO_DIM];
    const F x;
    complex<real> b[MAX_ORTHO_DIM];
    F y;

    caxpby3dArg(const std::vector<Complex> &a, const ColorSpinorField &x, const std::vector<Complex> &b,
                ColorSpinorField &y) :
      baseArg(dim3(x.VolumeCB(), x.SiteSubset(), 1), x), x(x), y(y)
    {
      if (x.X(3) > MAX_ORTHO_DIM) errorQuda("Orthogonal dimension %d exceeds maximum %d", x.X(3), MAX_ORTHO_DIM);
      for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
    }
  };

  template <typename Arg> struct caxpby3d {
    const Arg &arg;
    constexpr caxpby3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      int idx[4];
      getCoords(idx, x_cb, arg.X, parity);

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      // Get the coeffs for this dim slice
      complex<real> a = arg.a[idx[3]];
      complex<real> b = arg.b[idx[3]];

      // Compute the caxpby
      Vector out = a * x + b * y;

      // Write to y
      arg.y(x_cb, parity) = out;
    }
  };

  template <typename Float, int nColor_> struct reDotProduct3dArg : public ReduceArg<double> {
    using real = typename mapper<Float>::type;
    static constexpr int reduction_dim = 3; // DMH Template this
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    int_fastdiv Xh[4]; // checkerboard grid dimensions

    reDotProduct3dArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<double>(dim3(x.VolumeCB() / x.X()[reduction_dim], x.SiteSubset(), x.X()[reduction_dim]),
                        x.X()[reduction_dim]),
      x(x),
      y(y)
    {
      for (int i = 0; i < 4; i++) Xh[i] = x.SiteSubset() == 2 && i == 0 ? x.X()[i] / 2 : x.X()[i];
    }

    __device__ __host__ double init() const { return double(); }
  };

  template <typename Arg> struct reDotProduct3d : plus<double> {
    using reduce_t = double;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2;
    const Arg &arg;
    constexpr reDotProduct3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int parity, int t)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // Collect vector data
      int idx_cb = idx_from_t_xyz<3>(t, xyz, arg.Xh);

      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

      // Get the inner product
      reduce_t sum = innerProduct(x, y).real();

      // Apply reduction to t bucket
      return plus::operator()(sum, result);
    }
  };

  template <typename Float, int nColor_> struct cDotProduct3dArg : public ReduceArg<array<double, 2>> {
    using real = typename mapper<Float>::type;
    static constexpr int reduction_dim = 3; // DMH Template this
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    int_fastdiv X[4];  // grid dimensions
    int_fastdiv Xh[4]; // checkerboard grid dimensions

    cDotProduct3dArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<array<double, 2>>(dim3(x.VolumeCB() / x.X()[reduction_dim], x.SiteSubset(), x.X()[reduction_dim]),
                                  x.X()[reduction_dim]),
      x(x),
      y(y)
    {
      for (int i = 0; i < 4; i++) Xh[i] = x.SiteSubset() == 2 && i == 0 ? x.X()[i] / 2 : x.X()[i];
    }

    __device__ __host__ reduce_t init() const { return {0.0, 0.0}; }
  };

  template <typename Arg> struct cDotProduct3d : plus<array<double, 2>> {
    using reduce_t = array<double, 2>;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2;

    const Arg &arg;
    constexpr cDotProduct3d(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int parity, int t)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // Collect vector data
      int idx_cb = idx_from_t_xyz<3>(t, xyz, arg.Xh);

      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

      // Get the inner product
      complex<double> res = innerProduct(x, y);

      // Apply reduction to temporal bucket
      return plus::operator()({res.real(), res.imag()}, result);
    }
  };
} // namespace quda
