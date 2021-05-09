#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <kernels/contraction_helper.cuh>

namespace quda {

  template <typename Float, int nColor_> struct copy3dArg : kernel_param<> 
  {
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
    
    int_fastdiv X[4]; // grid dimensions
    
    copy3dArg(ColorSpinorField &y, ColorSpinorField &x, const int slice) :
      kernel_param(dim3(y.VolumeCB(), 2, 1)),
      y(y),
      x(x),
      slice(slice)
    {
      for(int i=0; i<4; i++) {
	X[i] = y.X()[i];
      }      
    }
  };
  
  
  template <typename Arg> struct copyTo3d {
    const Arg &arg;
    constexpr copyTo3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int idx[4];
      getCoords(idx, x_cb, X, parity);

      // Isolate the slice of the 4D array
      if(idx[3] == arg.slice) {
	// Get 4D data
	Vector data = arg.y(x_cb, parity);
	// Write to 3D
	int idx_3d = ((idx[2]) * X[1] + idx[1]) * X[0] + idx[0];
	arg.x(idx_3d/2, idx_3d%2) = data;
      }
    }
  };

  template <typename Arg> struct copyFrom3d {
    const Arg &arg;
    constexpr copyFrom3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int idx[4];
      getCoords(idx, x_cb, X, parity);

      // Isolate the slice of the 4D array
      if(idx[3] == arg.slice) {
	// Get 3D data
	int idx_3d = ((idx[2]) * X[1] + idx[1]) * X[0] + idx[0];
	Vector data = arg.x(idx_3d/2, idx_3d%2);
	// Write to 4D
	arg.y(x_cb, parity) = data;
      }
    }
  };

  
  template <typename Float, int nColor_> struct axpby3dArg : kernel_param<> 
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;    
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    const Float *a;
    F x;
    const Float *b;
    F y;

    int_fastdiv X[4]; // grid dimensions
    
    axpby3dArg(const Float *a, ColorSpinorField &x, const Float *b, ColorSpinorField &y) :
      kernel_param(dim3(x.VolumeCB(), 2, 1)),
      a(a),
      x(x),
      b(b),
      y(y)
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }
  };
  
  
  template <typename Arg> struct axpby3d {
    const Arg &arg;
    constexpr axpby3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int idx[4];
      getCoords(idx, x_cb, X, parity);

      //printf("axpby %d %d %d %d\n", idx[0], idx[1], idx[2], idx[3]);
      
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

  template <typename Float, int nColor_> struct caxpby3dArg : kernel_param<> 
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;    
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    const complex<Float> *a;
    F x;
    const complex<Float> *b;
    F y;

    int_fastdiv X[4]; // grid dimensions
    
    caxpby3dArg(const complex<Float> *a, ColorSpinorField &x, const complex<Float> *b, ColorSpinorField &y) :
      kernel_param(dim3(x.VolumeCB(), 2, 1)),
      a(a),
      x(x),
      b(b),
      y(y)
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }
  };
  
  
  template <typename Arg> struct caxpby3d {
    const Arg &arg;
    constexpr caxpby3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int idx[4];
      getCoords(idx, x_cb, X, parity);
      
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

  template <typename Float, int nColor_> struct reDotProduct3dArg : public ReduceArg<double>
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
    dim3 threads;     // number of active threads required  
    
    int_fastdiv X[4]; // grid dimensions
    
    reDotProduct3dArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<double>(x.X()[3]),
      x(x),
      y(y),
      // Launch xyz threads per t, t times.
      threads(x.Volume()/x.X()[3], x.X()[3])
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }

    __device__ __host__ double init() const { return double(); }
  };
    
  template <typename Arg> struct reDotProduct3d : plus<double> {
    using reduce_t = double;
    using plus<reduce_t>::operator();
    const Arg &arg;
    constexpr reDotProduct3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz<3>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
     
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);
      real sum = 0.0;
      
      // Get the inner product
      for (int s = 0; s < Arg::nSpin; s++) sum += innerProduct(x, y, s, s).real();
      
      // Apply reduction to t bucket
      return plus::operator()(sum, result);
    }
  };

  template <typename Float, int nColor_> struct cDotProduct3dArg : public ReduceArg<double2>
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
    dim3 threads;     // number of active threads required  
    
    int_fastdiv X[4]; // grid dimensions
    
    cDotProduct3dArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<double2>(x.X()[3]),
      x(x),
      y(y),
      // Launch xyz threads per t, t times.
      threads(x.Volume()/x.X()[3], x.X()[3])
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }

    __device__ __host__ double2 init() const { return double2(); }
  };
    
  template <typename Arg> struct cDotProduct3d : plus<double2> {
    using reduce_t = double2;
    using plus<reduce_t>::operator();
    const Arg &arg;
    constexpr cDotProduct3d(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz<3>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
     
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);
      reduce_t sum = {0.0, 0.0};
      complex<real> res;
      
      // Get the inner product
      for (int s = 0; s < Arg::nSpin; s++) {
	res = innerProduct(x, y, s, s);
	sum.x += res.real();
	sum.y += res.imag();
      }
      
      // Apply reduction to dim bucket
      return plus::operator()(sum, result);
    }
  };

  
}
