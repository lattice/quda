#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <kernels/contraction_helper.cuh>

namespace quda
{  
  template <typename Float, int nColor_>
  struct MomentumProjectArg : public ReduceArg<double2>
  {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    typedef typename colorspinor_mapper<Float, nSpin, nColor>::type F;

    F meta;
    const complex<Float> *cc_array;
    
    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    int L[3];
    int offset[3];
    int mom_mode[3];
    
    MomentumProjectArg(const ColorSpinorField &meta, const complex<Float> *cc_array_in,
		       const int mom_mode_in[3]) :
      ReduceArg<double2>(meta.X()[3]),
      meta(meta),
      cc_array(cc_array_in),      
      // Launch xyz threads per t, t times.
      threads(meta.Volume()/meta.X()[3], meta.X()[3])
    {
      for(int i=0; i<4; i++) {
	X[i] = meta.X()[i];
	if(i<3) {	  
	  L[i] = comm_dim(i) * meta.X()[i];
	  offset[i] = comm_coord(i) * meta.X()[i];
	  mom_mode[i] = mom_mode_in[i];
	}
      }
    }
    
    __device__ __host__ double2 init() const { return double2(); }
  };
  
  template <typename Arg> struct MomProj : plus<double2> {
    using reduce_t = double2;
    using plus<reduce_t>::operator();    
    const Arg &arg;
    constexpr MomProj(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      using real = typename Arg::real;
      
      reduce_t result_mom_mode = double2();

      // Collect index data
      int x[4] = { };
      getCoords(x, xyz/2, arg.X, xyz%2);
      if(x[3] != 0) printf("Womp, womp....\n");
           
      // Calculate the Fourier phase for this p mode at this x position
      // exp( -i x \dot p)
      double x_dot_p = (((arg.offset[0] + x[0]) * arg.mom_mode[0])*1.0/arg.L[0] +
			((arg.offset[1] + x[1]) * arg.mom_mode[1])*1.0/arg.L[1] +
			((arg.offset[2] + x[2]) * arg.mom_mode[2])*1.0/arg.L[2]);
      
      double phase_real = 0.0;
      double phase_imag = 0.0;
      sincos(2.0*M_PI*x_dot_p, &phase_imag, &phase_real);
      
      // Get the 4D coordinates for the < q | q > data
      int parity = 0;
      int idx = idx_from_t_xyz<3>(t, xyz, arg.X);
      int x_cb = getParityCBFromFull(parity, arg.X, idx);
      complex<real> cc = arg.cc_array[x_cb + parity*arg.threads.x];

      // The exp( -i x \dot p) convention carries a -ve sign in the
      // imaginary part of the phase
      result_mom_mode.x =   cc.real() * phase_real + cc.imag() * phase_imag;
      result_mom_mode.y = - cc.real() * phase_imag + cc.imag() * phase_real;
      
      return plus::operator()(result_mom_mode, result);
    }
  };
  
  
  template <typename Float, int nColor_> struct ColorContractArg : kernel_param<> 
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

    int_fastdiv X[4]; // grid dimensions
    
    ColorContractArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<Float> *s) :
      kernel_param(dim3(x.VolumeCB(), 2, 1)),
      x(x),
      y(y),
      s(s)
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }      
    }
  };


  template <typename Arg> struct ColorContraction {
    const Arg &arg;
    constexpr ColorContraction(const Arg &arg) : arg(arg) {}
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

  template <typename Float, int nColor_> struct ColorCrossArg : kernel_param<>
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

    int_fastdiv X[4]; // grid dimensions
    
    ColorCrossArg(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result) :
      kernel_param(dim3(x.VolumeCB(), 2, 1)),
      x(x),
      y(y),
      result(result)
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }            
    }
  };

  template <typename Arg> struct ColorCrossCompute {
    const Arg &arg;
    constexpr ColorCrossCompute(const Arg &arg) : arg(arg) {}
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
