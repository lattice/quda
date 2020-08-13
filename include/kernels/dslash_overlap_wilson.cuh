#pragma once

#include <kernels/dslash_wilson.cuh>
//#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool twist_ = false>
  struct hwilsonArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_>::nSpin;

    typedef typename mapper<Float>::type real;

    hwilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                    double a, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override)
    {
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct hwilson : dslash_default {

    Arg &arg;
    constexpr hwilson(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the dslash for the overlap fermion.
       out(x) = gamma5 * (1+a D) * in(x-mu)
       Note this routine only exists in xpay form.
    */
    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      // defined in dslash_wilson.cuh
      applyWilson<nParity, dagger, kernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      if (kernel_type == INTERIOR_KERNEL) {
	Vector tmp = arg.x(coord.x_cb, my_spinor_parity);
	tmp = tmp + arg.a * out;
	out = tmp.gamma(4);
      } else if (active) {
        Vector x = arg.out(coord.x_cb, my_spinor_parity);
        out = x + arg.a * out.gamma(4);
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(coord.x_cb, my_spinor_parity) = out;
    }
  };

  template <typename Float, int nColor>
  struct OverlapLinopArg {
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;
    typedef typename mapper<Float>::type RegType;

    F out;                // output vector field
    const F in;           // input vector field
    const int nParity;    // number of parities we're working on
    const int volumeCB;   // checkerboarded volume
    RegType k0;            // scale factor on x 
    RegType k1;            // scale factor on eps5
    RegType k2;            // scale factor on eps

    OverlapLinopArg(ColorSpinorField &out, const ColorSpinorField &in,
		double k0, double k1, double k2)
      : out(out), in(in), nParity(in.SiteSubset()),
	volumeCB(in.VolumeCB()),k0(k0),k1(k1),k2(k2){}
  };


  template <typename Float, int nColor, typename Arg>  
  __global__ void overlapLinop(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
    ColorSpinor<RegType,nColor,4> out = arg.out(x_cb, parity); 
    ColorSpinor<RegType,nColor,4> tmp=out.gamma(4);
    out = arg.k0*in + arg.k1* tmp + arg.k2*out;
    arg.out(x_cb, parity)=out;
  }


} // namespace quda
