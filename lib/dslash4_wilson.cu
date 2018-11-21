#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>

namespace quda {
#include <dslash_events.cuh>
#include <dslash_policy.cuh>
}

#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator

   TODO
   - gauge fix support
   - commDim vs ghostDim
   - ghost texture support in accessors
   - CPU support
   - peer-to-peer (copy engine race condition)
*/

namespace quda {

  template <typename Float, int nDim, int nColor, typename Arg>
  class Wilson : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;
    using Dslash<Float>::setParam;
    using Dslash<Float>::launch;

  public:

    Wilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in)
    {  }

    virtual ~Wilson() { }

    template <bool dagger, bool xpay>
    inline void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not implemented");
#if 0
        switch (arg.kernel_type) {
        case   INTERIOR_KERNEL: wilsonCPU<Float,nDim,nColor,dagger,xpay,INTERIOR_KERNEL  >(arg); break;
        case EXTERIOR_KERNEL_X: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_X>(arg); break;
        case EXTERIOR_KERNEL_Y: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Y>(arg); break;
        case EXTERIOR_KERNEL_Z: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Z>(arg); break;
        case EXTERIOR_KERNEL_T: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_T>(arg); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
#endif
      } else {
        switch(arg.kernel_type) {
        case INTERIOR_KERNEL:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,INTERIOR_KERNEL,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_X:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_X,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Y:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Y,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Z:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Z,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_T:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_T,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_ALL:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_ALL,Arg>, tp, arg, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      }
    }

    void apply(const cudaStream_t &stream) {
      setParam(arg);
      if (arg.xpay) arg.dagger ? apply<true, true>(stream) : apply<false, true>(stream);
      else          arg.dagger ? apply<true,false>(stream) : apply<false,false>(stream);
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    constexpr int nDim = 4;
    WilsonArg<Float,nColor,recon> arg(out, in, U, kappa, x, parity, dagger, comm_override);
    Wilson<Float,nDim,nColor,WilsonArg<Float,nColor,recon> > wilson(arg, out, in);

    TimeProfile profile("dummy");
    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity, dagger, comm_override);
#if 0
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity, dagger, comm_override);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                     double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    if (in.Ncolor() == 3) {
      ApplyWilson<Float,3>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  //Apply the Wilson operator
  //out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger,
                   const int *comm_override)
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilson<double>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilson<float>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilson<short>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilson<char>(out, in, U, kappa, x, parity, dagger, comm_override);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
  }


} // namespace quda
