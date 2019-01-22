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

#include <kernels/dslash_twisted_mass.cuh>

/**
   This is the basic gauged twisted-mass operator
*/

namespace quda {

#ifdef GPU_TWISTED_MASS_DIRAC

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct TwistedMassLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(xpay == true, "Twisted-mass operator only defined for xpay");
      dslash.launch(twistedMassGPU<Float,nDim,nColor,nParity,dagger,kernel_type,Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class TwistedMass : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    TwistedMass(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in) {  }

    virtual ~TwistedMass() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) Dslash<Float>::template instantiate<TwistedMassLaunch,nDim,nColor, true>(tp, arg, stream);
      else errorQuda("Twisted-mass operator only defined for xpay=true");
    }

    long long flops() const {
      long long flops = Dslash<Float>::flops();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break; // twisted-mass flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	flops += 2 * nColor * 4 * 2 * in.Volume(); // complex * Nc * Ns * fma * vol
	break;
      }
      return flops;
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                        double a, double b, const ColorSpinorField &x, int parity, bool dagger,
                        const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    TwistedMassArg<Float,nColor,recon> arg(out, in, U, a, b, x, parity, dagger, comm_override);
    TwistedMass<Float,nDim,nColor,TwistedMassArg<Float,nColor,recon> > twisted(arg, out, in);

    DslashPolicyTune<decltype(twisted)> policy(twisted, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                        double a, double b, const ColorSpinorField &x, int parity, bool dagger,
                        const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyTwistedMass<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyTwistedMass<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyTwistedMass<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                        double a, double b, const ColorSpinorField &x, int parity, bool dagger,
                        const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyTwistedMass<Float,3>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

#endif // GPU_TWISTED_MASS_DIRAC

  //Apply the twisted-mass Dslash operator
  //out(x) = M*in = (1 + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                        double a, double b, const ColorSpinorField &x, int parity, bool dagger,
                        const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyTwistedMass<double>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyTwistedMass<float>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyTwistedMass<short>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyTwistedMass<char>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Twisted-mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }


} // namespace quda
