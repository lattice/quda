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

#include <kernels/dslash_ndeg_twisted_mass.cuh>

/**
   This is the gauged twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda {

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct NdegTwistedMassLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(xpay == true, "Non-generate twisted-mass operator only defined for xpay");
      dslash.launch(ndegTwistedMassGPU<Float,nDim,nColor,nParity,dagger,kernel_type,Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class NdegTwistedMass : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    NdegTwistedMass(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in) {
      TunableVectorYZ::resizeVector(2,arg.nParity);
    }

    virtual ~NdegTwistedMass() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) Dslash<Float>::template instantiate<NdegTwistedMassLaunch,nDim,nColor, true>(tp, arg, stream);
      else errorQuda("Non-degenerate twisted-mass operator only defined for xpay=true");
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
	flops += 2 * nColor * 4 * 4 * in.Volume(); // complex * Nc * Ns * fma * vol
	break;
      }
      return flops;
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyNdegTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                            double a, double b, double c, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    NdegTwistedMassArg<Float,nColor,recon> arg(out, in, U, a, b, c, x, parity, dagger, comm_override);
    NdegTwistedMass<Float,nDim,nColor,NdegTwistedMassArg<Float,nColor,recon> > twisted(arg, out, in);

    DslashPolicyTune<decltype(twisted)> policy(twisted, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                               in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyNdegTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                            double a, double b, double c, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyNdegTwistedMass<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyNdegTwistedMass<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyNdegTwistedMass<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyNdegTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                            double a, double b, double c, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyNdegTwistedMass<Float,3>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

#endif // GPU_NDEG_TWISTED_MASS_DIRAC

  //Apply the non-degenerate twisted-mass Dslash operator
  //out(x) = M*in = (1 + i*b*gamma_5*tau_3 + c*tau_1)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyNdegTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                            double a, double b, double c, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, x, U);

    // check all locations match
    checkLocation(out, in, x, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyNdegTwistedMass<double>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyNdegTwistedMass<float>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyNdegTwistedMass<short>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyNdegTwistedMass<char>(out, in, U, a, b, c, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Non-degenerate twisted-mass dslash has not been built");
#endif // GPU_NDEG_TWISTED_MASS_DIRAC
  }


} // namespace quda
