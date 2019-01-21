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

#include <kernels/dslash_twisted_mass_preconditioned.cuh>

/**
   This is the preconditioned gauged twisted-mass operator
*/

namespace quda {

#ifdef GPU_TWISTED_MASS_DIRAC

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct TwistedMassPreconditionedLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(nParity == 1, "preconditioned twisted-mass operator only defined for nParity=1");
      if (arg.asymmetric) {
        // constrain template instantiation for compilation
        if (!dagger) errorQuda("asymmetric operator only defined for dagger");
        if (xpay) errorQuda("asymmetric operator not defined for xpay");
        dslash.launch(twistedMassGPU<Float,nDim,nColor,nParity,true,true,false,kernel_type,Arg>, tp, arg, stream);
      } else {
        dslash.launch(twistedMassGPU<Float,nDim,nColor,nParity,dagger,false,xpay,kernel_type,Arg>, tp, arg, stream);
      }
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class TwistedMassPreconditioned : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    TwistedMassPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in) {
      if (arg.asymmetric) for (int i=0; i<8; i++) strcat(Dslash<Float>::aux[arg.kernel_type],",asym");
    }

    virtual ~TwistedMassPreconditioned() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.nParity == 1) {
        if (arg.xpay) Dslash<Float>::template instantiate<TwistedMassPreconditionedLaunch,nDim,nColor,1, true>(tp, arg, stream);
        else          Dslash<Float>::template instantiate<TwistedMassPreconditionedLaunch,nDim,nColor,1,false>(tp, arg, stream);
      } else {
        errorQuda("Preconditioned twisted-mass operator not defined nParity=%d", arg.nParity);
      }
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                      double a, double b, bool xpay, const ColorSpinorField &x, int parity,
                                      bool dagger, bool asymmetric, const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    TwistedMassArg<Float,nColor,recon> arg(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override);
    TwistedMassPreconditioned<Float,nDim,nColor,TwistedMassArg<Float,nColor,recon> > twisted(arg, out, in);

    DslashPolicyTune<decltype(twisted)> policy(twisted, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                      double a, double b, bool xpay, const ColorSpinorField &x, int parity,
                                      bool dagger, bool asymmetric, const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyTwistedMassPreconditioned<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyTwistedMassPreconditioned<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyTwistedMassPreconditioned<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                      double a, double b, bool xpay, const ColorSpinorField &x, int parity,
                                      bool dagger, bool asymmetric, const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyTwistedMassPreconditioned<Float,3>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

#endif // GPU_TWISTED_MASS_DIRAC

  /*
    Apply the preconditioned twisted-mass Dslash operator

    out = x + A^{-1} D * in = x + a*(1 + i*b*gamma_5)*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  */
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                      double a, double b, bool xpay, const ColorSpinorField &x, int parity,
                                      bool dagger, bool asymmetric, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    // with symmetric dagger operator we must use kernel packing
    if (dagger && !asymmetric) pushKernelPackT(true);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyTwistedMassPreconditioned<double>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyTwistedMassPreconditioned<float>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyTwistedMassPreconditioned<short>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyTwistedMassPreconditioned<char>(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    if (dagger && !asymmetric) popKernelPackT();

#else
    errorQuda("Twisted-mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }


} // namespace quda
