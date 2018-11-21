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
   - ghost texture support in accessors
   - CPU support
*/

namespace quda {

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct WilsonLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      dslash.launch(wilsonGPU<Float,nDim,nColor,nParity,dagger,xpay,kernel_type,Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class Wilson : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    Wilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in) {  }

    virtual ~Wilson() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) Dslash<Float>::template instantiate<WilsonLaunch,nDim,nColor, true>(tp, arg, stream);
      else          Dslash<Float>::template instantiate<WilsonLaunch,nDim,nColor,false>(tp, arg, stream);
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger,
                   const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    WilsonArg<Float,nColor,recon> arg(out, in, U, kappa, x, parity, dagger, comm_override);
    Wilson<Float,nDim,nColor,WilsonArg<Float,nColor,recon> > wilson(arg, out, in);

    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger,
                   const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
#if 0
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                     double kappa, const ColorSpinorField &x, int parity, bool dagger,
                     const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyWilson<Float,3>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  //Apply the Wilson operator
  //out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger,
                   const int *comm_override, TimeProfile &profile)
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilson<double>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilson<float>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilson<short>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilson<char>(out, in, U, kappa, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }


} // namespace quda
