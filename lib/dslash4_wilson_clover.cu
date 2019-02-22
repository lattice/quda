#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <clover_field.h>
#include <clover_field_order.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>

namespace quda {
#include <dslash_events.cuh>
#include <dslash_policy.cuh>
}

#include <kernels/dslash_wilson_clover.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct WilsonCloverLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(xpay == true, "wilsonClover operator only defined for xpay");
      dslash.launch(wilsonCloverGPU<Float,nDim,nColor,nParity,dagger,kernel_type,Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class WilsonClover : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    WilsonClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in)
    {  }

    virtual ~WilsonClover() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) Dslash<Float>::template instantiate<WilsonCloverLaunch,nDim,nColor,true>(tp, arg, stream);
      else errorQuda("Wilson-clover operator only defined for xpay=true");
    }

    long long flops() const {
      int clover_flops = 504;
      long long flops = Dslash<Float>::flops();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break; // all clover flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	flops += clover_flops * in.Volume();	  
	break;
      }
      return flops;
    }

    long long bytes() const {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in.Precision() + (isFixed ? 2*sizeof(float) : 0);

      long long bytes = Dslash<Float>::bytes();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	bytes += clover_bytes*in.Volume();
	break;
      }

      return bytes;
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    WilsonCloverArg<Float,nColor,recon> arg(out, in, U, A, kappa, 0.0, x, parity, dagger, comm_override);
    WilsonClover<Float,nDim,nColor,WilsonCloverArg<Float,nColor,recon> > wilson(arg, out, in);

    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyWilsonClover<Float,3>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }
#endif


  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
			 const CloverField &A, double kappa, const ColorSpinorField &x, int parity, bool dagger,
			 const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_CLOVER_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U, A);

    // check all locations match
    checkLocation(out, in, U, A);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilsonClover<double>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilsonClover<float>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilsonClover<short>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilsonClover<char>(out, in, U, A, kappa, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Clover dslash has not been built");
#endif
  }


#ifdef GPU_CLOVER_DIRAC
  // Overloads to specify twists to the A-kD
  // Should produce (A - i g5 mu) x + k D y
  //
  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, double b, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4;
    WilsonCloverArg<Float,nColor,recon,true> arg(out, in, U, A, kappa, b, x, parity, dagger, comm_override);
    WilsonClover<Float,nDim,nColor,WilsonCloverArg<Float,nColor,recon,true> > wilson(arg, out, in);

    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, double b, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, double b, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyWilsonClover<Float,3>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

#endif // GPU_CLOVER_DIRAC

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
			 const CloverField &A, double kappa, double b, const ColorSpinorField &x, int parity, bool dagger,
			 const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_CLOVER_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, A);

    // check all locations match
    checkLocation(out, in, U, A);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilsonClover<double>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilsonClover<float>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilsonClover<short>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilsonClover<char>(out, in, U, A, kappa, b, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Clover dslash has not been built");
#endif
  }

} // namespace quda

#endif
