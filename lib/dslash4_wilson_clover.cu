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
   This is the Wilson-clover preconditioned linear operator
*/

namespace quda {

  template <typename Float, int nDim, int nColor, typename Arg>
  class WilsonClover : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;
    using Dslash<Float>::setParam;
    using Dslash<Float>::launch;

  public:

    WilsonClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in), arg(arg), in(in)
    {  }

    virtual ~WilsonClover() { }

    template <bool dagger, bool xpay>
    inline void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not implemented");
      } else {
        switch(arg.kernel_type) {
        case INTERIOR_KERNEL:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,INTERIOR_KERNEL,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_X:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_X,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Y:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Y,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Z:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Z,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_T:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_T,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_ALL:
          launch(wilsonCloverGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_ALL,Arg>, tp, arg, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      }
    }

    void apply(const cudaStream_t &stream) {
      setParam(arg);
      if (arg.xpay) arg.dagger ? apply<true, true>(stream) : apply<false, true>(stream);
      else          arg.dagger ? apply<true,false>(stream) : apply<false,false>(stream);
    }

    long long flops() const {
      int clover_flops = 504;
      long long flops = Dslash<Float>::flops();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
	flops += clover_flops * in.GhostFace()[arg.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	flops += clover_flops * 2 * (in.GhostFace()[0]+in.GhostFace()[1]+in.GhostFace()[2]+in.GhostFace()[3]);
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	flops += clover_flops * in.Volume();	  

	if (arg.kernel_type == KERNEL_POLICY) break;
	// now correct for flops done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
	flops -= clover_flops * ghost_sites;
	
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
	bytes += clover_bytes * 2 * in.GhostFace()[arg.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	bytes += clover_bytes * 2 * (in.GhostFace()[0]+in.GhostFace()[1]+in.GhostFace()[2]+in.GhostFace()[3]);
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	bytes += clover_bytes*in.Volume();

	if (arg.kernel_type == KERNEL_POLICY) break;
	// now correct for bytes done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (arg.commDim[d]) ghost_sites += 2*in.GhostFace()[d];
	bytes -= clover_bytes * ghost_sites;
	
	break;
      }

      return bytes;
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    constexpr int nDim = 4;
    constexpr bool dynamic_clover = false;
    WilsonCloverArg<Float,nColor,recon,dynamic_clover> arg(out, in, U, A, kappa, x, parity, dagger, comm_override);
    WilsonClover<Float,nDim,nColor,WilsonCloverArg<Float,nColor,recon,dynamic_clover> > wilson(arg, out, in);

    TimeProfile profile("dummy");
    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, A, kappa, x, parity, dagger, comm_override);
#if 0
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilsonClover<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, A, kappa, x, parity, dagger, comm_override);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
			 double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override)
  {
    if (in.Ncolor() == 3) {
      ApplyWilsonClover<Float,3>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  // Apply the Wilson-clover operator
  // out(x) = M*in = A(x)^{-1} (-kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
			 const CloverField &A, double kappa, const ColorSpinorField &x, int parity, bool dagger,
			 const int *comm_override)
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U, A);

    // check all locations match
    checkLocation(out, in, U, A);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilsonClover<double>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilsonClover<float>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilsonClover<short>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilsonClover<char>(out, in, U, A, kappa, x, parity, dagger, comm_override);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
  }


} // namespace quda
