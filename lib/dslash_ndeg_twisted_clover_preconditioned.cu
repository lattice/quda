#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_ndeg_twisted_clover_preconditioned.cuh>

/**
   This is the gauged preconditioned twisted-clover operator 
   acting on a non-degenerate quark doublet.
*/

namespace quda
{
  template <typename Arg> class NdegTwistedCloverPreconditioned : public Dslash<nDegTwistedCloverPreconditioned, Arg>
    {
      using Dslash = Dslash<nDegTwistedCloverPreconditioned, Arg>;
      using Dslash::arg;
      using Dslash::halo;
      using Dslash::in;

      unsigned int sharedBytesPerThread() const
      {
        return (in.Nspin() / 2) * in.Ncolor() * 2 * sizeof(typename mapper<typename Arg::Float>::type);
      }
      
    public:
      NdegTwistedCloverPreconditioned(Arg &arg, cvector_ref<ColorSpinorField> &out,
                                      cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo) :
        Dslash(arg, out, in, halo)
      {
        TunableKernel3D::resizeStep(2, 1); // this will force flavor to be contained in the block
      }
      
      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash::setParam(tp);
        if (arg.nParity != 1) errorQuda("Preconditioned non-degenerate twisted-clover operator not defined nParity=%d", arg.nParity);
       
        if (arg.xpay){
          if (arg.dagger) errorQuda("xpay operator not only defined for not dagger");
          Dslash::template instantiate<packShmem, 1, false, true>(tp, stream);
        } else {
          if (arg.dagger)
            Dslash::template instantiate<packShmem, 1, true, false>(tp, stream);
          else
            Dslash::template instantiate<packShmem, 1, false, false>(tp, stream);
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        Dslash::initTuneParam(param);
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }
      
      void defaultTuneParam(TuneParam &param) const
      {
        Dslash::defaultTuneParam(param);
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }
      
      long long flops() const
      {
        int clover_flops = 504;
        long long flops = Dslash::flops();
        switch (arg.kernel_type) {
        case INTERIOR_KERNEL:
        case KERNEL_POLICY:
          // b and c multiply (= 2 * 48 * in.Volume())
          flops += 2 * in.Ncolor() * 4 * 4 * halo.Volume(); // complex * Nc * Ns * fma * vol
          flops += clover_flops * halo.Volume();
          break;
        default: break; // twisted-mass flops are in the interior kernel
        }
        return flops;
      }
      
      long long bytes() const
      {
        int clover_bytes = 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);
        
        long long bytes = Dslash::bytes();
        switch (arg.kernel_type) {
        case INTERIOR_KERNEL:
        case KERNEL_POLICY: bytes += clover_bytes * halo.Volume(); break;
        default: break;
        }
        
        return bytes;
      }
    };
  
  template <typename Float, int nColor, QudaReconstructType recon> struct NdegTwistedCloverPreconditionedApply {

    NdegTwistedCloverPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                         const GaugeField &U, const CloverField &A, double a, double b, double c,
                                         bool xpay, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                         const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      NdegTwistedCloverPreconditionedArg<Float, nColor, nDim, recon> arg(out, in, halo, U, A, a, b, c, xpay, x, parity,
                                                                         dagger, comm_override);
      NdegTwistedCloverPreconditioned<decltype(arg)> twisted(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
    }
  };

  void ApplyNdegTwistedCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                            const GaugeField &U, const CloverField &A, double a, double b, double c,
                                            bool xpay, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                            const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<NdegTwistedCloverPreconditionedApply>(out, in, U, A, a, b, c, xpay, x, parity, dagger, comm_override,
                                                        profile);
    } else {
      errorQuda("Non-degenerate preconditioned twisted-clover operator has not been built");
    }
  }
  
} // namespace quda

