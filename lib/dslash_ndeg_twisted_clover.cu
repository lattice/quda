#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_ndeg_twisted_clover.cuh>

/**
   This is the gauged non-degenerate twisted-clover operator acting on a 
   quark doublet.
*/

namespace quda
{

  template <typename Arg> class NdegTwistedClover : public Dslash<nDegTwistedClover, Arg>
    {
      using Dslash = Dslash<nDegTwistedClover, Arg>;
      using Dslash::arg;
      using Dslash::halo;
      using Dslash::in;

      unsigned int sharedBytesPerThread() const
      {
        return 2 * in.Ncolor() * 4 * sizeof(typename mapper<typename Arg::Float>::type);
      }

    public:
      NdegTwistedClover(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                        const ColorSpinorField &halo) :
        Dslash(arg, out, in, halo)
      {
        TunableKernel3D::resizeStep(2, 1);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash::setParam(tp);
        if (arg.xpay)
          Dslash::template instantiate<packShmem, true>(tp, stream);
        else
          errorQuda("Non-degenerate twisted-clover operator only defined for xpay=true");
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
  
  template <typename Float, int nColor, QudaReconstructType recon> struct NdegTwistedCloverApply {

    NdegTwistedCloverApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &A, double a,
                           double b, double c, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      NdegTwistedCloverArg<Float, nColor, nDim, recon> arg(out, in, halo, U, A, a, b, c, x, parity, dagger,
                                                           comm_override);
      NdegTwistedClover<decltype(arg)> twisted(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
    }
  };

  void ApplyNdegTwistedClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                              const GaugeField &U, const CloverField &A, double a, double b, double c,
                              cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                              TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<NdegTwistedCloverApply>(out, in, x, U, A, a, b, c, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Non-degenerate twisted-clover operator has not been built");
    }
  }
  
} // namespace quda
