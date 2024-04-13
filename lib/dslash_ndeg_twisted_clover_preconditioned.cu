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
      using Dslash::in;

      unsigned int sharedBytesPerThread() const
      {
        return (in.Nspin() / 2) * in.Ncolor() * 2 * sizeof(typename mapper<typename Arg::Float>::type);
      }
      
    public:
    NdegTwistedCloverPreconditioned(Arg &arg, const ColorSpinorField &out,
                                    const ColorSpinorField &in) :
      Dslash(arg, out, in)
      {
        TunableKernel3D::resizeVector(2, arg.nParity);
        // this will force flavor to be contained in the block
        TunableKernel3D::resizeStep(2, 1); 
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
          flops += 2 * in.Ncolor() * 4 * 4 * in.Volume(); // complex * Nc * Ns * fma * vol
          flops += clover_flops * in.Volume();
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
        case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
        default: break;
        }
        
        return bytes;
      }
    };
  
  template <typename Float, int nColor, QudaReconstructType recon> struct NdegTwistedCloverPreconditionedApply {
    
    inline NdegTwistedCloverPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in,
                                                const GaugeField &U, const CloverField &A,
                                                double a, double b, double c, bool xpay,
                                                const ColorSpinorField &x, int parity, bool dagger,
                                                const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      NdegTwistedCloverPreconditionedArg<Float, nColor, nDim, recon> arg(out, in, U, A, a, b, c, xpay, x, parity, dagger, comm_override);
      NdegTwistedCloverPreconditioned<decltype(arg)> twisted(arg, out, in);
        
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
    }
  };
  
#ifdef GPU_NDEG_TWISTED_CLOVER_DIRAC
  void ApplyNdegTwistedCloverPreconditioned(ColorSpinorField &out, const ColorSpinorField &in,
                                            const GaugeField &U, const CloverField &A,
                                            double a, double b, double c, bool xpay,
                                            const ColorSpinorField &x, int parity, bool dagger,
                                            const int *comm_override, TimeProfile &profile)
  {
    instantiate<NdegTwistedCloverPreconditionedApply>(out, in, U, A, a, b, c, xpay, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyNdegTwistedCloverPreconditioned(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, const CloverField &,
                                            double, double, double, bool, const ColorSpinorField &, int, bool,
                                            const int *, TimeProfile &)
  {
    errorQuda("Non-degenerate preconditioned twisted-clover dslash has not been built");
  }
#endif // GPU_NDEG_TWISTED_CLOVER_DIRAC
  
} // namespace quda

