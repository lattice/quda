#ifndef USE_LEGACY_DSLASH

#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_staggered.cuh>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct StaggeredSextetLaunch {
    static constexpr const char *kernel = "quda::staggeredGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(staggeredGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class StaggeredSextet : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    StaggeredSextet(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_staggered.cuh"),
        arg(arg),
        in(in)
    {
    }

    virtual ~StaggeredSextet() {}

    void apply(const cudaStream_t &stream)
    {
      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Staggered Dslash not implemented on CPU");
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash<Float>::setParam(arg);
        Dslash<Float>::template instantiate<StaggeredSextetLaunch, nDim, nColor>(tp, arg, stream);
      }
    }

    // SU(N) times sextet fermion flops = N^2 matrix-vector multiplications
    virtual long long mv_flops() const { return in.Ncolor() * in.Ncolor() * Dslash<Float>::mv_flops(); }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u> struct StaggeredSextetApply {

    inline StaggeredSextetApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
      constexpr bool improved = false;
      constexpr bool sextet = true;

      if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
        StaggeredArg<Float, nColor, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_MILC, sextet> arg(
            out, in, U, U, a, x, parity, dagger, comm_override);
        StaggeredSextet<Float, nDim, nColor, decltype(arg)> staggered(arg, out, in);

        dslash::DslashPolicyTune<decltype(staggered)> policy(staggered,
            const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
            in.GhostFaceCB(), profile);
        policy.apply(0);
      } else {
        errorQuda("Unsupported staggered phase type %d", U.StaggeredPhase());
      }

      checkCudaError();
    }
  };

  void ApplyStaggeredSextet(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_STAGGERED_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    instantiate<StaggeredSextetApply, StaggeredReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Staggered dslash has not been built");
#endif
  }

} // namespace quda

#endif
