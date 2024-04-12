#include <dslash_wilson_clover.hpp>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverWithTwistApply {

    inline WilsonCloverWithTwistApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                      const CloverField &A, double a, double b, const ColorSpinorField &x, int parity,
                                      bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonCloverArg<Float, nColor, nDim, recon, true> arg(out, in, U, A, a, b, x, parity, dagger, comm_override);
      WilsonClover<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x)*in(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_CLOVER_DIRAC
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
                         double a, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                         TimeProfile &profile)
  {
    auto dummy = DistanceType<false>();
    instantiate<WilsonCloverApply>(out, in, U, A, a, 0, 0, x, parity, dagger, comm_override, dummy, profile);
  }
#else
  void ApplyWilsonClover(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, const CloverField &, double,
                         const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Clover dslash has not been built");
  }
#endif

} // namespace quda
