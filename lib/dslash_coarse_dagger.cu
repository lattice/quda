#include <dslash_coarse.hpp>

namespace quda {

  // dagger = true wrapper
  void ApplyCoarseDagger(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
                         const GaugeField &Y, const GaugeField &X, double kappa, int parity,
                         bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
#ifdef GPU_MULTIGRID
    constexpr bool dagger = true;
    DslashCoarseLaunch<dagger> Dslash(out, inA, inB, Y, X, kappa, parity, dslash,
                                      clover, commDim, halo_precision);

    DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
    policy.apply(0);
#else
    errorQuda("Multigrid has not been built");
#endif
  } //ApplyCoarseDagger

} // namespace quda
