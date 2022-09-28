#include <dslash_coarse.hpp>

namespace quda {

  // dagger = true wrapper
#ifdef GPU_MULTIGRID
  void ApplyCoarseDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                         cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                         double kappa, int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    constexpr bool dagger = true;
    DslashCoarseLaunch<dagger> Dslash(out, inA, inB, Y, X, kappa, parity, dslash,
                                      clover, commDim, halo_precision);

    DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
    policy.apply(device::get_default_stream());
  }
#else
  void ApplyCoarseDagger(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                         cvector_ref<const ColorSpinorField> &, const GaugeField &,
                         const GaugeField &, double, int, bool, bool, const int *, QudaPrecision)
  {
    errorQuda("Multigrid has not been built");
  }
#endif

} // namespace quda
