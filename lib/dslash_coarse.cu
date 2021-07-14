#include <dslash_coarse.hpp>

namespace quda {

  // declaration for dagger wrapper - defined in dslash_coarse_dagger.cu
  void ApplyCoarseDagger(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
                         const GaugeField &Y, const GaugeField &X, double kappa, int parity,
                         bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision);

  // dagger = false wrapper
#ifdef GPU_MULTIGRID
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
	           const GaugeField &Y, const GaugeField &X, double kappa, int parity,
		   bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    constexpr bool dagger = false;
    DslashCoarseLaunch<dagger> Dslash(out, inA, inB, Y, X, kappa, parity, dslash,
                                      clover, commDim, halo_precision);

    DslashCoarsePolicyTune<decltype(Dslash)>  policy(Dslash);
    policy.apply(device::get_default_stream());
  }
#else
  void ApplyCoarse(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                   const GaugeField &, double, int, bool, bool, const int *, QudaPrecision)
  {
    errorQuda("Multigrid has not been built");
  }
#endif

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //  or
  //out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
	           const GaugeField &Y, const GaugeField &X, double kappa, int parity,
		   bool dslash, bool clover, bool dagger, const int *commDim, QudaPrecision halo_precision) {

    if (dagger)
      ApplyCoarseDagger(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
    else
      ApplyCoarse(out, inA, inB, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);

  } //ApplyCoarse

} // namespace quda
