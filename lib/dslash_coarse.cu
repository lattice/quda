#include <dslash_coarse.hpp>

namespace quda {

  // declaration for dagger wrapper - defined in dslash_coarse_dagger.cu
  void ApplyCoarseDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                         cvector_ref<const ColorSpinorField> &inB, const ColorSpinorField &halo,
                         const GaugeField &Y, const GaugeField &X, double kappa, int parity,
                         bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision);


  // dagger = false wrapper
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const ColorSpinorField &halo,
                   const GaugeField &Y, const GaugeField &X, double kappa, int parity,
                   bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      constexpr bool dagger = false;
      DslashCoarseLaunch<dagger> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                        clover, commDim, halo_precision);

      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //  or
  //out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                   double kappa, int parity, bool dslash, bool clover, bool dagger,
                   const int *commDim, QudaPrecision halo_precision)
  {
    // create a halo ndim+1 field for batched comms
    auto halo = ColorSpinorField::create_comms_batch(inA);

    if (dagger)
      ApplyCoarseDagger(out, inA, inB, halo, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
    else
      ApplyCoarse(out, inA, inB, halo, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
  } //ApplyCoarse

} // namespace quda
