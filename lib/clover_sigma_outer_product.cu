#include <cstdio>
#include <cstdlib>

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/clover_sigma_outer_product.cuh>

namespace quda {

  template <typename Float, int nColor> class CloverSigmaOprod : public TunableKernel3D
  {
    template <bool doublet> using Arg = CloverSigmaOprodArg<Float, nColor, doublet>;
    GaugeField &oprod;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const std::vector<array<double, 2>> &coeff;
    const bool doublet; // whether we are applying the operator to a doublet
    unsigned int minThreads() const override { return oprod.VolumeCB(); }

  public:
    CloverSigmaOprod(GaugeField &oprod, cvector_ref<const ColorSpinorField> &inA,
                     cvector_ref<const ColorSpinorField> &inB, const std::vector<array<double, 2>> &coeff) :
      TunableKernel3D(oprod, 2, 6),
      oprod(oprod),
      inA(inA),
      inB(inB),
      coeff(coeff),
      doublet(inA[0].TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET)
    {
      if (doublet) strcat(aux, ",doublet");
      setRHSstring(aux, inA.size());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (doublet)
        launch<SigmaOprod>(tp, stream, Arg<true>(oprod, inA, inB, coeff));
      else
        launch<SigmaOprod>(tp, stream, Arg<false>(oprod, inA, inB, coeff));
    } // apply

    void preTune() override { oprod.backup(); }
    void postTune() override { oprod.restore(); }

    long long flops() const override
    {
      int n_flavor = doublet ? 2 : 1;
      int oprod_flops = inA.Ncolor() * inA.Ncolor() * (8 * inA.Nspin() - 2);
      int mat_size = 2 * inA.Ncolor() * inA.Ncolor();
      // ((spin trace + multiply-add) * n_flavor * n_vector + projection) * 6 dir * sites
      return ((oprod_flops + 2 * mat_size) * n_flavor * inA.size() + mat_size) * 6 * oprod.Volume();
    }
    long long bytes() const override { return (inA.Bytes() + inB.Bytes()) * 6 + 2 * oprod.Bytes(); }
  }; // CloverSigmaOprod

  void computeCloverSigmaOprod(GaugeField &oprod, cvector_ref<const ColorSpinorField> &x,
                               cvector_ref<const ColorSpinorField> &p, const std::vector<array<double, 2>> &coeff)
  {
    if constexpr (is_enabled_clover()) {
      if (x.size() > MAX_MULTI_RHS) {
        // divide and conquer
        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField> {x.begin(), x.begin() + x.size() / 2},
                                cvector_ref<const ColorSpinorField> {p.begin(), p.begin() + p.size() / 2},
                                {coeff.begin(), coeff.begin() + coeff.size() / 2});

        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField> {x.begin() + x.size() / 2, x.end()},
                                cvector_ref<const ColorSpinorField> {p.begin() + p.size() / 2, p.end()},
                                {coeff.begin() + coeff.size() / 2, coeff.end()});
        return;
      }

      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      instantiate<CloverSigmaOprod>(oprod, x, p, coeff);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Clover Dirac operator has not been built!");
    }
  }

} // namespace quda
