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
    template <int nvector> using Arg = CloverSigmaOprodArg<Float, nColor, nvector>;
    GaugeField &oprod;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const std::vector<array<double, 2>> &coeff;
    unsigned int minThreads() const override { return oprod.VolumeCB(); }

  public:
    CloverSigmaOprod(GaugeField &oprod, cvector_ref<const ColorSpinorField> &inA,
                     cvector_ref<const ColorSpinorField> &inB, const std::vector<array<double, 2>> &coeff) :
      TunableKernel3D(oprod, 2, 6),
      oprod(oprod),
      inA(inA),
      inB(inB),
      coeff(coeff)
    {
      char tmp[16];
      sprintf(tmp, ",nvector=%lu", inA.size());
      strcat(aux, tmp);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (inA.size()) {
      case 1: launch<SigmaOprod>(tp, stream, Arg<1>(oprod, inA, inB, coeff)); break;
      default: errorQuda("Unsupported nvector = %lu\n", inA.size());
      }
    } // apply

    void preTune() override { oprod.backup(); }
    void postTune() override { oprod.restore(); }

    long long flops() const override
    {
      return ((144 + 18) * inA.size() + 18) * 6 * oprod.Volume(); // spin trace + multiply-add
    }
    long long bytes() const override
    {
      return (inA[0].Bytes() + inB[0].Bytes()) * inA.size() * 6 + 2 * oprod.Bytes();
    }
  }; // CloverSigmaOprod

  void computeCloverSigmaOprod(GaugeField& oprod, cvector_ref<const ColorSpinorField> &x,
			       cvector_ref<const ColorSpinorField> &p, const std::vector<array<double, 2> > &coeff)
  {
    if constexpr (is_enabled_clover()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      if (x.size() > MAX_NVECTOR) {
        // divide and conquer
        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField>{x.begin(), x.begin() + x.size()/2},
                                cvector_ref<const ColorSpinorField>{p.begin(), p.begin() + p.size() / 2},
                                {coeff.begin(), coeff.begin() + coeff.size() / 2});

        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField>{x.begin() + x.size() / 2, x.end()},
                                cvector_ref<const ColorSpinorField>{p.begin() + p.size() / 2, p.end()},
                                {coeff.begin() + coeff.size() / 2, coeff.end()});
        return;
      }

      instantiate<CloverSigmaOprod>(oprod, x, p, coeff);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Clover Dirac operator has not been built!");
    }
  }

} // namespace quda
