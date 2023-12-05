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
    const std::vector<std::vector<double>> &coeff;
    unsigned int minThreads() const { return oprod.VolumeCB(); }

  public:
    CloverSigmaOprod(GaugeField &oprod, cvector_ref<const ColorSpinorField> &inA,
                     cvector_ref<const ColorSpinorField> &inB, const std::vector<std::vector<double>> &coeff) :
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

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (inA.size()) {
      case 1: launch<SigmaOprod>(tp, stream, Arg<1>(oprod, inA, inB, coeff)); break;
      default: errorQuda("Unsupported nvector = %lu\n", inA.size());
      }
    } // apply

    void preTune() { oprod.backup(); }
    void postTune() { oprod.restore(); }

    long long flops() const
    {
      return ((144 + 18) * inA.size() + 18) * 6 * oprod.Volume(); // spin trace + multiply-add
    }
    long long bytes() const
    {
      return (inA[0].Bytes() + inB[0].Bytes()) * inA.size() * 6 + 2 * oprod.Bytes();
    }
  }; // CloverSigmaOprod

  void computeCloverSigmaOprod(GaugeField& oprod, cvector_ref<const ColorSpinorField> &x,
			       cvector_ref<const ColorSpinorField> &p, const std::vector<std::vector<double> > &coeff)
  {
    if constexpr (clover::is_enabled()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      if (x.size() > MAX_NVECTOR) {
        // divide and conquer
        std::vector<std::vector<double> > coeff_1(coeff.size() / 2);
        std::vector<std::vector<double> > coeff_2(coeff.size()- coeff.size() / 2);
        for (auto i = 0u; i<coeff.size() / 2 ; i++)
          coeff_1[i] =std::vector<double>{coeff[i][0],coeff[i][1]};
        for (auto i=coeff.size() / 2; i<coeff.size() ;i++)
          coeff_2[i-coeff.size() / 2] =std::vector<double>{coeff[i][0],coeff[i][1]};

        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField>{x.begin(), x.begin() + x.size()/2},
                                cvector_ref<const ColorSpinorField>{p.begin(), p.begin() + p.size() / 2},
                                coeff_1);

        computeCloverSigmaOprod(oprod, cvector_ref<const ColorSpinorField>{x.begin() + x.size() / 2, x.end()},
                                cvector_ref<const ColorSpinorField>{p.begin() + p.size() / 2, p.end()},
                                coeff_2);

        return;
      }

      instantiate<CloverSigmaOprod>(oprod, x, p, coeff);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Clover Dirac operator has not been built!");
    }
  }

} // namespace quda
