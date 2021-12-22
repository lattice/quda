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
    const std::vector<ColorSpinorField*> &inA;
    const std::vector<ColorSpinorField*> &inB;
    const std::vector<std::vector<double>> &coeff;
    unsigned int minThreads() const { return oprod.VolumeCB(); }

  public:
    CloverSigmaOprod(GaugeField &oprod, const std::vector<ColorSpinorField*> &inA,
                     const std::vector<ColorSpinorField*> &inB,
                     const std::vector<std::vector<double>> &coeff) :
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
      return (inA[0]->Bytes() + inB[0]->Bytes()) * inA.size() * 6 + 2 * oprod.Bytes();
    }
  }; // CloverSigmaOprod

#ifdef GPU_CLOVER_DIRAC
  void computeCloverSigmaOprod(GaugeField& oprod, std::vector<ColorSpinorField*> &x,
			       std::vector<ColorSpinorField*> &p, std::vector<std::vector<double> > &coeff)
  {
    if (x.size() > MAX_NVECTOR) {
      // divide and conquer
      std::vector<ColorSpinorField*> x0(x.begin(), x.begin()+x.size()/2);
      std::vector<ColorSpinorField*> p0(p.begin(), p.begin()+p.size()/2);
      std::vector<std::vector<double> > coeff0(coeff.begin(), coeff.begin()+coeff.size()/2);
      for (unsigned int i=0; i<coeff0.size(); i++) {
	coeff0[i].reserve(2); coeff0[i][0] = coeff[i][0]; coeff0[i][1] = coeff[i][1];
      }
      computeCloverSigmaOprod(oprod, x0, p0, coeff0);

      std::vector<ColorSpinorField*> x1(x.begin()+x.size()/2, x.end());
      std::vector<ColorSpinorField*> p1(p.begin()+p.size()/2, p.end());
      std::vector<std::vector<double> > coeff1(coeff.begin()+coeff.size()/2, coeff.end());
      for (unsigned int i=0; i<coeff1.size(); i++) {
	coeff1[i].reserve(2); coeff1[i][0] = coeff[coeff.size()/2 + i][0]; coeff1[i][1] = coeff[coeff.size()/2 + i][1];
      }
      computeCloverSigmaOprod(oprod, x1, p1, coeff1);

      return;
    }

    instantiate<CloverSigmaOprod>(oprod, x, p, coeff);
  }
#else // GPU_CLOVER_DIRAC not defined
  void computeCloverSigmaOprod(GaugeField &, std::vector<ColorSpinorField*> &,
			       std::vector<ColorSpinorField*> &, std::vector<std::vector<double> > &)
  {
    errorQuda("Clover Dirac operator has not been built!");
  }
#endif

} // namespace quda
