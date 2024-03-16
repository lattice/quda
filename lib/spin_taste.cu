#include <comm_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/spin_taste.cuh>

namespace quda {

  template <typename Float, int nColor>
  class SpinTastePhase_ : TunableKernel2D {
    const ColorSpinorField &in;
    ColorSpinorField &out;
    QudaSpinTasteGamma gamma; // used for meta data only
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    template <QudaSpinTasteGamma gamma> using Arg = SpinTasteArg<Float, nColor, gamma>;

    SpinTastePhase_(ColorSpinorField &out, const ColorSpinorField &in, QudaSpinTasteGamma gamma) :
      TunableKernel2D(in, in.SiteSubset()),
      in(in),
      out(out),
      gamma(gamma)
    {
      strcat(aux, "gamma=");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (gamma == QUDA_SPIN_TASTE_G1) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G1>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GX) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GX>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GY) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GY>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GZ) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GZ>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GT) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GT>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_G5) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G5>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GYGZ) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GYGZ>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GZGX) {
        launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GZGX>(out, in));	
      } else if (gamma == QUDA_SPIN_TASTE_GXGY) {
        launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GXGY>(out, in));	
      } else if (gamma == QUDA_SPIN_TASTE_GXGT) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GXGT>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GYGT) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GYGT>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_GZGT) {
        launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_GZGT>(out, in));	      
      } else if (gamma == QUDA_SPIN_TASTE_G5GX) {
        launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G5GX>(out, in));	      
      } else if (gamma == QUDA_SPIN_TASTE_G5GY) {
        launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G5GY>(out, in));	      
      } else if (gamma == QUDA_SPIN_TASTE_G5GZ) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G5GZ>(out, in));
      } else if (gamma == QUDA_SPIN_TASTE_G5GT) {
	launch<SpinTastePhase>(tp, stream, Arg<QUDA_SPIN_TASTE_G5GT>(out, in));
      } else {
        errorQuda("Undefined gamma type");
      }
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }

    long long flops() const { return 0; }
    long long bytes() const { return 2 * in.Bytes(); }
  };

#ifdef GPU_STAGGERED_DIRAC
  void applySpinTaste(ColorSpinorField &out, const ColorSpinorField &in, QudaSpinTasteGamma gamma)
  {
    instantiate<SpinTastePhase_>(out, in, gamma);
    //// ensure that ghosts are updated if needed
    //if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) u.exchangeGhost();
  }
#else
  void applySpinTaste(ColorSpinorField &out, const ColorSpinorField &in, QudaSpinTasteGamma gamma)
  {
    errorQuda("Gauge tools are not build");
  }
#endif

} // namespace quda
