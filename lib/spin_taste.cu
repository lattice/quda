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
    SpinTastePhase_(ColorSpinorField &out, const ColorSpinorField &in, QudaSpinTasteGamma gamma) :
      TunableKernel2D(in, 2),
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
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G1> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GX) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GX> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GY) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GY> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GZ) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GZ> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GT) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GT> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_G5) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G5> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GYGZ) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GYGZ> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GZGX) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GZGX> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GXGY) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GXGY> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GXGT) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GXGT> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GYGT) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GYGT> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_GZGT) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_GZGT> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_G5GX) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G5GX> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_G5GY) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G5GY> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_G5GZ) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G5GZ> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
      } else if (gamma == QUDA_SPIN_TASTE_G5GT) {
        SpinTasteArg<Float, nColor, QUDA_SPIN_TASTE_G5GT> arg(out, in);
        launch<SpinTastePhase>(tp, stream, arg);
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
