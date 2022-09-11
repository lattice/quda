#include <comm_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_phase.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class GaugePhase_ : TunableKernel2D {
    GaugeField &u; // used for meta data only
    unsigned int minThreads() const { return u.VolumeCB(); }

  public:
    GaugePhase_(GaugeField &u) :
      TunableKernel2D(u, 2),
      u(u)
    {
      strcat(aux, "phase=");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_MILC> arg(u);
        launch<GaugePhase>(tp, stream, arg);
      } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_CPS) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_CPS> arg(u);
        launch<GaugePhase>(tp, stream, arg);
      } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_TIFR> arg(u);
        launch<GaugePhase>(tp, stream, arg);
      } else {
        errorQuda("Undefined phase type %d", u.StaggeredPhase());
      }
    }

    void preTune() { u.backup(); }
    void postTune() { u.restore(); }

    long long flops() const { return 0; }
    long long bytes() const { return 2 * u.Bytes(); }
  };

  void applyGaugePhase(GaugeField &u)
  {
    instantiate<GaugePhase_, ReconstructNone>(u);
    // ensure that ghosts are updated if needed
    if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) u.exchangeGhost();
  }

} // namespace quda
