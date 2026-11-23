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
      } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_CHROMA) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_CHROMA> arg(u);
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
    if (u.LinkType() == QUDA_THREE_LINKS && u.iMu() != 0.0)
      errorQuda("The imaginary chemical potential (i_mu = %g) is not supported on long links",
                static_cast<double>(u.iMu()));
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugePhase_, ReconstructNone>(u);
    // ensure that ghosts are updated if needed
    if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) u.exchangeGhost();
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeTBoundary_ : TunableKernel2D
  {
    GaugeField &u; // used for meta data only
    long long boundary_bytes;
    unsigned int minThreads() const { return u.VolumeCB(); }

  public:
    GaugeTBoundary_(GaugeField &u) :
      TunableKernel2D(u, 2),
      u(u),
      boundary_bytes(2 * (u.Bytes() / 4) * GaugePhaseGeometry(u).boundaryTimeSlices() / u.X()[3])
    {
      strcat(aux, "t_boundary");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GaugeTBoundary>(tp, stream, GaugeTBoundaryArg<Float, nColor, recon>(u));
    }

    void preTune() { u.backup(); }
    void postTune() { u.restore(); }

    long long flops() const { return 0; }
    long long bytes() const { return boundary_bytes; }
  };

  void applyGaugeTBoundary(GaugeField &u)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeTBoundary_, ReconstructGauge>(u);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
