#include <quda_internal.h>
#include <gauge_field.h>
#include <random_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_random.cuh>
#include "timer.h"

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class GaugeGauss : TunableKernel2D
  {
    GaugeField &U;
    RNG &rng;
    real_t sigma;
    bool group;
    unsigned int minThreads() const { return U.VolumeCB(); }

  public:
    GaugeGauss(GaugeField &U, RNG &rng, real_t sigma) :
      TunableKernel2D(U, 2),
      U(U),
      rng(rng),
      sigma(sigma),
      group(U.LinkType() == QUDA_SU3_LINKS)
    {
      if (group) {
        logQuda(QUDA_SUMMARIZE, "Creating Gaussian distributed Lie group field with sigma = %e\n", double(sigma));
      } else {
        logQuda(QUDA_SUMMARIZE, "Creating Gaussian distributed Lie algebra field\n");
      }
      strcat(aux, group ? ",lie_group" : "lie_algebra");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (group) {
        launch<GaussGauge>(tp, stream, GaugeGaussArg<Float, nColor, recon, true>(U, rng.State(), sigma));
      } else {
        launch<GaussGauge>(tp, stream, GaugeGaussArg<Float, nColor, recon, false>(U, rng.State(), sigma));
      }
    }

    long long bytes() const { return U.Bytes(); }

    void preTune() { rng.backup(); }
    void postTune() { rng.restore(); }
  };

  void gaugeGauss(GaugeField &U, RNG &rng, real_t sigma)
  {
    if (!U.isNative()) errorQuda("Order %d with %d reconstruct not supported", U.Order(), U.Reconstruct());
    if (U.LinkType() != QUDA_SU3_LINKS && U.LinkType() != QUDA_MOMENTUM_LINKS)
      errorQuda("Unexpected link type %d", U.LinkType());

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeGauss, ReconstructFull>(U, rng, sigma);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    // ensure multi-gpu consistency if required
    getProfile().TPSTART(QUDA_PROFILE_COMMS);
    if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) {
      U.exchangeExtendedGhost(U.R());
    } else if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) {
      U.exchangeGhost();
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMMS);
  }

  void gaugeGauss(GaugeField &U, unsigned long long seed, real_t sigma)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMMS);
    RNG randstates(U, seed);
    getProfile().TPSTOP(QUDA_PROFILE_COMMS);

    gaugeGauss(U, randstates, sigma);
  }

}
