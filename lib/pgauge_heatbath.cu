#include <quda_internal.h>
#include <timer.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <pgauge_monte.h>
#include <gauge_tools.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_heatbath.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class GaugeHB : TunableKernel1D {
    GaugeField &U;
    Float beta;
    RNG &rng;
    int mu;
    int parity;
    bool heatbath; // true = heatbath, false = over relaxation
    char aux2[TuneKey::aux_n];
    unsigned int minThreads() const { return U.LocalVolumeCB(); }

  public:
    GaugeHB(GaugeField &U, double beta, RNG &rng, int mu, int parity, bool heatbath) :
      TunableKernel1D(U),
      U(U),
      beta(static_cast<Float>(beta)),
      rng(rng),
      mu(mu),
      parity(parity),
      heatbath(heatbath)
    {
      strcat(aux, mu == 0 ? ",mu=0" : mu == 1 ? ",mu=1" : mu == 2 ? ",mu=2" : ",mu=3");
      strcat(aux, parity ? ",parity=1" : ",parity=0");
      strcat(aux, heatbath ? ",heatbath" : ",ovr");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (heatbath) {
        launch<HB>(tp, stream, MonteArg<Float, nColor, recon, true>(U, beta, rng.State(), mu, parity));
      } else {
        launch<HB>(tp, stream, MonteArg<Float, nColor, recon, false>(U, beta, rng.State(), mu, parity));
      }
    }

    void preTune() {
      U.backup();
      if (heatbath) rng.backup();
    }

    void postTune() {
      U.restore();
      if (heatbath) rng.restore();
    }

    long long flops() const
    {
      //NEED TO CHECK THIS!!!!!!
      if ( nColor == 3 ) {
        long long flop = 2268LL;
        if (heatbath) {
          flop += 801LL;
        } else {
          flop += 843LL;
        }
        flop *= U.LocalVolumeCB();
        return flop;
      } else {
        long long flop = nColor * nColor * nColor * 84LL;
        if (heatbath) {
          flop += nColor * nColor * nColor + (nColor * ( nColor - 1) / 2) * (46LL + 48LL + 56LL * nColor);
        } else {
          flop += nColor * nColor * nColor + (nColor * ( nColor - 1) / 2) * (17LL + 112LL * nColor);
        }
        flop *= U.LocalVolumeCB();
        return flop;
      }
    }

    long long bytes() const
    {
      //NEED TO CHECK THIS!!!!!!
      if ( nColor == 3 ) {
        long long byte = 20LL * recon * sizeof(Float);
        if (heatbath) byte += 2LL * sizeof(RNGState);
        byte *= U.LocalVolumeCB();
        return byte;
      } else {
        long long byte = 20LL * nColor * nColor * 2 * sizeof(Float);
        if (heatbath) byte += 2LL * sizeof(RNGState);
        byte *= U.LocalVolumeCB();
        return byte;
      }
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct MonteAlg {
    MonteAlg(GaugeField& data, RNG &rngstate, Float Beta, int nhb, int nover)
    {
      host_timer_t timer;
      double hb_time = 0.0, ovr_time = 0.0;
      if (getVerbosity() >= QUDA_VERBOSE) timer.start();

      for (int step = 0; step < nhb; step++) {
        for (int parity = 0; parity < 2; parity++) {
          for (int mu = 0; mu < 4; ++mu) {
            GaugeHB<Float, nColor, recon>(data, Beta, rngstate, mu, parity, true);
            PGaugeExchange(data, mu, parity);
          }
        }
      }

      if (getVerbosity() >= QUDA_VERBOSE) {
        qudaDeviceSynchronize();
        timer.stop();
        hb_time = timer.last();
        timer.start();
      }

      for (int step = 0; step < nover; step++) {
        for (int parity = 0; parity < 2; parity++) {
          for (int mu = 0; mu < 4; mu++) {
            GaugeHB<Float, nColor, recon>(data, Beta, rngstate, mu, parity, false);
            PGaugeExchange(data, mu, parity);
          }
        }
      }

      if (getVerbosity() >= QUDA_VERBOSE) {
        qudaDeviceSynchronize();
        timer.stop();
        ovr_time = timer.last();
        printfQuda("Heatbath time = %6.6f, Over-relaxation time = %6.6f\n", hb_time, ovr_time);
      }
    }
  };

  /** @brief Perform heatbath and overrelaxation. Performs nhb heatbath steps followed by nover overrelaxation steps.
   *
   * @param[in,out] data Gauge field
   * @param[in,out] rngstate state of the CURAND random number generator
   * @param[in] Beta inverse of the gauge coupling, beta = 2 Nc / g_0^2
   * @param[in] nhb number of heatbath steps
   * @param[in] nover number of overrelaxation steps
   */
  void Monte(GaugeField& data, RNG &rngstate, double Beta, int nhb, int nover)
  {
    instantiate<MonteAlg>(data, rngstate, (float)Beta, nhb, nover);
  }

}
