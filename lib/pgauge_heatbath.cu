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
    GaugeHB(GaugeField &U, double beta, RNG &rng) :
      TunableKernel1D(U),
      U(U),
      beta(static_cast<Float>(beta)),
      rng(rng),
      mu(0),
      parity(0),
      heatbath(false)
    {
      strcpy(aux2, aux);
    }

    void set_param(int _mu, int _parity, bool _heatbath)
    {
      mu = _mu;
      parity = _parity;
      heatbath = _heatbath;
      strcpy(aux, aux2);
      strcat(aux, heatbath ? ",heatbath" : ",ovr");
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (heatbath) {
        MonteArg<Float, nColor, recon, true> arg(U, beta, rng.State(), mu, parity);
        launch<HB>(tp, stream, arg);
      } else {
        MonteArg<Float, nColor, recon, false> arg(U, beta, rng.State(), mu, parity);
        launch<HB>(tp, stream, arg);
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
      TimeProfile profileHBOVR("HeatBath_OR_Relax", false);

      if (getVerbosity() >= QUDA_SUMMARIZE) profileHBOVR.TPSTART(QUDA_PROFILE_COMPUTE);
      GaugeHB<Float, nColor, recon> hb(data, Beta, rngstate);
      for ( int step = 0; step < nhb; ++step ) {
        for ( int parity = 0; parity < 2; ++parity ) {
          for ( int mu = 0; mu < 4; ++mu ) {
            hb.set_param(mu, parity, true);
            hb.apply(device::get_default_stream());
            PGaugeExchange(data, mu, parity);
          }
        }
      }
      if (getVerbosity() >= QUDA_VERBOSE) {
        qudaDeviceSynchronize();
        profileHBOVR.TPSTOP(QUDA_PROFILE_COMPUTE);
        double secs = profileHBOVR.Last(QUDA_PROFILE_COMPUTE);
        double gflops = (hb.flops() * 8 * nhb * 1e-9) / (secs);
        double gbytes = hb.bytes() * 8 * nhb / (secs * 1e9);
        printfQuda("HB: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops * comm_size(), gbytes * comm_size());
      }

      if (getVerbosity() >= QUDA_VERBOSE) profileHBOVR.TPSTART(QUDA_PROFILE_COMPUTE);
      GaugeHB<Float, nColor, recon> relax(data, Beta, rngstate);
      for ( int step = 0; step < nover; ++step ) {
        for ( int parity = 0; parity < 2; ++parity ) {
          for ( int mu = 0; mu < 4; ++mu ) {
            relax.set_param(mu, parity, false);
            relax.apply(device::get_default_stream());
            PGaugeExchange(data, mu, parity);
          }
        }
      }
      if (getVerbosity() >= QUDA_VERBOSE) {
        qudaDeviceSynchronize();
        profileHBOVR.TPSTOP(QUDA_PROFILE_COMPUTE);
        double secs = profileHBOVR.Last(QUDA_PROFILE_COMPUTE);
        double gflops = (relax.flops() * 8 * nover * 1e-9) / (secs);
        double gbytes = relax.bytes() * 8 * nover / (secs * 1e9);
        printfQuda("OVR: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops * comm_size(), gbytes * comm_size());
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
#ifdef GPU_GAUGE_ALG
  void Monte(GaugeField& data, RNG &rngstate, double Beta, int nhb, int nover)
  {
    instantiate<MonteAlg>(data, rngstate, (float)Beta, nhb, nover);
  }
#else
  void Monte(GaugeField &, RNG &, double, int, int)
  {
    errorQuda("Pure gauge code has not been built");
  }
#endif // GPU_GAUGE_ALG

}
