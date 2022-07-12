#include <quda_internal.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <unitarization_links.h>
#include <pgauge_monte.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/pgauge_init.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class InitGaugeCold : TunableKernel2D {
    GaugeField &U;
    unsigned int minThreads() const { return U.VolumeCB(); } // includes any extended volume

  public:
    InitGaugeCold(GaugeField &U) :
      TunableKernel2D(U, 2),
      U(U)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ColdStart>(tp, stream, InitGaugeColdArg<Float, nColor, recon>(U));
    }

    long long flops() const { return 0; }
    long long bytes() const { return U.Bytes(); }
  };

  template<typename Float, int nColors, QudaReconstructType recon>
  class InitGaugeHot : TunableKernel1D {
    const GaugeField &U;
    RNG &rng;
    unsigned int minThreads() const { return U.LocalVolumeCB(); }

  public:
    InitGaugeHot(GaugeField &U, RNG &rng) :
      TunableKernel1D(U),
      U(U),
      rng(rng)
    {
      apply(device::get_default_stream());
      qudaDeviceSynchronize();
      U.exchangeExtendedGhost(U.R(),false);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HotStart>(tp, stream, InitGaugeHotArg<Float, nColors, recon>(U, rng.State()));
    }

    void preTune() { rng.backup(); }
    void postTune() { rng.restore(); }
    long long flops() const { return 0; }
    long long bytes() const { return U.Bytes(); }
  };

  /**
   * @brief Perform a cold start to the gauge field, identity SU(3)
   * matrix, also fills the ghost links in multi-GPU case (no need to
   * exchange data)
   *
   * @param[in,out] data Gauge field
   */
  void InitGaugeField(GaugeField& data)
  {
    instantiate<InitGaugeCold>(data);
  }

  /** @brief Perform a hot start to the gauge field, random SU(3)
   * matrix, followed by reunitarization, also exchange borders links
   * in multi-GPU case.
   *
   * @param[in,out] data Gauge field
   * @param[in,out] rngstate state of the CURAND random number generator
   */
  void InitGaugeField(GaugeField& data, RNG &rngstate)
  {
    instantiate<InitGaugeHot>(data, rngstate);
  }

}
