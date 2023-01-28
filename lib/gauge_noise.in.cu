#include <quda_internal.h>
#include <gauge_field.h>
#include <random_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_noise.cuh>

namespace quda {

  template <typename real, int nColor>
  class GaugeNoise : TunableKernel2D
  {
    GaugeField &U;
    RNG &rng;
    QudaNoiseType type;
    unsigned int minThreads() const { return U.VolumeCB(); }

  public:
    GaugeNoise(GaugeField &U, RNG &rng, QudaNoiseType type) :
      TunableKernel2D(U, 2),
      U(U),
      rng(rng),
      type(type)
    {
      strcat(aux, type == QUDA_NOISE_GAUSS ? ",gauss" : ",uniform");
      if (type == QUDA_NOISE_GAUSS) {
        logQuda(QUDA_SUMMARIZE, "Creating Gaussian distributed field\n");
      } else {
        logQuda(QUDA_SUMMARIZE, "Creating uniformly distributed field\n");
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (type == QUDA_NOISE_UNIFORM)
        launch<NoiseGauge>(tp, stream, GaugeNoiseArg<real, nColor, QUDA_NOISE_UNIFORM>(U, rng.State()));
      else
        launch<NoiseGauge>(tp, stream, GaugeNoiseArg<real, nColor, QUDA_NOISE_GAUSS>(U, rng.State()));
        
    }

    long long bytes() const { return U.Bytes(); }
    void preTune() { rng.backup(); }
    void postTune() { rng.restore(); }
  };

  template <int...> struct IntList { };

  template <typename real, int nColor, int...N>
  void gaugeNoise(GaugeField &U, RNG &rng, QudaNoiseType type, IntList<nColor, N...>)
  {
    if ((U.Ncolor() == 3 && U.Ncolor() == nColor) ||
        (U.Ncolor() > 3 && (U.Ncolor() / 2 == nColor))) {
      if constexpr (nColor == 3) GaugeNoise<real, nColor>(U, rng, type);
      else GaugeNoise<real, 2 * nColor>(U, rng, type);
    } else {
      if constexpr (sizeof...(N) > 0) {
        gaugeNoise<real>(U, rng, type, IntList<N...>());
      } else {
        errorQuda("Nc = %d not instantiated", U.Ncolor());
      }
    }
  }
    
  void gaugeNoise(GaugeField &U_, RNG &rng, QudaNoiseType type)
  {
    GaugeFieldParam param(U_);
    GaugeField *U = nullptr;
    bool copy_back = false;
    if (U_.Location() == QUDA_CPU_FIELD_LOCATION || U_.Precision() < QUDA_SINGLE_PRECISION ||
        U_.Reconstruct() != QUDA_RECONSTRUCT_NO || !U_.isNative()) {
      QudaPrecision prec = std::max(U_.Precision(), QUDA_SINGLE_PRECISION);
      param.setPrecision(prec, true);
      if (param.order != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Unexpected order %d", param.order);
      param.create = QUDA_NULL_FIELD_CREATE;
      param.location = QUDA_CUDA_FIELD_LOCATION;
      U = GaugeField::Create(param);
      copy_back = true;
    } else {
      U = &U_;
    }

    if (U->Precision() == QUDA_DOUBLE_PRECISION) {
      gaugeNoise<double>(*U, rng, type, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
    } else if (U->Precision() == QUDA_SINGLE_PRECISION) {
      gaugeNoise<float>(*U, rng, type, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
    } else {
      errorQuda("Unsupported precision %d", U->Precision());      
    }

    if (copy_back) {
      U_.copy(*U);
      delete U;
    }

    // ensure multi-gpu consistency if required
    if (U_.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) {
      U_.exchangeExtendedGhost(U_.R());
    } else if (U_.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) {
      if (U_.Geometry() == QUDA_COARSE_GEOMETRY)
        U_.exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
      else
        U_.exchangeGhost(QUDA_LINK_BACKWARDS);
    }
  }

  void gaugeNoise(GaugeField &U, unsigned long long seed, QudaNoiseType type)
  {
    RNG randstates(U, seed);
    gaugeNoise(U, randstates, type);
  }
  
}
