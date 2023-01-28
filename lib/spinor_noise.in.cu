#include <color_spinor_field.h>
#include <random_quda.h>
#include <tunable_nd.h>
#include <kernels/spinor_noise.cuh>
#include <instantiate.h>

namespace quda {

  template <typename real, int Ns, int Nc>
  class SpinorNoise : TunableKernel2D {
    ColorSpinorField &v;
    RNG &rng;
    QudaNoiseType type;
    unsigned int minThreads() const { return v.VolumeCB(); }

  public:
    SpinorNoise(ColorSpinorField &v, RNG &rng, QudaNoiseType type) :
      TunableKernel2D(v, v.SiteSubset()),
      v(v),
      rng(rng),
      type(type)
    {
      strcat(aux, type == QUDA_NOISE_GAUSS ? ",gauss" : ",uniform");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (type) {
      case QUDA_NOISE_GAUSS:
        launch<NoiseSpinor>(tp, stream, SpinorNoiseArg<real, Ns, Nc, QUDA_NOISE_GAUSS>(v, rng.State()));
        break;
      case QUDA_NOISE_UNIFORM:
        launch<NoiseSpinor>(tp, stream, SpinorNoiseArg<real, Ns, Nc, QUDA_NOISE_UNIFORM>(v, rng.State()));
        break;
      default: errorQuda("Noise type %d not implemented", type);
      }
    }

    long long bytes() const { return v.Bytes(); }
    void preTune() { rng.backup(); }
    void postTune(){ rng.restore(); }
  };

  template <int...> struct IntList { };

  template <typename real, int Ns, int Nc, int...N>
  void spinorNoise(ColorSpinorField &src, RNG& randstates, QudaNoiseType type, IntList<Nc, N...>)
  {
    if (src.Ncolor() == Nc) {
      SpinorNoise<real, Ns, Nc>(src, randstates, type);
    } else {
      if constexpr (sizeof...(N) > 0) spinorNoise<real, Ns>(src, randstates, type, IntList<N...>());
      else errorQuda("nColor = %d not implemented", src.Ncolor());
    }
  }

  template <typename real>
  void spinorNoise(ColorSpinorField &src, RNG& randstates, QudaNoiseType type)
  {
    checkNative(src);
    if (!is_enabled_spin(src.Nspin()))
      errorQuda("spinorNoise has not been built for nSpin=%d fields", src.Nspin());

    if (src.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) spinorNoise<real, 4>(src, randstates, type, IntList<3>());
    } else if (src.Nspin() == 2) {
      if constexpr (is_enabled_spin(2)) spinorNoise<real, 2>(src, randstates, type, IntList<3, @QUDA_MULTIGRID_NVEC_LIST@>());
    } else if (src.Nspin() == 1) {
      if constexpr (is_enabled_spin(1)) spinorNoise<real, 1>(src, randstates, type, IntList<3>());
    } else {
      errorQuda("Nspin = %d not implemented", src.Nspin());
    }
  }

  void spinorNoise(ColorSpinorField &src_, RNG &randstates, QudaNoiseType type)
  {
    // if src is a CPU field then create GPU field
    ColorSpinorField src;
    ColorSpinorParam param(src_);
    bool copy_back = false;
    if (src_.Location() == QUDA_CPU_FIELD_LOCATION || src_.Precision() < QUDA_SINGLE_PRECISION) {
      QudaPrecision prec = std::max(src_.Precision(), QUDA_SINGLE_PRECISION);
      param.setPrecision(prec, prec, true); // change to native field order
      param.create = QUDA_NULL_FIELD_CREATE;
      param.location = QUDA_CUDA_FIELD_LOCATION;
      src = ColorSpinorField(param);
      copy_back = true;
    } else {
      src = src_.create_alias(param);
    }

    switch (src.Precision()) {
    case QUDA_DOUBLE_PRECISION: spinorNoise<double>(src, randstates, type); break;
    case QUDA_SINGLE_PRECISION: spinorNoise<float>(src, randstates, type); break;
    default: errorQuda("Precision %d not implemented", src.Precision());
    }

    if (copy_back) src_ = src; // copy back if needed
  }

  void spinorNoise(ColorSpinorField &src, unsigned long long seed, QudaNoiseType type)
  {
    RNG randstates(src, seed);
    spinorNoise(src, randstates, type);
  }

} // namespace quda
