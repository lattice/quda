#include <color_spinor_field.h>
#include <random_quda.h>
#include <tunable_nd.h>
#include <kernels/spinor_noise.cuh>

namespace quda {

  template <typename real, int Ns, int Nc, QudaFieldOrder order>
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
        launch<NoiseSpinor>(tp, stream, SpinorNoiseArg<real, Ns, Nc, order, QUDA_NOISE_GAUSS>(v, rng.State()));
        break;
      case QUDA_NOISE_UNIFORM:
        launch<NoiseSpinor>(tp, stream, SpinorNoiseArg<real, Ns, Nc, order, QUDA_NOISE_UNIFORM>(v, rng.State()));
        break;
      default: errorQuda("Noise type %d not implemented", type);
      }
    }

    long long flops() const { return 0; }
    long long bytes() const { return v.Bytes(); }
    void preTune() { rng.backup(); }
    void postTune(){ rng.restore(); }
  };

  /** Decide on the input order*/
  template <typename real, int Ns, int Nc>
  void spinorNoise(ColorSpinorField &in, RNG &rng, QudaNoiseType type)
  {
    if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      SpinorNoise<real,Ns,Nc,QUDA_FLOAT2_FIELD_ORDER>(in, rng, type);
    } else if (in.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      SpinorNoise<real,Ns,Nc,QUDA_FLOAT4_FIELD_ORDER>(in, rng, type);
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", in.FieldOrder(), Ns, Nc);
    }
  }

  template <typename real, int Ns>
  void spinorNoise(ColorSpinorField &src, RNG& randstates, QudaNoiseType type)
  {
    if (src.Ncolor() == 3) {
      spinorNoise<real,Ns,3>(src, randstates, type);
    } else if (src.Ncolor() == 6) {
      spinorNoise<real,Ns,6>(src, randstates, type);
    } else if (src.Ncolor() == 24) {
      spinorNoise<real,Ns,24>(src, randstates, type);
    } else if (src.Ncolor() == 32) {
      spinorNoise<real,Ns,32>(src, randstates, type);
    } else if (src.Ncolor() == 64) {
      spinorNoise<real,Ns,64>(src, randstates, type);
    } else if (src.Ncolor() == 96) {
      spinorNoise<real,Ns,96>(src, randstates, type);
    } else {
      errorQuda("nColor = %d not implemented", src.Ncolor());
    }
  }

  template <typename real>
#if defined(NSPIN1) || defined(NSPIN2) || defined(NSPIN4)
  void spinorNoise(ColorSpinorField &src, RNG& randstates, QudaNoiseType type)
#else
  void spinorNoise(ColorSpinorField &src, RNG &, QudaNoiseType)
#endif
  {
    if (src.Nspin() == 4) {
#ifdef NSPIN4
      spinorNoise<real,4>(src, randstates, type);
#else
      errorQuda("spinorNoise has not been built for nSpin=%d fields", src.Nspin());
#endif
    } else if (src.Nspin() == 2) {
#ifdef NSPIN2
      spinorNoise<real,2>(src, randstates, type);
#else
      errorQuda("spinorNoise has not been built for nSpin=%d fields", src.Nspin());
#endif
    } else if (src.Nspin() == 1) {
#ifdef NSPIN1
      spinorNoise<real,1>(src, randstates, type);
#else
      errorQuda("spinorNoise has not been built for nSpin=%d fields", src.Nspin());
#endif
    } else {
      errorQuda("Nspin = %d not implemented", src.Nspin());
    }
  }

  void spinorNoise(ColorSpinorField &src_, RNG &randstates, QudaNoiseType type)
  {
    // if src is a CPU field then create GPU field
    ColorSpinorField *src = &src_;
    if (src_.Location() == QUDA_CPU_FIELD_LOCATION || src_.Precision() < QUDA_SINGLE_PRECISION) {
      ColorSpinorParam param(src_);
      QudaPrecision prec = std::max(src_.Precision(), QUDA_SINGLE_PRECISION);
      param.setPrecision(prec, prec, true); // change to native field order
      param.create = QUDA_NULL_FIELD_CREATE;
      param.location = QUDA_CUDA_FIELD_LOCATION;
      src = ColorSpinorField::Create(param);
    }

    switch (src->Precision()) {
    case QUDA_DOUBLE_PRECISION: spinorNoise<double>(*src, randstates, type); break;
    case QUDA_SINGLE_PRECISION: spinorNoise<float>(*src, randstates, type); break;
    default: errorQuda("Precision %d not implemented", src->Precision());
    }

    if (src != &src_) {
      src_ = *src; // upload result
      delete src;
    }
  }

  void spinorNoise(ColorSpinorField &src, unsigned long long seed, QudaNoiseType type)
  {
    RNG randstates(src, seed);
    spinorNoise(src, randstates, type);
  }

} // namespace quda
