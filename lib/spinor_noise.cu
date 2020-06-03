/*
  Spinor reordering and copying routines.  These are implemented to
  un on both CPU and GPU.  Here we are templating on the following:
  - input precision
  - output precision
  - number of colors
  - number of spins
  - field ordering
*/

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <utility> // for std::swap
#include <random_quda.h>

namespace quda {

  using namespace colorspinor;

  template<typename real, int Ns, int Nc, QudaFieldOrder order>
  struct Arg {
    typedef typename colorspinor::FieldOrderCB<real,Ns,Nc,1,order> V;
    V v;
    const int nParity;
    const int volumeCB;
    RNG rng;
    Arg(ColorSpinorField &v, RNG &rng) : v(v), nParity(v.SiteSubset()), volumeCB(v.VolumeCB()), rng(rng) { }
  };

  template<typename real, typename Arg> // Gauss
  __device__ __host__ inline void genGauss(Arg &arg, cuRNGState& localState, int parity, int x_cb, int s, int c) {
    real phi = 2.0*M_PI*Random<real>(localState);
    real radius = Random<real>(localState);
    radius = sqrt(-1.0 * log(radius));
    arg.v(parity, x_cb, s, c) = complex<real>(radius*cos(phi),radius*sin(phi));
  }

  template<typename real, typename Arg> // Uniform
  __device__ __host__ inline void genUniform(Arg &arg, cuRNGState& localState, int parity, int x_cb, int s, int c) {
    real x = Random<real>(localState);
    real y = Random<real>(localState);
    arg.v(parity, x_cb, s, c) = complex<real>(x, y);
  }

  /** CPU function to reorder spinor fields.  */
  template <typename real, int Ns, int Nc, QudaNoiseType type, typename Arg> void SpinorNoiseCPU(Arg &arg)
  {

    for (int parity = 0; parity < arg.nParity; parity++) {
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            cuRNGState localState = arg.rng.State()[parity * arg.volumeCB + x_cb];
            if (type == QUDA_NOISE_GAUSS)
              genGauss<real>(arg, localState, parity, x_cb, s, c);
            else if (type == QUDA_NOISE_UNIFORM)
              genUniform<real>(arg, localState, parity, x_cb, s, c);
            arg.rng.State()[parity * arg.volumeCB + x_cb] = localState;
          }
        }
      }
    }
  }

  /** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
  template <typename real, int Ns, int Nc, QudaNoiseType type, typename Arg>
    __global__ void SpinorNoiseGPU(Arg arg) {

    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    if (parity >= arg.nParity) return;

    cuRNGState localState = arg.rng.State()[parity * arg.volumeCB + x_cb];
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        if (type == QUDA_NOISE_GAUSS) genGauss<real>(arg, localState, parity, x_cb, s, c);
        else if (type == QUDA_NOISE_UNIFORM) genUniform<real>(arg, localState, parity, x_cb, s, c);
      }
    }
    arg.rng.State()[parity * arg.volumeCB + x_cb] = localState;
  }

  template <typename real, int Ns, int Nc, QudaNoiseType type, typename Arg>
  class SpinorNoise : TunableVectorY {
    Arg &arg;
    const ColorSpinorField &meta; // this reference is for meta data only

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return meta.VolumeCB(); }

  public:
    SpinorNoise(Arg &arg, const ColorSpinorField &meta)
      : TunableVectorY(meta.SiteSubset()), arg(arg), meta(meta) {
      strcpy(aux, meta.AuxString());
      strcat(aux, meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU" : ",CPU");
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      SpinorNoiseGPU<real, Ns, Nc, type><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; }
    long long bytes() const { return meta.Bytes(); }
    void preTune() { arg.rng.backup(); }
    void postTune(){ arg.rng.restore(); }
  };

  template <typename real, int Ns, int Nc, QudaFieldOrder order>
  void spinorNoise(ColorSpinorField &in, RNG &rngstate, QudaNoiseType type) {
    Arg<real, Ns, Nc, order> arg(in, rngstate);
    switch (type) {
    case QUDA_NOISE_GAUSS:
      {
        SpinorNoise<real, Ns, Nc, QUDA_NOISE_GAUSS, Arg<real, Ns, Nc, order> > noise(arg, in);
        noise.apply(0);
        break;
      }
    case QUDA_NOISE_UNIFORM:
      {
        SpinorNoise<real, Ns, Nc, QUDA_NOISE_UNIFORM, Arg<real, Ns, Nc, order> > noise(arg, in);
        noise.apply(0);
        break;
      }
    default:
      errorQuda("Noise type %d not implemented", type);
    }
  }

  /** Decide on the input order*/
  template <typename real, int Ns, int Nc>
  void spinorNoise(ColorSpinorField &in, RNG &rngstate, QudaNoiseType type)
  {
    if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      spinorNoise<real,Ns,Nc,QUDA_FLOAT2_FIELD_ORDER>(in, rngstate, type);
    } else if (in.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      spinorNoise<real,Ns,Nc,QUDA_FLOAT4_FIELD_ORDER>(in, rngstate, type);
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
  void spinorNoise(ColorSpinorField &src, RNG& randstates, QudaNoiseType type)
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
    RNG *randstates = new RNG(src, seed);
    randstates->Init();
    spinorNoise(src, *randstates, type);
    randstates->Release();
    delete randstates;
  }

} // namespace quda
