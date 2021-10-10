#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "kernels/dslash_gamma_helper.cuh"

namespace quda {

  template <typename Float, int nColor> class GammaApply : public TunableKernel2D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const int d;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    GammaApply(ColorSpinorField &out, const ColorSpinorField &in, int d) :
      TunableKernel2D(in, in.SiteSubset()),
      out(out),
      in(in),
      d(d)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Gamma>(tp, stream, GammaArg<Float, nColor>(out, in, d));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long flops() const { return 0; }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyGamma(ColorSpinorField &out, const ColorSpinorField &in, int d)
  {
    instantiate<GammaApply>(out, in, d);
  }

  template <typename Float, int nColor> class TwistGammaApply : public TunableKernel2D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    int d;
    double kappa;
    double mu;
    double epsilon;
    int dagger;
    QudaTwistGamma5Type type;
    unsigned int minThreads() const { return in.VolumeCB() / (in.Ndim() == 5 ? in.X(4) : 1); }

  public:
    TwistGammaApply(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu,
                    double epsilon, int dagger, QudaTwistGamma5Type type) :
      TunableKernel2D(in, in.SiteSubset()),
      out(out),
      in(in),
      d(d),
      kappa(kappa),
      mu(mu),
      epsilon(epsilon),
      dagger(dagger),
      type(type)
    {
      if (d != 4) errorQuda("Unexpected d=%d", d);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<TwistGamma>(tp, stream, GammaArg<Float, nColor>(out, in, d, kappa, mu, epsilon, dagger, type));
    }

    void preTune() { if (out.V() == in.V()) out.backup(); }
    void postTune() { if (out.V() == in.V()) out.restore(); }
    long long flops() const { return 0; }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
#ifdef GPU_TWISTED_MASS_DIRAC
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
    instantiate<TwistGammaApply>(out, in, d, kappa, mu, epsilon, dagger, type);
  }
#else
  void ApplyTwistGamma(ColorSpinorField &, const ColorSpinorField &, int, double, double, double, int, QudaTwistGamma5Type)
  {
    errorQuda("Twisted mass dslash has not been built");
  }
#endif // GPU_TWISTED_MASS_DIRAC

  // Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in) { ApplyGamma(out,in,4); }

}
