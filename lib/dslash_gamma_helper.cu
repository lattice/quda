#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "kernels/dslash_gamma_helper.cuh"

namespace quda {

  template <typename Float, int nColor> class GammaApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const int d;
    const int proj;
    unsigned int minThreads() const { return in.VolumeCB() / (in.Ndim() == 5 ? in.X(4) : 1); }

  public:
    GammaApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d, int proj = 0) :
      TunableKernel3D(in[0], in.size(), in.SiteSubset()), out(out), in(in), d(d), proj(proj)
    {
      setRHSstring(aux, in.size());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (proj == 0)
        launch<Gamma>(tp, stream, GammaArg<Float, nColor>(out, in, d));
      else
        launch<ChiralProject>(tp, stream, GammaArg<Float, nColor>(out, in, d, proj));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyGamma(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d)
  {
    instantiate_recurse2<GammaApply>(out, in, d);
  }

  // Applies out(x) = 1/2 * [(1 +/- gamma5) * in] + out
  void ApplyChiralProj(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const int proj)
  {
    checkPrecision(out, in); // check all precisions match
    checkLocation(out, in);  // check all locations match
    // Launch with 4 as the gamma matrix arg to stop the constructor from erroring out,
    // but this parameter is not used for chiral projection.
    instantiate<GammaApply>(out, in, 4, proj);
  }

  template <typename Float, int nColor> class TwistGammaApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    int d;
    double kappa;
    double mu;
    double epsilon;
    int dagger;
    QudaTwistGamma5Type type;
    unsigned int minThreads() const { return in.VolumeCB() / (in.Ndim() == 5 ? in.X(4) : 1); }

  public:
    TwistGammaApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d, double kappa,
                    double mu, double epsilon, int dagger, QudaTwistGamma5Type type) :
      TunableKernel3D(in[0], in.size(), in.SiteSubset()),
      out(out),
      in(in),
      d(d),
      kappa(kappa),
      mu(mu),
      epsilon(epsilon),
      dagger(dagger),
      type(type)
    {
      setRHSstring(aux, in.size());
      if (d != 4) errorQuda("Unexpected d=%d", d);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<TwistGamma>(tp, stream, GammaArg<Float, nColor>(out, in, d, 0, kappa, mu, epsilon, dagger, type));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyTwistGamma(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d, double kappa,
                       double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      instantiate_recurse2<TwistGammaApply>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else {
      errorQuda("Twisted mass operator has not been built");
    }
  }

  // Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
  void gamma5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) { ApplyGamma(out, in, 4); }

  template <typename Float, int nColor> class TauApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const int d;
    unsigned int minThreads() const { return in.VolumeCB() / 2; }

  public:
    TauApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d) :
      TunableKernel3D(in[0], in.size(), in.SiteSubset()), out(out), in(in), d(d)
    {
      setRHSstring(aux, in.size());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Tau>(tp, stream, GammaArg<Float, nColor>(out, in, d));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  // Apply the tau1 matrix to a doublet colorspinor field
  // out(x) = tau_1*in
  void ApplyTau(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      instantiate_recurse2<TauApply>(out, in, d);
    } else {
      errorQuda("Twisted mass operator has not been built");
    }
  }
}
