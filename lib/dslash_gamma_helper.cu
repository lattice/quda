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

  template <typename Float, int nColor> class Split5DTo4DApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    const ColorSpinorField &in;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Split5DTo4DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(out[0], out.size(), in[0].SiteSubset()), out(out), in(in[0])
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Split5DTo4D>(tp, stream, Split5DTo4DArgs<Float, nColor>(out, in));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  void Split5DTo4DFields(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>() || is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      instantiate_recurse2<Split5DTo4DApply>(out, cvector_ref<const ColorSpinorField>{in});
    } else {
      errorQuda("Twisted clover operator or domain wall operator has not been built");
    }
  }

  template <typename Float, int nColor> class Join4DTo5DApply : public TunableKernel3D
  {
    ColorSpinorField &out;
    cvector_ref<const ColorSpinorField> &in;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    Join4DTo5DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(in[0], in.size(), out[0].SiteSubset()), out(out[0]), in(in)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Join4DTo5D>(tp, stream, Join4DTo5DArgs<Float, nColor>(out, in));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  void Join4DTo5DField(ColorSpinorField &out, cvector_ref<const ColorSpinorField> &in)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>() || is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      instantiate_recurse2<Join4DTo5DApply>(cvector_ref<ColorSpinorField>{out}, in);
    } else {
      errorQuda("Twisted mass operator or domain wall operator has not been built");
    }
  }
}
