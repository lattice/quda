#include "color_spinor_field.h"
#include "clover_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "kernels/dslash_clover_helper.cuh"

namespace quda {

  template <typename Float, int nColor> class Clover : public TunableKernel3D {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const CloverField &clover;
    bool inverse;
    int parity;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Clover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const CloverField &clover,
           bool inverse, int parity) :
      TunableKernel3D(in[0], in.size(), in.SiteSubset()), out(out), in(in), clover(clover), inverse(inverse), parity(parity)
    {
      setRHSstring(aux, in.size());
      if (in.Nspin() != 4 || out.Nspin() != 4) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      if (!inverse) errorQuda("Unsupported direct application");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverApply>(tp, stream, CloverArg<Float, nColor>(out, in, clover, parity));
    }

    // Backup if in and out fields alias
    void preTune() { out.backup(); }
    void postTune() { out.restore(); }

    long long flops() const { return in.size() * in.Volume() * 504ll; }

    long long bytes() const { return in.size() * (out.Bytes() + in.Bytes() + clover.Bytes() / (3 - in.SiteSubset())); }
  };

  //Apply the clover matrix field to a colorspinor field
  //out(x) = clover*in
  void ApplyClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   const CloverField &clover, bool inverse, int parity)
  {
    if (in.size() > MAX_MULTI_RHS) {
      ApplyClover({out.begin(), out.begin() + out.size() / 2}, {in.begin(), in.begin() + in.size() / 2},
                  clover, inverse, parity);
      ApplyClover({out.begin() + out.size() / 2, out.end()}, {in.begin() + in.size() / 2, in.end()},
                  clover, inverse, parity);
      return;
    }

    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>()) {
      instantiate<Clover>(out, in, clover, inverse, parity);
    } else {
      errorQuda("Clover dslash has not been built");
    }
  }

  template <typename Float, int nColor> class TwistClover : public TunableKernel3D {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const CloverField &clover;
    double kappa;
    double mu;
    double epsilon;
    int parity;
    bool inverse;
    int dagger;
    QudaTwistGamma5Type twist;
    unsigned int minThreads() const { return in.VolumeCB(); }

    unsigned int sharedBytesPerThread() const
    {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
        return 0;
      } else {
        return (in.Nspin() / 2) * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type);
      }
    }

  public:
    TwistClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const CloverField &clover,
                double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist) :
      TunableKernel3D(in[0], in.size() * in.TwistFlavor(), in.SiteSubset()),
      out(out),
      in(in),
      clover(clover),
      kappa(kappa),
      mu(mu),
      epsilon(epsilon),
      parity(parity),
      inverse(twist != QUDA_TWIST_GAMMA5_DIRECT),
      dagger(dagger),
      twist(twist)
    {
      if (in.Nspin() != 4 || out.Nspin() != 4) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      strcat(aux, inverse ? ",inverse" : ",direct");
      setRHSstring(aux, in.size());
      resizeStep(in.TwistFlavor(), 1); // this will force flavor to be contained in the block
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (inverse) {
        CloverArg<Float, nColor, true> arg(out, in, clover, parity, kappa, mu, epsilon, dagger, twist);
        if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
          launch<TwistCloverApply>(tp, stream, arg);
        } else {
          launch<NdegTwistCloverApply>(tp, stream, arg);
        }
      } else {
        CloverArg<Float, nColor, false> arg(out, in, clover, parity, kappa, mu, epsilon, dagger, twist);
        if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
          launch<TwistCloverApply>(tp, stream, arg);
        } else {
          launch<NdegTwistCloverApply>(tp, stream, arg);
        }
      }
    }

    // Restore if the in and out fields alias
    void preTune() { out.backup(); }
    void postTune() { out.restore(); }

    long long flops() const { return in.size() * (inverse ? 1056ll : 552ll) * in.Volume(); }
    long long bytes() const {
      long long rtn = out.Bytes() + in.Bytes() + clover.Bytes() / (3 - in.SiteSubset());
      if (twist == QUDA_TWIST_GAMMA5_INVERSE && !clover::dynamic_inverse())
	rtn += clover.Bytes() / (3 - in.SiteSubset());
      return in.size() * rtn;
    }
  };

  //Apply the twisted-clover matrix field to a colorspinor field
  void ApplyTwistClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                        const CloverField &clover, double kappa, double mu, double epsilon, int parity, int dagger,
                        QudaTwistGamma5Type twist)
  {
    if (in.size() > MAX_MULTI_RHS) {
      ApplyTwistClover({out.begin(), out.begin() + out.size() / 2}, {in.begin(), in.begin() + in.size() / 2},
                       clover, kappa, mu, epsilon, parity, dagger, twist);
      ApplyTwistClover({out.begin() + out.size() / 2, out.end()}, {in.begin() + in.size() / 2, in.end()},
                       clover, kappa, mu, epsilon, parity, dagger, twist);
      return;
    }

    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>()) {
      instantiate<TwistClover>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else {
      errorQuda("Twisted-clover operator has not been built");
    }
  }

}
