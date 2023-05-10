#include "color_spinor_field.h"
#include "clover_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "kernels/dslash_clover_helper.cuh"

namespace quda {

  template <typename Float, int nColor> class Clover : public TunableKernel3D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const CloverField &clover;
    bool inverse;
    int parity;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Clover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity) :
      TunableKernel3D(in, 1, in.SiteSubset()),
      out(out),
      in(in),
      clover(clover),
      inverse(inverse),
      parity(parity)
    {
      if (in.Nspin() != 4 || out.Nspin() != 4) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      if (!inverse) errorQuda("Unsupported direct application");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverApply>(tp, stream, CloverArg<Float, nColor>(out, in, clover, parity));
    }

    void preTune() { if (out.V() == in.V()) out.backup(); }  // Backup if in and out fields alias
    void postTune() { if (out.V() == in.V()) out.restore(); } // Restore if the in and out fields alias
    long long flops() const { return in.Volume()*504ll; }
    long long bytes() const { return out.Bytes() + in.Bytes() + clover.Bytes() / (3 - in.SiteSubset()); }
  };

#ifdef GPU_CLOVER_DIRAC
  //Apply the clover matrix field to a colorspinor field
  //out(x) = clover*in
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity)
  {
    instantiate<Clover>(out, in, clover, inverse, parity);
  }
#else
  void ApplyClover(ColorSpinorField &, const ColorSpinorField &, const CloverField &, bool, int)
  {
    errorQuda("Clover dslash has not been built");
  }
#endif // GPU_CLOVER_DIRAC

  template <typename Float, int nColor> class TwistClover : public TunableKernel3D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
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
      return (in.Nspin() / 2) * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type);
    }

  public:
    TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
                double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist) :
      TunableKernel3D(in, in.TwistFlavor(), in.SiteSubset()),
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
      resizeVector(2, in.SiteSubset());
      resizeStep(2, 1); // this will force flavor to be contained in the block
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

    void preTune() { if (out.V() == in.V()) out.backup(); } // Restore if the in and out fields alias
    void postTune() { if (out.V() == in.V()) out.restore(); } // Restore if the in and out fields alias
    long long flops() const { return (inverse ? 1056ll : 552ll) * in.Volume(); }
    long long bytes() const {
      long long rtn = out.Bytes() + in.Bytes() + clover.Bytes() / (3 - in.SiteSubset());
      if (twist == QUDA_TWIST_GAMMA5_INVERSE && !clover::dynamic_inverse())
	rtn += clover.Bytes() / (3 - in.SiteSubset());
      return rtn;
    }
  };

#ifdef GPU_TWISTED_CLOVER_DIRAC
  //Apply the twisted-clover matrix field to a colorspinor field
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist)
  {
    instantiate<TwistClover>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
  }
#else
  void ApplyTwistClover(ColorSpinorField &, const ColorSpinorField &, const CloverField &,
			double, double, double, int, int, QudaTwistGamma5Type)
  {
    errorQuda("Twisted-clover dslash has not been built");
  }
#endif // GPU_TWISTED_MASS_DIRAC

}
