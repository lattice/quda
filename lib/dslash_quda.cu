#include <stack>

#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <dslash_policy.cuh>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/dslash_misc.cuh>

namespace quda {

  // these should not be namespaced!!
  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple memcpys
  static bool kernelPackT = false;

  void setKernelPackT(bool packT) { kernelPackT = packT; }

  bool getKernelPackT() { return kernelPackT; }

  static std::stack<bool> kptstack;

  void pushKernelPackT(bool packT)
  {
    kptstack.push(getKernelPackT());
    setKernelPackT(packT);

    if (kptstack.size() > 10)
    {
      warningQuda("KernelPackT stack contains %u elements.  Is there a missing popKernelPackT() somewhere?",
      static_cast<unsigned int>(kptstack.size()));
    }
  }

  void popKernelPackT()
  {
    if (kptstack.empty())
    {
      errorQuda("popKernelPackT() called with empty stack");
    }
    setKernelPackT(kptstack.top());
    kptstack.pop();
  }

  namespace dslash {
    int it = 0;

    static constexpr int nDim = 4;
    static constexpr int nDir = 2;
    static constexpr int nStream = nDim * nDir + 1;

    qudaEvent_t packEnd[2];
    qudaEvent_t gatherEnd[nStream];
    qudaEvent_t scatterEnd[nStream];
    qudaEvent_t dslashStart[2];

    // these variables are used for benchmarking the dslash components in isolation
    bool dslash_pack_compute;
    bool dslash_interior_compute;
    bool dslash_exterior_compute;
    bool dslash_comms;
    bool dslash_copy;

    // whether the dslash policy tuner has been enabled
    bool dslash_policy_init;

    // used to keep track of which policy to start the autotuning
    int first_active_policy;
    int first_active_p2p_policy;

    // list of dslash policies that are enabled
    std::vector<QudaDslashPolicy> policies;

    // list of p2p policies that are enabled
    std::vector<QudaP2PPolicy> p2p_policies;

    // string used as a tunekey to ensure we retune if the dslash policy env changes
    char policy_string[TuneKey::aux_n];

    // FIX this is a hack from hell
    // Auxiliary work that can be done while waiting on comms to finis
    Worker *aux_worker;
  }

  void createDslashEvents()
  {
    using namespace dslash;
    for (int i=0; i<nStream; i++) {
      gatherEnd[i] = qudaEventCreate();
      scatterEnd[i] = qudaEventCreate();
    }
    for (int i=0; i<2; i++) {
      packEnd[i] = qudaEventCreate();
      dslashStart[i] = qudaEventCreate();
    }

    aux_worker = NULL;

    dslash_pack_compute = true;
    dslash_interior_compute = true;
    dslash_exterior_compute = true;
    dslash_comms = true;
    dslash_copy = true;

    dslash_policy_init = false;
    first_active_policy = 0;
    first_active_p2p_policy = 0;

    // list of dslash policies that are enabled
    policies = std::vector<QudaDslashPolicy>(
        static_cast<int>(QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED), QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED);

    // list of p2p policies that are enabled
    p2p_policies = std::vector<QudaP2PPolicy>(
        static_cast<int>(QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED), QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED);

    strcat(policy_string, ",pol=");
  }

  void destroyDslashEvents()
  {
    using namespace dslash;

    for (int i=0; i<nStream; i++) {
      qudaEventDestroy(gatherEnd[i]);
      qudaEventDestroy(scatterEnd[i]);
    }

    for (int i=0; i<2; i++) {
      qudaEventDestroy(packEnd[i]);
      qudaEventDestroy(dslashStart[i]);
    }
  }

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

  template <typename Float, int nColor> class Clover : public TunableKernel2D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const CloverField &clover;
    bool inverse;
    int parity;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Clover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity) :
      TunableKernel2D(in, in.SiteSubset()),
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
#endif // GPU_TWISTED_MASS_DIRAC

  template <typename Float, int nColor> class TwistClover : public TunableKernel2D {
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

  public:
    TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
                double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist) :
      TunableKernel2D(in, in.SiteSubset()),
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
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (inverse) {
        CloverArg<Float, nColor, true> arg(out, in, clover, parity, kappa, mu, epsilon, dagger, twist);
        launch<TwistCloverApply>(tp, stream, arg);
      } else {
        CloverArg<Float, nColor, false> arg(out, in, clover, parity, kappa, mu, epsilon, dagger, twist);
        launch<TwistCloverApply>(tp, stream, arg);
      }
    }

    void preTune() { if (out.V() == in.V()) out.backup(); } // Restore if the in and out fields alias
    void postTune() { if (out.V() == in.V()) out.restore(); } // Restore if the in and out fields alias
    long long flops() const { return (inverse ? 1056ll : 552ll) * in.Volume(); }
    long long bytes() const {
      long long rtn = out.Bytes() + in.Bytes() + clover.Bytes() / (3 - in.SiteSubset());
      if (twist == QUDA_TWIST_GAMMA5_INVERSE && !dynamic_clover_inverse())
	rtn += clover.Bytes() / (3 - in.SiteSubset());
      return rtn;
    }
  };

#ifdef GPU_CLOVER_DIRAC
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
    errorQuda("Clover dslash has not been built");
  }
#endif // GPU_TWISTED_MASS_DIRAC

} // namespace quda
