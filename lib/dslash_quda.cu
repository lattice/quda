#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <stack>

#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <clover_field_order.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <linalg.cuh>
#include <dslash_policy.cuh>

namespace quda {

  // these should not be namespaced!!
  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple calls to cudaMemcpy()
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

    cudaEvent_t packEnd[2];
    cudaEvent_t gatherStart[Nstream];
    cudaEvent_t gatherEnd[Nstream];
    cudaEvent_t scatterStart[Nstream];
    cudaEvent_t scatterEnd[Nstream];
    cudaEvent_t dslashStart[2];

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

#if CUDA_VERSION >= 8000
    cuuint32_t *commsEnd_h;
    CUdeviceptr commsEnd_d[Nstream];
#endif
  }

  void createDslashEvents()
  {
    using namespace dslash;
    // add cudaEventDisableTiming for lower sync overhead
    for (int i=0; i<Nstream; i++) {
      cudaEventCreateWithFlags(&gatherStart[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&gatherEnd[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
    }
    for (int i=0; i<2; i++) {
      cudaEventCreateWithFlags(&packEnd[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&dslashStart[i], cudaEventDisableTiming);
    }

    aux_worker = NULL;

#if CUDA_VERSION >= 8000
    commsEnd_h = static_cast<cuuint32_t*>(mapped_malloc(Nstream*sizeof(int)));
    for (int i=0; i<Nstream; i++) {
      cudaHostGetDevicePointer((void**)&commsEnd_d[i], commsEnd_h+i, 0);
      commsEnd_h[i] = 0;
    }
#endif

    checkCudaError();

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

#if CUDA_VERSION >= 8000
    host_free(commsEnd_h);
    commsEnd_h = 0;
#endif

    for (int i=0; i<Nstream; i++) {
      cudaEventDestroy(gatherStart[i]);
      cudaEventDestroy(gatherEnd[i]);
      cudaEventDestroy(scatterStart[i]);
      cudaEventDestroy(scatterEnd[i]);
    }

    for (int i=0; i<2; i++) {
      cudaEventDestroy(packEnd[i]);
      cudaEventDestroy(dslashStart[i]);
    }

    checkCudaError();
  }

  /**
     @brief Parameter structure for driving the Gamma operator
   */
  template <typename Float, int nColor>
  struct GammaArg {
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;
    typedef typename mapper<Float>::type RegType;

    F out;                // output vector field
    const F in;           // input vector field
    const int d;          // which gamma matrix are we applying
    const int nParity;    // number of parities we're working on
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    RegType a;            // scale factor
    RegType b;            // chiral twist
    RegType c;            // flavor twist

    GammaArg(ColorSpinorField &out, const ColorSpinorField &in, int d,
	     RegType kappa=0.0, RegType mu=0.0, RegType epsilon=0.0,
	     bool dagger=false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID)
      : out(out), in(in), d(d), nParity(in.SiteSubset()),
	doublet(in.TwistFlavor() == QUDA_TWIST_DEG_DOUBLET || in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
	volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0)
    {
      if (d < 0 || d > 4) errorQuda("Undefined gamma matrix %d", d);
      if (in.Nspin() != 4) errorQuda("Cannot apply gamma5 to nSpin=%d field", in.Nspin());
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
	if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
          b = 2.0 * kappa * mu;
          a = 1.0;
        } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
          b = -2.0 * kappa * mu;
          a = 1.0 / (1.0 + b * b);
        }
	c = 0.0;
        if (dagger) b *= -1.0;
      } else if (doublet) {
        if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
          b = 2.0 * kappa * mu;
          c = -2.0 * kappa * epsilon;
          a = 1.0;
        } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
          b = -2.0 * kappa * mu;
          c = 2.0 * kappa * epsilon;
          a = 1.0 / (1.0 + b * b - c * c);
          if (a <= 0) errorQuda("Invalid twisted mass parameters (kappa=%e, mu=%e, epsilon=%e)\n", kappa, mu, epsilon);
        }
        if (dagger) b *= -1.0;
      }
    }
  };

  // CPU kernel for applying the gamma matrix to a colorspinor
  template <typename Float, int nColor, typename Arg>
  void gammaCPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    for (int parity= 0; parity < arg.nParity; parity++) {

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
	arg.out(x_cb, parity) = in.gamma(arg.d);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the gamma matrix to a colorspinor
  template <typename Float, int nColor, int d, typename Arg>
  __global__ void gammaGPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
    arg.out(x_cb, parity) = in.gamma(d);
  }

  template <typename Float, int nColor, typename Arg>
  class Gamma : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Gamma(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
    }
    virtual ~Gamma() { }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	gammaCPU<Float,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	switch (arg.d) {
	case 4: gammaGPU<Float,nColor,4> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	default: errorQuda("%d not instantiated", arg.d);
	}
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); }
    void postTune() { arg.out.load(); }
  };


  template <typename Float, int nColor>
  void ApplyGamma(ColorSpinorField &out, const ColorSpinorField &in, int d)
  {
    GammaArg<Float,nColor> arg(out, in, d);
    Gamma<Float,nColor,GammaArg<Float,nColor> > gamma(arg, in);
    gamma.apply(streams[Nstream-1]);
  }

  // template on the number of colors
  template <typename Float>
  void ApplyGamma(ColorSpinorField &out, const ColorSpinorField &in, int d)
  {
    if (in.Ncolor() == 3) {
      ApplyGamma<Float,3>(out, in, d);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyGamma(ColorSpinorField &out, const ColorSpinorField &in, int d)
  {
    checkPrecision(out, in);    // check all precisions match
    checkLocation(out, in);     // check all locations match

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyGamma<double>(out, in, d);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyGamma<float>(out, in, d);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyGamma<short>(out, in, d);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyGamma<char>(out, in, d);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
  }

  // CPU kernel for applying the gamma matrix to a colorspinor
  template <bool doublet, typename Float, int nColor, typename Arg>
  void twistGammaCPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    for (int parity= 0; parity < arg.nParity; parity++) {
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	if (!doublet) {
	  ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
          arg.out(x_cb, parity) = arg.a * (in + arg.b * in.igamma(arg.d));
        } else {
	  ColorSpinor<RegType,nColor,4> in_1 = arg.in(x_cb+0*arg.volumeCB, parity);
	  ColorSpinor<RegType,nColor,4> in_2 = arg.in(x_cb+1*arg.volumeCB, parity);
          arg.out(x_cb + 0 * arg.volumeCB, parity) = arg.a * (in_1 + arg.b * in_1.igamma(arg.d) + arg.c * in_2);
          arg.out(x_cb + 1 * arg.volumeCB, parity) = arg.a * (in_2 - arg.b * in_2.igamma(arg.d) + arg.c * in_1);
        }
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the gamma matrix to a colorspinor
  template <bool doublet, typename Float, int nColor, int d, typename Arg>
  __global__ void twistGammaGPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (x_cb >= arg.volumeCB) return;

    if (!doublet) {
      ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
      arg.out(x_cb, parity) = arg.a * (in + arg.b * in.igamma(d));
    } else {
      ColorSpinor<RegType,nColor,4> in_1 = arg.in(x_cb+0*arg.volumeCB, parity);
      ColorSpinor<RegType,nColor,4> in_2 = arg.in(x_cb+1*arg.volumeCB, parity);
      arg.out(x_cb + 0 * arg.volumeCB, parity) = arg.a * (in_1 + arg.b * in_1.igamma(d) + arg.c * in_2);
      arg.out(x_cb + 1 * arg.volumeCB, parity) = arg.a * (in_2 - arg.b * in_2.igamma(d) + arg.c * in_1);
    }
  }

  template <typename Float, int nColor, typename Arg>
  class TwistGamma : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    TwistGamma(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
    }
    virtual ~TwistGamma() { }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.doublet) twistGammaCPU<true,Float,nColor>(arg);
	twistGammaCPU<false,Float,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.doublet)
	  switch (arg.d) {
	  case 4: twistGammaGPU<true,Float,nColor,4> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	  default: errorQuda("%d not instantiated", arg.d);
	  }
	else
	  switch (arg.d) {
	  case 4: twistGammaGPU<false,Float,nColor,4> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	  default: errorQuda("%d not instantiated", arg.d);
	  }
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); }
  };


  template <typename Float, int nColor>
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
    GammaArg<Float,nColor> arg(out, in, d, kappa, mu, epsilon, dagger, type);
    TwistGamma<Float,nColor,GammaArg<Float,nColor> > gamma(arg, in);
    gamma.apply(streams[Nstream-1]);

    checkCudaError();
  }

  // template on the number of colors
  template <typename Float>
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
    if (in.Ncolor() == 3) {
      ApplyTwistGamma<Float,3>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
    checkPrecision(out, in);    // check all precisions match
    checkLocation(out, in);     // check all locations match

#ifdef GPU_TWISTED_MASS_DIRAC
    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyTwistGamma<double>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyTwistGamma<float>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyTwistGamma<short>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyTwistGamma<char>(out, in, d, kappa, mu, epsilon, dagger, type);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

  // Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in) { ApplyGamma(out,in,4); }

  /**
     @brief Parameteter structure for driving the clover and twist-clover application kernels
     @tparam Float Underlying storage precision
     @tparam nSpin Number of spin components
     @tparam nColor Number of colors
     @tparam dynamic_clover Whether we are inverting the clover field on the fly
  */
  template <typename Float, int nSpin, int nColor>
  struct CloverArg {
    static constexpr int length = (nSpin / (nSpin/2)) * 2 * nColor * nColor * (nSpin/2) * (nSpin/2) / 2;
    static constexpr bool dynamic_clover = dynamic_clover_inverse();

    typedef typename colorspinor_mapper<Float,nSpin,nColor>::type F;
    typedef typename clover_mapper<Float,length>::type C;
    typedef typename mapper<Float>::type RegType;

    F out;                // output vector field
    const F in;           // input vector field
    const C clover;       // clover field
    const C cloverInv;    // inverse clover field (only set if not dynamic clover and doing twisted clover)
    const int nParity;    // number of parities we're working on
    const int parity;     // which parity we're acting on (if nParity=1)
    bool inverse;         // whether we are applying the inverse
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    RegType a;
    RegType b;
    RegType c;
    QudaTwistGamma5Type twist;

    CloverArg(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
	      bool inverse, int parity, RegType kappa=0.0, RegType mu=0.0, RegType epsilon=0.0,
	      bool dagger = false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID)
      : out(out), clover(clover, twist == QUDA_TWIST_GAMMA5_INVALID ? inverse : false),
	cloverInv(clover, (twist != QUDA_TWIST_GAMMA5_INVALID && !dynamic_clover) ? true : false),
	in(in), nParity(in.SiteSubset()), parity(parity), inverse(inverse),
	doublet(in.TwistFlavor() == QUDA_TWIST_DEG_DOUBLET || in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
        volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0), twist(twist)
    {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
	if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
	  a = 2.0 * kappa * mu;
	  b = 1.0;
	} else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
	  a = -2.0 * kappa * mu;
	  b = 1.0 / (1.0 + a*a);
	}
	c = 0.0;
	if (dagger) a *= -1.0;
      } else if (doublet) {
	errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      }
    }
  };

  template <typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ inline void cloverApply(Arg &arg, int x_cb, int parity) {
    using namespace linalg; // for Cholesky
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType, nColor, nSpin> Spinor;
    typedef ColorSpinor<RegType, nColor, nSpin / 2> HalfSpinor;
    int spinor_parity = arg.nParity == 2 ? parity : 0;
    Spinor in = arg.in(x_cb, spinor_parity);
    Spinor out;

    in.toRel(); // change to chiral basis here

#pragma unroll
    for (int chirality=0; chirality<2; chirality++) {

      HMatrix<RegType,nColor*nSpin/2> A = arg.clover(x_cb, parity, chirality);
      HalfSpinor chi = in.chiral_project(chirality);

      if (arg.dynamic_clover) {
        Cholesky<HMatrix, RegType, nColor * nSpin / 2> cholesky(A);
        chi = static_cast<RegType>(0.25) * cholesky.backward(cholesky.forward(chi));
      } else {
        chi = A * chi;
      }

      out += chi.chiral_reconstruct(chirality);
    }

    out.toNonRel(); // change basis back

    arg.out(x_cb, spinor_parity) = out;
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  void cloverCPU(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) cloverApply<Float,nSpin,nColor>(arg, x_cb, parity);
    }
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  __global__ void cloverGPU(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.volumeCB) return;
    cloverApply<Float,nSpin,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  class Clover : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

  protected:
    long long flops() const { return arg.nParity*arg.volumeCB*504ll; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes() + arg.nParity*arg.volumeCB*arg.clover.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Clover(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
    }
    virtual ~Clover() { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	cloverCPU<Float,nSpin,nColor>(arg);
      } else {
	cloverGPU<Float,nSpin,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }  // Need to save the out field if it aliases the in field
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); } // Restore if the in and out fields alias
  };


  template <typename Float, int nColor>
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity)
  {
    if (in.Nspin() != 4) errorQuda("Unsupported nSpin=%d", in.Nspin());
    constexpr int nSpin = 4;

    if (inverse) {
      CloverArg<Float, nSpin, nColor> arg(out, in, clover, inverse, parity);
      Clover<Float, nSpin, nColor, decltype(arg)> worker(arg, in);
      worker.apply(streams[Nstream - 1]);
    } else {
      CloverArg<Float, nSpin, nColor> arg(out, in, clover, inverse, parity);
      Clover<Float, nSpin, nColor, decltype(arg)> worker(arg, in);
      worker.apply(streams[Nstream - 1]);
    }

    checkCudaError();
  }

  // template on the number of colors
  template <typename Float>
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity)
  {
    if (in.Ncolor() == 3) {
      ApplyClover<Float,3>(out, in, clover, inverse, parity);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  //Apply the clvoer matrix field to a colorspinor field
  //out(x) = clover*in
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity)
  {
    checkPrecision(out, clover, in);    // check all precisions match
    checkLocation(out, clover, in);     // check all locations match

#ifdef GPU_CLOVER_DIRAC
    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyClover<double>(out, in, clover, inverse, parity);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyClover<float>(out, in, clover, inverse, parity);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyClover<short>(out, in, clover, inverse, parity);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyClover<char>(out, in, clover, inverse, parity);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

  // if (!inverse) apply (Clover + i*a*gamma_5) to the input spinor
  // else apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  template <bool inverse, typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ inline void twistCloverApply(Arg &arg, int x_cb, int parity) {
    using namespace linalg; // for Cholesky
    constexpr int N = nColor*nSpin/2;
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType,nColor,nSpin> Spinor;
    typedef ColorSpinor<RegType,nColor,nSpin/2> HalfSpinor;
    typedef HMatrix<RegType,N> Mat;
    int spinor_parity = arg.nParity == 2 ? parity : 0;
    Spinor in = arg.in(x_cb, spinor_parity);
    Spinor out;

    in.toRel(); // change to chiral basis here

#pragma unroll
    for (int chirality=0; chirality<2; chirality++) {
      // factor of 2 comes from clover normalization we need to correct for
      const complex<RegType> j(0.0, chirality == 0 ? static_cast<RegType>(0.5) : -static_cast<RegType>(0.5));

      Mat A = arg.clover(x_cb, parity, chirality);

      HalfSpinor in_chi = in.chiral_project(chirality);
      HalfSpinor out_chi = A*in_chi + j*arg.a*in_chi;

      if (inverse) {
	if (arg.dynamic_clover) {
	  Mat A2 = A.square();
	  A2 += arg.a*arg.a*static_cast<RegType>(0.25);
	  Cholesky<HMatrix,RegType,N> cholesky(A2);
	  out_chi = static_cast<RegType>(0.25)*cholesky.backward(cholesky.forward(out_chi));
	} else {
	  Mat Ainv = arg.cloverInv(x_cb, parity, chirality);
	  out_chi = static_cast<RegType>(2.0)*(Ainv*out_chi);
	}
      }

      out += (out_chi).chiral_reconstruct(chirality);
    }

    out.toNonRel(); // change basis back

    arg.out(x_cb, spinor_parity) = out;
  }

  template <bool inverse, typename Float, int nSpin, int nColor, typename Arg>
  void twistCloverCPU(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) twistCloverApply<inverse,Float,nSpin,nColor>(arg, x_cb, parity);
    }
  }

  template <bool inverse, typename Float, int nSpin, int nColor, typename Arg>
  __global__ void twistCloverGPU(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.volumeCB) return;
    twistCloverApply<inverse,Float,nSpin,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  class TwistClover : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

  protected:
    long long flops() const { return (arg.inverse ? 1056ll : 552ll) * arg.nParity*arg.volumeCB; }
    long long bytes() const {
      long long rtn = arg.out.Bytes() + arg.in.Bytes() + arg.nParity*arg.volumeCB*arg.clover.Bytes();
      if (arg.twist == QUDA_TWIST_GAMMA5_INVERSE && !arg.dynamic_clover)
	rtn += arg.nParity*arg.volumeCB*arg.cloverInv.Bytes();
      return rtn;
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    TwistClover(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, arg.inverse ? ",inverse" : ",direct");
    }
    virtual ~TwistClover() { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.inverse) twistCloverCPU<true,Float,nSpin,nColor>(arg);
	else twistCloverCPU<false,Float,nSpin,nColor>(arg);
      } else {
	if (arg.inverse) twistCloverGPU<true,Float,nSpin,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	else twistCloverGPU<false,Float,nSpin,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }  // Need to save the out field if it aliases the in field
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); } // Restore if the in and out fields alias
  };


  template <typename Float, int nColor>
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist)
  {
    if (in.Nspin() != 4) errorQuda("Unsupported nSpin=%d", in.Nspin());
    constexpr int nSpin = 4;
    bool inverse = twist == QUDA_TWIST_GAMMA5_DIRECT ? false : true;

    CloverArg<Float,nSpin,nColor> arg(out, in, clover, inverse, parity, kappa, mu, epsilon, dagger, twist);
    TwistClover<Float,nSpin,nColor,decltype(arg)> worker(arg, in);
    worker.apply(streams[Nstream-1]);

    checkCudaError();
  }

  // template on the number of colors
  template <typename Float>
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist)
  {
    if (in.Ncolor() == 3) {
      ApplyTwistClover<Float,3>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  //Apply the twisted-clover matrix field to a colorspinor field
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist)
  {
    checkPrecision(out, clover, in);    // check all precisions match
    checkLocation(out, clover, in);     // check all locations match

#ifdef GPU_CLOVER_DIRAC
    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyTwistClover<double>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyTwistClover<float>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyTwistClover<short>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyTwistClover<char>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

} // namespace quda
