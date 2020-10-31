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
#include <kernels/dslash_quda.cuh>
#include <instantiate.h>

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
  
  // CPU kernel for applying the gamma matrix to a colorspinor
  template <typename Float, int nColor, int d, typename Arg>
  void gammaCPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    
    for (int parity= 0; parity < arg.nParity; parity++) {      
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
	arg.out(x_cb, parity) = in.gamma(d);
      } // 4-d volumeCB
    } // parity
  }

  // CPU kernel for applying the chiral projection matrix to a colorspinor
  template <typename Float, int nColor, typename Arg>
  void chiralProjCPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType, nColor, 4> Spinor;
    typedef ColorSpinor<RegType, nColor, 2> HalfSpinor;
    
    // arg.proj is either +1 or -1
    // chiral_project() expects +1 or 0
    int proj = arg.proj == 1 ? 1 : 0;
    
    for (int parity = 0; parity < arg.nParity; parity++) {      
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	
	Spinor in = arg.in(x_cb, parity);
	HalfSpinor chi = in.chiral_project(proj);
	
	// out += P_{L/R} * in
	arg.out(x_cb, parity) = 0.5*chi.chiral_reconstruct(proj);
      }
    }
  }

  template <typename Float, int nColor>
  class Gamma : public TunableVectorY {

    GammaArg<Float, nColor> arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Gamma(ColorSpinorField &out, const ColorSpinorField &in, int d, int proj) :
      TunableVectorY(in.SiteSubset()),
      arg(out, in, d, proj),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      
      apply(streams[Nstream-1]);
    }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if(arg.proj == 0) gammaCPU<Float,nColor,4,decltype(arg)>(arg);
	else chiralProjCPU<Float,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	switch (arg.d) {
	case 4:
	  if(arg.proj == 0) qudaLaunchKernel(gammaGPU<Float,nColor,4,decltype(arg)>, tp, stream, arg);
	  else if (arg.proj == -1 || arg.proj == 1) qudaLaunchKernel(chiralProjGPU<Float,nColor,decltype(arg)>, tp, stream, arg);
	  else errorQuda("Unexpected chrial projection %d", arg.proj);
	  break;
	default: errorQuda("%d not instantiated", arg.d);
	}
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); }
    void postTune() { arg.out.load(); }
  };

  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyGamma(ColorSpinorField &out, const ColorSpinorField &in, int d)
  {
    instantiate<Gamma>(out, in, d, 0);
  }
  
  // Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in) {
    ApplyGamma(out,in,4);
  }
  
  void ApplyChiralProj(ColorSpinorField &out, const ColorSpinorField &in, const int proj)
  {
    instantiate<Gamma>(out, in, 4, proj);
  }
  
  // Applies out(x) = 1/2 * [(1 + gamma5) * in_right(x) + (1 - gamma5) * in_left(x)
  void chiralProject(ColorSpinorField &out, const ColorSpinorField &in, const int proj) {
    ApplyChiralProj(out, in, proj);
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
  
  template <typename Float, int nColor>
  class TwistGamma : public TunableVectorY {

    GammaArg<Float, nColor> arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    TwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type) :
      TunableVectorY(in.SiteSubset()),
      arg(out, in, d, kappa, mu, epsilon, dagger, type),
      meta(in)
    {
      strcpy(aux, meta.AuxString());

      apply(streams[Nstream-1]);
    }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.doublet) twistGammaCPU<true,Float,nColor>(arg);
	twistGammaCPU<false,Float,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.doublet)
	  switch (arg.d) {
	  case 4: qudaLaunchKernel(twistGammaGPU<true,Float,nColor,4,decltype(arg)>, tp, stream, arg); break;
	  default: errorQuda("%d not instantiated", arg.d);
	  }
	else
	  switch (arg.d) {
	  case 4: qudaLaunchKernel(twistGammaGPU<false,Float,nColor,4,decltype(arg)>, tp, stream, arg); break;
	  default: errorQuda("%d not instantiated", arg.d);
	  }
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); }
  };
  
  //Apply the Gamma matrix to a colorspinor field
  //out(x) = gamma_d*in
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu, double epsilon, int dagger, QudaTwistGamma5Type type)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    instantiate<TwistGamma>(out, in, d, kappa, mu, epsilon, dagger, type);
#else
    errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  void cloverCPU(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) cloverApply<Float,nSpin,nColor>(arg, x_cb, parity);
    }
  }
  
  template <typename Float, int nColor>
  class Clover : public TunableVectorY {

    static constexpr int nSpin = 4;
    CloverArg<Float, nSpin, nColor> arg;
    const ColorSpinorField &meta;

    long long flops() const { return arg.nParity*arg.volumeCB*504ll; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes() + arg.nParity*arg.volumeCB*arg.clover.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Clover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity) :
      TunableVectorY(in.SiteSubset()),
      arg(out, in, clover, inverse, parity),
      meta(in)
    {
      if (in.Nspin() != 4 || out.Nspin() != 4) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      if (!inverse) errorQuda("Unsupported direct application");
      strcpy(aux, meta.AuxString());

      apply(streams[Nstream-1]);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	cloverCPU<Float,nSpin,nColor>(arg);
      } else {
	qudaLaunchKernel(cloverGPU<Float,nSpin,nColor,decltype(arg)>, tp, stream, arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }  // Need to save the out field if it aliases the in field
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); } // Restore if the in and out fields alias
  };

  //Apply the clover matrix field to a colorspinor field
  //out(x) = clover*in
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity)
  {
#ifdef GPU_CLOVER_DIRAC
    instantiate<Clover>(out, in, clover, inverse, parity);
#else
    errorQuda("Clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
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

  template <typename Float, int nColor>
  class TwistClover : public TunableVectorY {

    static constexpr int nSpin = 4;
    CloverArg<Float,nSpin,nColor> arg;
    const ColorSpinorField &meta;

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
    TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
                double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist) :
      TunableVectorY(in.SiteSubset()),
      arg(out, in, clover, twist != QUDA_TWIST_GAMMA5_DIRECT, parity, kappa, mu, epsilon, dagger, twist),
      meta(in)
    {
      if (in.Nspin() != 4 || out.Nspin() != 4) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      strcpy(aux, meta.AuxString());
      strcat(aux, arg.inverse ? ",inverse" : ",direct");

      apply(streams[Nstream-1]);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.inverse) twistCloverCPU<true,Float,nSpin,nColor>(arg);
	else twistCloverCPU<false,Float,nSpin,nColor>(arg);
      } else {
	if (arg.inverse) qudaLaunchKernel(twistCloverGPU<true,Float,nSpin,nColor,decltype(arg)>, tp, stream, arg);
	else qudaLaunchKernel(twistCloverGPU<false,Float,nSpin,nColor,decltype(arg)>, tp, stream, arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    void preTune() { if (arg.out.field == arg.in.field) arg.out.save(); }  // Need to save the out field if it aliases the in field
    void postTune() { if (arg.out.field == arg.in.field) arg.out.load(); } // Restore if the in and out fields alias
  };

  //Apply the twisted-clover matrix field to a colorspinor field
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist)
  {
#ifdef GPU_CLOVER_DIRAC
    instantiate<TwistClover>(out, in, clover, kappa, mu, epsilon, parity, dagger, twist);
#else
    errorQuda("Clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

} // namespace quda
