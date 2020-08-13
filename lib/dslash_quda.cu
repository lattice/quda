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
#include <kernels/dslash_quda.cuh>

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

  // CPU kernel for applying the chiral projection matrix to a colorspinor
  template <typename Float, int nColor, typename Arg>
  void chiralProjCPU(Arg arg)
  {
  /*
    typedef typename mapper<Float>::type RegType;
    
    for (int parity= 0; parity < arg.nParity; parity++) {
      
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
	arg.out(x_cb, parity) = in.chiralProj(arg.proj);
      } // 4-d volumeCB
    } // parity
  */
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
        if(arg.proj == 0) gammaCPU<Float, nColor, Arg>(arg);
        else chiralProjCPU<Float, nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());	
        if(arg.proj == 0) {
          qudaLaunchKernel(gammaGPU<Float, nColor, Arg>, tp, stream, arg);
        } else {
          qudaLaunchKernel(chiralProjGPU<Float, nColor, Arg>, tp, stream, arg);
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

  // Applies a gamma matrix to a spinor (wrapper to ApplyGamma)
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in) {
    ApplyGamma(out, in, 4);
  }
  
  // Applies a gamma matrix to a spinor (wrapper to ApplyGamma)
  void gamma(ColorSpinorField &out, const ColorSpinorField &in, int gamma_mat) {
    ApplyGamma(out, in, gamma_mat);
  }

  template <typename Float, int nColor>
  void ApplyChiralProj(ColorSpinorField &out, const ColorSpinorField &in, const int proj)
  {
    // Hardcode 4 to specify gamma5 in 3rd GammaArg.
    //printfQuda("Proj = %d\n", proj);
    GammaArg<Float,nColor> arg(out, in, 4, proj);
    //printfQuda("Flag 1\n");
    Gamma<Float,nColor,GammaArg<Float,nColor> > gamma(arg, in);
    //printfQuda("Flag 2\n");
    gamma.apply(streams[Nstream-1]);
    //printfQuda("Flag 3\n");
  }
  
  // template on the number of colors
  template <typename Float>
  void ApplyChiralProj(ColorSpinorField &out, const ColorSpinorField &in, const int proj)
  {
    if (in.Ncolor() == 3) {
      //printfQuda("Proj = %d\n", proj);
      ApplyChiralProj<Float,3>(out, in, proj);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }
  
  void ApplyChiralProj(ColorSpinorField &out, const ColorSpinorField &in, const int proj)
  {
    checkPrecision(out, in);    // check all precisions match
    checkLocation(out, in);     // check all locations match

    //printfQuda("Proj = %d\n", proj);
    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyChiralProj<double>(out, in, proj);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyChiralProj<float>(out, in, proj);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyChiralProj<short>(out, in, proj);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyChiralProj<char>(out, in, proj);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
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
	  case 4: qudaLaunchKernel(twistGammaGPU<true,Float,nColor,4,Arg>, tp, stream, arg); break;
	  default: errorQuda("%d not instantiated", arg.d);
	  }
	else
	  switch (arg.d) {
	  case 4: qudaLaunchKernel(twistGammaGPU<false,Float,nColor,4,Arg>, tp, stream, arg); break;
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
    GammaArg<Float,nColor> arg(out, in, d, 0, kappa, mu, epsilon, dagger, type);
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


  template <typename Float, int nSpin, int nColor, typename Arg>
  void cloverCPU(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) cloverApply<Float,nSpin,nColor>(arg, x_cb, parity);
    }
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
	qudaLaunchKernel(cloverGPU<Float,nSpin,nColor,Arg>, tp, stream, arg);
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
	if (arg.inverse) qudaLaunchKernel(twistCloverGPU<true,Float,nSpin,nColor,Arg>, tp, stream, arg);
	else qudaLaunchKernel(twistCloverGPU<false,Float,nSpin,nColor,Arg>, tp, stream, arg);
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
