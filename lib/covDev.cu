#ifndef USE_LEGACY_DSLASH


#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.cuh>
#include <kernels/covDev.cuh>

/**
   This is the covariant derivative based on the basic gauged Laplace operator
*/

namespace quda {

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct CovDevLaunch {
    
    // kernel name for jit compilation
    static constexpr const char *kernel = "quda::covDevGPU";
    
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      dslash.launch(covDevGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };
  
  template <typename Float, int nDim, int nColor, typename Arg>
  class CovDev : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:
    
    CovDev(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in, "kernels/covDev.cuh"), arg(arg), in(in) { }    
    
    virtual ~CovDev() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      Dslash<Float>::template instantiate<CovDevLaunch, nDim, nColor>(tp, arg, stream);
    }
    
    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };
  
  template <typename Float, int nColor, QudaReconstructType recon>
  struct CovDevApply {

    inline CovDevApply(ColorSpinorField &out, const ColorSpinorField &in,
		       const GaugeField &U, int mu, double a, const ColorSpinorField &x,
		       int parity, bool dagger, const int *comm_override,
		       TimeProfile &profile)
      
    {
      constexpr int nDim = 4; 
      CovDevArg<Float,nColor,recon> arg(out,in,U,mu,a,x,parity,dagger,comm_override);
      CovDev<Float,nDim,nColor,CovDevArg<Float,nColor,recon>>covDev(arg, out, in);

      dslash::DslashPolicyTune<decltype(covDev)> policy(covDev, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(), in.GhostFaceCB(), profile);
      policy.apply(0);
      
      checkCudaError();
    }
  };

  //Apply the covariant derivative operator
  //out(x) = U_{\mu}(x)in(x+mu) for mu = 0...3
  //out(x) = U^\dagger_mu'(x-mu')in(x-mu') for mu = 4...7 and we set mu' = mu-4
  void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in,
		   const GaugeField &U, int mu, double a, const ColorSpinorField &x,
		   int parity, bool dagger, const int *comm_override,
		   TimeProfile &profile)
  {
    
    //DMH: Put in own build preprocessor guards?
    //#ifdef GPU_COVDEV
#if 1
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder()) errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);
    
    instantiate<CovDevApply,WilsonReconstruct>(out, in, U, mu, a, x, parity, dagger, comm_override, profile);
    //instantiate<CovDevApply>(out, in, U, a, x, parity, dagger, comm_override, profile, mu);
#else
    errorQuda("GPU CovDev has not been built");
#endif //GPU_COVDEV
  }
  
} // namespace quda

#endif
