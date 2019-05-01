#ifdef USE_LEGACY_DSLASH

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <typeinfo>

#include <color_spinor_field.h>
#include <clover_field.h>

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_INTER
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
//#define DIRECT_ACCESS_CLOVER
#endif // GPU_WILSON_DIRAC


#include <quda_internal.h>
#include <dslash_quda.h>
#include <dslash.h>
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>
#include <dslash_policy.cuh>

namespace quda {

  namespace twisted {

#undef GPU_STAGGERED_DIRAC
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_TWISTED_MASS_DIRAC
#include <tm_dslash_def.h>        // Twisted Mass kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace twisted
  
  using namespace twisted;

#ifdef GPU_TWISTED_MASS_DIRAC
  template <typename sFloat, typename gFloat>
  class TwistedDslashCuda : public SharedDslashCuda {

  private:
    const QudaTwistDslashType dslashType;
    double a, b, c, d;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
	return 0;
      }
    }

  public:
    TwistedDslashCuda(cudaColorSpinorField *out, const GaugeField &gauge,
                      const cudaColorSpinorField *in, const cudaColorSpinorField *x,
		      const QudaTwistDslashType dslashType, const double kappa, const double mu,
		      const double epsilon, const double k, const int parity, const int dagger,
                      const int *commOverride)
      : SharedDslashCuda(out, in, x, gauge, parity, dagger, commOverride), dslashType(dslashType)
    { 
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;
      if (dslashType == QUDA_NONDEG_DSLASH) errorQuda("Invalid dslashType for twisted-mass Dslash");

      dslashParam.twist_a = (dslashType == QUDA_DEG_TWIST_INV_DSLASH) ? kappa : 0.0;
      dslashParam.twist_b = (dslashType == QUDA_DEG_TWIST_INV_DSLASH) ? mu : 0.0;
      dslashParam.a = kappa;
      dslashParam.a_f = kappa;
      dslashParam.b = mu;
      dslashParam.b_f = mu;
      dslashParam.fl_stride = in->VolumeCB();
    }
    virtual ~TwistedDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      switch(dslashType){
      case QUDA_DEG_TWIST_INV_DSLASH:
	strcat(key.aux,",TwistInvDslash");
	break;
      case QUDA_DEG_DSLASH_TWIST_INV:
	strcat(key.aux,",");
	break;
      case QUDA_DEG_DSLASH_TWIST_XPAY:
	strcat(key.aux,",DslashTwist");
	break;
      default:
	errorQuda("Unsupported twisted-dslash type %d", dslashType);
      }
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
#ifndef USE_TEXTURE_OBJECTS
      if (dslashParam.kernel_type == INTERIOR_KERNEL) bindSpinorTex<sFloat>(in, out, x);
#endif // USE_TEXTURE_OBJECTS
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      setParam();
      dslashParam.block[0] = tp.aux.x; dslashParam.block[1] = tp.aux.y; dslashParam.block[2] = tp.aux.z; dslashParam.block[3] = tp.aux.w;
      for (int i=0; i<4; i++) dslashParam.grid[i] = ( (i==0 ? 2 : 1) * in->X(i)) / dslashParam.block[i];

      switch(dslashType){
      case QUDA_DEG_TWIST_INV_DSLASH:
	DSLASH(twistedMassTwistInvDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      case QUDA_DEG_DSLASH_TWIST_INV:
	DSLASH(twistedMassDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      case QUDA_DEG_DSLASH_TWIST_XPAY:
	DSLASH(twistedMassDslashTwist, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      default: errorQuda("Invalid twisted mass dslash type");
      }
    }

    long long flops() const {
      int twisted_flops = 48;
      long long flops = DslashCuda::flops();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	// twisted mass flops are done in the interior kernel
	flops += twisted_flops * in->VolumeCB();	  
	break;
      }
      return flops;
    }
  };
#endif // GPU_TWISTED_MASS_DIRAC

  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge,
			     const cudaColorSpinorField *in, const int parity, const int dagger,
			     const cudaColorSpinorField *x, const QudaTwistDslashType type,
			     const double &kappa, const double &mu, const double &epsilon,
			     const double &k,  const int *commOverride, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    if (type == QUDA_DEG_TWIST_INV_DSLASH) setKernelPackT(true);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new TwistedDslashCuda<double2,double2>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedDslashCuda<float4,float4>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedDslashCuda<short4,short4>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    }

    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), in->Volume(), in->GhostFace(), profile);
    dslash_policy.apply(0);

    delete dslash;

    if (type == QUDA_DEG_TWIST_INV_DSLASH) setKernelPackT(false);
#else
    errorQuda("Twisted mass dslash has not been built");
#endif
  }

}

#endif
