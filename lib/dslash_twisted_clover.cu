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

  namespace twistedclover {

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_TWISTED_CLOVER_DIRAC
#include <tmc_dslash_def.h>       // Twisted Clover kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace twisted_clover

  using namespace twistedclover;

#ifdef GPU_TWISTED_CLOVER_DIRAC
  template <typename sFloat, typename gFloat, typename cFloat>
  class TwistedCloverDslashCuda : public SharedDslashCuda {

  private:
    const QudaTwistCloverDslashType dslashType;
    double a, b, c, d;
    const FullClover &clover;
    const FullClover &cloverInv;

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
    TwistedCloverDslashCuda(cudaColorSpinorField *out, const GaugeField &gauge, const FullClover &clover, const FullClover &cloverInv,
                            //const cFloat *clover, const float *cNorm, const cFloat *cloverInv, const float *cNrm2, int cl_stride,
                            const cudaColorSpinorField *in, const cudaColorSpinorField *x, const QudaTwistCloverDslashType dslashType,
                            const double kappa, const double mu, const double epsilon, const double k,
                            const int parity, const int dagger, const int *commOverride)
      : SharedDslashCuda(out, in, x, gauge, parity, dagger, commOverride), clover(clover), cloverInv(cloverInv), dslashType(dslashType)
    { 
      QudaPrecision clover_prec = bindTwistedCloverTex(clover, cloverInv, parity, dslashParam);
      if (in->Precision() != clover_prec) errorQuda("Mixing clover and spinor precision not supported");

#ifndef DYNAMIC_CLOVER
      if (clover.stride != cloverInv.stride)
        errorQuda("clover and cloverInv must have matching strides (%d != %d)", clover.stride, cloverInv.stride);
#endif

      a = kappa;
      b = mu;
      c = epsilon;
      d = k;

      dslashParam.twist_a = 0.0;
      dslashParam.twist_b = 0.0;
      dslashParam.a = kappa;
      dslashParam.a_f = kappa;
      dslashParam.b = mu;
      dslashParam.b_f = mu;
      dslashParam.cl_stride = clover.stride;
      dslashParam.fl_stride = in->VolumeCB();
    }

    virtual ~TwistedCloverDslashCuda() {
      unbindSpinorTex<sFloat>(in, out, x);
      unbindTwistedCloverTex(clover);
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
      case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
	DSLASH(twistedCloverInvDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
	DSLASH(twistedCloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
	DSLASH(twistedCloverDslashTwist, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
	break;
      default:
	errorQuda("Invalid twisted clover dslash type");
      }
    }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      switch (dslashType) {
      case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
#ifndef DYNAMIC_CLOVER
	strcat(key.aux,",CloverTwistInvDslash");
#else
	strcat(key.aux,",CloverTwistInvDynDslash");
#endif
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
#ifndef DYNAMIC_CLOVER
	strcat(key.aux,",Dslash");
#else
	strcat(key.aux,",DynDslash");
#endif
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
#ifndef DYNAMIC_CLOVER
        strcat(key.aux,",DslashCloverTwist");
#else
        strcat(key.aux,",DynDslashCloverTwist");
#endif
	break;
      default:
	errorQuda("Unsupported twisted-dslash type %d", dslashType);
      }
      return key;
    }

    long long flops() const {
      int clover_flops = 504 + 48;
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
	// clover flops are done in the interior kernel
	flops += clover_flops * in->VolumeCB();	  
	break;
      }
      return flops;
    }

    long long bytes() const {
      bool isFixed = (in->Precision() == sizeof(short) || in->Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in->Precision() + (isFixed ? 2*sizeof(float) : 0);
      long long bytes = DslashCuda::bytes();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	bytes += clover_bytes*in->VolumeCB();
	break;
      }

      return bytes;
    }

  };
#endif // GPU_TWISTED_CLOVER_DIRAC

  void twistedCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover *clover, const FullClover *cloverInv,
			       const cudaColorSpinorField *in, const int parity, const int dagger, 
			       const cudaColorSpinorField *x, const QudaTwistCloverDslashType type, const double &kappa, const double &mu, 
			       const double &epsilon, const double &k,  const int *commOverride, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_CLOVER_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
        dslash = new TwistedCloverDslashCuda<double2,double2,double2>
          (out, gauge, *clover, *cloverInv, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedCloverDslashCuda<float4,float4,float4>
          (out, gauge, *clover, *cloverInv, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedCloverDslashCuda<short4,short4,short4>
          (out, gauge, *clover, *cloverInv, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    }

    int ghost_threads[4] = {0};
    int bulk_threads = (in->TwistFlavor() == QUDA_TWIST_SINGLET) ? in->Volume() : in->Volume() / 2;
    for (int i=0;i<4;i++) ghost_threads[i] = (in->TwistFlavor() == QUDA_TWIST_SINGLET) ? in->GhostFace()[i] : in->GhostFace()[i] / 2;

    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), bulk_threads, ghost_threads, profile);
    dslash_policy.apply(0);

    delete dslash;
#else
    errorQuda("Twisted clover dslash has not been built");
#endif
  }

}

#endif
