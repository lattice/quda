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

  namespace ndegtwisted {

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
#include <tm_ndeg_dslash_def.h>   // Non-degenerate twisted Mass
#endif

#ifndef NDEGTM_SHARED_FLOATS_PER_THREAD
#define NDEGTM_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace twisted
  
  using namespace ndegtwisted;

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  template <typename sFloat, typename gFloat>
  class NdegTwistedDslashCuda : public SharedDslashCuda {

  private:
    const QudaTwistDslashType dslashType;
    double a, b, c, d;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return NDEGTM_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
	return 0;
      }
    }

  public:
    NdegTwistedDslashCuda(cudaColorSpinorField *out, const GaugeField &gauge,
                          const cudaColorSpinorField *in, const cudaColorSpinorField *x,
                          const QudaTwistDslashType dslashType, const double kappa, const double mu,
                          const double epsilon, const double k, const int parity, const int dagger, const int *commOverride)
      : SharedDslashCuda(out, in, x, gauge, parity, dagger, commOverride), dslashType(dslashType)
    { 
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;
      dslashParam.a = kappa;
      dslashParam.a_f = kappa;
      dslashParam.b = mu;
      dslashParam.b_f = mu;
      dslashParam.c = epsilon;
      dslashParam.c_f = epsilon;
      dslashParam.d = k;
      dslashParam.d_f = k;

      if (dslashType != QUDA_NONDEG_DSLASH) errorQuda("Invalid dslashType for non-degenerate twisted-mass Dslash");
      dslashParam.fl_stride = in->VolumeCB()/2;
    }
    virtual ~NdegTwistedDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      strcat(key.aux,",NdegDslash");
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
      NDEG_TM_DSLASH(twistedNdegMassDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
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
	// twisted-mass flops are done in the interior kernel
	flops += twisted_flops * in->VolumeCB();	  
	break;
      }
      return flops;
    }
  };
#endif // GPU_NDEG_TWISTED_MASS_DIRAC

  void ndegTwistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge,
				 const cudaColorSpinorField *in, const int parity, const int dagger,
				 const cudaColorSpinorField *x, const QudaTwistDslashType type,
				 const double &kappa, const double &mu, const double &epsilon,
				 const double &k,  const int *commOverride, TimeProfile &profile)
  {
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new NdegTwistedDslashCuda<double2,double2>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new NdegTwistedDslashCuda<float4,float4>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new NdegTwistedDslashCuda<short4,short4>(out, gauge, in, x, type, kappa, mu, epsilon, k, parity, dagger, commOverride);
    }

    int bulk_threads = in->Volume() / 2;
    int ghost_threads[4] = {0};
    for(int i=0;i<4;i++) ghost_threads[i] = in->GhostFace()[i] / 2;
    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), bulk_threads, ghost_threads, profile);
    dslash_policy.apply(0);

    delete dslash;
#else
    errorQuda("Non-degenerate twisted mass dslash has not been built");
#endif
  }

}

#endif
