#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

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
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>

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
  
  // declare the dslash events
#include <dslash_events.cuh>

  using namespace twisted;

#ifdef GPU_TWISTED_MASS_DIRAC
  template <typename sFloat, typename gFloat>
  class TwistedDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
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
    TwistedDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		      const QudaReconstructType reconstruct, const cudaColorSpinorField *in,  const cudaColorSpinorField *x, 
		      const QudaTwistDslashType dslashType, const double kappa, const double mu, 
		      const double epsilon, const double k, const int dagger)
      : SharedDslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), dslashType(dslashType)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;
      if (dslashType == QUDA_NONDEG_DSLASH) errorQuda("Invalid dslashType for twisted-mass Dslash");

      dslashParam.gauge0 = (void*)gauge0;
      dslashParam.gauge1 = (void*)gauge1;
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
      // factor of 2 (or 1) for T-dimensional spin projection (FIXME - unnecessary)
      dslashParam.tProjScale = getKernelPackT() ? 1.0 : 2.0;
      dslashParam.tProjScale_f = (float)(dslashParam.tProjScale);

#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
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

#include <dslash_policy.cuh> 

  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			     const cudaColorSpinorField *in, const int parity, const int dagger, 
			     const cudaColorSpinorField *x, const QudaTwistDslashType type, 
			     const double &kappa, const double &mu, const double &epsilon, 
			     const double &k,  const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inSpinor->createComms(1);

#ifdef GPU_TWISTED_MASS_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

    int ghost_threads[4] = {0};
    int bulk_threads = in->Volume();

    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = comm_dim_partitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i][0] = in->GhostOffset(i,0)/in->FieldOrder();
      dslashParam.ghostOffset[i][1] = in->GhostOffset(i,1)/in->FieldOrder();
      dslashParam.ghostNormOffset[i][0] = in->GhostNormOffset(i,0);
      dslashParam.ghostNormOffset[i][1] = in->GhostNormOffset(i,1);
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : comm_dim_partitioned(i); // switch off comms if override = 0
      ghost_threads[i] = in->GhostFace()[i];
    }

#ifdef MULTI_GPU
    if(type == QUDA_DEG_TWIST_INV_DSLASH){
      setKernelPackT(true);
      twist_a = kappa; 
      twist_b = mu;
    }
#endif

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new TwistedDslashCuda<double2,double2>(out, (double2*)gauge0,(double2*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);
      regSize = sizeof(double);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedDslashCuda<float4,float4>(out, (float4*)gauge0,(float4*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedDslashCuda<short4,short4>(out, (short4*)gauge0,(short4*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);
    }

    DslashPolicyTune dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, bulk_threads, ghost_threads, profile);
    dslash_policy.apply(0);

    delete dslash;
#ifdef MULTI_GPU
    if(type == QUDA_DEG_TWIST_INV_DSLASH){
      setKernelPackT(false);
      twist_a = 0.0; 
      twist_b = 0.0;
    }
#endif

    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Twisted mass dslash has not been built");
#endif
  }

}
