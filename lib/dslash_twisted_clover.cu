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
#include <face_quda.h>

#include <inline_ptx.h>

namespace quda {

  namespace twistedclover {

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#if (__COMPUTE_CAPABILITY__ >= 200) && defined(GPU_TWISTED_CLOVER_DIRAC)
#include <tmc_dslash_def.h>       // Twisted Clover kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace twisted_clover

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace twistedclover;

#if (__COMPUTE_CAPABILITY__ >= 200) && defined(GPU_TWISTED_CLOVER_DIRAC)
  template <typename sFloat, typename gFloat, typename cFloat>
  class TwistedCloverDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaTwistCloverDslashType dslashType;
    double a, b, c, d;
    const cFloat *clover;
    const float *cNorm;
    const cFloat *cloverInv;
    const float *cNrm2;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
	return 0;
      }
#else
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
    }

  public:
    TwistedCloverDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
			    const QudaReconstructType reconstruct, const cFloat *clover, const float *cNorm,
			    const cFloat *cloverInv, const float *cNrm2, int cl_stride, const cudaColorSpinorField *in,
			    const cudaColorSpinorField *x, const QudaTwistCloverDslashType dslashType, const double kappa,
			    const double mu, const double epsilon, const double k, const int dagger)
      : SharedDslashCuda(out, in, x, reconstruct,dagger),gauge0(gauge0), gauge1(gauge1), clover(clover),
	cNorm(cNorm), cloverInv(cloverInv), cNrm2(cNrm2), dslashType(dslashType)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
      dslashParam.cl_stride = cl_stride;
      dslashParam.fl_stride = in->VolumeCB();
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;

      switch(dslashType){
      case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
#ifdef MULTI_GPU 
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=interior,CloverTwistInvDslash");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,CloverTwistInvDslash");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,CloverTwistInvDslash");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,CloverTwistInvDslash");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,CloverTwistInvDslash");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,CloverTwistInvDslash");
#else
        fillAux(INTERIOR_KERNEL, "type=interior,CloverTwistInvDynDslash");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,CloverTwistInvDynDslash");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,CloverTwistInvDynDslash");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,CloverTwistInvDynDslash");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,CloverTwistInvDynDslash");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,CloverTwistInvDynDslash");
#endif // DYNAMIC_CLOVER
#else
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=single-GPU,CloverTwistInvDslash");
#else
        fillAux(INTERIOR_KERNEL, "type=single-GPU,CloverTwistInvDynDslash");
#endif // DYNAMIC_CLOVER
#endif // MULTI_GPU
        break;

      case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
#ifdef MULTI_GPU
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=interior,Dslash");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,Dslash");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,Dslash");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,Dslash");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,Dslash");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,Dslash");
#else
        fillAux(INTERIOR_KERNEL, "type=interior,DynDslash");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,DynDslash");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,DynDslash");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,DynDslash");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,DynDslash");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,DynDslash");
#endif // DYNAMIC_CLOVER
#else
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=single-GPU,Dslash");
#else
        fillAux(INTERIOR_KERNEL, "type=single-GPU,DynDslash");
#endif // DYNAMIC_CLOVER
#endif // MULTI_GPU
        break;

      case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
#ifdef MULTI_GPU 
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=interior,DslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,DslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,DslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,DslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,DslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,DslashCloverTwist");
#else
        fillAux(INTERIOR_KERNEL, "type=interior,DynDslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all,DynDslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_X, "type=exterior_x,DynDslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y,DynDslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z,DynDslashCloverTwist");
        fillAux(EXTERIOR_KERNEL_T, "type=exterior_t,DynDslashCloverTwist");
#endif // DYNAMIC_CLOVER
#else
#ifndef DYNAMIC_CLOVER
        fillAux(INTERIOR_KERNEL, "type=single-GPU,DslashCloverTwist");
#else
        fillAux(INTERIOR_KERNEL, "type=single-GPU,DynDslashCloverTwist");
#endif // DYNAMIC_CLOVER
#endif // MULTI_GPU
        break;
      }
    }

    virtual ~TwistedCloverDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch(dslashType){

      case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
	DSLASH(twistedCloverInvDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	       (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
	DSLASH(twistedCloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	       (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
	break;
      case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
	DSLASH(twistedCloverDslashTwist, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	       (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)x->V(), (float*)x->Norm());
	break;
      default: errorQuda("Invalid twisted clover dslash type");
      }
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
	// clover flops are done in the interior kernel
	flops += clover_flops * in->VolumeCB();	  
	break;
      }
      return flops;
    }

    long long bytes() const {
      bool isHalf = in->Precision() == sizeof(short) ? true : false;
      int clover_bytes = 72 * in->Precision() + (isHalf ? 2*sizeof(float) : 0);
      long long bytes = DslashCuda::bytes();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
	bytes += clover_bytes*in->VolumeCB();
	break;
      }

      return bytes;
    }

  };
#endif // GPU_TWISTED_CLOVER_DIRAC

#include <dslash_policy.cuh>

  void twistedCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover *clover, const FullClover *cloverInv,
			       const cudaColorSpinorField *in, const int parity, const int dagger, 
			       const cudaColorSpinorField *x, const QudaTwistCloverDslashType type, const double &kappa, const double &mu, 
			       const double &epsilon, const double &k,  const int *commOverride,
			       TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
#if (__COMPUTE_CAPABILITY__ >= 200) && defined(GPU_TWISTED_CLOVER_DIRAC)
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

    int ghost_threads[4] = {0};
    int bulk_threads = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->Volume() : in->Volume() / 2;

    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
      ghost_threads[i] = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->GhostFace()[i] : in->GhostFace()[i] / 2;
    }

#ifdef MULTI_GPU
    twist_a	= 0.;
    twist_b	= 0.;
#endif

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    void *cloverP, *cloverNormP, *cloverInvP, *cloverInvNormP;
    QudaPrecision clover_prec = bindTwistedCloverTex(*clover, *cloverInv, parity, &cloverP, &cloverNormP, &cloverInvP, &cloverInvNormP);

    if (in->Precision() != clover_prec)
      errorQuda("Mixing clover and spinor precision not supported");
	
    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

#ifndef DYNAMIC_CLOVER
    if (clover->stride != cloverInv->stride) 
      errorQuda("clover and cloverInv must have matching strides (%d != %d)", clover->stride, cloverInv->stride);
#endif

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);
	
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new TwistedCloverDslashCuda<double2,double2,double2>(out, (double2*)gauge0,(double2*)gauge1, gauge.Reconstruct(), (double2*)cloverP, (float*)cloverNormP,
								  (double2*)cloverInvP, (float*)cloverInvNormP, clover->stride, in, x, type, kappa, mu, epsilon, k, dagger);
	  
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedCloverDslashCuda<float4,float4,float4>(out, (float4*)gauge0,(float4*)gauge1, gauge.Reconstruct(), (float4*)cloverP, (float*)cloverNormP,
								 (float4*)cloverInvP, (float*)cloverInvNormP, clover->stride, in, x, type, kappa, mu, epsilon, k, dagger);

    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedCloverDslashCuda<short4,short4,short4>(out, (short4*)gauge0,(short4*)gauge1, gauge.Reconstruct(), (short4*)cloverP, (float*)cloverNormP,
								 (short4*)cloverInvP, (float*)cloverInvNormP, clover->stride, in, x, type, kappa, mu, epsilon, k, dagger);
    }

#ifndef GPU_COMMS
    DslashPolicyImp* dslashImp = DslashFactory::create(dslashPolicy);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
#endif
    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, bulk_threads, ghost_threads, profile);
    delete dslashImp;
	
    delete dslash;

    unbindGaugeTex(gauge);
    unbindTwistedCloverTex(*clover);
	
    checkCudaError();
#else

#if (__COMPUTE_CAPABILITY__ < 200)
  errorQuda("Twisted-clover fermions not supported on pre-Fermi architecture");
#else
  errorQuda("Twisted clover dslash has not been built");
#endif

#endif
  }

}
