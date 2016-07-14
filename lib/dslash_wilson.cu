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

  namespace wilson {

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

  // Enable shared memory dslash for Fermi architecture
  //#define SHARED_WILSON_DSLASH
  //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_WILSON_DIRAC
#define DD_CLOVER 0
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#undef DD_CLOVER
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace wilson

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace wilson;

#ifdef GPU_WILSON_DIRAC
  template <typename sFloat, typename gFloat>
  class WilsonDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      if (dslashParam.kernel_type == INTERIOR_KERNEL) { // Interior kernels use shared memory for common iunput
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else { // Exterior kernels use no shared memory
	return 0;
      }
    }

  public:
    WilsonDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		     const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x, const double a, const int dagger)
      : SharedDslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
    }

    virtual ~WilsonDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

#define dk(BX,BY,BZ,BT) DSLASH(dslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam, (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a)

    //#define dk(BX,BY,BZ,BT) DSLASH(dslash, BX, BY, BZ, BT, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam, (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a)

#define bz(x,y)								\
    if (tp.aux.z==1) {dk(x,y,1,1);}					\
    else if (tp.aux.z==2) {dk(x,y,2,1);}				\
    else if (tp.aux.z==4) {dk(x,y,4,1);}				\
    else if (tp.aux.z==8) {dk(x,y,8,1);}				\
    else if (tp.aux.z==16) {dk(x,y,16,1);}				\
    else if (tp.aux.z==32) {dk(x,y,32,1);}

#define by(x)						\
    {							\
      if (tp.aux.y==1) { bz(x,1) }			\
      else if (tp.aux.y==2) { bz(x,2) }			\
      else if (tp.aux.y==4) { bz(x,4) }			\
      else if (tp.aux.y==8) { bz(x,8) }			\
      else if (tp.aux.y==16) { bz(x,16) }		\
      else if (tp.aux.y==32) { bz(x,32) }		\
    }

#define bxbybz						\
    if (tp.aux.x==1) { by(1) }				\
    else if (tp.aux.x==2) { by(2) }			\
    else if (tp.aux.x==4) { by(4) }			\
    else if (tp.aux.x==8) { by(8) }			\
    else if (tp.aux.x==16) { by(16) }			\
    else if (tp.aux.x==32) { by(32) }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dk(1,1,1,1);
      //bxbybz;
    }

#if 0
    // Experimental autotuning of the thread ordering
    bool advanceAux(TuneParam &param) const
    {
      const int *X = in->X();

      if (param.aux.w < X[3] && param.aux.x > 1 && param.aux.w < 32 && 0) {

	do { param.aux.w++; } while( (X[3]) % param.aux.w != 0);

	if (param.aux.w <= X[3]) return true;
      } else {
	param.aux.w = 1;

	if (param.aux.z < X[2] && param.aux.x > 1 && param.aux.z < 32) {

	  do { param.aux.z++; } while( (X[2]) % param.aux.z != 0);

	  if (param.aux.z <= X[2]) return true;

	} else {
	  param.aux.z = 1;

	  if (param.aux.y < X[1] && param.aux.x > 1 && param.aux.y < 32) {

	    do { param.aux.y++; } while( X[1] % param.aux.y != 0);

	    if (param.aux.y <= X[1]) return true;
	  } else {
	    param.aux.y = 1;

	    if (param.aux.x < (2*X[0]) && param.aux.x < 32) {

	      do { param.aux.x++; } while( (2*X[0]) % param.aux.x != 0);

	      if (param.aux.x <= (2*X[0]) ) return true;
	    }
	  }
	}
      }
      param.aux = make_int4(1,1,1,1);
      return false;
    }
#endif

    void initTuneParam(TuneParam &param) const
    {
      SharedDslashCuda::initTuneParam(param);
      param.aux = make_int4(1,1,1,1);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      SharedDslashCuda::defaultTuneParam(param);
      param.aux = make_int4(1,1,1,1);
    }

  };
#endif // GPU_WILSON_DIRAC

#include <dslash_policy.cuh>

  // Wilson wrappers
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, 
			const int parity, const int dagger, const cudaColorSpinorField *x, const double &k, 
			const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inSpinor->allocateGhostBuffer(1);

#ifdef GPU_WILSON_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
        
      dslashParam.ghostOffset[i][0] = in->GhostOffset(i,0)/in->FieldOrder();
      dslashParam.ghostOffset[i][1] = in->GhostOffset(i,1)/in->FieldOrder();

      if(in->GhostOffset(i,0)%in->FieldOrder()) errorQuda("ghostOffset(%d,0) %d is not a multiple of FloatN\n", i, in->GhostOffset(i,0));
      if(in->GhostOffset(i,1)%in->FieldOrder()) errorQuda("ghostOffset(%d,1) %d is not a multiple of FloatN\n", i, in->GhostOffset(i,1));

      dslashParam.ghostNormOffset[i][0] = in->GhostNormOffset(i,0);
      dslashParam.ghostNormOffset[i][1] = in->GhostNormOffset(i,1);

      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge %d and spinor %d precision not supported", 
		gauge.Precision(), in->Precision());

    DslashCuda *dslash = nullptr;
    size_t regSize = in->Precision() == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new WilsonDslashCuda<double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
						      gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new WilsonDslashCuda<float4, float4>(out, (float4*)gauge0, (float4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new WilsonDslashCuda<short4, short4>(out, (short4*)gauge0, (short4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    }

#ifndef GPU_COMMS
    DslashPolicyTune dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    dslash_policy.apply(0);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    delete dslashImp;
#endif

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

  }

}
