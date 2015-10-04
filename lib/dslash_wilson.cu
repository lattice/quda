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
#if (__COMPUTE_CAPABILITY__ >= 200) // Fermi uses shared memory for common input
      if (dslashParam.kernel_type == INTERIOR_KERNEL) { // Interior kernels use shared memory for common iunput
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else { // Exterior kernels use no shared memory
	return 0;
      }
#else // Pre-Fermi uses shared memory only for pseudo-registers
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
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

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DSLASH(dslash, tp.grid, tp.block, tp.shared_bytes, stream, 
	     dslashParam, (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a);
    }
  };
#endif // GPU_WILSON_DIRAC

#include <dslash_policy.cuh>

  // Wilson wrappers
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, 
			const int parity, const int dagger, const cudaColorSpinorField *x, const double &k, 
			const int *commOverride, TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_WILSON_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge %d and spinor %d precision not supported", 
		gauge.Precision(), in->Precision());

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new WilsonDslashCuda<double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
						      gauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new WilsonDslashCuda<float4, float4>(out, (float4*)gauge0, (float4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new WilsonDslashCuda<short4, short4>(out, (short4*)gauge0, (short4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    }

#ifndef GPU_COMMS
    DslashPolicyImp* dslashImp = DslashFactory::create(dslashPolicy);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
#endif

    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    delete dslashImp;

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

  }

}
