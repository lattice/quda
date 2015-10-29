#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>

//these are access control for staggered action
#ifdef GPU_STAGGERED_DIRAC
#if (__COMPUTE_CAPABILITY__ >= 300) // Kepler works best with texture loads only
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#elif (__COMPUTE_CAPABILITY__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#endif
#endif // GPU_STAGGERED_DIRAC

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>

namespace quda {

  namespace staggered {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

#undef GPU_CLOVER_DIRAC
#undef GPU_DOMAIN_WALL_DIRAC
#define DD_IMPROVED 0
#include <staggered_dslash_def.h> // staggered Dslash kernels
#undef DD_IMPROVED

#include <dslash_quda.cuh>
  } // end namespace staggered

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace staggered;

  template<typename T> struct RealType {};
  template<> struct RealType<double2> { typedef double type; };
  template<> struct RealType<float2> { typedef float type; };
  template<> struct RealType<float4> { typedef float type; };
  template<> struct RealType<short2> { typedef short type; };
  template<> struct RealType<short4> { typedef short type; };

#ifdef GPU_STAGGERED_DIRAC
  template <typename sFloat, typename gFloat>
  class StaggeredDslashCuda : public DslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return 6 * reg_size;
    }

  public:
    StaggeredDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1,
			const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			const cudaColorSpinorField *x, const double a, const int dagger)
      : DslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
		       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
		       (sFloat*)in->V(), (float*)in->Norm(), 
		       (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
    }

    int Nface() { return 2; } 
  };
#endif // GPU_STAGGERED_DIRAC

#include <dslash_policy.cuh>

  void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, 
			   const int dagger, const cudaColorSpinorField *x,
			   const double &k, const int *commOverride, TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

    dslashParam.parity = parity;
    dslashParam.gauge_stride = gauge.Stride();
    dslashParam.fat_link_max = gauge.LinkMax(); // May need to use this in the preconditioning step 
    // in the solver for the improved staggered action

    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }
    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision()) {
      errorQuda("Mixing precisions gauge=%d and spinor=%d not supported",
		gauge.Precision(), in->Precision());
    }

    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_9 || gauge.Reconstruct() == QUDA_RECONSTRUCT_13) {
      errorQuda("Reconstruct %d not supported", gauge.Reconstruct());
    }

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new StaggeredDslashCuda<double2, double2>
	(out, (double2*)gauge0, (double2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2>
	(out, (float2*)gauge0, (float2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2>
	(out, (short2*)gauge0, (short2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
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
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

}
