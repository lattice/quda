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

  namespace improvedstaggered {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

#define STAGGERED_TESLA_HACK
#undef GPU_NDEG_TWISTED_MASS_DIRAC
#undef GPU_CLOVER_DIRAC
#undef GPU_DOMAIN_WALL_DIRAC
#define DD_IMPROVED 1
#include <staggered_dslash_def.h> // staggered Dslash kernels
#undef DD_IMPROVED

#include <dslash_quda.cuh>
  } // end namespace improvedstaggered

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace improvedstaggered;

  template<typename T> struct RealType {};
  template<> struct RealType<double2> { typedef double type; };
  template<> struct RealType<float2> { typedef float type; };
  template<> struct RealType<float4> { typedef float type; };
  template<> struct RealType<short2> { typedef short type; };
  template<> struct RealType<short4> { typedef short type; };

#ifdef GPU_STAGGERED_DIRAC
  template <typename sFloat, typename fatGFloat, typename longGFloat, typename phaseFloat>
  class StaggeredDslashCuda : public DslashCuda {

  private:
    const fatGFloat *fat0, *fat1;
    const longGFloat *long0, *long1;
    const phaseFloat *phase0, *phase1;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return 6 * reg_size;
    }

  public:
    StaggeredDslashCuda(cudaColorSpinorField *out, const fatGFloat *fat0, const fatGFloat *fat1,
			const longGFloat *long0, const longGFloat *long1,
			const phaseFloat *phase0, const phaseFloat *phase1, 
			const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			const cudaColorSpinorField *x, const double a, const int dagger)
      : DslashCuda(out, in, x, reconstruct, dagger), fat0(fat0), fat1(fat1), long0(long0), 
	long1(long1), phase0(phase0), phase1(phase1), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
#if (__COMPUTE_CAPABILITY__ >= 200)
      IMPROVED_STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
				(sFloat*)out->V(), (float*)out->Norm(), 
				fat0, fat1, long0, long1, phase0, phase1, 
				(sFloat*)in->V(), (float*)in->Norm(), 
				(sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
#else
      IMPROVED_STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
				(sFloat*)out->V(), (float*)out->Norm(), 
				fat0, fat1, long0, long1,
				(sFloat*)in->V(), (float*)in->Norm(), 
				(sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
#endif
    }

    int Nface() { return 6; } 

    long long flops() const { 
      long long flops;
      flops = (x ? 1158ll : 1146ll) * in->VolumeCB();
      return flops;
    } 
  };
#endif // GPU_STAGGERED_DIRAC

#include <dslash_policy.cuh>

  void improvedStaggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
				   const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
				   const int parity, const int dagger, const cudaColorSpinorField *x,
				   const double &k, const int *commOverride, TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

#ifdef MULTI_GPU
    for(int i=0;i < 4; i++){
      if(commDimPartitioned(i) && (fatGauge.X()[i] < 6)){
	errorQuda("ERROR: partitioned dimension with local size less than 6 is not supported in staggered dslash\n");
      }    
    }
#endif

    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

    dslashParam.parity = parity;
    dslashParam.gauge_stride = fatGauge.Stride();
    dslashParam.long_gauge_stride = longGauge.Stride();
    dslashParam.fat_link_max = fatGauge.LinkMax();

    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *fatGauge0, *fatGauge1;
    void* longGauge0, *longGauge1;
    bindFatGaugeTex(fatGauge, parity, &fatGauge0, &fatGauge1);
    bindLongGaugeTex(longGauge, parity, &longGauge0, &longGauge1);
    void *longPhase0 = (char*)longGauge0 + longGauge.PhaseOffset();
    void *longPhase1 = (char*)longGauge1 + longGauge.PhaseOffset();   

    if (in->Precision() != fatGauge.Precision() || in->Precision() != longGauge.Precision()){
      errorQuda("Mixing gauge and spinor precision not supported"
		"(precision=%d, fatlinkGauge.precision=%d, longGauge.precision=%d",
		in->Precision(), fatGauge.Precision(), longGauge.Precision());
    }

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new StaggeredDslashCuda<double2, double2, double2, double>
	(out, (double2*)fatGauge0, (double2*)fatGauge1,
	 (double2*)longGauge0, (double2*)longGauge1,
	 (double*)longPhase0, (double*)longPhase1, 
	 longGauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2, float4, float>
	(out, (float2*)fatGauge0, (float2*)fatGauge1,
	 (float4*)longGauge0, (float4*)longGauge1, 
	 (float*)longPhase0, (float*)longPhase1,
	 longGauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2, short4, short>
	(out, (short2*)fatGauge0, (short2*)fatGauge1,
	 (short4*)longGauge0, (short4*)longGauge1, 
	 (short*)longPhase0, (short*)longPhase1,
	 longGauge.Reconstruct(), in, x, k, dagger);
    }

#ifndef GPU_COMMS
    DslashPolicyImp* dslashImp = DslashFactory::create(dslashPolicy);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
#endif
    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    delete dslashImp;

    delete dslash;
    unbindFatGaugeTex(fatGauge);
    unbindLongGaugeTex(longGauge);

    checkCudaError();

#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

}
