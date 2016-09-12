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
#else // Fermi
//#define DIRECT_ACCESS_FAT_LINK
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
    const int nSrc;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#ifdef PARALLEL_DIR
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return 6 * reg_size;
#else
      return 0;
#endif
    }

  public:
    StaggeredDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1,
			const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			const cudaColorSpinorField *x, const double a, const int dagger)
      : DslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), a(a), nSrc(in->X(4))
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      STAGGERED_DSLASH(tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
		       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
		       (sFloat*)in->V(), (float*)in->Norm(), 
		       (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
    }

    bool advanceBlockDim(TuneParam &param) const
    {
      const unsigned int max_shared = deviceProp.sharedMemPerBlock;
      // first try to advance block.y (number of right-hand sides per block)
      if (param.block.y < nSrc && param.block.y < deviceProp.maxThreadsDim[1] &&
	  sharedBytesPerThread()*param.block.x*param.block.y < max_shared &&
	  (param.block.x*(param.block.y+1)) <= deviceProp.maxThreadsPerBlock) {
	param.block.y++;
	param.grid.y = (nSrc + param.block.y - 1) / param.block.y;
	return true;
      } else {
	param.block.y = 1;
	param.grid.y = nSrc;
	bool rtn = DslashCuda::advanceBlockDim(param);
	param.block.y = 1;
	param.grid.y = nSrc;
	return rtn;
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      DslashCuda::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = nSrc;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    int Nface() { return 2; } 
  };
#endif // GPU_STAGGERED_DIRAC

#include <dslash_policy.cuh>

  void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, 
			   const int dagger, const cudaColorSpinorField *x,
			   const double &k, const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inSpinor->allocateGhostBuffer(1);

#ifdef GPU_STAGGERED_DIRAC

    dslashParam.Ls = out->X(4);

    dslashParam.parity = parity;
    dslashParam.gauge_stride = gauge.Stride();
    dslashParam.fat_link_max = gauge.LinkMax(); // May need to use this in the preconditioning step 
    //experimental for U1 emulation:
    dslashParam.staggered_u1_emulation = gauge.StaggeredU1Emulation() ? 1 : 0;
    // in the solver for the improved staggered action

    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i][0] = in->GhostOffset(i,0)/in->FieldOrder();
      dslashParam.ghostOffset[i][1] = in->GhostOffset(i,1)/in->FieldOrder();
      dslashParam.ghostNormOffset[i][0] = in->GhostNormOffset(i,0);
      dslashParam.ghostNormOffset[i][1] = in->GhostNormOffset(i,1);
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
      dslash = new StaggeredDslashCuda<double2, double2>
	(out, (double2*)gauge0, (double2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2>
	(out, (float2*)gauge0, (float2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2>
	(out, (short2*)gauge0, (short2*)gauge1, gauge.Reconstruct(), in, x, k, dagger);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

#ifndef GPU_COMMS
    DslashPolicyTune dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger,  in->Volume()/in->X(4), ghostFace, profile);
    dslash_policy.apply(0);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume()/in->X(4), ghostFace, profile);
    delete dslashImp;
#endif

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();

#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

}
