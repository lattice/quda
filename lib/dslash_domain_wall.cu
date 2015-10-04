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

  namespace domainwall {

#undef GPU_STAGGERED_DIRAC
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>
    
    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_DOMAIN_WALL_DIRAC
#include <dw_dslash_def.h>        // Domain Wall kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>
  }

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace domainwall;

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <typename sFloat, typename gFloat>
  class DomainWallDslashCuda : public DslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const double mferm;
    const double a;

    bool checkGrid(TuneParam &param) const {
      if (param.grid.x > deviceProp.maxGridSize[0] || param.grid.y > deviceProp.maxGridSize[1]) {
	warningQuda("Autotuner is skipping blockDim=(%u,%u,%u), gridDim=(%u,%u,%u) because lattice volume is too large",
                    param.block.x, param.block.y, param.block.z, 
                    param.grid.x, param.grid.y, param.grid.z);
	return false;
      } else {
	return true;
      }
    }

  protected:
    bool advanceBlockDim(TuneParam &param) const
    {
      const unsigned int max_shared = 16384; // FIXME: use deviceProp.sharedMemPerBlock;
      const int step[2] = { deviceProp.warpSize, 1 };
      bool advance[2] = { false, false };

      // first try to advance block.x
      param.block.x += step[0];
      if (param.block.x > deviceProp.maxThreadsDim[0] || 
	  sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	advance[0] = false;
	param.block.x = step[0]; // reset block.x
      } else {
	advance[0] = true; // successfully advanced block.x
      }

      if (!advance[0]) {  // if failed to advance block.x, now try block.y
	param.block.y += step[1];

	if (param.block.y > in->X(4) || 
	    sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	  advance[1] = false;
	  param.block.y = step[1]; // reset block.x
	} else {
	  advance[1] = true; // successfully advanced block.y
	}
      }

      if (advance[0] || advance[1]) {
	param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			   (in->X(4)+param.block.y-1) / param.block.y, 1);

	bool advance = true;
	if (!checkGrid(param)) advance = advanceBlockDim(param);
	return advance;
      } else {
	return false;
      }
    }

    unsigned int sharedBytesPerThread() const { return 0; }

  public:
    DomainWallDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
			 const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			 const cudaColorSpinorField *x, const double mferm, 
			 const double a, const int dagger)
      : DslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), 
	gauge1(gauge1), mferm(mferm), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }
    virtual ~DomainWallDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      bool ok = true;
      if (!checkGrid(param)) ok = advanceBlockDim(param);
      if (!ok) errorQuda("Lattice volume is too large for even the largest blockDim");
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      bool ok = true;
      if (!checkGrid(param)) ok = advanceBlockDim(param);
      if (!ok) errorQuda("Lattice volume is too large for even the largest blockDim");
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DSLASH(domainWallDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a);
    }

    long long flops() const {
      long long flops = DslashCuda::flops();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
	int Ls = in->X(4);
	long long bulk = (Ls-2)*(in->VolumeCB()/Ls);
	long long wall = 2*(in->VolumeCB()/Ls);
	flops += 96ll*bulk + 120ll*wall;
	break;
      }
      return flops;
    }

    virtual long long bytes() const {
      bool isHalf = in->Precision() == sizeof(short) ? true : false;
      int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isHalf ? sizeof(float) : 0);
      long long bytes = DslashCuda::bytes();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break;
      case INTERIOR_KERNEL:
	bytes += 2 * spinor_bytes * in->VolumeCB();
	break;
      }
      return bytes;
    }
  };
#endif // GPU_DOMAIN_WALL_DIRAC

#include <dslash_policy.cuh>

  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			    const cudaColorSpinorField *in, const int parity, const int dagger, 
			    const cudaColorSpinorField *x, const double &m_f, const double &k2, 
			    const int *commOverride, TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

    dslashParam.parity = parity;

#ifdef GPU_DOMAIN_WALL_DIRAC
    //currently splitting in space-time is impelemented:
    int dirs = 4;
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i = 0;i < dirs; i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }  

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new DomainWallDslashCuda<double2,double2>(out, (double2*)gauge0, (double2*)gauge1, 
							 gauge.Reconstruct(), in, x, m_f, k2, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new DomainWallDslashCuda<float4,float4>(out, (float4*)gauge0, (float4*)gauge1, 
						       gauge.Reconstruct(), in, x, m_f, k2, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new DomainWallDslashCuda<short4,short4>(out, (short4*)gauge0, (short4*)gauge1, 
						       gauge.Reconstruct(), in, x, m_f, k2, dagger);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

#ifndef GPU_COMMS
    DslashPolicyImp* dslashImp = DslashFactory::create(dslashPolicy);
#else
    DslashPolicyImp* dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
#endif

    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume()/in->X(4), ghostFace, profile);
    delete dslashImp;

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

}
