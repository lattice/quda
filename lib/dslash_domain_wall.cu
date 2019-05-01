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

  using namespace domainwall;

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <typename sFloat, typename gFloat>
  class DomainWallDslashCuda : public DslashCuda {

  private:
    bool checkGrid(TuneParam &param) const {
      if (param.grid.x > (unsigned int)deviceProp.maxGridSize[0] || param.grid.y > (unsigned int)deviceProp.maxGridSize[1]) {
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
      //memory constraint
      if (param.block.x > (unsigned int)deviceProp.maxThreadsDim[0] ||
	  sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	advance[0] = false;
	param.block.x = step[0]; // reset block.x
      } else {
	advance[0] = true; // successfully advanced block.x
      }

      if (!advance[0]) {  // if failed to advance block.x, now try block.y
	param.block.y += step[1];

	//memory constraint
	if (param.block.y > (unsigned)in->X(4) ||
	    sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	  advance[1] = false;
	  param.block.y = step[1]; // reset block.y
	} else {
	  advance[1] = true; // successfully advanced block.y
	}
      }

      //thread constraint
      if ( (advance[0] || advance[1]) && param.block.x*param.block.y*param.block.z <= (unsigned)deviceProp.maxThreadsPerBlock) {
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
    DomainWallDslashCuda(cudaColorSpinorField *out, const GaugeField &gauge, const cudaColorSpinorField *in,
			 const cudaColorSpinorField *x, const double mferm, const double a,
                         const int parity, const int dagger, const int* commOverride)
      : DslashCuda(out, in, x, gauge, parity, dagger, commOverride)
    { 
      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.a_inv = 1.0/a;
      dslashParam.a_inv_f = 1.0/a;
      dslashParam.mferm = mferm;
      dslashParam.mferm_f = mferm;
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
#ifndef USE_TEXTURE_OBJECTS
      if (dslashParam.kernel_type == INTERIOR_KERNEL) bindSpinorTex<sFloat>(in, out, x);
#endif // USE_TEXTURE_OBJECTS
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      setParam();
      DSLASH(domainWallDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
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
      case KERNEL_POLICY:
	int Ls = in->X(4);
	long long bulk = (Ls-2)*(in->VolumeCB()/Ls);
	long long wall = 2*(in->VolumeCB()/Ls);
	flops += 96ll*bulk + 120ll*wall;
	break;
      }
      return flops;
    }

    virtual long long bytes() const {
      bool isFixed = (in->Precision() == sizeof(short) || in->Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isFixed ? sizeof(float) : 0);
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
	bytes += 2 * spinor_bytes * in->VolumeCB();
	break;
      }
      return bytes;
    }
  };
#endif // GPU_DOMAIN_WALL_DIRAC

  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			    const cudaColorSpinorField *in, const int parity, const int dagger, 
			    const cudaColorSpinorField *x, const double &m_f, const double &k2, 
			    const int *commOverride, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    // with 5-d checkerboarding we must use kernel packing
    pushKernelPackT(true);

    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = 0;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new DomainWallDslashCuda<double2,double2>(out, gauge, in, x, m_f, k2, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new DomainWallDslashCuda<float4,float4>(out, gauge, in, x, m_f, k2, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new DomainWallDslashCuda<short4,short4>(out, gauge, in, x, m_f, k2, parity, dagger, commOverride);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), in->Volume() / in->X(4), ghostFace, profile);
    dslash_policy.apply(0);

    delete dslash;

    popKernelPackT();
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

}

#endif
