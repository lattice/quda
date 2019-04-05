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

  namespace clover {

#undef GPU_STAGGERED_DIRAC // do not delete - hack for Tesla architecture

#ifndef GPU_DOMAIN_WALL_DIRAC
#define GPU_DOMAIN_WALL_DIRAC // do not delete - work around for CUDA 6.5 alignment bug
#endif

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_CLOVER_DIRAC
#define DD_CLOVER 1
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#undef DD_CLOVER
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace clover

  using namespace clover;

#ifdef GPU_CLOVER_DIRAC
  template <typename sFloat, typename gFloat, typename cFloat>
  class CloverDslashCuda : public SharedDslashCuda {

  protected:
    const FullClover &cloverInv;

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
    CloverDslashCuda(cudaColorSpinorField *out,  const GaugeField &gauge, const FullClover &cloverInv, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x, const double a, const int parity, const int dagger, const int *commOverride)
      : SharedDslashCuda(out, in, x, gauge, parity, dagger, commOverride), cloverInv(cloverInv)
    { 
      QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, dslashParam);
      if (in->Precision() != clover_prec) errorQuda("Mixing clover and spinor precision not supported");
      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.cl_stride = cloverInv.stride;
    }

    virtual ~CloverDslashCuda() {
      unbindSpinorTex<sFloat>(in, out, x);
      unbindCloverTex(cloverInv);
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
      dslashParam.block[0] = tp.aux.x; dslashParam.block[1] = tp.aux.y; dslashParam.block[2] = tp.aux.z; dslashParam.block[3] = tp.aux.w;
      for (int i=0; i<4; i++) dslashParam.grid[i] = ( (i==0 ? 2 : 1) * in->X(i)) / dslashParam.block[i];
      DSLASH(cloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
    }

    long long flops() const {
      int clover_flops = 504;
      long long flops = DslashCuda::flops();
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
	flops += clover_flops * in->GhostFace()[dslashParam.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	flops += clover_flops * 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	flops += clover_flops * in->VolumeCB();	  

	if (dslashParam.kernel_type == KERNEL_POLICY) break;
	// now correct for flops done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2 * in->GhostFace()[d];
	flops -= clover_flops * ghost_sites;
	
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
	bytes += clover_bytes * 2 * in->GhostFace()[dslashParam.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	bytes += clover_bytes * 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	bytes += clover_bytes*in->VolumeCB();

	if (dslashParam.kernel_type == KERNEL_POLICY) break;
	// now correct for bytes done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2*in->GhostFace()[d];
	bytes -= clover_bytes * ghost_sites;
	
	break;
      }

      return bytes;
    }

  };
#endif // GPU_CLOVER_DIRAC

  void cloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover &cloverInv,
			const cudaColorSpinorField *in, const int parity, const int dagger, 
			const cudaColorSpinorField *x, const double &a, const int *commOverride,
			TimeProfile &profile)
  {
#ifdef GPU_CLOVER_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new CloverDslashCuda<double2, double2, double2>(out, gauge, cloverInv, in, x, a, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new CloverDslashCuda<float4, float4, float4>(out, gauge, cloverInv, in, x, a, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new CloverDslashCuda<short4, short4, short4>(out, gauge, cloverInv, in, x, a, parity, dagger, commOverride);
    }

    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), in->Volume(), in->GhostFace(), profile);
    dslash_policy.apply(0);

    delete dslash;
#else
    errorQuda("Clover dslash has not been built");
#endif

  }

}

#endif
