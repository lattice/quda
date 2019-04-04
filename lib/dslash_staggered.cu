#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <typeinfo>

#include <color_spinor_field.h>
#include <clover_field.h>

//these are access control for staggered action
#if (defined GPU_STAGGERED_DIRAC && defined USE_LEGACY_DSLASH)
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
#include <dslash.h>
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>
#include <dslash_policy.cuh>

namespace quda {
#ifdef USE_LEGACY_DSLASH
  namespace staggered {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

#undef GPU_CLOVER_DIRAC
#undef GPU_DOMAIN_WALL_DIRAC
#define DD_IMPROVED 0

#define DD_DAG 0
#include <staggered_dslash_def.h> // staggered Dslash kernels
#undef DD_DAG
#define DD_DAG 1
#include <staggered_dslash_def.h> // staggered Dslash dagger kernels
#undef DD_DAG

#define TIFR
#define DD_DAG 0
#include <staggered_dslash_def.h> // staggered Dslash kernels
#undef DD_DAG
#define DD_DAG 1
#include <staggered_dslash_def.h> // staggered Dslash dagger kernels
#undef DD_DAG
#undef TIFR

#undef DD_IMPROVED

#include <dslash_quda.cuh>
  } // end namespace staggered

#endif

  using namespace staggered;

#if (defined GPU_STAGGERED_DIRAC && defined USE_LEGACY_DSLASH)
  template <typename sFloat, typename gFloat>
  class StaggeredDslashCuda : public DslashCuda {

  private:
    const unsigned int nSrc;

  protected:
    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
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
    StaggeredDslashCuda(cudaColorSpinorField *out, const GaugeField &gauge, const cudaColorSpinorField *in,
			const cudaColorSpinorField *x, const double a,
                        const int parity, const int dagger, const int *commOverride)
      : DslashCuda(out, in, x, gauge, parity, dagger, commOverride), nSrc(in->X(4))
    {
      if (gauge.Reconstruct() == QUDA_RECONSTRUCT_9 || gauge.Reconstruct() == QUDA_RECONSTRUCT_13) {
        errorQuda("Reconstruct %d not supported", gauge.Reconstruct());
      }
      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.fat_link_max = gauge.LinkMax();

      if (gauge.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef MULTI_GPU
        strcat(aux[INTERIOR_KERNEL],",TIFR");
        strcat(aux[EXTERIOR_KERNEL_ALL],",TIFR");
        strcat(aux[EXTERIOR_KERNEL_X],",TIFR");
        strcat(aux[EXTERIOR_KERNEL_Y],",TIFR");
        strcat(aux[EXTERIOR_KERNEL_Z],",TIFR");
        strcat(aux[EXTERIOR_KERNEL_T],",TIFR");
#else
        strcat(aux[INTERIOR_KERNEL],",TIFR");
#endif // MULTI_GPU
        strcat(aux[KERNEL_POLICY],",TIFR");
      }
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
#ifndef USE_TEXTURE_OBJECTS
      if (dslashParam.kernel_type == INTERIOR_KERNEL) bindSpinorTex<sFloat>(in, out, x);
#endif // USE_TEXTURE_OBJECTS
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      setParam();
      dslashParam.swizzle = tp.aux.x;
      if (gauge.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
        STAGGERED_DSLASH_TIFR(tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
#else
        errorQuda("TIFR interface has not been built");
#endif
      } else {
        STAGGERED_DSLASH(tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
      }
    }

    bool advanceBlockDim(TuneParam &param) const
    {
      const unsigned int max_shared = deviceProp.sharedMemPerBlock;
      // first try to advance block.y (number of right-hand sides per block)
      if (param.block.y < nSrc && param.block.y < (unsigned int)deviceProp.maxThreadsDim[1] &&
	  sharedBytesPerThread()*param.block.x*param.block.y < max_shared &&
	  (param.block.x*(param.block.y+1u)) <= (unsigned int)deviceProp.maxThreadsPerBlock) {
	param.block.y++;
	param.grid.y = (nSrc + param.block.y - 1) / param.block.y;
	return true;
      } else {
	bool rtn = DslashCuda::advanceBlockDim(param);
	param.block.y = 1;
	param.grid.y = nSrc;
	return rtn;
      }
    }

    bool advanceAux(TuneParam &param) const
    {
#ifdef SWIZZLE
      if (param.aux.x < 2*deviceProp.multiProcessorCount) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      return false;
#endif
    }

    void initTuneParam(TuneParam &param) const
    {
      DslashCuda::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = nSrc;
      param.aux.x = 1;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    int Nface() const { return 2; }
  };
#endif // GPU_STAGGERED_DIRAC

  void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, 
			   const int dagger, const cudaColorSpinorField *x,
			   const double &k, const int *commOverride, TimeProfile &profile)
  {
#if (defined GPU_STAGGERED_DIRAC && defined USE_LEGACY_DSLASH)
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new StaggeredDslashCuda<double2, double2>(out, gauge, in, x, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2>(out, gauge, in, x, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2>(out, gauge, in, x, k, parity, dagger, commOverride);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    dslash::DslashPolicyTune<DslashCuda> dslash_policy(
        *dslash, const_cast<cudaColorSpinorField *>(in), in->Volume() / in->X(4), ghostFace, profile);
    dslash_policy.apply(0);

    delete dslash;
#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

}
