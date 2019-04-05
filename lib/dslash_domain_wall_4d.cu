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

  namespace domainwall4d {

#undef GPU_STAGGERED_DIRAC
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_DOMAIN_WALL_DIRAC
#include <dw_dslash4_def.h>       // Dslash4 Domain Wall kernels
#include <dw_dslash5_def.h>       // Dslash5 Domain Wall kernels
#include <dw_dslash5inv_def.h>    // Dslash5inv Domain Wall kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>
  }

  using namespace domainwall4d;

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <typename sFloat, typename gFloat>
  class DomainWallDslash4DPCCuda : public DslashCuda {

  private:
    const int DS_type;

    bool checkGrid(TuneParam &param) const {
      if (param.grid.x > (unsigned int)deviceProp.maxGridSize[0] || param.grid.y > (unsigned int)deviceProp.maxGridSize[1]) {
        warningQuda("Autotuner is skipping blockDim=(%u,%u,%u), gridDim=(%u,%u,%u) because lattice volume is too large",
                    param.block.x, param.block.y, param.block.z, param.grid.x, param.grid.y, param.grid.z);
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
    DomainWallDslash4DPCCuda(cudaColorSpinorField *out, const GaugeField &gauge, const cudaColorSpinorField *in,
			     const cudaColorSpinorField *x, const double mferm,
			     const double a, const double b, const int parity, const int dagger, const int *commOverride, const int DS_type)
      : DslashCuda(out, in, x, gauge, parity, dagger, commOverride), DS_type(DS_type)
    { 
      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.b = b;
      dslashParam.b_f = b;
      dslashParam.mferm = mferm;
      dslashParam.mferm_f = mferm;
    }
    virtual ~DomainWallDslash4DPCCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      switch(DS_type){
      case 0:
	strcat(key.aux,",Dslash4");
	break;
      case 1:
	strcat(key.aux,",Dslash5");
	break;
      case 2:
	strcat(key.aux,",Dslash5inv");
	break;
      }
      return key;
    }

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
      
      switch(DS_type){
        case 0:
          DSLASH(domainWallDslash4, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 1:
          DSLASH(domainWallDslash5, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 2:
          DSLASH(domainWallDslash5inv, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        default:
          errorQuda("invalid Dslash type");
      }
    }

    long long flops() const {
      long long Ls = in->X(4);
      long long vol4d = in->VolumeCB() / Ls;
      long long bulk = (Ls-2)*vol4d;
      long long wall = 2*vol4d;
      long long flops = 0;
      switch(DS_type){
        case 0:
          flops = DslashCuda::flops();
          break;
        case 1:
          flops = (x ? 48ll : 0 ) * in->VolumeCB() + 96ll*bulk + 120ll*wall;
          break;
        case 2:
          flops = 144ll*in->VolumeCB()*Ls + 3ll*Ls*(Ls-1ll);
          break;
        default:
          errorQuda("invalid Dslash type");
      }
      return flops;
    }

    long long bytes() const {
      bool isFixed = (in->Precision() == sizeof(short) || in->Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isFixed ? sizeof(float) : 0);
      long long Ls = in->X(4);
      long long bytes = 0;

      switch(DS_type){
      case 0:
	bytes = DslashCuda::bytes();
	break;
      case 1:
	bytes = (x ? 5ll : 4ll ) * spinor_bytes * in->VolumeCB();
	break;
      case 2:
	bytes = (x ? Ls + 2 : Ls + 1) * spinor_bytes * in->VolumeCB();
	break;
      default:
	errorQuda("invalid Dslash type");
      }
      return bytes;
    }
  };
#endif // GPU_DOMAIN_WALL_DIRAC

  //-----------------------------------------------------
  // Modification for 4D preconditioned DWF operator
  // Additional Arg. is added to give a function name.
  //
  // pre-defined DS_type list
  // 0 = dslash4
  // 1 = dslash5
  // 2 = dslash5inv
  //-----------------------------------------------------

  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			    const cudaColorSpinorField *in, const int parity, const int dagger, 
			    const cudaColorSpinorField *x, const double &m_f, const double &a, const double &b,
			    const int *commOverride, const int DS_type, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    using namespace dslash;
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new DomainWallDslash4DPCCuda<double2,double2>(out, gauge, in, x, m_f, a, b, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new DomainWallDslash4DPCCuda<float4,float4>(out, gauge, in, x, m_f, a, b, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new DomainWallDslash4DPCCuda<short4,short4>(out, gauge, in, x, m_f, a, b, parity, dagger, commOverride, DS_type);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    DslashPolicyImp<DslashCuda> *dslashImp = nullptr;
    if (DS_type != 0) {
      dslashImp = DslashFactory<DslashCuda>::create(QudaDslashPolicy::QUDA_DSLASH_NC);
      (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), in->Volume()/in->X(4), ghostFace, profile);
      delete dslashImp;
    } else {
      DslashPolicyTune<DslashCuda> dslash_policy(
          *dslash, const_cast<cudaColorSpinorField *>(in), in->Volume() / in->X(4), ghostFace, profile);
      dslash_policy.apply(0);
    }

    delete dslash;
#else
    errorQuda("4D preconditioned Domain wall dslash has not been built");
#endif
  }

}

#endif
