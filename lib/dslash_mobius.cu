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

  namespace mobius {

#undef GPU_STAGGERED_DIRAC
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_DOMAIN_WALL_DIRAC
#include <mdw_dslash4_def.h>      // Dslash4, intermediate operator for Mobius Mat_4 kernels
#include <mdw_dslash4pre_def.h>   // Dslash4pre, intermediate operator for Mobius Mat_4 kernels
#include <mdw_dslash5_def.h>      // Dslash5 Mobius Domain Wall kernels
#include <mdw_dslash5inv_def.h>   // Dslash5inv Mobius Domain Wall kernels
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>
  }

  // declare the dslash events
#include <dslash_events.cuh>

  using namespace mobius;

#ifdef GPU_DOMAIN_WALL_DIRAC
  //Dslash class definition for Mobius Domain Wall Fermion
  template <typename sFloat, typename gFloat>
  class MDWFDslashPCCuda : public DslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const double mferm, a; 
    double *b5, *c5;
    const int DS_type;

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
    MDWFDslashPCCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1,
		     const QudaReconstructType reconstruct, const cudaColorSpinorField *in, 
		     const cudaColorSpinorField *x, const double mferm, 
		     const double a, const int dagger, const int DS_type)
      : DslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), 
	mferm(mferm), a(a), DS_type(DS_type)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }
    virtual ~MDWFDslashPCCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      switch(DS_type){
      case 0:
	strcat(key.aux,",Dslash4");
	break;
      case 1:
	strcat(key.aux,",Dslash4pre");
	break;
      case 2:
	strcat(key.aux,",Dslash5");
	break;
      case 3:
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
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      
      switch(DS_type){
      case 0:
	DSLASH(MDWFDslash4, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), 
	       (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0),  (float*)(x ? x->Norm() : 0), a);
	break;
      case 1:
	DSLASH(MDWFDslash4pre, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), 
	       (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0),  (float*)(x ? x->Norm() : 0), a);
	break;
      case 2:
	DSLASH(MDWFDslash5, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), 
	       (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0),  (float*)(x ? x->Norm() : 0), a);
	break;
      case 3:
	DSLASH(MDWFDslash5inv, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	       (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), 
	       (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0),  (float*)(x ? x->Norm() : 0), a);
	break;
      default:
	errorQuda("invalid Dslash type");
      }
    }

    long long flops() const { // FIXME for multi-GPU
      long long Ls = in->X(4);
      long long vol4d = in->VolumeCB() / Ls;
      long long bulk = (Ls-2)*vol4d;
      long long wall = 2*vol4d;
      long long flops_Tmp; 
      switch(DS_type){
      case 0:
	flops_Tmp = (x ? 1368ll : 1320ll)*in->VolumeCB();
	break;
      case 1:
	flops_Tmp = 72ll*in->VolumeCB() + 96ll*bulk + 120ll*wall;
	break;
      case 2:
	flops_Tmp = (x ? 96ll : 48ll)*in->VolumeCB() + 96ll*bulk + 120ll*wall;
	break;
      case 3:
	flops_Tmp = 144ll*in->VolumeCB()*Ls + 3ll*Ls*(Ls-1ll);
	break;
      default:
	errorQuda("invalid Dslash type");
      }
      return flops_Tmp;
    }
  };
#endif // GPU_DOMAIN_WALL_DIRAC

#include <dslash_policy.cuh>

  //-----------------------------------------------------
  // Modification for 4D preconditioned Mobius DWF operator
  // Additional Arg. is added to give a function name.
  //
  // pre-defined DS_type list
  // 0 = MDWF dslash4
  // 1 = MDWF dslash4pre
  // 2 = MDWF dslash5
  // 3 = MDWF dslash5inv
  //-----------------------------------------------------

  void MDWFDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
		      const cudaColorSpinorField *in, const int parity, const int dagger, 
		      const cudaColorSpinorField *x, const double &m_f, const double &k2, 
		      const int *commOverride, const int DS_type, TimeProfile &profile, const QudaDslashPolicy &dslashPolicy)
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
      dslash = new MDWFDslashPCCuda<double2,double2>(out, (double2*)gauge0, (double2*)gauge1, 
						     gauge.Reconstruct(), in, x, m_f, k2, dagger, DS_type);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new MDWFDslashPCCuda<float4,float4>(out, (float4*)gauge0, (float4*)gauge1, 
						   gauge.Reconstruct(), in, x, m_f, k2, dagger, DS_type);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new MDWFDslashPCCuda<short4,short4>(out, (short4*)gauge0, (short4*)gauge1, 
						   gauge.Reconstruct(), in, x, m_f, k2, dagger, DS_type);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    DslashPolicyImp* dslashImp = NULL;
    if (DS_type != 0) {
      dslashImp = DslashFactory::create(QUDA_DSLASH_NC);
    } else {
#ifndef GPU_COMMS
      dslashImp = DslashFactory::create(dslashPolicy);
#else
      dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
#endif
    }
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
