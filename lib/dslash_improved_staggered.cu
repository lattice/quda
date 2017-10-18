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

#include <inline_ptx.h>

namespace quda {

  namespace improvedstaggered {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

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

#ifdef GPU_STAGGERED_DIRAC
  template <typename sFloat, typename fatGFloat, typename longGFloat, typename phaseFloat>
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
    StaggeredDslashCuda(cudaColorSpinorField *out, const fatGFloat *fat0, const fatGFloat *fat1,
			const longGFloat *long0, const longGFloat *long1,
			const phaseFloat *phase0, const phaseFloat *phase1, 
			const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			const cudaColorSpinorField *x, const double a, const int dagger)
      : DslashCuda(out, in, x, reconstruct, dagger), nSrc(in->X(4))
    { 
      bindSpinorTex<sFloat>(in, out, x);
      dslashParam.gauge0 = (void*)fat0;
      dslashParam.gauge1 = (void*)fat1;
      dslashParam.longGauge0 = (void*)long0;
      dslashParam.longGauge1 = (void*)long1;
      dslashParam.longPhase0 = (void*)phase0;
      dslashParam.longPhase1 = (void*)phase1;
      dslashParam.a = a;
      dslashParam.a_f = a;
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dslashParam.swizzle = tp.aux.x;
      IMPROVED_STAGGERED_DSLASH(tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
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

    int Nface() { return 6; } 

    /*
      per direction / dimension flops
      SU(3) matrix-vector flops = (8 Nc - 2) * Nc
      xpay = 2 * 2 * Nc * Ns
      
      So for the full dslash we have      
      flops = (2 * 2 * Nd * (8*Nc-2) * Nc)  +  ((2 * 2 * Nd - 1) * 2 * Nc * Ns)
      flops_xpay = flops + 2 * 2 * Nc * Ns
      
      For Asqtad this should give 1146 for Nc=3,Ns=2 and 1158 for the axpy equivalent
    */
    virtual long long flops() const {
      int mv_flops = (8 * in->Ncolor() - 2) * in->Ncolor(); // SU(3) matrix-vector flops
      int ghost_flops = (3 + 1) * (mv_flops + 2*in->Ncolor()*in->Nspin());
      int xpay_flops = 2 * 2 * in->Ncolor() * in->Nspin(); // multiply and add per real component
      int num_dir = 2 * 4; // dir * dim

      long long flops = 0;
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
	flops = ghost_flops * 2 * in->GhostFace()[dslashParam.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	{
	  long long ghost_sites = 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	  flops = ghost_flops * ghost_sites;
	  break;
	}
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	{
	  long long sites = in->VolumeCB();
	  flops = (2*num_dir*mv_flops +                   // SU(3) matrix-vector multiplies
		   (2*num_dir-1)*2*in->Ncolor()*in->Nspin()) * sites;   // accumulation
	  if (x) flops += xpay_flops * sites; // axpy is always on interior

	  if (dslashParam.kernel_type == KERNEL_POLICY) break;
	  // now correct for flops done by exterior kernel
	  long long ghost_sites = 0;
	  for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2 * in->GhostFace()[d];
	  flops -= ghost_flops * ghost_sites;
	  
	  break;
	}
      }
      return flops;
    }

    virtual long long bytes() const {
      int gauge_bytes_fat = QUDA_RECONSTRUCT_NO * in->Precision();
      int gauge_bytes_long = reconstruct * in->Precision();
      bool isHalf = in->Precision() == sizeof(short) ? true : false;
      int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isHalf ? sizeof(float) : 0);
      int ghost_bytes = 3 * (spinor_bytes + gauge_bytes_long) + (spinor_bytes + gauge_bytes_fat) + spinor_bytes;
      int num_dir = 2 * 4; // set to 4 dimensions since we take care of 5-d fermions in derived classes where necessary

      long long bytes = 0;
      switch(dslashParam.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
	bytes = ghost_bytes * 2 * in->GhostFace()[dslashParam.kernel_type];
	break;
      case EXTERIOR_KERNEL_ALL:
	{
	  long long ghost_sites = 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	  bytes = ghost_bytes * ghost_sites;
	  break;
	}
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
	{
	  long long sites = in->VolumeCB();
	  bytes = (num_dir*(gauge_bytes_fat + gauge_bytes_long) + // gauge reads
		   num_dir*2*spinor_bytes +                       // spinor reads
		   spinor_bytes)*sites;                           // spinor write
	  if (x) bytes += spinor_bytes;

	  if (dslashParam.kernel_type == KERNEL_POLICY) break;
	  // now correct for bytes done by exterior kernel
	  long long ghost_sites = 0;
	  for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2*in->GhostFace()[d];
	  bytes -= ghost_bytes * ghost_sites;
	  
	  break;
	}
      }
      return bytes;
    }

  };
#endif // GPU_STAGGERED_DIRAC

#include <dslash_policy.cuh>

  void improvedStaggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
				   const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
				   const int parity, const int dagger, const cudaColorSpinorField *x,
				   const double &k, const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inSpinor->createComms(3);

#ifdef GPU_STAGGERED_DIRAC

    dslashParam.Ls = out->X(4); // use Ls as the number of sources

#ifdef MULTI_GPU
    for(int i=0;i < 4; i++){
      if(comm_dim_partitioned(i) && (fatGauge.X()[i] < 6)){
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
      dslashParam.ghostDim[i] = comm_dim_partitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i][0] = in->GhostOffset(i,0)/in->FieldOrder();
      dslashParam.ghostOffset[i][1] = in->GhostOffset(i,1)/in->FieldOrder();
      dslashParam.ghostNormOffset[i][0] = in->GhostNormOffset(i,0);
      dslashParam.ghostNormOffset[i][1] = in->GhostNormOffset(i,1);
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : comm_dim_partitioned(i); // switch off comms if override = 0
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
      dslash = new StaggeredDslashCuda<double2, double2, double2, double>
	(out, (double2*)fatGauge0, (double2*)fatGauge1,
	 (double2*)longGauge0, (double2*)longGauge1,
	 (double*)longPhase0, (double*)longPhase1, 
	 longGauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
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

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    DslashPolicyTune dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume()/in->X(4), ghostFace, profile);
    dslash_policy.apply(0);

    delete dslash;
    unbindFatGaugeTex(fatGauge);
    unbindLongGaugeTex(longGauge);

    checkCudaError();

#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

}
