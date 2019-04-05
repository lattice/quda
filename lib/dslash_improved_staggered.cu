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
  namespace improvedstaggered {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

#undef GPU_NDEG_TWISTED_MASS_DIRAC
#undef GPU_CLOVER_DIRAC
#undef GPU_DOMAIN_WALL_DIRAC
#define DD_IMPROVED 1

#define DD_DAG 0
#include <staggered_dslash_def.h> // staggered Dslash kernels
#undef DD_DAG
#define DD_DAG 1
#include <staggered_dslash_def.h> // staggered Dslash dagger kernels

#undef DD_IMPROVED

#include <dslash_quda.cuh>
  } // end namespace improvedstaggered
#endif

  using namespace improvedstaggered;

#if (defined GPU_STAGGERED_DIRAC && defined USE_LEGACY_DSLASH)
  template <typename sFloat, typename fatGFloat, typename longGFloat, typename phaseFloat>
  class StaggeredDslashCuda : public DslashCuda {

  private:
    const GaugeField &fatGauge;
    const GaugeField &longGauge;
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
    StaggeredDslashCuda(cudaColorSpinorField *out, const GaugeField &fatGauge, const GaugeField &longGauge,
                        const cudaColorSpinorField *in, const cudaColorSpinorField *x, const double a,
                        const int parity, const int dagger, const int *commOverride)
      : DslashCuda(out, in, x, longGauge, parity, dagger, commOverride),
        fatGauge(fatGauge), longGauge(longGauge), nSrc(in->X(4))
    { 
#ifdef MULTI_GPU
      for(int i=0;i < 4; i++){
        if(comm_dim_partitioned(i) && (fatGauge.X()[i] < 6)){
          errorQuda("ERROR: partitioned dimension with local size less than 6 is not supported in improved staggered dslash\n");
        }
      }
#endif

      bindFatGaugeTex(static_cast<const cudaGaugeField&>(fatGauge), parity, dslashParam);
      bindLongGaugeTex(static_cast<const cudaGaugeField&>(longGauge), parity, dslashParam);

      if (in->Precision() != fatGauge.Precision() || in->Precision() != longGauge.Precision()){
        errorQuda("Mixing gauge and spinor precision not supported"
                  "(precision=%d, fatlinkGauge.precision=%d, longGauge.precision=%d",
                  in->Precision(), fatGauge.Precision(), longGauge.Precision());
      }

      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.fat_link_max = fatGauge.LinkMax();
      dslashParam.coeff = 1.0/longGauge.Scale();
      dslashParam.coeff_f = (float)dslashParam.coeff;
    }

    virtual ~StaggeredDslashCuda() {
      unbindSpinorTex<sFloat>(in, out, x);
      unbindFatGaugeTex(static_cast<const cudaGaugeField&>(fatGauge));
      unbindLongGaugeTex(static_cast<const cudaGaugeField&>(longGauge));
    }

    void apply(const cudaStream_t &stream)
    {
#ifndef USE_TEXTURE_OBJECTS
      if (dslashParam.kernel_type == INTERIOR_KERNEL) bindSpinorTex<sFloat>(in, out, x);
#endif // USE_TEXTURE_OBJECTS
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      setParam();
      dslashParam.gauge_stride = fatGauge.Stride();
      dslashParam.long_gauge_stride = longGauge.Stride();
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

    int Nface() const { return 6; }

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
      bool isFixed = (in->Precision() == sizeof(short) || in->Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isFixed ? sizeof(float) : 0);
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

  void improvedStaggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
				   const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
				   const int parity, const int dagger, const cudaColorSpinorField *x,
				   const double &k, const int *commOverride, TimeProfile &profile)
  {
#if (defined GPU_STAGGERED_DIRAC && defined USE_LEGACY_DSLASH)
    const_cast<cudaColorSpinorField*>(in)->createComms(3);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new StaggeredDslashCuda<double2, double2, double2, double>
        (out, fatGauge, longGauge, in, x, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2, float4, float>
	(out, fatGauge, longGauge, in, x, k, parity, dagger, commOverride);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2, short4, short>
	(out, fatGauge, longGauge, in, x, k, parity, dagger, commOverride);
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
