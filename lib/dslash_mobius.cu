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
#include <dslash_helper.cuh>
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>

#if (__COMPUTE_CAPABILITY__ >= 700)
#include <cublas_v2.h>
#include <mma.h>
#endif

namespace quda {

  namespace mobius {

    template<class T>
    struct MDWFSharedMemory
    {
      __device__ inline operator T*()
      {
        extern __shared__ int __smem[];
        return (T*)__smem;
      }

      __device__ inline operator const T*() const
      {
        extern __shared__ int __smem[];
        return (T*)__smem;
      }
    };

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
#include <mdw_dslash4_dslash5inv_dslash4pre_def.h>   // Dslash5inv Mobius Domain Wall kernels
#include <mdw_dslash4_dslash5inv_xpay_dslash5inv_dagger_def.h>   // Dslash5inv Mobius Domain Wall kernels
#include <mdw_dslash4_dagger_dslash4pre_dagger_dslash5inv_dagger_def.h>
#include <mdw_dslash4_dagger_dslash4pre_dagger_xpay_def.h>
#include <mdw_dslash5inv_def_sm.h>
#include <mdw_dslash5inv_def_sm_tc.h>
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
    const int DS_type;

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
    
    virtual bool tuneGridDim() const {         
      if(DS_type == 9){
        return true;
      }
      return false;
    }
    
    virtual bool advanceGridDim(TuneParam &param) const
    {
      if (tuneGridDim()) {
        const unsigned int max_blocks = maxGridSize();
        const int step = deviceProp.multiProcessorCount/4;
        param.grid.x += step;
        if (param.grid.x > max_blocks) {
          param.grid.x = minGridSize();
          return false;
        } else {
          return true;
        }
      } else {
        return false;
      }
    }

    virtual unsigned int maxGridSize() const { return 32*deviceProp.multiProcessorCount; }

    bool advanceBlockDim(TuneParam &param) const
    {
//      const unsigned int max_shared = deviceProp.sharedMemPerBlock;
      const unsigned int max_shared = deviceProp.major>=7 ? 96*1024 : deviceProp.sharedMemPerBlock;
      const int step[2] = { deviceProp.warpSize, 1 };
//      const int step[2] = { 16, 1 };
      bool advance[2] = { false, false };

      // first try to advance block.x
      param.block.x += step[0];
      if (param.block.x > (unsigned int)deviceProp.maxThreadsDim[0] ||
          shared_bytes_per_block(param.block.x, param.block.y) > max_shared) {
        advance[0] = false;
        param.block.x = step[0]; // reset block.x
      } else {
        advance[0] = true; // successfully advanced block.x
      }
      
      if(DS_type < 4){
        if (!advance[0]) {  // if failed to advance block.x, now try block.y
          param.block.y += step[1];
  
          if (param.block.y > (unsigned)in->X(4) ||
              sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
            advance[1] = false;
            param.block.y = step[1]; // reset block.x
          } else {
            advance[1] = true; // successfully advanced block.y
          }
        }
      }
      
      if (advance[0] || advance[1]) {
        if( param.block.x*param.block.y*param.block.z > (unsigned)deviceProp.maxThreadsPerBlock ){
          return false;
        }
        
        if(DS_type < 9){
          param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			      (in->X(4)+param.block.y-1) / param.block.y, 1);
        }
        param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y);
        
        bool advance = true;
        if (!checkGrid(param)) advance = advanceBlockDim(param);
        return advance;
      } else {
        return false;
      }
    }

		bool advanceSharedBytes(TuneParam &param) const
    {
      if (tuneSharedBytes()) {
//				const int max_shared = deviceProp.sharedMemPerBlock;
				const unsigned int max_shared = deviceProp.major>=7 ? 96*1024 : deviceProp.sharedMemPerBlock;
				const int max_blocks_per_sm = std::min(deviceProp.maxThreadsPerMultiProcessor / (param.block.x*param.block.y*param.block.z), maxBlocksPerSM());
				int blocks_per_sm = max_shared / (param.shared_bytes ? param.shared_bytes : 1);
				if (blocks_per_sm > max_blocks_per_sm) blocks_per_sm = max_blocks_per_sm;
				param.shared_bytes = (blocks_per_sm > 0 ? max_shared / blocks_per_sm + 1 : max_shared + 1);
			
				if ((size_t)param.shared_bytes > max_shared) {
				  TuneParam next(param);
				  advanceBlockDim(next); // to get next blockDim
				  // int nthreads = next.block.x * next.block.y * next.block.z;
				  param.shared_bytes = shared_bytes_per_block(next.block.x,next.block.y) > sharedBytesPerBlock(param) ?
				     shared_bytes_per_block(next.block.x,next.block.y) : sharedBytesPerBlock(param);
				  return false;
				} else {
				  return true;
				}
      } else {
				return false;
      }
    }
    
//    virtual bool advanceTuneParam(TuneParam &param) const
//    {
//      return advanceSharedBytes(param) || advanceBlockDim(param) || advanceGridDim(param) || advanceAux(param);
//    }

    unsigned int sharedBytesPerThread() const { 
      if(DS_type >= 4){
        return 24*(in->Precision()==8?8:4);
      }else{
        return 0;
      }
    }
 
    unsigned int shared_bytes_per_block(int x, int y) const { 
      if(DS_type == 9){
        return ( (y*4)*(y*4+0)+(y*4)*(x*6+16)*1 )*2; // 4*4*2 TODO: fix this!
      }else{
        return sharedBytesPerThread()*x*y;
      }
    }
 
  public:
    MDWFDslashPCCuda(cudaColorSpinorField *out, const GaugeField &gauge, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x, const double mferm, const double a,
                     const double *b_5, const double *c_5, const double m5,
                     const int parity, const int dagger, const int *commOverride, const int DS_type)
      : DslashCuda(out, in, x, gauge, parity, dagger, commOverride), DS_type(DS_type)
    { 
      dslashParam.a = a;
      dslashParam.a_f = a;
      dslashParam.mferm = mferm;
      dslashParam.mferm_f = mferm;

      memcpy(dslashParam.mdwf_b5_d, b_5, out->X(4)*sizeof(double));
      memcpy(dslashParam.mdwf_c5_d, c_5, out->X(4)*sizeof(double));
      for (int s=0; s<out->X(4); s++) {
        dslashParam.mdwf_b5_f[s] = (float)dslashParam.mdwf_b5_d[s];
        dslashParam.mdwf_c5_f[s] = (float)dslashParam.mdwf_c5_d[s];
      }

      dslashParam.m5_d = m5;
      dslashParam.m5_f = (float)m5;
    }
    virtual ~MDWFDslashPCCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      if(dslashParam.partial_length){
        
        char config[256];
        switch(DS_type){
          case 0:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash4,partial%d,%d,%d,%d,expand%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash4,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
          case 1:
            sprintf(config, ",Dslash4pre,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            strcat(key.aux,config);
            break;
          case 2:
            sprintf(config, ",Dslash5,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            strcat(key.aux,config);
            break;
          case 3:
            sprintf(config, ",Dslash5inv,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            strcat(key.aux,config);
          	break;
					case 4:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash4Dslash5invDslash4pre,partial%d,%d,%d,%d,expand%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash4Dslash5invDslash4pre,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
					case 5:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash4Dslash5invXpayDslash5invDagger,partial%d,%d,%d,%d,expand%d,%d,%d,%d", 
								dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash4Dslash5invXpayDslash5invDagger,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
					case 6:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash4DaggerDslash4preDaggerDslash5invDagger,partial%d,%d,%d,%d,expand%d,%d,%d,%d", 
								dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash4DaggerDslash4preDaggerDslash5invDagger,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
          case 7:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash4DaggerDslash4preDaggerXpay,partial%d,%d,%d,%d,expand%d,%d,%d,%d", 
								dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash4DaggerDslash4preDaggerXpay,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
					case 8:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash5invSm,partial%d,%d,%d,%d,expand%d,%d,%d,%d", 
								dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash5invSm,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
					case 9:
            if(dslashParam.expanding){
              sprintf(config, ",Dslash5invSmTc,partial%d,%d,%d,%d,expand%d,%d,%d,%d", 
								dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3],
                dslashParam.Rz[0], dslashParam.Rz[1], dslashParam.Rz[2], dslashParam.Rz[3]);
            }else{
              sprintf(config, ",Dslash5invSmTc,partial%d,%d,%d,%d", dslashParam.R[0], dslashParam.R[1], dslashParam.R[2], dslashParam.R[3]);
            }
            strcat(key.aux,config);
            break;
        }
      
      }else{

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
          case 4:
            strcat(key.aux,",Dslash4Dslash5invDslash4pre");
            break;
					case 5:
            strcat(key.aux,",Dslash4Dslash5invXpayDslash5invDagger");
            break;
					case 6:
            strcat(key.aux,",Dslash4DaggerDslash4preDaggerDslash5invDagger");
            break;
          case 7:
            strcat(key.aux,",Dslash4DaggerDslash4preDaggerXpay");
            break;
					case 8:
            strcat(key.aux,",Dslash5invSm");
            break;
          case 9:
            strcat(key.aux,",Dslash5invSmTc");
            break;
        }
      
      }
      return key;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      if(DS_type >= 4){ 
        // For these kernels, for one 4D-site all corresponding 5D-sites have to be within the same block,
        // since shared memory is used.
        param.block = dim3( param.block.x, in->X(4), 1);
//        param.block = dim3( 16, in->X(4), 1);
      }
      param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y);
        printfQuda( "Shared memory %08lu is larger than limit %08lu?\n", (size_t)param.shared_bytes, (size_t)(deviceProp.major>=7 ? 96*1024 : deviceProp.sharedMemPerBlock) );
//      if( (size_t)param.shared_bytes > (size_t)deviceProp.sharedMemPerBlock ) 
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      if(DS_type == 9){
        param.grid = dim3(80,1,1);
      }
      bool ok = true;
      if (!checkGrid(param)) ok = advanceBlockDim(param);
      if (!ok) errorQuda("Lattice volume is too large for even the largest blockDim");
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      if(DS_type >= 4){ 
        // For these kernels, for one 4D-site all corresponding 5D-sites have to be within the same block,
        // since shared memory is used.
        param.block = dim3( param.block.x, in->X(4), 1);
//        param.block = dim3( 16, in->X(4), 1);
      }
      param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y);
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      if(DS_type == 9){
        param.grid = dim3(80,1,1);
      }
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
          DSLASH(MDWFDslash4, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 1:
          DSLASH(MDWFDslash4pre, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 2:
          DSLASH(MDWFDslash5, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 3:
          DSLASH(MDWFDslash5inv, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 4:
          DSLASH(MDWFDslash4Dslash5invDslash4pre, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
				case 5:
          DSLASH(MDWFDslash4Dslash5invXpayDslash5invDagger, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
				case 6:
          DSLASH(MDWFDslash4DaggerDslash4preDaggerDslash5invDagger, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 7:
          DSLASH(MDWFDslash4DaggerDslash4preDaggerXpay, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
				case 8:
          DSLASH(MDWFDslash5invSm, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        case 9:
          DSLASH(MDWFDslash5invSmTc, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam);
          break;
        default:
          errorQuda("invalid Dslash type");
      }
    }

    long long flops() const {
      long long Ls = in->X(4);
			long long vol4d = 0;
			if( dslashParam.partial_length ){
      	vol4d = dslashParam.partial_length;
			}else{
        vol4d = in->VolumeCB() / Ls;
			}
      long long bulk = (Ls-2)*vol4d;
      long long wall = 2*vol4d;
      long long flops = 0;
      switch(DS_type){
        case 0:
          if( dslashParam.partial_length ){
            flops = 1320ll*dslashParam.partial_length*Ls;
          }else{
            flops = DslashCuda::flops();
          }
          break;
        case 1:
          flops = 72ll*vol4d*Ls + 96ll*bulk + 120ll*wall;
          break;
        case 2:
          flops = (x ? 96ll : 48ll)*vol4d*Ls + 96ll*bulk + 120ll*wall;
          break;
        case 3:
				case 8:
				case 9:
						flops = 144ll*vol4d*Ls*Ls + 3ll*Ls*(Ls-1ll);
					break;
        case 4:
				case 6:
            flops = 1320ll*vol4d*Ls + 144ll*vol4d*Ls*Ls + 3ll*Ls*(Ls-1ll) + 72ll*vol4d*Ls + 96ll*bulk + 120ll*wall;
					break;
				case 5:
            flops = (x?1368ll:1320ll)*vol4d*Ls + 144ll*vol4d*Ls*Ls + 3ll*Ls*(Ls-1ll);
          break;
        case 7:
            flops = (x?1368ll:1320ll)*vol4d*Ls + 72ll*vol4d*Ls + 96ll*bulk + 120ll*wall;
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
        case 4:
				case 5:
				case 6:
        case 7:
          if( dslashParam.partial_length ){
            bytes = (x?16ll:15ll)*spinor_bytes*(long long)dslashParam.partial_length*Ls;
          }else{
            bytes = DslashCuda::bytes();
          }
          break;
        case 1:
        case 2:
          bytes = (x ? 5ll : 4ll) * spinor_bytes * in->VolumeCB();
          break;
        case 3:
				case 8:
        case 9:
          bytes = (x ? Ls + 2 : Ls + 1) * spinor_bytes * in->VolumeCB();
          break;
        default:
          errorQuda("invalid Dslash type");
      }
      return bytes;
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
                      const double *b_5, const double *c_5, const double &m5,
		      const int *commOverride, const int DS_type, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new MDWFDslashPCCuda<double2,double2>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new MDWFDslashPCCuda<float4,float4>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new MDWFDslashPCCuda<short4,short4>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    DslashPolicyImp<DslashCuda>* dslashImp = nullptr;
    if (DS_type != 0) {
      dslashImp = DslashFactory<DslashCuda>::create(QudaDslashPolicy::QUDA_DSLASH_NC);
      (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), in->Volume()/in->X(4), ghostFace, profile);
      delete dslashImp;
    } else {
      DslashPolicyTune<DslashCuda> dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), in->Volume()/in->X(4), ghostFace, profile);
      dslash_policy.apply(0);
    }

    delete dslash;
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }
 
	void set_shared_memory_on_volta(const void* f, const char* name){
			cudaDeviceProp device_prop;
			cudaGetDeviceProperties( &device_prop, 0 );
			if(device_prop.major < 7) return;
			
			auto found = qudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
			printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));
			
			found = qudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
			printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));
			
			cudaFuncAttributes cfa;
			found = cudaFuncGetAttributes(&cfa, f);
			printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));
			
			printfQuda("Actual maximum:         %d\n", (int)cfa.maxDynamicSharedSizeBytes);
			printfQuda("Actual maximum percent: %d\n", (int)cfa.preferredShmemCarveout);
	}

  void mdwf_dslash_cuda_partial(cudaColorSpinorField *out, const cudaGaugeField &gauge,
		      const cudaColorSpinorField *in, const int parity, const int dagger,
		      const cudaColorSpinorField *x, const double &m_f, const double &k2,
                      const double *b_5, const double *c_5, const double &m5,
		      const int *commOverride, const int DS_type, TimeProfile &profile, int sp_idx_length, int R_[4], int_fastdiv Xs_[4],
          bool expanding_, std::array<int,4> Rz_)
  {
		static bool init = false;
#ifdef GPU_DOMAIN_WALL_DIRAC
    const_cast<cudaColorSpinorField*>(in)->createComms(1);

//    if(DS_type == 9){
//      cudaDeviceProp device_prop;
//      cudaGetDeviceProperties( &device_prop, 0 );
//      if(device_prop.major < 7 || in->Precision() != QUDA_HALF_PRECISION){
//        errorQuda("Your are either NOT rich enough to buy a Volta or TOO rich to buy a Volta.\n");
//      }
//    }

		if(!init){
			set_shared_memory_on_volta((const void*)MDWFDslash4Dslash5invDslash4preH18Kernel<INTERIOR_KERNEL>, 
				"MDWFDslash4Dslash5invDslash4preH18Kernel<INTERIOR_KERNEL>");
			set_shared_memory_on_volta((const void*)MDWFDslash4Dslash5invXpayDslash5invDaggerH18XpayKernel<INTERIOR_KERNEL>, 
				"MDWFDslash4Dslash5invXpayDslash5invDaggerH18XpayKernel<INTERIOR_KERNEL>");
			set_shared_memory_on_volta((const void*)MDWFDslash4DaggerDslash4preDaggerDslash5invDaggerH18Kernel<INTERIOR_KERNEL>, 
				"MDWFDslash4DaggerDslash4preDaggerDslash5invDaggerH18Kernel<INTERIOR_KERNEL>");
      
      set_shared_memory_on_volta((const void*)MDWFDslash5invSmTcH18DaggerKernel<INTERIOR_KERNEL>, 
				"MDWFDslash5invSmTcH18DaggerKernel<INTERIOR_KERNEL>");
      set_shared_memory_on_volta((const void*)MDWFDslash5invSmH18DaggerKernel<INTERIOR_KERNEL>, 
				"MDWFDslash5invSmH18DaggerKernel<INTERIOR_KERNEL>");
			
      set_shared_memory_on_volta((const void*)MDWFDslash4DaggerDslash4preDaggerXpayH18XpayKernel<INTERIOR_KERNEL>, 
				"MDWFDslash4DaggerDslash4preDaggerXpayH18XpayKernel<INTERIOR_KERNEL>");
			init = true;
		  // cudaFuncSetSharedMemConfig((const void*)MDWFDslash5invSmTcH18DaggerKernel<INTERIOR_KERNEL>, cudaSharedMemBankSizeEightByte);
    }

    DslashCuda *dslash = nullptr;
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new MDWFDslashPCCuda<double2,double2>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new MDWFDslashPCCuda<float4,float4>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new MDWFDslashPCCuda<short4,short4>(out, gauge, in, x, m_f, k2, b_5, c_5, m5, parity, dagger, commOverride, DS_type);
    }

    dslash->dslashParam.partial_length = sp_idx_length;
    dslash->dslashParam.R[0] = R_[0];
    dslash->dslashParam.R[1] = R_[1];
    dslash->dslashParam.R[2] = R_[2];
    dslash->dslashParam.R[3] = R_[3];

    dslash->dslashParam.Xs[0] = Xs_[0];
    dslash->dslashParam.Xs[1] = Xs_[1];
    dslash->dslashParam.Xs[2] = Xs_[2];
    dslash->dslashParam.Xs[3] = Xs_[3];

//    printfQuda("volume: %dx%dx%dx%d; R: %dx%dx%dx%d; partial_length=%d.\n", 
//                                                               int(dslash->dslashParam.Xs[0]),
//                                                               int(dslash->dslashParam.Xs[1]),
//                                                               int(dslash->dslashParam.Xs[2]),
//                                                               int(dslash->dslashParam.Xs[3]), 
//                                                               int(dslash->dslashParam.R[0]), 
//                                                               int(dslash->dslashParam.R[1]), 
//                                                               int(dslash->dslashParam.R[2]), 
//                                                               int(dslash->dslashParam.R[3]), 
//                                                               sp_idx_length);
    
    if(expanding_){
      dslash->dslashParam.expanding = true;
      dslash->dslashParam.Rz[0] = Rz_[0];
      dslash->dslashParam.Rz[1] = Rz_[1];
      dslash->dslashParam.Rz[2] = Rz_[2];
      dslash->dslashParam.Rz[3] = Rz_[3];
    } 

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);

    DslashPolicyImp<DslashCuda>* dslashImp = nullptr;
    if (DS_type != 0) {
      dslashImp = DslashFactory<DslashCuda>::create(QudaDslashPolicy::QUDA_DSLASH_NC);
      (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), sp_idx_length, ghostFace, profile);
      delete dslashImp;
    } else {
      DslashPolicyTune<DslashCuda> dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), sp_idx_length, ghostFace, profile);
      dslash_policy.apply(0);
    }
    // sp_idx_length is the param.threads

    delete dslash;
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }
}
