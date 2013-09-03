#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>
#include <gauge_field.h>
#include <quda_matrix.h>

namespace quda {

  struct Stride {
    const int link;
    const int momentum;
    Stride(int link, int mom) : momentum(mom), link(link) {}
  };


  template<class Cmplx, int N>
  __global__ void updateGaugeField(Cmplx* evenOutput, Cmplx* oddOutput, 
                                   typename RealTypeId<Cmplx>::Type eps,
                                   Cmplx* evenInput, Cmplx* oddInput,
                                   Cmplx* evenMomentum, Cmplx* oddMomentum, Stride strides, int volume){
 
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(mem_idx >= volume) return;
  
    const int halfVolume = volume/2;
  
    Cmplx* inGaugeField;
    Cmplx* outGaugeField;
    Cmplx* momField;
    int odd = 0; 

    if(mem_idx >= halfVolume){
      odd = 1;
      mem_idx -= halfVolume;
      inGaugeField = oddInput;
      outGaugeField = oddOutput;
      momField = oddMomentum;
    }else{
      inGaugeField = evenInput;
      outGaugeField = evenOutput;
      momField = evenMomentum;
    }

    Matrix<Cmplx,3> link, result, mom;



    for(int dir=0; dir<4; ++dir){

      loadLinkVariableFromArray(inGaugeField, dir, mem_idx, strides.link, &link);
      loadMomentumFromArray(momField, dir, mem_idx, strides.momentum, &mom);
      result = link;

      // Nth order expansion of exponential
      for(int r=N; r>0; r--){
        result = (eps/r)*mom*result + link;
      }
      
      writeLinkVariableToArray(result, dir, mem_idx, strides.link, outGaugeField);
      
    } // dir

    return;
  }

  class UpdateGaugeFieldCuda : public Tunable {
    private: 
      const cudaGaugeField& momentum;
      const cudaGaugeField& inGauge;
      const cudaGaugeField* outGauge;
      const double eps;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }

      bool advanceBlockDim(TuneParam &param) const 
      {
        const unsigned int max_threads = deviceProp.maxThreadsDim[0];
        const unsigned int max_blocks = deviceProp.maxGridSize[0];
        const unsigned int max_shared = 16384;
        const int step = deviceProp.warpSize;
        const int threads = inGauge.Volume();
        bool ret;
      
        param.block.x += step;
        if(param.block.x > max_threads || sharedBytesPerThread()*param.block.x > max_shared){
          param.block = dim3((threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough given the limit on gridDim
          param.block.x = ((param.block.x+step-1)/step)*step;
          if(param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
          ret = false;
        }else{
          ret=true;
        }
        param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
        return ret; 
      }

    public:
      UpdateGaugeFieldCuda(cudaGaugeField* const outGauge, double eps, const cudaGaugeField& inGauge, const cudaGaugeField& momentum) : outGauge(outGauge), eps(eps), inGauge(inGauge), momentum(momentum) 
    {}

      virtual ~UpdateGaugeFieldCuda() {}

      void apply(const cudaStream_t &stream){
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
       

        if(inGauge.Precision() == QUDA_SINGLE_PRECISION){

          updateGaugeField<float2,6><<<tp.grid,tp.block>>>((float2*)outGauge->Even_p(), 
              (float2*)outGauge->Odd_p(), 
              (float)eps, 
              (float2*)inGauge.Even_p(),
              (float2*)inGauge.Odd_p(),    
              (float2*)momentum.Even_p(), 
              (float2*)momentum.Odd_p(),
              Stride(inGauge.Stride(), momentum.Stride()),  
              inGauge.Volume());  

        }else if(inGauge.Precision() == QUDA_DOUBLE_PRECISION){

          updateGaugeField<double2,6><<<tp.grid,tp.block>>>((double2*)outGauge->Even_p(), 
              (double2*)outGauge->Odd_p(), 
              eps,       
              (double2*)inGauge.Even_p(),
              (double2*)inGauge.Odd_p(),    
              (double2*)momentum.Even_p(), 
              (double2*)momentum.Odd_p(),
              Stride(inGauge.Stride(), momentum.Stride()),
              inGauge.Volume());  
        } // precision
      } // apply

      void preTune(){}
      void postTune(){}

      virtual void initTuneParam(TuneParam &param) const
      {
        const unsigned int max_threads = deviceProp.maxThreadsDim[0];
        const unsigned int max_blocks = deviceProp.maxGridSize[0];
        const int threads = inGauge.Volume();
        const int step = deviceProp.warpSize;
        param.block = dim3((threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough, given the limit on gridDim
        param.block.x = ((param.block.x+step-1) / step) * step; // round up to the nearest "step"
        if (param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
        param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
        param.shared_bytes = sharedBytesPerThread()*param.block.x > sharedBytesPerBlock(param) ?
          sharedBytesPerThread()*param.block.x : sharedBytesPerBlock(param);
      }

      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const {
        initTuneParam(param);
      }

      long long flops() const { return 0; } // FIXME: add flops counter

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << inGauge.X()[0] << "x";
        vol << inGauge.X()[1] << "x";
        vol << inGauge.X()[2] << "x";
        vol << inGauge.X()[3] << "x";
        aux << "threads=" << inGauge.Volume() << ",prec=" << inGauge.Precision();
        aux << "stride=" << inGauge.Stride();
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }
  };


  void updateGaugeFieldCuda(cudaGaugeField* const outGauge, double eps, const cudaGaugeField& inGauge, const cudaGaugeField& momentum)
  {
    UpdateGaugeFieldCuda updateGauge(outGauge, eps, inGauge, momentum);
    updateGauge.apply(0); 
    checkCudaError();
  }

} // namespace quda
