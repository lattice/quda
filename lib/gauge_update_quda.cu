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
  __global__ void updateGaugeField(Cmplx* evenGauge, Cmplx* oddGauge, 
                                   typename RealTypeId<Cmplx>::Type eps,
                                   Cmplx* evenMomentum, Cmplx* oddMomentum, Stride strides, int volume){
 
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(mem_idx >= volume) return;
  
    const int halfVolume = volume/2;
  
    Cmplx* gaugeField;
    Cmplx* momField;
 
    if(mem_idx >= halfVolume){
      mem_idx -= halfVolume;
      gaugeField = oddGauge;
      momField = oddMomentum;
    }else{
      gaugeField = evenGauge;
      momField = evenMomentum;
    }

    Matrix<Cmplx,3> link, result, mom;

    for(int dir=0; dir<4; ++dir){

      loadLinkVariableFromArray(gaugeField, dir, mem_idx, strides.link, &link);
      loadMomentumFromArray(momField, dir, mem_idx, strides.momentum, &mom);
      result = link;
      // Nth order expansion of exponential
      for(int r=N; r>0; r--){
        result = (eps/r)*mom*result + link;
      }
      writeLinkVariableToArray(result, dir, mem_idx, strides.link, gaugeField);
      
    } // dir

    return;
  }

  class UpdateGaugeFieldCuda : public Tunable {
    private: 
      const cudaGaugeField& momentum;
      cudaGaugeField* gauge;
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
        const int threads = gauge->Volume();
        bool ret;

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
      UpdateGaugeFieldCuda(cudaGaugeField* const gauge, double eps, const cudaGaugeField& momentum) : gauge(gauge), eps(eps), momentum(momentum) 
    {}

      virtual ~UpdateGaugeFieldCuda() {}

      void apply(const cudaStream_t &stream){
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        

        if(gauge->Precision() == QUDA_SINGLE_PRECISION){

          updateGaugeField<float2,6><<<tp.grid,tp.block>>>((float2*)gauge->Even_p(), 
              (float2*)gauge->Odd_p(), 
              (float)eps,     
              (float2*)momentum.Even_p(), 
              (float2*)momentum.Odd_p(),
              Stride(gauge->Stride(), momentum.Stride()),  
              gauge->Volume());  

        }else if(gauge->Precision() == QUDA_DOUBLE_PRECISION){

          updateGaugeField<double2,6><<<tp.grid,tp.block>>>((double2*)gauge->Even_p(), 
              (double2*)gauge->Odd_p(), 
              eps,       
              (double2*)momentum.Even_p(), 
              (double2*)momentum.Odd_p(),
              Stride(gauge->Stride(), momentum.Stride()),
              gauge->Volume());  
        } // precision
      } // apply

      void preTune(){}
      void postTune(){}

      virtual void initTuneParam(TuneParam &param) const
      {
        const unsigned int max_threads = deviceProp.maxThreadsDim[0];
        const unsigned int max_blocks = deviceProp.maxGridSize[0];
        const int threads = gauge->Volume();
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
        vol << gauge->X()[0] << "x";
        vol << gauge->X()[1] << "x";
        vol << gauge->X()[2] << "x";
        vol << gauge->X()[3] << "x";
        aux << "threads=" << gauge->Volume() << ",prec=" << gauge->Precision();
        aux << "stride=" << gauge->Stride();
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }
  };


  void updateGaugeFieldCuda(cudaGaugeField* const gauge, double eps, const cudaGaugeField& momentum)
  {
    UpdateGaugeFieldCuda updateGauge(gauge, eps, momentum);
    updateGauge.apply(0); 
    checkCudaError();
  }

} // namespace quda
