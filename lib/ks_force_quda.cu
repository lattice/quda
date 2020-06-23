#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <ks_force_quda.h>
#include <index_helper.cuh>

namespace quda {

  using namespace gauge;

  template<typename Oprod, typename Gauge, typename Mom>
    struct KSForceArg {
      int threads;
      int X[4]; // grid dimensions
#ifndef BUILD_TIFR_INTERFACE
#ifdef MULTI_GPU
      int border[4];
#endif
#endif
      Oprod oprod;
      Gauge gauge;
      Mom mom;

      KSForceArg(Oprod& oprod, Gauge &gauge, Mom& mom, int dim[4])
        : oprod(oprod), gauge(gauge), mom(mom){

          threads = 1;
          for(int dir=0; dir<4; ++dir) threads *= dim[dir];

          for(int dir=0; dir<4; ++dir) X[dir] = dim[dir];
#ifndef BUILD_TIFR_INTERFACE
#ifdef MULTI_GPU
          for(int dir=0; dir<4; ++dir) border[dir] = 2;
#endif
#endif
        }

    };

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    __host__ __device__ void completeKSForceCore(KSForceArg<Oprod,Gauge,Mom>& arg, int idx){

      int parity = 0;
      if(idx >= arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }

      int X[4];
      for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

      int x[4];
      getCoords(x, idx, X, parity);
#ifndef BUILD_TIFR_INTERFACE
#ifdef MULTI_GPU
      for(int dir=0; dir<4; ++dir){
        x[dir] += arg.border[dir];
        X[dir] += 2*arg.border[dir];
      }
#endif
#endif

      Matrix<complex<Float>,3> O, G, M;

      int dx[4] = {0,0,0,0};
      for(int dir=0; dir<4; ++dir){
        G = arg.gauge(dir, linkIndexShift(x,dx,X), parity);
        O = arg.oprod(dir, linkIndexShift(x,dx,X), parity);
        if(parity==0){
          M = G*O;
        }else{
          M = -G*O;
        }

        makeAntiHerm(M);

        arg.mom(dir, idx, parity) = M;
      }
    }

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    __global__ void completeKSForceKernel(KSForceArg<Oprod,Gauge,Mom> arg)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;

      if(idx >= arg.threads) return;
      completeKSForceCore<Float,Oprod,Gauge,Mom>(arg,idx);
    }

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    void completeKSForceCPU(KSForceArg<Oprod,Gauge,Mom>& arg)
    {
      for(int idx=0; idx<arg.threads; idx++){
        completeKSForceCore<Float,Oprod,Gauge,Mom>(arg,idx);
      }
    }

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    class KSForceComplete : Tunable {

      KSForceArg<Oprod, Gauge, Mom> arg;
      const GaugeField &meta;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      KSForceComplete(KSForceArg<Oprod,Gauge,Mom> &arg, const GaugeField &meta, QudaFieldLocation location)
        : arg(arg), meta(meta), location(location) {
	writeAuxString("prec=%lu,stride=%d",sizeof(Float),arg.mom.stride);
      }

      virtual ~KSForceComplete() {}

      void apply(const qudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          // Fix this
          dim3 blockDim(128, 1, 1);
          dim3 gridDim((arg.threads + blockDim.x - 1) / blockDim.x, 1, 1);
          completeKSForceKernel<Float><<<gridDim,blockDim>>>(arg);
        }else{
          completeKSForceCPU<Float>(arg);
        }
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 792*arg.X[0]*arg.X[1]*arg.X[2]*arg.X[3]; }
      long long bytes() const { return 0; } // Fix this
    };

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
  void completeKSForce(Oprod oprod, Gauge gauge, Mom mom, int dim[4], const GaugeField &meta, QudaFieldLocation location, long long *flops)
    {
      KSForceArg<Oprod,Gauge,Mom> arg(oprod, gauge, mom, dim);
      KSForceComplete<Float,Oprod,Gauge,Mom> completeForce(arg,meta,location);
      completeForce.apply(0);
      if(flops) *flops = completeForce.flops();
      qudaDeviceSynchronize();
    }


  template<typename Float>
    void completeKSForce(GaugeField& mom, const GaugeField& oprod, const GaugeField& gauge, QudaFieldLocation location, long long *flops)
    {

      if(location != QUDA_CUDA_FIELD_LOCATION){
        errorQuda("Only QUDA_CUDA_FIELD_LOCATION currently supported");
      }else{
        if((oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) || (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) || (mom.Reconstruct() != QUDA_RECONSTRUCT_10)){
          errorQuda("Reconstruct type not supported");
        }else{
          completeKSForce<Float>(FloatNOrder<Float, 18, 2, 18>(oprod),
				 FloatNOrder<Float, 18, 2, 18>(gauge),
				 FloatNOrder<Float, 10, 2, 10>(mom),
				 const_cast<int*>(mom.X()),
				 gauge, location, flops);
        }
      }
      return;
    }


  void completeKSForce(GaugeField &mom, const GaugeField &oprod, const GaugeField &gauge, QudaFieldLocation location, long long *flops)
  {
    if(mom.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported");
    }

    if(mom.Precision() == QUDA_SINGLE_PRECISION){
      completeKSForce<float>(mom, oprod, gauge, location, flops);
    }else if(mom.Precision() == QUDA_DOUBLE_PRECISION){
      completeKSForce<double>(mom, oprod, gauge, location, flops);
    }else{
      errorQuda("Precision %d not supported", mom.Precision());
    }
    return;
  }




  template<typename Result, typename Oprod, typename Gauge>
    struct KSLongLinkArg {
      int threads;
      int X[4]; // grid dimensions
#ifdef MULTI_GPU
      int border[4];
#endif
      double coeff;
      Result res;
      Oprod oprod;
      Gauge gauge;

      KSLongLinkArg(Result& res, Oprod& oprod, Gauge &gauge, int dim[4])
        : coeff(1.0), res(res), oprod(oprod), gauge(gauge){

          threads = 1;
#ifdef MULTI_GPU
          for(int dir=0; dir<4; ++dir) threads *= (dim[dir]-2);
          for(int dir=0; dir<4; ++dir) X[dir] = dim[dir]-2;
          for(int dir=0; dir<4; ++dir) border[dir] = 2;
#else
          for(int dir=0; dir<4; ++dir) threads *= dim[dir];
          for(int dir=0; dir<4; ++dir) X[dir] = dim[dir];
#endif
        }

    };



  template<typename Float, typename Result, typename Oprod, typename Gauge>
    __host__ __device__ void computeKSLongLinkForceCore(KSLongLinkArg<Result,Oprod,Gauge>& arg, int idx){

      /*
         int parity = 0;
         if(idx >= arg.threads/2){
         parity = 1;
         idx -= arg.threads/2;
         }

         int X[4];
         for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

         int x[4];
         getCoords(x, idx, X, parity);
#ifndef BUILD_TIFR_INTERFACE
#ifdef MULTI_GPU
for(int dir=0; dir<4; ++dir){
x[dir] += arg.border[dir];
X[dir] += 2*arg.border[dir];
}
#endif
#endif

typedef complex<Float> Cmplx;

Matrix<Cmplx,3> O;
Matrix<Cmplx,3> G;
Matrix<Cmplx,3> M;


int dx[4] = {0,0,0,0};
for(int dir=0; dir<4; ++dir){
arg.gauge.load((Float*)(G.data), linkIndexShift(x,dx,X), dir, parity);
arg.oprod.load((Float*)(O.data), linkIndexShift(x,dx,X), dir, parity);
if(parity==0){
M = G*O;
}else{
M = -G*O;
}

Float sub = getTrace(M).y/(static_cast<Float>(3));
Float temp[10];


temp[0] = (M.data[1].x - M.data[3].x)*0.5;
temp[1] = (M.data[1].y + M.data[3].y)*0.5;

temp[2] = (M.data[2].x - M.data[6].x)*0.5;
temp[3] = (M.data[2].y + M.data[6].y)*0.5;

temp[4] = (M.data[5].x - M.data[7].x)*0.5;
temp[5] = (M.data[5].y + M.data[7].y)*0.5;

temp[6] = (M.data[0].y-sub);
temp[7] = (M.data[4].y-sub);
temp[8] = (M.data[8].y-sub);
temp[9] = 0.0;

arg.mom.save(temp, idx, dir, parity);
}
       */
    }

  template<typename Float, typename Result, typename Oprod, typename Gauge>
__global__ void computeKSLongLinkForceKernel(KSLongLinkArg<Result,Oprod,Gauge> arg)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx >= arg.threads) return;
  computeKSLongLinkForceCore<Float,Result,Oprod,Gauge>(arg,idx);
}




  template<typename Float, typename Result, typename Oprod, typename Gauge>
void computeKSLongLinkForceCPU(KSLongLinkArg<Result,Oprod,Gauge>& arg)
{
  for(int idx=0; idx<arg.threads; idx++){
    computeKSLongLinkForceCore<Float,Result,Oprod,Gauge>(arg,idx);
  }
}



// should be tunable
template<typename Float, typename Result, typename Oprod, typename Gauge>
class KSLongLinkForce : Tunable {


  KSLongLinkArg<Result,Oprod,Gauge> arg;
  const GaugeField &meta;
  const QudaFieldLocation location;

  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  KSLongLinkForce(KSLongLinkArg<Result,Oprod,Gauge> &arg, const GaugeField &meta, QudaFieldLocation location)
    : arg(arg), meta(meta), location(location) {
    writeAuxString("prec=%lu,stride=%d",sizeof(Float),arg.res.stride);
  }

  virtual ~KSLongLinkForce() {}

  void apply(const qudaStream_t &stream) {
    if(location == QUDA_CUDA_FIELD_LOCATION){
      // Fix this
      dim3 blockDim(128, 1, 1);
      dim3 gridDim((arg.threads + blockDim.x - 1) / blockDim.x, 1, 1);
      computeKSLongLinkForceKernel<Float><<<gridDim,blockDim>>>(arg);
    }else{
      computeKSLongLinkForceCPU<Float>(arg);
    }
  }

  TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

  long long flops() const { return 0; } // Fix this
  long long bytes() const { return 0; } // Fix this
};




template<typename Float, typename Result, typename Oprod, typename Gauge>
void computeKSLongLinkForce(Result res, Oprod oprod, Gauge gauge, int dim[4], const GaugeField &meta, QudaFieldLocation location)
{
  KSLongLinkArg<Result,Oprod,Gauge> arg(res, oprod, gauge, dim);
  KSLongLinkForce<Float,Result,Oprod,Gauge> computeLongLink(arg,meta,location);
  computeLongLink.apply(0);
  qudaDeviceSynchronize();
}

  template<typename Float>
void computeKSLongLinkForce(GaugeField& result, const GaugeField &oprod, const GaugeField &gauge, QudaFieldLocation location)
{
  if(location != QUDA_CUDA_FIELD_LOCATION){
    errorQuda("Only QUDA_CUDA_FIELD_LOCATION currently supported");
  }else{
    if((oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) || (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) ||
        (result.Reconstruct() != QUDA_RECONSTRUCT_10)){

      errorQuda("Reconstruct type not supported");
    }else{
      computeKSLongLinkForce<Float>(FloatNOrder<Float, 18, 2, 18>(result),
				    FloatNOrder<Float, 18, 2, 18>(oprod),
				    FloatNOrder<Float, 18, 2, 18>(gauge),
				    const_cast<int*>(result.X()),
				    gauge, location);
    }
  }
  return;
}


void computeKSLongLinkForce(GaugeField &result, const GaugeField &oprod, const GaugeField &gauge, QudaFieldLocation location)
{
  if(result.Precision() == QUDA_HALF_PRECISION){
    errorQuda("Half precision not supported");
  }

  if(result.Precision() == QUDA_SINGLE_PRECISION){
    computeKSLongLinkForce<float>(result, oprod, gauge, location);
  }else if(result.Precision() == QUDA_DOUBLE_PRECISION){
    computeKSLongLinkForce<double>(result, oprod, gauge, location);
  }
  errorQuda("Precision %d not supported", result.Precision());
  return;
}

} // namespace quda
