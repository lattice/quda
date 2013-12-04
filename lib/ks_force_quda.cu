#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <ks_force_quda.h>


namespace quda {

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

  __device__ __host__ inline int linkIndex(int x[], int dx[], const int X[4]) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
    return idx;
  }


  __device__ __host__ inline void getCoords(int x[4], int cb_index, const int X[4], int parity)
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }

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

      typedef typename ComplexTypeId<Float>::Type Cmplx;

      Matrix<Cmplx,3> O;
      Matrix<Cmplx,3> G;
      Matrix<Cmplx,3> M;


      int dx[4] = {0,0,0,0};
      for(int dir=0; dir<4; ++dir){
        arg.gauge.load((Float*)(G.data), linkIndex(x,dx,X), dir, parity); 
        arg.oprod.load((Float*)(O.data), linkIndex(x,dx,X), dir, parity); 
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
    }

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    __global__ void completeKSForceKernel(KSForceArg<Oprod,Gauge,Mom> arg)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;

      if(idx==0){ 
        printf("arg.threads = %d\n", arg.threads);
      }

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
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      KSForceComplete(KSForceArg<Oprod,Gauge,Mom> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}

      virtual ~KSForceComplete() {}

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          // Fix this
          dim3 blockDim(128, 1, 1);
          dim3 gridDim((arg.threads + blockDim.x - 1) / blockDim.x, 1, 1);
          completeKSForceKernel<Float><<<gridDim,blockDim>>>(arg);
        }else{
          completeKSForceCPU<Float>(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.threads;
        aux << "stride=" << arg.mom.stride;
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }

      std::string paramString(const TuneParam &param) const { // Don't print the grid dim.
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }


      long long flops() const { return 0; } // Fix this
      long long bytes() const { return 0; } // Fix this
    };

  template<typename Float, typename Oprod, typename Gauge, typename Mom>
    void completeKSForce(Oprod oprod, Gauge gauge, Mom mom, int dim[4], QudaFieldLocation location)
    {
      KSForceArg<Oprod,Gauge,Mom> arg(oprod, gauge, mom, dim);
      KSForceComplete<Float,Oprod,Gauge,Mom> completeForce(arg,location);
      completeForce.apply(0);
      cudaDeviceSynchronize();
    }

  template<typename Float>
    void completeKSForce(GaugeField& mom, const GaugeField& oprod, const GaugeField& gauge, QudaFieldLocation location)
    {

      if(location != QUDA_CUDA_FIELD_LOCATION){
        errorQuda("Only QUDA_CUDA_FIELD_LOCATION currently supported");
      }else{
        if((oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) || (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) || (mom.Reconstruct() != QUDA_RECONSTRUCT_10)){
          errorQuda("Reconstruct type not supported");
        }else{
          completeKSForce<Float>(FloatNOrder<Float, 18, 2, 18>(oprod),
              FloatNOrder<Float, 18, 2, 18>(gauge),
              FloatNOrder<Float, 18, 2, 10>(mom),
              const_cast<int*>(mom.X()),
              location);
        }
      }
    }


  void completeKSForce(GaugeField &mom, const GaugeField &oprod, const GaugeField &gauge, QudaFieldLocation location){
    if(mom.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported");
    }

    if(mom.Precision() == QUDA_SINGLE_PRECISION){
      completeKSForce<float>(mom, oprod, gauge, location);
    }else if(mom.Precision() == QUDA_DOUBLE_PRECISION){
      completeKSForce<double>(mom, oprod, gauge, location);
    }else{
      errorQuda("Precision %d not supported", mom.Precision());
    }


  }


} // namespace quda
