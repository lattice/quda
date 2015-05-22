#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>

#include <hisq_links_quda.h> //reunit gauge links!!!!!

#include <comm_quda.h>


#include <gauge_fix_ovr_extra.h>


#include <gauge_fix_ovr_hit_devf.cuh>


namespace quda {

#ifdef GPU_GAUGE_ALG

static int numParams = 18;

#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, arg, parity, ...)     \
  if(tp.block.z==0){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<0, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:                \
    kernel<0, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:                \
    kernel<0, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<0, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==1){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<1, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:                \
    kernel<1, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:                \
    kernel<1, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<1, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==2){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<2, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:                \
    kernel<2, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:                \
    kernel<2, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<2, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==3){\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<3, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 256:                \
    kernel<3, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 384:                \
    kernel<3, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:               \
    kernel<3, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 640:               \
    kernel<3, 160,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:               \
    kernel<3, 192,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 896:               \
    kernel<3, 224,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<3, 256,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==4){\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<4, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 256:                \
    kernel<4, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 384:                \
    kernel<4, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:               \
    kernel<4, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 640:               \
    kernel<4, 160,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:               \
    kernel<4, 192,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 896:               \
    kernel<4, 224,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<4, 256,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==5){\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<5, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 256:                \
    kernel<5, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 384:                \
    kernel<5, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:               \
    kernel<5, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 640:               \
    kernel<5, 160,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:               \
    kernel<5, 192,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 896:               \
    kernel<5, 224,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<5, 256,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else{\
    errorQuda("Not implemented for %d", tp.block.z);\
  }



template<class T>
__device__ __host__ inline Matrix<T,3> getSubTraceUnit(const Matrix<T,3>& a){
  T tr = (a(0,0) + a(1,1) + a(2,2)) / 3.0;
  Matrix<T,3> res;
  res(0,0) = a(0,0)- tr; res(0,1) = a(0,1); res(0,2) = a(0,2);
  res(1,0) = a(1,0); res(1,1) = a(1,1)-tr; res(1,2) = a(1,2);
  res(2,0) = a(2,0); res(2,1) = a(2,1); res(2,2) = a(2,2)-tr;
  return res;
}

template<class T>
__device__ __host__ inline void SubTraceUnit(Matrix<T,3>& a){
  T tr = (a(0,0) + a(1,1) + a(2,2)) / 3.0;
  a(0,0)-= tr; a(1,1) -= tr; a(2,2) -= tr;
}

template<class T>
__device__ __host__ inline double getRealTraceUVdagger(const Matrix<T,3>& a, const Matrix<T,3>& b){
   double sum = (double)(a(0,0).x * b(0,0).x  + a(0,0).y * b(0,0).y);
   sum += (double)(a(0,1).x * b(0,1).x  + a(0,1).y * b(0,1).y);
   sum += (double)(a(0,2).x * b(0,2).x  + a(0,2).y * b(0,2).y);
   sum += (double)(a(1,0).x * b(1,0).x  + a(1,0).y * b(1,0).y);
   sum += (double)(a(1,1).x * b(1,1).x  + a(1,1).y * b(1,1).y);
   sum += (double)(a(1,2).x * b(1,2).x  + a(1,2).y * b(1,2).y);
   sum += (double)(a(2,0).x * b(2,0).x  + a(2,0).y * b(2,0).y);
   sum += (double)(a(2,1).x * b(2,1).x  + a(2,1).y * b(2,1).y);
   sum += (double)(a(2,2).x * b(2,2).x  + a(2,2).y * b(2,2).y);
  return sum;
}



template <typename T>
struct Summ {
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b){
        return a + b;
    }
};
template <>
struct Summ<double2>{
    __host__ __device__ __forceinline__ double2 operator()(const double2 &a, const double2 &b){
        return make_double2(a.x+b.x, a.y+b.y);
    }
};




static __device__ __host__ inline int linkIndex3(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndex(int x[], const int X[4]) {
  int idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndexM1(int x[], const int X[4], const int mu) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = (y[mu] -1 + X[mu]) % X[mu];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}


static __device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) {
  /*x[3] = cb_index/(X[2]*X[1]*X[0]/2);
  x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  x[1] = (cb_index/(X[0]/2)) % X[1];
  x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);*/
  int za = (cb_index / (X[0]/2));
  int zb =  (za / X[1]);
  x[1] = za - zb * X[1];
  x[3] = (zb / X[2]);
  x[2] = zb - x[3] * X[2];
  int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = (2 * cb_index + x1odd)  - za * X[0];
  return;
}








template <typename Gauge>
struct GaugeFixQualityArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  double2 *quality;
  double2 *quality_h;
  GaugeFixQualityArg(const Gauge &dataOr, const cudaGaugeField &data)
    : dataOr(dataOr) {
    //: dataOr(dataOr), quality_h(static_cast<double2*>(pinned_malloc(sizeof(double2)))) {

    for(int dir=0; dir<4; ++dir){
      X[dir] = data.X()[dir] - data.R()[dir]*2;
      #ifdef MULTI_GPU
      border[dir] = data.R()[dir];
      #endif
    }
    threads = X[0]*X[1]*X[2]*X[3];
    quality = (double2*)device_malloc(sizeof(double2));
    quality_h = (double2*)safe_malloc(sizeof(double2));
    //cudaHostGetDevicePointer(&quality, quality_h, 0);
  }
  double getAction(){return quality_h[0].x;}
  double getTheta(){return quality_h[0].y;}
};



template<int blockSize, typename Float, typename Gauge, int gauge_dir>
__global__ void computeFix_quality(GaugeFixQualityArg<Gauge> argQ){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  typedef cub::BlockReduce<double2, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  
  //AVOID SHAREDMEM PROBLEMS!!!!!!! cub::BlockReduce<double2, blockSize> not initialize memory with 0? 
  double2 data = make_double2(0.0,0.0);
  if(idx < argQ.threads) {
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    int parity = 0;
    if(idx >= argQ.threads/2) {
      parity = 1;
      idx -= argQ.threads/2;
    }
    int X[4]; 
    #pragma unroll
    for(int dr=0; dr<4; ++dr) X[dr] = argQ.X[dr];

    int x[4];
    getCoords3(x, idx, X, parity);
#ifdef MULTI_GPU
    #pragma unroll
    for(int dr=0; dr<4; ++dr) {
         x[dr] += argQ.border[dr];
         X[dr] += 2*argQ.border[dr];
    }
#endif
    Matrix<Cmplx,3> delta;
    setZero(&delta);
    idx = linkIndex(x,X);
  
    for (int mu = 0; mu < gauge_dir; mu++) { 
      Matrix<Cmplx,3> U; 
      argQ.dataOr.load((Float*)(U.data),idx, mu, parity);
      delta -= U;
    }
    //18*gauge_dir
    data.x = -delta(0,0).x - delta(1,1).x - delta(2,2).x ;
    //2
    for (int mu = 0; mu < gauge_dir; mu++) {
      Matrix<Cmplx,3> U; 
      argQ.dataOr.load((Float*)(U.data),linkIndexM1(x,X,mu), mu, 1 - parity);
      delta += U;
    }
    //18*gauge_dir
    delta -= conj(delta);
    //18
    SubTraceUnit(delta);
    //12
    data.y = getRealTraceUVdagger(delta, delta);
    //35
    //T=36*gauge_dir+65
  }
  //This must be here for the case when the total number of threads is not multiple of blocksize!!!!
  //HOW TO pre-initialize temp_storage to 0?
  double2 aggregate = BlockReduce(temp_storage).Reduce(data, Summ<double2>());
  if (threadIdx.x == 0) atomicAdd(argQ.quality, aggregate);
}



template<typename Float, typename Gauge, int gauge_dir>
class GaugeFixQuality : Tunable {
  GaugeFixQualityArg<Gauge> argQ;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return argQ.threads; }

  public:
  GaugeFixQuality(GaugeFixQualityArg<Gauge> &argQ)
    : argQ(argQ) {}
  ~GaugeFixQuality () { host_free(argQ.quality_h);device_free(argQ.quality);}

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      //argQ.quality_h[0] = make_double2(0.0,0.0);
      cudaMemset(argQ.quality, 0, sizeof(double2));
      LAUNCH_KERNEL(computeFix_quality, tp, stream, argQ, Float, Gauge, gauge_dir);
      cudaMemcpy(argQ.quality_h, argQ.quality, sizeof(double2), cudaMemcpyDeviceToHost);
      //cudaDeviceSynchronize();
      #ifdef MULTI_GPU        
        if(comm_size() != 1) comm_allreduce_array((double*)argQ.quality_h, 2);
        const int nNodes = comm_dim(0)*comm_dim(1)*comm_dim(2)*comm_dim(3);
        argQ.quality_h[0].x  /= (double)(3*gauge_dir*argQ.threads*nNodes);
        argQ.quality_h[0].y  /= (double)(3*argQ.threads*nNodes);
      #else
        argQ.quality_h[0].x  /= (double)(3*gauge_dir*argQ.threads);
        argQ.quality_h[0].y  /= (double)(3*argQ.threads);
      #endif
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << argQ.X[0] << "x";
    vol << argQ.X[1] << "x";
    vol << argQ.X[2] << "x";
    vol << argQ.X[3];
    sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",argQ.threads, sizeof(Float),gauge_dir);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
    
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune(){}
  void postTune(){}
  long long flops() const { return (36LL*gauge_dir+65LL)*argQ.threads; }// Only correct if there is no link reconstruction, no cub reduction accounted also
  //long long bytes() const { return (1)*2*gauge_dir*argQ.dataOr.Bytes(); }//no accounting the reduction!!!! argQ.dataOr.Bytes() return 0....
  long long bytes() const { return 2LL*gauge_dir*argQ.threads*numParams*sizeof(Float); }//no accounting the reduction!!!!

}; 


//  template <typename Float, typename Gauge>
template <typename Float, typename Gauge>
struct GaugeFixArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  cudaGaugeField &data;
  const Float relax_boost;

  GaugeFixArg(Gauge &dataOr, cudaGaugeField &data, const Float relax_boost)
    : dataOr(dataOr), data(data), relax_boost(relax_boost) {

    for(int dir=0; dir<4; ++dir){
      X[dir] = data.X()[dir] - data.R()[dir]*2;
      #ifdef MULTI_GPU
      border[dir] = data.R()[dir];
      #endif
    }
    threads = X[0]*X[1]*X[2]*X[3] >> 1;
  }
};





template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
__global__ void computeFix(GaugeFixArg<Float, Gauge> arg, int parity){
  int tid = (threadIdx.x + blockSize) % blockSize;  
  int idx = blockIdx.x * blockSize + tid;

  if(idx >= arg.threads) return;

  typedef typename ComplexTypeId<Float>::Type Cmplx;

  if(ImplementationType<3){
    int X[4]; 
    #pragma unroll
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords3(x, idx, X, parity);
  #ifdef MULTI_GPU
    #pragma unroll
    for(int dr=0; dr<4; ++dr) {
         x[dr] += arg.border[dr];
         X[dr] += 2*arg.border[dr];
    }
  #endif
    int mu = (threadIdx.x / blockSize);
    int oddbit = parity;
    if(threadIdx.x >= blockSize * 4){
      mu -= 4;
      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      oddbit = 1 - parity;
    }
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, oddbit);
    if(ImplementationType==0) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==1) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==2)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    arg.dataOr.save((Float*)(link.data),idx, mu, oddbit);
  }
  else{
    int X[4]; 
    #pragma unroll
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords3(x, idx, X, parity);
  #ifdef MULTI_GPU
    #pragma unroll
    for(int dr=0; dr<4; ++dr) {
         x[dr] += arg.border[dr];
         X[dr] += 2*arg.border[dr];
    }
  #endif
    int mu = (threadIdx.x / blockSize);
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, parity);


    x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
    int idx1 = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link1;
    arg.dataOr.load((Float*)(link1.data),idx1, mu, 1-parity);

    if(ImplementationType==3) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==4) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==5)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

    arg.dataOr.save((Float*)(link.data),idx, mu, parity);
    arg.dataOr.save((Float*)(link1.data),idx1, mu, 1-parity);

  }
}






template<typename Float, typename Gauge, int gauge_dir>
class GaugeFix : Tunable {
  GaugeFixArg<Float, Gauge> arg;
  int parity;
  mutable char aux_string[128]; // used as a label in the autotuner
protected:

  dim3 createGrid   (const dim3 &block) const {
    unsigned int blockx = block.x / 8;
    if(block.z > 2) blockx = block.x / 4;
    unsigned int  gx  = (arg.threads + blockx - 1) / blockx;
    return  dim3(gx, 1, 1);
  }
  bool advanceBlockDim  (TuneParam &param) const {
    //Use param.block.z to tune and save state for best kernel option
    // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
    const unsigned int min_threads0 = 32 * 8;
    const unsigned int min_threads1 = 32 * 4;
    const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
    const unsigned int atmadd = 0;
    unsigned int min_threads = min_threads0;
    param.block.z += atmadd;    //USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
    if(param.block.z > 2) min_threads = 32 * 4;
    param.block.x += min_threads;
    param.block.y = 1;    
    param.grid  = createGrid(param.block);


    
    if  ((param.block.x >= min_threads) && (param.block.x <= max_threads)){
      if(param.block.z == 0) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 4 || param.block.z == 5) param.shared_bytes = param.block.x * sizeof(Float);
      return  true;
    }
    else if(param.block.z == 0){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 1;    //USE FOR ATOMIC ADD
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 1){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 2;    //USE FOR NO ATOMIC ADD and LESS SHARED MEM
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 2){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 3;        //USE FOR NO ATOMIC ADD 
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float);
      return true;
    }
    else if(param.block.z == 3){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 4;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else if(param.block.z == 4){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 5;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else
      return  false;
  }
  private:
  unsigned int sharedBytesPerThread() const { 
    return 0; 
  }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { 
      if(param.block.z == 0) return param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) return param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) return param.block.x * 4 * sizeof(Float);
      else return param.block.x * sizeof(Float);
  }

  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  virtual void initTuneParam(TuneParam &param) const{
    param.block = dim3(256, 1, 0);
    param.grid = createGrid(param.block);
    param.shared_bytes = param.block.x * 4 * sizeof(Float);
  }

  GaugeFix(GaugeFixArg<Float, Gauge> &arg) : arg(arg) {
      int parity = 0;
    }
  ~GaugeFix () { }
  void setParity(const int par){ parity = par; }

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    LAUNCH_KERNEL_GAUGEFIX(computeFix, tp, stream, arg, parity, Float, Gauge, gauge_dir);
  }

  /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const{ initTuneParam(param); }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",arg.threads,sizeof(Float),gauge_dir);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
  }

  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    ps << ", atomicadd=" << param.block.z;
    return ps.str();
  }

  //need this
  void preTune() { arg.data.backup(); }
  void postTune() { arg.data.restore(); }
  long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3)*arg.threads; }// Only correct if there is no link reconstruction
  //long long bytes() const { return (1)*8*2*arg.dataOr.Bytes(); } // Only correct if there is no link reconstruction load+save
  long long bytes() const { return 8LL*2*arg.threads*numParams*sizeof(Float); }//no accounting the reduction!!!!
}; 




#ifdef MULTI_GPU
template <typename Float, typename Gauge>
struct GaugeFixInteriorPointsArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  cudaGaugeField &data;
  const Float relax_boost;
  GaugeFixInteriorPointsArg(Gauge &dataOr, cudaGaugeField &data, const Float relax_boost)
    : dataOr(dataOr), data(data), relax_boost(relax_boost) {

#ifdef MULTI_GPU   
    for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = data.R()[dir] + 1; //skip BORDER_RADIUS + face border point
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3] >> 1;
  }
};




template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
__global__ void computeFixInteriorPoints(GaugeFixInteriorPointsArg<Float, Gauge> arg, int parity){
  int tid = (threadIdx.x + blockSize) % blockSize;  
  int idx = blockIdx.x * blockSize + tid;
  if(idx >= arg.threads) return;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  int X[4];
  #pragma unroll 
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
  int x[4];
#ifdef MULTI_GPU
  int za = (idx / (X[0]/2));
  int zb =  (za / X[1]);
  x[1] = za - zb * X[1];
  x[3] = (zb / X[2]);
  x[2] = zb - x[3] * X[2];
  int p=0; for(int dr=0; dr<4; ++dr) p += arg.border[dr]; 
  p = p & 1;
  int x1odd = (x[1] + x[2] + x[3] + parity + p) & 1;
  //int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = (2 * idx + x1odd)  - za * X[0];
  for(int dr=0; dr<4; ++dr) {
       x[dr] += arg.border[dr];
       X[dr] += 2 * arg.border[dr];
  }
#else
  getCoords3(x, idx, X, parity);
#endif
  int mu = (threadIdx.x / blockSize);

  if(ImplementationType<3){
    if(threadIdx.x >= blockSize * 4){
      mu -= 4;
      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      parity = 1 - parity;
    }
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, parity);
    if(ImplementationType==0) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==1) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==2)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    arg.dataOr.save((Float*)(link.data),idx, mu, parity);
  }
 else{
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, parity);


    x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
    int idx1 = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link1;
    arg.dataOr.load((Float*)(link1.data),idx1, mu, 1-parity);

    if(ImplementationType==3) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==4) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==5)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

    arg.dataOr.save((Float*)(link.data),idx, mu, parity);
    arg.dataOr.save((Float*)(link1.data),idx1, mu, 1-parity);

  }
}








template<typename Float, typename Gauge, int gauge_dir>
class GaugeFixInteriorPoints : Tunable {
  GaugeFixInteriorPointsArg<Float, Gauge> arg;
  int parity;
  mutable char aux_string[128]; // used as a label in the autotuner
protected:

  dim3 createGrid   (const dim3 &block) const {
    unsigned int blockx = block.x / 8;
    if(block.z > 2) blockx = block.x / 4;
    unsigned int  gx  = (arg.threads + blockx - 1) / blockx;
    return  dim3(gx, 1, 1);
  }
  bool advanceBlockDim  (TuneParam &param) const {
    //Use param.block.z to tune and save state for best kernel option
    // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
    const unsigned int min_threads0 = 32 * 8;
    const unsigned int min_threads1 = 32 * 4;
    const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
    const unsigned int atmadd = 0;
    unsigned int min_threads = min_threads0;
    param.block.z += atmadd;    //USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
    if(param.block.z > 2) min_threads = 32 * 4;
    param.block.x += min_threads;
    param.block.y = 1;    
    param.grid  = createGrid(param.block);


    
    if  ((param.block.x >= min_threads) && (param.block.x <= max_threads)){
      if(param.block.z == 0) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 4 || param.block.z == 5) param.shared_bytes = param.block.x * sizeof(Float);
      return  true;
    }
    else if(param.block.z == 0){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 1;    //USE FOR ATOMIC ADD
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 1){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 2;    //USE FOR NO ATOMIC ADD and LESS SHARED MEM
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 2){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 3;        //USE FOR NO ATOMIC ADD 
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float);
      return true;
    }
    else if(param.block.z == 3){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 4;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else if(param.block.z == 4){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 5;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else
      return  false;
  }
  private:
  unsigned int sharedBytesPerThread() const { 
    return 0; 
  }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { 
      if(param.block.z == 0) return param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) return param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) return param.block.x * 4 * sizeof(Float);
      else return param.block.x * sizeof(Float);
  }

  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  virtual void initTuneParam(TuneParam &param) const{
    param.block = dim3(256, 1, 0);
    param.grid = createGrid(param.block);
    param.shared_bytes = param.block.x * 4 * sizeof(Float);
  }
  GaugeFixInteriorPoints(GaugeFixInteriorPointsArg<Float, Gauge> &arg) : arg(arg) {
      int parity = 0;
    }
  ~GaugeFixInteriorPoints () { }
  void setParity(const int par){ parity = par; }

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    LAUNCH_KERNEL_GAUGEFIX(computeFixInteriorPoints, tp, stream, arg, parity, Float, Gauge, gauge_dir);
  }


  /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const{ initTuneParam(param); }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",arg.threads,sizeof(Float),gauge_dir);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
  }

  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    ps << ", atomicadd=" << param.block.z;
    return ps.str();
  }

  //need this
  void preTune() { arg.data.backup(); }
  void postTune() { arg.data.restore(); }
  long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3)*arg.threads; }// Only correct if there is no link reconstruction
  //long long bytes() const { return (1)*8*2*arg.dataOr.Bytes(); } // Only correct if there is no link reconstruction load+save
  long long bytes() const { return 8LL*2*arg.threads*numParams*sizeof(Float); } // Only correct if there is no link reconstruction load+save
}; 
















template <typename Float, typename Gauge>
struct GaugeFixBorderPointsArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
  int border[4]; 
  int *borderpoints[2];
  int *faceindicessize[2];
  size_t faceVolume[4];
  size_t faceVolumeCB[4];
  Gauge dataOr;
  cudaGaugeField &data;
  const Float relax_boost;

  GaugeFixBorderPointsArg(Gauge &dataOr, cudaGaugeField &data, const Float relax_boost, size_t faceVolume_[4], size_t faceVolumeCB_[4])
    : dataOr(dataOr), data(data), relax_boost(relax_boost) {


    for(int dir=0; dir<4; ++dir){
      X[dir] = data.X()[dir] - data.R()[dir]*2;
      border[dir] = data.R()[dir];
    }

    /*for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;*/
    for(int dir=0; dir<4; ++dir){
      faceVolume[dir] = faceVolume_[dir];
      faceVolumeCB[dir] = faceVolumeCB_[dir];
    }
    if(comm_size() !=1) PreCalculateLatticeIndices(faceVolume, faceVolumeCB, X, border, threads, borderpoints);
  }
};

template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
__global__ void computeFixBorderPoints(GaugeFixBorderPointsArg<Float, Gauge> arg, int parity){
  int tid = (threadIdx.x + blockSize) % blockSize;  
  int idx = blockIdx.x * blockSize + tid;
  if(idx >= arg.threads) return;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  int mu = (threadIdx.x / blockSize);
  idx = arg.borderpoints[parity][idx];
  int X[4], x[4];
  x[3] = idx/(arg.X[0] * arg.X[1]  * arg.X[2]);
  x[2] = (idx/(arg.X[0] * arg.X[1])) % arg.X[2];
  x[1] = (idx/arg.X[0]) % arg.X[1];
  x[0] = idx % arg.X[0];
  #pragma unroll
  for(int dr=0; dr<4; ++dr) x[dr] += arg.border[dr];
  #pragma unroll
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr] + 2 * arg.border[dr];

  if(ImplementationType<3){
    if(threadIdx.x >= blockSize * 4){
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        parity = 1 - parity;
    }
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, parity);
    if(ImplementationType==0) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==1) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    if(ImplementationType==2)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
    arg.dataOr.save((Float*)(link.data),idx, mu, parity);
  }
  else{
    idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link;
    arg.dataOr.load((Float*)(link.data),idx, mu, parity);


    x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
    int idx1 = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    Matrix<Cmplx,3> link1;
    arg.dataOr.load((Float*)(link1.data),idx1, mu, 1-parity);

    if(ImplementationType==3) GaugeFixHit_NoAtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==4) GaugeFixHit_AtomicAdd<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
    if(ImplementationType==5)GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Cmplx, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

    arg.dataOr.save((Float*)(link.data),idx, mu, parity);
    arg.dataOr.save((Float*)(link1.data),idx1, mu, 1-parity);
  }
}




template<typename Float, typename Gauge, int gauge_dir>
class GaugeFixBorderPoints : Tunable {
  GaugeFixBorderPointsArg<Float, Gauge> arg;
  int parity;
  mutable char aux_string[128]; // used as a label in the autotuner
protected:

  dim3 createGrid   (const dim3 &block) const {
    unsigned int blockx = block.x / 8;
    if(block.z > 2) blockx = block.x / 4;
    unsigned int  gx  = (arg.threads + blockx - 1) / blockx;
    return  dim3(gx, 1, 1);
  }
  bool advanceBlockDim  (TuneParam &param) const {
    //Use param.block.z to tune and save state for best kernel option
    // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
    const unsigned int min_threads0 = 32 * 8;
    const unsigned int min_threads1 = 32 * 4;
    const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
    const unsigned int atmadd = 0;
    unsigned int min_threads = min_threads0;
    param.block.z += atmadd;    //USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
    if(param.block.z > 2) min_threads = 32 * 4;
    param.block.x += min_threads;
    param.block.y = 1;    
    param.grid  = createGrid(param.block);


    
    if  ((param.block.x >= min_threads) && (param.block.x <= max_threads)){
      if(param.block.z == 0) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) param.shared_bytes = param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 4 || param.block.z == 5) param.shared_bytes = param.block.x * sizeof(Float);
      return  true;
    }
    else if(param.block.z == 0){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 1;    //USE FOR ATOMIC ADD
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 1){
      param.block.x = min_threads0;   
      param.block.y = 1;    
      param.block.z = 2;    //USE FOR NO ATOMIC ADD and LESS SHARED MEM
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
      return true;
    }
    else if(param.block.z == 2){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 3;        //USE FOR NO ATOMIC ADD 
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * 4 * sizeof(Float);
      return true;
    }
    else if(param.block.z == 3){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 4;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else if(param.block.z == 4){
      param.block.x = min_threads1;   
      param.block.y = 1;    
      param.block.z = 5;
      param.grid  = createGrid(param.block);
      param.shared_bytes = param.block.x * sizeof(Float);
      return true;
    }
    else
      return  false;
  }
  private:
  unsigned int sharedBytesPerThread() const { 
    return 0; 
  }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { 
      if(param.block.z == 0) return param.block.x * 4 * sizeof(Float);
      else if(param.block.z == 1 || param.block.z == 2) return param.block.x * 4 * sizeof(Float) / 8;
      else if(param.block.z == 3) return param.block.x * 4 * sizeof(Float);
      else return param.block.x * sizeof(Float);
  }

  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  virtual void initTuneParam(TuneParam &param) const{
    param.block = dim3(256, 1, 0);
    param.grid = createGrid(param.block);
    param.shared_bytes = param.block.x * 4 * sizeof(Float);
  }
  GaugeFixBorderPoints(GaugeFixBorderPointsArg<Float, Gauge> &arg) : arg(arg) {
      int parity = 0;
  }
  ~GaugeFixBorderPoints () { 
    if(comm_size() !=1) for(int i = 0; i < 2; i++) cudaFree(arg.borderpoints[i]);
   }
  void setParity(const int par){ parity = par; }

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    LAUNCH_KERNEL_GAUGEFIX(computeFixBorderPoints, tp, stream, arg, parity, Float, Gauge, gauge_dir);
  }

  /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const{ initTuneParam(param); }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",arg.threads,sizeof(Float),gauge_dir);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
  }

  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    ps << ", atomicadd=" << param.block.z;
    return ps.str();
  }

  //need this
  void preTune() { arg.data.backup(); }
  void postTune() { arg.data.restore(); }
  long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3)*arg.threads; }// Only correct if there is no link reconstruction
  //long long bytes() const { return (1)*8*2*arg.dataOr.Bytes(); } // Only correct if there is no link reconstruction load+save
  long long bytes() const { return 8LL*2*arg.threads*numParams*sizeof(Float); } // Only correct if there is no link reconstruction load+save

}; 














template <typename Gauge>
struct GaugeFixUnPackArg {
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  GaugeFixUnPackArg(Gauge &dataOr, cudaGaugeField &data)
    : dataOr(dataOr) {
    for(int dir=0; dir<4; ++dir){
      X[dir] = data.X()[dir] - data.R()[dir]*2;
      #ifdef MULTI_GPU
      border[dir] = data.R()[dir];
      #endif
    }
  }
};


template<int NElems, typename Float, typename Gauge, bool pack>
__global__ void Kernel_UnPackGhost(int size, GaugeFixUnPackArg<Gauge> arg, typename ComplexTypeId<Float>::Type *array, int parity, int face, int dir){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size) return;
  int X[4]; 
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
  int x[4];
  int za, xodd;
  int borderid = 0;
  parity = 1 - parity;
  switch(face){
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
    break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
  }
  for(int dr=0; dr<4; ++dr) {
       x[dr] += arg.border[dr];
       X[dr] += 2*arg.border[dr];
  }
  x[face] -= 1;
  parity = 1 - parity;
  int id = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  typedef typename mapper<Float>::type RegType;
  RegType tmp[NElems];
  RegType data[18];
  if(pack){
    arg.dataOr.load(data, id, dir, parity);
    arg.dataOr.reconstruct.Pack(tmp, data, id);
    for(int i=0; i<NElems/2; ++i) array[idx + size * i] = ((Cmplx*)tmp)[i];
  }
else{
    for(int i=0; i<NElems/2; ++i) ((Cmplx*)tmp)[i] = array[idx + size * i];
    arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0);
    arg.dataOr.save(data, id, dir, parity);
  }
}




template<int NElems, typename Float, typename Gauge, bool pack>
__global__ void Kernel_UnPackTop(int size, GaugeFixUnPackArg<Gauge> arg, typename ComplexTypeId<Float>::Type *array, int parity, int face, int dir){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size) return;
  int X[4]; 
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
  int x[4];
  int za, xodd;
  int borderid = arg.X[face] - 1;
  switch(face){
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
    break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
    break;
  }
  for(int dr=0; dr<4; ++dr) {
       x[dr] += arg.border[dr];
       X[dr] += 2*arg.border[dr];
  }
  int id = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  typedef typename mapper<Float>::type RegType;
  RegType tmp[NElems];
  RegType data[18];
  if(pack){
    arg.dataOr.load(data, id, dir, parity);
    arg.dataOr.reconstruct.Pack(tmp, data, id);
    for(int i=0; i<NElems/2; ++i) array[idx + size * i] = ((Cmplx*)tmp)[i];
  }
  else{
    for(int i=0; i<NElems/2; ++i) ((Cmplx*)tmp)[i] = array[idx + size * i];
    arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0);
    arg.dataOr.save(data, id, dir, parity);
  }
}
#endif


















template<typename Float, typename Gauge, int NElems, int gauge_dir>
void gaugefixingOVR( Gauge dataOr,  cudaGaugeField& data, \
  const unsigned int Nsteps, const unsigned int verbose_interval, \
  const Float relax_boost, const double tolerance, \
  const unsigned int reunit_interval, \
  const unsigned int stopWtheta) {


  TimeProfile profileGaugeFix("GaugeFixCuda");

  profileGaugeFix.Start(QUDA_PROFILE_COMPUTE);
  double flop = 0;
  double byte = 0;

  

  printfQuda("\tOverrelaxation boost parameter: %lf\n", (double)relax_boost);
  printfQuda("\tStop criterium: %lf\n", tolerance);
  if(stopWtheta) printfQuda("\tStop criterium method: theta\n");
  else           printfQuda("\tStop criterium method: Delta\n");
  printfQuda("\tMaximum number of iterations: %d\n", Nsteps);
  printfQuda("\tReunitarize at every %d steps\n", reunit_interval);
  printfQuda("\tPrint convergence results at every %d steps\n", verbose_interval);

  
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
  const int reunit_allow_svd = 1;
  const int reunit_svd_only  = 0;
  const double svd_rel_error = 1e-6;
  const double svd_abs_error = 1e-6;
  setUnitarizeLinksConstants(unitarize_eps, max_error,
      reunit_allow_svd, reunit_svd_only,
      svd_rel_error, svd_abs_error);
  int num_failures=0;
  int* num_failures_dev;
  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));
  if(num_failures_dev == NULL) errorQuda("cudaMalloc failed for dev_pointer\n");

  GaugeFixQualityArg<Gauge> argQ(dataOr, data);
  GaugeFixQuality<Float,Gauge, gauge_dir> GaugeFixQuality(argQ);


  GaugeFixArg<Float, Gauge> arg(dataOr, data, relax_boost);
  GaugeFix<Float,Gauge, gauge_dir> gaugeFix(arg);





#ifdef MULTI_GPU
  void *send[4];
  void *recv[4];
  void *sendg[4];
  void *recvg[4];
  void *send_d[4];
  void *recv_d[4];
  void *sendg_d[4];
  void *recvg_d[4];
  void *hostbuffer_h[4];
  cudaStream_t GFStream[9];
  size_t offset[4];
  size_t bytes[4];
  size_t faceVolume[4];
  size_t faceVolumeCB[4];
  // do the exchange
  MsgHandle *mh_recv_back[4];
  MsgHandle *mh_recv_fwd[4];
  MsgHandle *mh_send_fwd[4];
  MsgHandle *mh_send_back[4];
  int X[4];
  dim3 block[4];
  dim3 grid[4];

  if(comm_size() != 1){

    for(int dir=0; dir<4; ++dir){
      X[dir] = data.X()[dir] - data.R()[dir]*2;
      if (!commDimPartitioned(dir) && data.R()[dir] != 0) errorQuda("Not supported!\n");
    }
    for (int i=0; i<4; i++) {
      faceVolume[i] = 1;
      for (int j=0; j<4; j++) {
        if (i==j) continue;
        faceVolume[i] *= X[j];
      }
      faceVolumeCB[i] = faceVolume[i]/2;
    }

    for (int d=0; d<4; d++) {
      if (!commDimPartitioned(d)) continue;
      offset[d] = faceVolumeCB[d] * NElems;
      bytes[d] =  sizeof(Float) * offset[d];
      send_d[d] = device_malloc(bytes[d]);
      recv_d[d] = device_malloc(bytes[d]);
      sendg_d[d] = device_malloc(bytes[d]);
      recvg_d[d] = device_malloc(bytes[d]);
      cudaStreamCreate(&GFStream[d]);
      cudaStreamCreate(&GFStream[4 + d]);
      #ifndef GPU_COMMS
      hostbuffer_h[d] = (void*)pinned_malloc(4*bytes[d]);
      #endif
      block[d] = make_uint3(128, 1, 1);
      grid[d] = make_uint3((faceVolumeCB[d] + block[d].x - 1) / block[d].x, 1, 1);
    }
    cudaStreamCreate(&GFStream[8]);
    for (int d=0; d<4; d++) {
      if (!commDimPartitioned(d)) continue;
      #ifdef GPU_COMMS
      recv[d] = recv_d[d];
      send[d] = send_d[d];
      recvg[d] = recvg_d[d];
      sendg[d] = sendg_d[d];
      #else
      recv[d] = hostbuffer_h[d];
      send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
      recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3*bytes[d];
      sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2*bytes[d];      
      #endif
      mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
      mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
      mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
      mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
    }
  }
  GaugeFixUnPackArg<Gauge> dataexarg(dataOr, data);
  GaugeFixBorderPointsArg<Float, Gauge> argBorder(dataOr, data, relax_boost, faceVolume, faceVolumeCB);
  GaugeFixBorderPoints<Float,Gauge, gauge_dir> gfixBorderPoints(argBorder);
  GaugeFixInteriorPointsArg<Float, Gauge> argInt(dataOr, data, relax_boost);
  GaugeFixInteriorPoints<Float,Gauge, gauge_dir> gfixIntPoints(argInt);
  #endif

  GaugeFixQuality.apply(0);
  flop += (double)GaugeFixQuality.flops();
  byte += (double)GaugeFixQuality.bytes();
  double action0 = argQ.getAction();
  printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());


unitarizeLinksQuda(data, num_failures_dev);
      cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
      if(num_failures>0){
        cudaFree(num_failures_dev); 
        errorQuda("Error in the unitarization\n"); 
        exit(1);
      }
      cudaMemset(num_failures_dev, 0, sizeof(int));

  int iter = 0;
  for(iter = 0; iter < Nsteps; iter++){
    for(int p = 0; p < 2; p++){
      #ifndef MULTI_GPU      
        gaugeFix.setParity(p);
        gaugeFix.apply(0);
        flop += (double)gaugeFix.flops();
        byte += (double)gaugeFix.bytes();
      #else
      if(comm_size() == 1){
        gaugeFix.setParity(p);
        gaugeFix.apply(0);
        flop += (double)gaugeFix.flops();
        byte += (double)gaugeFix.bytes();
      }
      else{
        gfixIntPoints.setParity(p);
        gfixBorderPoints.setParity(p);//compute border points
        gfixBorderPoints.apply(0);
        flop += (double)gfixBorderPoints.flops();
        byte += (double)gfixBorderPoints.bytes();
        flop += (double)gfixIntPoints.flops();
        byte += (double)gfixIntPoints.bytes();
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_start(mh_recv_back[d]);  
          comm_start(mh_recv_fwd[d]);  
        }   
        //wait for the update to the halo points before start packing...
        cudaDeviceSynchronize();
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          //extract top face
          Kernel_UnPackTop<NElems, Float, Gauge, true><<<grid[d], block[d], 0, GFStream[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(send_d[d]), p, d, d);
          //extract bottom ghost
          Kernel_UnPackGhost<NElems, Float, Gauge, true><<<grid[d], block[d], 0, GFStream[4+d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(sendg_d[d]), 1-p, d, d);
        }  
        #ifdef GPU_COMMS
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          cudaStreamSynchronize(GFStream[d]);
          comm_start(mh_send_fwd[d]);
          cudaStreamSynchronize(GFStream[4+d]);
          comm_start(mh_send_back[d]);
        }   
        #else
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          cudaMemcpyAsync(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[d]);
        }
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          cudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[4+d]);
        }    
        #endif
        //compute interior points
        gfixIntPoints.apply(GFStream[8]);

        #ifndef GPU_COMMS
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          cudaStreamSynchronize(GFStream[d]);
          comm_start(mh_send_fwd[d]);
          cudaStreamSynchronize(GFStream[4+d]);
          comm_start(mh_send_back[d]);
        }
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_recv_back[d]);
          cudaMemcpyAsync(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice, GFStream[d]);
        }
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_recv_fwd[d]);
          cudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice, GFStream[4 + d]);
        }
        #endif
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          #ifdef GPU_COMMS
          comm_wait(mh_recv_back[d]);
          #endif
          Kernel_UnPackGhost<NElems, Float, Gauge, false><<<grid[d], block[d], 0, GFStream[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(recv_d[d]), p, d, d);
        }
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          #ifdef GPU_COMMS
          comm_wait(mh_recv_fwd[d]);
          #endif
          Kernel_UnPackTop<NElems, Float, Gauge, false><<<grid[d], block[d], 0, GFStream[4 + d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(recvg_d[d]), 1-p, d, d); 
        }
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_send_back[d]);
          comm_wait(mh_send_fwd[d]);
          cudaStreamSynchronize(GFStream[d]);
          cudaStreamSynchronize(GFStream[4+d]);
        }
        cudaStreamSynchronize(GFStream[8]);
      }
      #endif 
      /*gaugeFix.setParity(p);
      gaugeFix.apply(0);
      flop += (double)gaugeFix.flops();
      byte += (double)gaugeFix.bytes();
      #ifdef MULTI_GPU
      if(comm_size() != 1){//exchange updated top face links in current parity
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_start(mh_recv_back[d]);      
          //extract top face
          Kernel_UnPackTop<NElems, Float, Gauge><<<grid[d], block[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<Float*>(send_d[d]), p, d, d, true);
          #ifndef GPU_COMMS
          cudaMemcpy(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost);
          #else
          cudaDeviceSynchronize();
          #endif
          comm_start(mh_send_fwd[d]);
          comm_wait(mh_recv_back[d]);
          comm_wait(mh_send_fwd[d]);
          #ifndef GPU_COMMS
          cudaMemcpy(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice);
          #endif
          //inject top face in ghost
          Kernel_UnPackGhost<NElems, Float, Gauge><<<grid[d], block[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<Float*>(recv_d[d]), p, d, d, false);
        }
        //exchange updated ghost links in opposite parity
        for (int d=0; d<4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_start(mh_recv_fwd[d]);  
          Kernel_UnPackGhost<NElems, Float, Gauge><<<grid[d], block[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<Float*>(sendg_d[d]), 1-p, d, d, true); 
          #ifndef GPU_COMMS
          cudaMemcpy(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost);
          #else
          cudaDeviceSynchronize();
          #endif
          comm_start(mh_send_back[d]);
          comm_wait(mh_recv_fwd[d]);
          comm_wait(mh_send_back[d]);
          #ifndef GPU_COMMS
          cudaMemcpy(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice);
          #endif
          Kernel_UnPackTop<NElems, Float, Gauge><<<grid[d], block[d]>>>(faceVolumeCB[d], dataexarg, reinterpret_cast<Float*>(recvg_d[d]), 1-p, d, d, false);
        }
      }
      #endif*/
    }
    if((iter % reunit_interval) == (reunit_interval - 1)) {
      unitarizeLinksQuda(data, num_failures_dev);
      cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
      if(num_failures>0){
        cudaFree(num_failures_dev); 
        errorQuda("Error in the unitarization\n"); 
        exit(1);
      }
      cudaMemset(num_failures_dev, 0, sizeof(int));
      //flop += (double)????????????????????????????????????????????????
      //byte += (double)????????????????????????????????????????????????
    }
    GaugeFixQuality.apply(0);
    flop += (double)GaugeFixQuality.flops();
    byte += (double)GaugeFixQuality.bytes();
    double action = argQ.getAction();
    double diff = abs(action0 - action);
    if((iter % verbose_interval) == (verbose_interval - 1))
    printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter+1, argQ.getAction(), argQ.getTheta(), diff);
    if(stopWtheta){
      if(argQ.getTheta() < tolerance) break;
    }
    else{
      if(diff < tolerance) break;
    } 
    action0 = action;
  }
  if((iter % reunit_interval) != 0)  {
    unitarizeLinksQuda(data, num_failures_dev);
    cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if(num_failures>0){
      cudaFree(num_failures_dev); 
      errorQuda("Error in the unitarization\n"); 
      exit(1);
    }
    cudaMemset(num_failures_dev, 0, sizeof(int));
    //flop += (double)????????????????????????????????????????????????
    //byte += (double)????????????????????????????????????????????????
  }
  if((iter % verbose_interval) != 0){
    GaugeFixQuality.apply(0);
    flop += (double)GaugeFixQuality.flops();
    byte += (double)GaugeFixQuality.bytes();
    double action = argQ.getAction();
    double diff = abs(action0 - action);
    printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter+1, argQ.getAction(), argQ.getTheta(), diff);
  }
  cudaFree(num_failures_dev); 
  #ifdef MULTI_GPU
  if(comm_size() != 1){
    for (int d=0; d<4; d++) {
      if (commDimPartitioned(d)) {
        comm_free(mh_send_fwd[d]);
        comm_free(mh_send_back[d]);
        comm_free(mh_recv_back[d]);
        comm_free(mh_recv_fwd[d]);
        device_free(send_d[d]);
        device_free(recv_d[d]);
        device_free(sendg_d[d]);
        device_free(recvg_d[d]);
        cudaStreamDestroy(GFStream[d]);
        cudaStreamDestroy(GFStream[4 + d]);
        #ifndef GPU_COMMS
        free(hostbuffer_h[d]);
        #endif
      }
    }
    cudaStreamDestroy(GFStream[8]);
  }
  #endif
  checkCudaError();
  cudaDeviceSynchronize();
  profileGaugeFix.Stop(QUDA_PROFILE_COMPUTE);
  double secs = profileGaugeFix.Last(QUDA_PROFILE_COMPUTE);
  double gflops = (flop*1e-9)/(secs);
  double gbytes = byte/(secs*1e9);
  #ifdef MULTI_GPU
  printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
  #else
  printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
  #endif
  printfQuda("Reunitarization flops and bandwidth not accounted!!!!!!\n");
}

template<typename Float, int NElems, typename Gauge>
void gaugefixingOVR( Gauge dataOr,  cudaGaugeField& data, const unsigned int gauge_dir, \
  const unsigned int Nsteps, const unsigned int verbose_interval, \
  const Float relax_boost, const double tolerance, const unsigned int reunit_interval, const unsigned int stopWtheta) {
  if( gauge_dir !=3 ){
    printfQuda("Starting Landau gauge fixing...\n");
    gaugefixingOVR<Float, Gauge, NElems, 4>(dataOr, data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
  } 
  else {        
    printfQuda("Starting Coulomb gauge fixing...\n");
    gaugefixingOVR<Float, Gauge, NElems, 3>(dataOr, data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
  }
}



template<typename Float>
void gaugefixingOVR( cudaGaugeField& data, const unsigned int gauge_dir, \
  const unsigned int Nsteps, const unsigned int verbose_interval, const Float relax_boost, const double tolerance, \
  const unsigned int reunit_interval, const unsigned int stopWtheta) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    printfQuda("QUDA_RECONSTRUCT_NO\n");
      numParams = 18;
      gaugefixingOVR<Float, 18>(FloatNOrder<Float, 18, 2, 18>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    printfQuda("QUDA_RECONSTRUCT_12\n");
      numParams = 12;
      gaugefixingOVR<Float, 12>(FloatNOrder<Float, 18, 2, 12>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    printfQuda("QUDA_RECONSTRUCT_8\n");
      numParams = 8;
      gaugefixingOVR<Float, 8>(FloatNOrder<Float, 18, 2,  8>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    printfQuda("QUDA_RECONSTRUCT_NO\n");
      numParams = 18;
      gaugefixingOVR<Float, 18>(FloatNOrder<Float, 18, 4, 18>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    printfQuda("QUDA_RECONSTRUCT_12\n");
      numParams = 12;
      gaugefixingOVR<Float, 12>(FloatNOrder<Float, 18, 4, 12>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    printfQuda("QUDA_RECONSTRUCT_8\n");
      numParams = 8;
      gaugefixingOVR<Float, 8>(FloatNOrder<Float, 18, 4,  8>(data), data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}

#endif // GPU_GAUGE_ALG

  void gaugefixingOVR( cudaGaugeField& data, const unsigned int gauge_dir,
		       const unsigned int Nsteps, const unsigned int verbose_interval, const double relax_boost,
		       const double tolerance, const unsigned int reunit_interval, const unsigned int stopWtheta) {
#ifdef GPU_GAUGE_ALG
    if(data.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported\n");
    }
    if (data.Precision() == QUDA_SINGLE_PRECISION) {
      gaugefixingOVR<float> (data, gauge_dir, Nsteps, verbose_interval, (float)relax_boost, tolerance, reunit_interval, stopWtheta);
    } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
      gaugefixingOVR<double>(data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
#else
    errorQuda("Gauge fixinghas not been built");
#endif // GPU_GAUGE_ALG
  }


} //namespace quda
