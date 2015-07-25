#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>


#include <comm_quda.h>


#include <pgauge_monte.h> 


namespace quda {

#ifdef GPU_GAUGE_ALG


static  __inline__ __device__ double atomicAdd(double *addr, double val){
  double old=*addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
            __double_as_longlong(assumed),
            __double_as_longlong(val+assumed)));
  } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );
  
  return old;
}

static  __inline__ __device__ double2 atomicAdd(double2 *addr, double2 val){
    double2 old=*addr;
    old.x = atomicAdd((double*)addr, val.x);
    old.y = atomicAdd((double*)addr+1, val.y);
    return old;
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



__device__ __host__ inline int linkIndex2(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}



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
struct KernelArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  double2 *value;
  double2 *value_h;
  KernelArg(const Gauge &dataOr, const cudaGaugeField &data)
    : dataOr(dataOr) {
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = data.R()[dir];
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3];
    value = (double2*)device_malloc(sizeof(double2));
    value_h = (double2*)safe_malloc(sizeof(double2));
  }
  double2 getValue(){return value_h[0];}
};

template<class Cmplx>
 __device__ __host__ inline double2 CmplxToDouble2(const Cmplx b){
  return make_double2(b.x , b.y);
}




template<int blockSize, typename Float, typename Gauge, int NCOLORS, int functiontype>
__global__ void compute_Value(KernelArg<Gauge> arg){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  typedef cub::BlockReduce<double2, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  
  double2 val = make_double2(0.0, 0.0);
  if(idx < arg.threads) {
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    int parity = 0;
    if(idx >= arg.threads/2) {
      parity = 1;
      idx -= arg.threads/2;
    }
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
    idx = linkIndex(x,X);
  #endif
    for (int mu = 0; mu < 4; mu++) {
      Matrix<Cmplx,NCOLORS> U;
      arg.dataOr.load((Float*)(U.data), idx, mu, parity);
      if(functiontype == 0) val += CmplxToDouble2(getDeterminant(U));
      if(functiontype == 1) val += CmplxToDouble2(getTrace(U));
    }
  }
  double2 aggregate = BlockReduce(temp_storage).Reduce(val, Summ<double2>());
  if (threadIdx.x == 0) atomicAdd(arg.value, aggregate);
}



template<typename Float, typename Gauge, int NCOLORS, int functiontype>
class CalcFunc : Tunable {
  KernelArg<Gauge> arg;
  TuneParam tp;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  CalcFunc(KernelArg<Gauge> &arg) : arg(arg) {}
  ~CalcFunc () { 
      host_free(arg.value_h); device_free(arg.value);
    }

  void apply(const cudaStream_t &stream){
      tp = tuneLaunch(*this, getTuning(), getVerbosity());
      cudaMemset(arg.value, 0, sizeof(double2));
      LAUNCH_KERNEL(compute_Value, tp, stream, arg, Float, Gauge, NCOLORS, functiontype);
      cudaMemcpy(arg.value_h, arg.value, sizeof(double2), cudaMemcpyDeviceToHost);
      #ifdef MULTI_GPU
        comm_allreduce_array((double*)arg.value_h, 2);
        const int nNodes = comm_dim(0)*comm_dim(1)*comm_dim(2)*comm_dim(3);
        arg.value_h[0].x  /= (double)(4*arg.threads*nNodes);
        arg.value_h[0].y  /= (double)(4*arg.threads*nNodes);
      #else
        arg.value_h[0].x  /= (double)(4*arg.threads);
        arg.value_h[0].y  /= (double)(4*arg.threads);
      #endif
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
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

  long long flops() const { 
    if(NCOLORS==3 && functiontype == 0) return 264LL*arg.threads+2LL*tp.block.x ; 
    if(NCOLORS==3 && functiontype == 1) return 24LL*arg.threads+2LL*tp.block.x ; 
    else return 0; 
  }// Only correct if there is no link reconstruction
  long long bytes() const { return 4LL*NCOLORS * NCOLORS * sizeof(Float)*2*arg.threads + tp.block.x * sizeof(double2); }

}; 





template<typename Float, int NCOLORS, int functiontype, typename Gauge>
double2 computeValue( Gauge dataOr,  cudaGaugeField& data) {
  TimeProfile profileGenericFunc("GenericFunc");
  if (getVerbosity() >= QUDA_SUMMARIZE) profileGenericFunc.Start(QUDA_PROFILE_COMPUTE);
  KernelArg<Gauge> arg(dataOr, data);
  CalcFunc<Float, Gauge, NCOLORS, functiontype> func(arg);
  func.apply(0);
  if(getVerbosity() >= QUDA_SUMMARIZE && functiontype == 0) printfQuda("Determinant: %.16e, %.16e\n", arg.getValue().x, arg.getValue().y);
  if(getVerbosity() >= QUDA_SUMMARIZE && functiontype == 1) printfQuda("Trace: %.16e, %.16e\n", arg.getValue().x, arg.getValue().y);
  checkCudaError();
  cudaDeviceSynchronize();
  if (getVerbosity() >= QUDA_SUMMARIZE){
    profileGenericFunc.Stop(QUDA_PROFILE_COMPUTE);
    double secs = profileGenericFunc.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (func.flops()*1e-9)/(secs);
    double gbytes = func.bytes()/(secs*1e9);
    if(functiontype == 0){
      #ifdef MULTI_GPU
      printfQuda("Determinant: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
      #else
      printfQuda("Determinant: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
      #endif
    }
    if(functiontype == 1){
      #ifdef MULTI_GPU
      printfQuda("Trace: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
      #else
      printfQuda("Trace: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
      #endif
    }
  }
  return arg.getValue();
}



template<typename Float, int functiontype>
double2 computeValue( cudaGaugeField& data) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
    return  computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 2, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      return computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 2, 12>(data), data);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      return computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 2,  8>(data), data);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      return computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 4, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      return computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 4, 12>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      return computeValue<Float, 3, functiontype>(FloatNOrder<Float, 18, 4,  8>(data), data);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}
#endif // GPU_GAUGE_ALG

/** @brief Calculate the Determinant
* 
* @param[in] data Gauge field
* @returns double2 complex Determinant value
*/
double2 getLinkDeterminant( cudaGaugeField& data) {
#ifdef GPU_GAUGE_ALG
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    return computeValue<float, 0> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    return computeValue<double, 0>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
#else
  errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
}

/** @brief Calculate the Trace
* 
* @param[in] data Gauge field
* @returns double2 complex trace value
*/
double2 getLinkTrace( cudaGaugeField& data) {
#ifdef GPU_GAUGE_ALG
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    return computeValue<float, 1> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    return computeValue<double, 1>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
#else
  errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
}


}
