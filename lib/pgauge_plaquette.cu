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


#include <pgauge_monte.h> 


#include <random.h>


#define BORDER_RADIUS 2

#ifndef PI
#define PI    3.1415926535897932384626433832795    // pi
#endif
#ifndef PII
#define PII   6.2831853071795864769252867665590    // 2 * pi
#endif

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


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
//CUDA 6.5 NOT DETECTING ATOMICADD FOR FLOAT TYPE!!!!!!!
static __inline__ __device__ float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 200 */



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
struct PlaquetteArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  double2 *plaq;
  double2 *plaq_h;
  PlaquetteArg(const Gauge &dataOr, const cudaGaugeField &data)
    : dataOr(dataOr), plaq_h(static_cast<double2*>(pinned_malloc(sizeof(double2)))) {
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3];
    cudaHostGetDevicePointer(&plaq, plaq_h, 0);
  }
  double2 getPlaq(){return plaq_h[0];}
};


template<class Float>
__device__ __host__ inline Float getRealTrace(const Matrix<typename ComplexTypeId<Float>::Type,3>& a){
  return a(0,0).x + a(1,1).x + a(2,2).x;
}

template<int blockSize, typename Float, typename Gauge, int NCOLORS>
__global__ void compute_Plaquette(PlaquetteArg<Gauge> arg){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  typedef cub::BlockReduce<double2, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  
  double2 plaq = make_double2(0.0, 0.0);
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
    int dx[4] = {0, 0, 0, 0};
    for (int mu = 0; mu < 3; mu++) {
      Matrix<Cmplx,NCOLORS> U1;
      arg.dataOr.load((Float*)(U1.data), idx, mu, parity);
      dx[mu]++;
      int idxmu = linkIndex3(x,dx,X);
      dx[mu]--;
      for (int nu = (mu+1); nu < 4; nu++) {
        Matrix<Cmplx,NCOLORS> U2, U3;
        arg.dataOr.load((Float*)(U2.data),idxmu, nu, 1-parity);
        dx[nu]++;
        arg.dataOr.load((Float*)(U3.data),linkIndex3(x,dx,X), mu, 1-parity);
        U2 *= conj(U3);
        dx[nu]--;
        arg.dataOr.load((Float*)(U3.data),idx, nu, parity);
        U2 *= conj(U3);
        U3  = U1 * U2;
        //if(mu == 3 || nu == 3) plaq.y += getTrace(U3).x;
        if(nu == 3) plaq.y += getRealTrace<Float>(U3);
        else plaq.x += getRealTrace<Float>(U3);
      }
    }
  }
  double2 aggregate = BlockReduce(temp_storage).Reduce(plaq, Summ<double2>());
  if (threadIdx.x == 0) atomicAdd(arg.plaq, aggregate);
  //Flops(3) = 6*3*198 + 6*3 + 2*blockDim.x=3582 + 2*blockDim.x
  //Flops(NCOLORS other case) = 6*3*NCOLORS * NCOLORS * NCOLORS*8 + 6 + NCOLORS*6+2*blockDim.x=144*NCOLORS * NCOLORS * NCOLORS + NCOLORS*6+2*blockDim.x
  //Bytes = (3 + 6*3) * NCOLORS * NCOLORS * sizeof(Float)*2 + blockDim.x * 8 * 2

}



template<typename Float, typename Gauge, int NCOLORS>
class CalcPlaquette : Tunable {
  PlaquetteArg<Gauge> arg;
  TuneParam tp;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  CalcPlaquette(PlaquetteArg<Gauge> &arg)
    : arg(arg) {}
  ~CalcPlaquette () { host_free(arg.plaq_h);}

  void apply(const cudaStream_t &stream){
      tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.plaq_h[0] = make_double2(0.0,0.0);
      LAUNCH_KERNEL(compute_Plaquette, tp, stream, arg, Float, Gauge, NCOLORS);
      cudaDeviceSynchronize();
      #ifdef MULTI_GPU
        comm_allreduce_array((double*)arg.plaq_h, 2);
        const int nNodes = comm_dim(0)*comm_dim(1)*comm_dim(2)*comm_dim(3);
        arg.plaq_h[0].x  /= (double)(3*NCOLORS*arg.threads*nNodes);
        arg.plaq_h[0].y  /= (double)(3*NCOLORS*arg.threads*nNodes);
      #else
        arg.plaq_h[0].x  /= (double)(3*NCOLORS*arg.threads);
        arg.plaq_h[0].y  /= (double)(3*NCOLORS*arg.threads);
      #endif
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",arg.threads, sizeof(Float));
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
  //Flops(3) = 6*3*198 + 6 + 3*2 + 2*blockDim.x=3576 + 2*blockDim.x
  //Flops(NCOLORS other case) = 6*3*NCOLORS * NCOLORS * NCOLORS*8 + 6 + NCOLORS*2+2*blockDim.x=144*NCOLORS * NCOLORS * NCOLORS + 6 + NCOLORS*2+2*blockDim.x
  //Bytes = (3 + 6*3) * NCOLORS * NCOLORS * sizeof(Float)*2 + blockDim.x * 8 * 2

  long long flops() const { 
    if(NCOLORS==3) return 3582LL*arg.threads + 2LL*tp.block.x; 
    else return 144LL*NCOLORS * NCOLORS * NCOLORS + NCOLORS*6LL+2LL*tp.block.x;
  }// Only correct if there is no link reconstruction
  long long bytes() const { return 21LL*NCOLORS * NCOLORS * sizeof(Float)*2*arg.threads + tp.block.x * sizeof(double2); }

}; 





template<typename Float, int NCOLORS, typename Gauge>
double2 Plaquette( Gauge dataOr,  cudaGaugeField& data) {


  TimeProfile profileGaugeFix("Plaquette");
  if (getVerbosity() >= QUDA_SUMMARIZE) profileGaugeFix.Start(QUDA_PROFILE_COMPUTE);
  PlaquetteArg<Gauge> plaqarg(dataOr, data);
  CalcPlaquette<Float, Gauge, NCOLORS> plaq(plaqarg);
  plaq.apply(0);
  printfQuda("Plaquette: %.16e, %.16e, %.16e\n", plaqarg.getPlaq().x, plaqarg.getPlaq().y, (plaqarg.getPlaq().x+plaqarg.getPlaq().y) / 2);
  checkCudaError();
  cudaDeviceSynchronize();
  if (getVerbosity() >= QUDA_SUMMARIZE){
    profileGaugeFix.Stop(QUDA_PROFILE_COMPUTE);
    double secs = profileGaugeFix.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (plaq.flops()*1e-9)/(secs);
    double gbytes = plaq.bytes()/(secs*1e9);
    #ifdef MULTI_GPU
    printfQuda("Plaq: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
    #else
    printfQuda("Plaq: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    #endif
  }
  return plaqarg.getPlaq();
}



template<typename Float>
double2 Plaquette( cudaGaugeField& data) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
    return  Plaquette<Float, 3>(FloatNOrder<Float, 18, 2, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      return Plaquette<Float, 3>(FloatNOrder<Float, 18, 2, 12>(data), data);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      return Plaquette<Float, 3>(FloatNOrder<Float, 18, 2,  8>(data), data);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      return Plaquette<Float, 3>(FloatNOrder<Float, 18, 4, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      return Plaquette<Float, 3>(FloatNOrder<Float, 18, 4, 12>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      return Plaquette<Float, 3>(FloatNOrder<Float, 18, 4,  8>(data), data);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}
#endif // GPU_GAUGE_ALG

/** @brief Calculate the plaquette
* 
* @param[in] data Gauge field
* @returns double2 .x stores the space-space plaquette value and .y stores the space-time plaquette value
*/
double2 Plaquette( cudaGaugeField& data) {
#ifdef GPU_GAUGE_ALG
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    return Plaquette<float> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    return Plaquette<double>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
#else
  errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
}


}
