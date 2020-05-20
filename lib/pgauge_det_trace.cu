#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <comm_quda.h>
#include <pgauge_monte.h> 
#include <atomic.cuh>
#include <cub_helper.cuh> 
#include <index_helper.cuh> 

namespace quda {

#ifdef GPU_GAUGE_ALG

template <typename Gauge>
struct KernelArg : public ReduceArg<double2> {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;

  KernelArg(const Gauge &dataOr, const cudaGaugeField &data)
    : ReduceArg<double2>(), dataOr(dataOr) {
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      border[dir] = data.R()[dir];
      X[dir] = data.X()[dir] - border[dir]*2;
    }
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3]/2;
  }
  double2 getValue(){return result_h[0];}
};



template<int blockSize, typename Float, typename Gauge, int NCOLORS, int functiontype>
__global__ void compute_Value(KernelArg<Gauge> arg){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int parity = threadIdx.y;

  complex<double> val(0.0, 0.0);
  while (idx < arg.threads) {
    int X[4]; 
    #pragma unroll
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
  #ifdef MULTI_GPU
    #pragma unroll
    for(int dr=0; dr<4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2*arg.border[dr];
    }
    idx = linkIndex(x,X);
  #endif
#pragma unroll
    for (int mu = 0; mu < 4; mu++) {
      Matrix<complex<Float>,NCOLORS> U = arg.dataOr(mu, idx, parity);
      if(functiontype == 0) val += getDeterminant(U);
      if(functiontype == 1) val += getTrace(U);
    }

    idx += blockDim.x*gridDim.x;
  }

  double2 sum = make_double2(val.real(), val.imag());
  reduce2d<blockSize,2>(arg, sum);
}



template<typename Float, typename Gauge, int NCOLORS, int functiontype>
class CalcFunc : TunableLocalParity {
  KernelArg<Gauge> arg;
  TuneParam tp;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  bool tuneGridDim() const { return true; }

  public:
  CalcFunc(KernelArg<Gauge> &arg) : arg(arg) {}
  ~CalcFunc () { }

  void apply(const qudaStream_t &stream){
    tp = tuneLaunch(*this, getTuning(), getVerbosity());
    arg.result_h[0] = make_double2(0.0, 0.0);
    LAUNCH_KERNEL_LOCAL_PARITY(compute_Value, (*this), tp, stream, arg, Float, Gauge, NCOLORS, functiontype);
    qudaDeviceSynchronize();

    comm_allreduce_array((double*)arg.result_h, 2);
    arg.result_h[0].x  /= (double)(4*2*arg.threads*comm_size());
    arg.result_h[0].y  /= (double)(4*2*arg.threads*comm_size());
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x" << arg.X[1] << "x" << arg.X[2] << "x" << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%lu", arg.threads, sizeof(Float));
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
    
  }

  long long flops() const { 
    if(NCOLORS==3 && functiontype == 0) return 264LL*2*arg.threads+2LL*tp.block.x;
    else if(NCOLORS==3 && functiontype == 1) return 24LL*2*arg.threads+2LL*tp.block.x;
    else return 0; 
  }// Only correct if there is no link reconstruction
  long long bytes() const { return 4LL*NCOLORS * NCOLORS * sizeof(Float)*2*2*arg.threads + tp.block.x * sizeof(double2); }

}; 





template<typename Float, int NCOLORS, int functiontype, typename Gauge>
double2 computeValue( Gauge dataOr,  cudaGaugeField& data) {
  TimeProfile profileGenericFunc("GenericFunc", false);
  if (getVerbosity() >= QUDA_SUMMARIZE) profileGenericFunc.TPSTART(QUDA_PROFILE_COMPUTE);
  KernelArg<Gauge> arg(dataOr, data);
  CalcFunc<Float, Gauge, NCOLORS, functiontype> func(arg);
  func.apply(0);
  if(getVerbosity() >= QUDA_SUMMARIZE && functiontype == 0) printfQuda("Determinant: %.16e, %.16e\n", arg.getValue().x, arg.getValue().y);
  if(getVerbosity() >= QUDA_SUMMARIZE && functiontype == 1) printfQuda("Trace: %.16e, %.16e\n", arg.getValue().x, arg.getValue().y);
  checkCudaError();
  qudaDeviceSynchronize();
  if (getVerbosity() >= QUDA_SUMMARIZE){
    profileGenericFunc.TPSTOP(QUDA_PROFILE_COMPUTE);
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
double2 computeValue(cudaGaugeField& data) {

  double2 rtn = make_double2(0.0,0.0);

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  if(data.isNative()) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
      rtn = computeValue<Float, 3, functiontype>(Gauge(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
      rtn = computeValue<Float, 3, functiontype>(Gauge(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
      rtn = computeValue<Float, 3, functiontype>(Gauge(data), data);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
  return rtn;
}
#endif // GPU_GAUGE_ALG

/** @brief Calculate the Determinant
* 
* @param[in] data Gauge field
* @returns double2 complex Determinant value
*/
double2 getLinkDeterminant( cudaGaugeField& data) {
  double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    det = computeValue<float, 0> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    det = computeValue<double, 0>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
#else
  errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
  return det;
}

/** @brief Calculate the Trace
* 
* @param[in] data Gauge field
* @returns double2 complex trace value
*/
double2 getLinkTrace( cudaGaugeField& data) {
  double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    det = computeValue<float, 1> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    det = computeValue<double, 1>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
#else
  errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
  return det;
}


} // namespace quda
