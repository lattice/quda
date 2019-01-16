#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

#include <cub/cub.cuh> 
#include <launch_kernel.cuh>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <index_helper.cuh>

#ifndef Pi2
#define Pi2 6.2831853071795864769252867665590
#endif

namespace quda {

#ifdef GPU_GAUGE_TOOLS
  template<typename Float, typename Gauge>
  struct QChargeArg : public ReduceArg<double> {
    int threads; // number of active threads required
    Gauge data;
    QChargeArg(const Gauge &data, GaugeField& Fmunu)
      : ReduceArg<double>(), data(data), threads(Fmunu.VolumeCB()) {}
  };

  // Core routine for computing the topological charge from the field strength
  template<int blockSize, typename Float, typename Gauge>
  __global__ void qChargeComputeKernel(QChargeArg<Float,Gauge> arg) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;

    double Q = 0.0;

    while (idx < arg.threads) {
      // Load the field-strength tensor from global memory
      Matrix<complex<Float>,3> F[6];
      for (int i=0; i<6; ++i) F[i] = arg.data(i, idx, parity);

      double Q1 = getTrace(F[0]*F[5]).real();
      double Q2 = getTrace(F[1]*F[4]).real();
      double Q3 = getTrace(F[3]*F[2]).real();
      Q += (Q1 + Q3 - Q2);

      idx += blockDim.x*gridDim.x;
    }
    Q /= (Pi2*Pi2);

    reduce2d<blockSize,2>(arg, Q);
  }

  template<typename Float, typename Gauge>
    class QChargeCompute : TunableLocalParity {
      QChargeArg<Float,Gauge> arg;
      const QudaFieldLocation location;
      GaugeField *vol;

    private:
      bool tuneGridDim() const { return true; }

    public:
      QChargeCompute(QChargeArg<Float,Gauge> &arg, GaugeField *vol, QudaFieldLocation location) 
        : arg(arg), vol(vol), location(location) {
	writeAuxString("threads=%d,prec=%lu",arg.threads,sizeof(Float));
      }

      virtual ~QChargeCompute() { }

      void apply(const cudaStream_t &stream) {
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          arg.result_h[0] = 0.;
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          LAUNCH_KERNEL(qChargeComputeKernel, tp, stream, arg, Float);
          qudaDeviceSynchronize();
        } else { // run the CPU code
	  errorQuda("qChargeComputeKernel not supported on CPU");
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(vol->VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return 2*arg.threads*(3*198+9); }
      long long bytes() const { return 2*arg.threads*(6*18)*sizeof(Float); }
    };

  template<typename Float, typename Gauge>
    void computeQCharge(const Gauge data, GaugeField& Fmunu, QudaFieldLocation location, Float &qChg){
      QChargeArg<Float,Gauge> arg(data,Fmunu);
      QChargeCompute<Float,Gauge> qChargeCompute(arg, &Fmunu, location);
      qChargeCompute.apply(0);
      checkCudaError();
      comm_allreduce((double*) arg.result_h);
      qChg = arg.result_h[0];
    }

  template<typename Float>
    Float computeQCharge(GaugeField &Fmunu, QudaFieldLocation location){
      Float res = 0.;

      if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

      if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
        computeQCharge<Float>(Gauge(Fmunu), Fmunu, location, res);
      } else if(Fmunu.Reconstruct() == QUDA_RECONSTRUCT_12){
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
        computeQCharge<Float>(Gauge(Fmunu), Fmunu, location, res);
      } else if(Fmunu.Reconstruct() == QUDA_RECONSTRUCT_8){
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
        computeQCharge<Float>(Gauge(Fmunu), Fmunu, location, res);
      } else {
        errorQuda("Reconstruction type %d of gauge field not supported", Fmunu.Reconstruct());
      }
//      computeQCharge(Fmunu, location, res);

      return res;
    }
#endif

  double computeQCharge(GaugeField& Fmunu, QudaFieldLocation location){

    double charge = 0;
#ifdef GPU_GAUGE_TOOLS
    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION){
      charge = computeQCharge<float>(Fmunu, location);
    } else if(Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      charge = computeQCharge<double>(Fmunu, location);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
#else
    errorQuda("Gauge tools are not build");
#endif
    return charge;

  }

} // namespace quda

