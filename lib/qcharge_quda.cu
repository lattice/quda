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
#define Pi2   6.2831853071795864769252867665590
#endif

namespace quda {

#ifdef GPU_GAUGE_TOOLS
  template<typename Float, typename Gauge>
  struct QChargeArg : public ReduceArg<double> {
    int threads; // number of active threads required

    typename ComplexTypeId<Float>::Type* Fmunu;

    Gauge data;
    
      QChargeArg(const Gauge &data, GaugeField& Fmunu) : ReduceArg<double>(), data(data), 
        threads(Fmunu.Volume()) {}
    };

  // Core routine for computing the topological charge from the field strength
  template<int blockSize, typename Float, typename Gauge>
    __global__
    void qChargeComputeKernel(QChargeArg<Float,Gauge> arg) {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;

      double tmpQ1 = 0.;

      if(idx < arg.threads) {
        int parity = 0;  
        if(idx >= arg.threads/2) {
          parity = 1;
          idx -= arg.threads/2;
        }
        typedef typename ComplexTypeId<Float>::Type Cmplx;

        // Load the field-strength tensor from global memory
        Matrix<Cmplx,3> F[6], temp1, temp2, temp3;
        double tmpQ2, tmpQ3;
        for(int i=0; i<6; ++i){
          arg.data.load((Float*)(F[i].data), idx, i, parity);
        }

        temp1 = F[0]*F[5];
        temp2 = F[1]*F[4];
        temp3 = F[3]*F[2];

        tmpQ1 = (getTrace(temp1)).x;
        tmpQ2 = (getTrace(temp2)).x;
        tmpQ3 = (getTrace(temp3)).x;
        tmpQ1 += (tmpQ3 - tmpQ2);
        tmpQ1 /= (Pi2*Pi2);
      }

      double Q = tmpQ1;
      reduce<blockSize>(arg, Q);
    }

  template<typename Float, typename Gauge>
    class QChargeCompute : Tunable {
      QChargeArg<Float,Gauge> arg;
      const QudaFieldLocation location;
      GaugeField *vol;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; };
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

//      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      QChargeCompute(QChargeArg<Float,Gauge> &arg, GaugeField *vol, QudaFieldLocation location) 
        : arg(arg), vol(vol), location(location) {
	writeAuxString("threads=%d,prec=%lu",arg.threads,sizeof(Float));
      }

      virtual ~QChargeCompute() { }

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          arg.result_h[0] = 0.;
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          LAUNCH_KERNEL(qChargeComputeKernel, tp, stream, arg, Float);
          cudaDeviceSynchronize();
        }else{ // run the CPU code
	  errorQuda("qChargeComputeKernel not supported on CPU");
//          qChargeComputeCPU(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(vol->VolString(), typeid(*this).name(), aux);
      }

      std::string paramString(const TuneParam &param) const { // Don't print the grid dim.
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return arg.threads*(3*198+9); }
      long long bytes() const { return arg.threads*(6*18)*sizeof(Float); }
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

#ifdef GPU_GAUGE_TOOLS
    if(Fmunu.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION){
      return computeQCharge<float>(Fmunu, location);
    } else if(Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      return computeQCharge<double>(Fmunu, location);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
    return;
#else
    errorQuda("Gauge tools are not build");
#endif

  }

} // namespace quda

