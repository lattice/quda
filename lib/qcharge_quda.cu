#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#ifndef Pi2
#define Pi2   6.2831853071795864769252867665590
#endif

namespace quda {

  template<typename Float>
    struct QChargeArg {
      int threads; // number of active threads required

      int FmunuStride; // stride used on Fmunu field
      int FmunuOffset; // parity offset 

      typename ComplexTypeId<Float>::Type* Fmunu;

      double *Qch;
      double *Q_h;

      QChargeArg(GaugeField& Fmunu) : threads(Fmunu.Volume()), 
        FmunuStride(Fmunu.Stride()), FmunuOffset(Fmunu.Bytes()/(4*sizeof(Float))),
        Fmunu(reinterpret_cast<typename ComplexTypeId<Float>::Type*>(Fmunu.Gauge_p())),
        Q_h(static_cast<double*>(pinned_malloc(sizeof(double)))) {
	  if (cudaHostGetDevicePointer(&Qch, Q_h, 0) != cudaSuccess)
	    errorQuda("ERROR: Failed to allocate pinned memory.\n");
        }
    };

  static __inline__ __device__ double atomicAdd(double *addr, double val)
  {
    double old=*addr, assumed;
    
    do {
      assumed = old;
      old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
					    __double_as_longlong(assumed),
					    __double_as_longlong(val+assumed)));
    } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );
    
    return old;
  }

  // Core routine for computing the topological charge from the field strength
  template<int blockSize, typename Float>
    __global__
    void qChargeComputeKernel(QChargeArg<Float> arg) {
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
          loadLinkVariableFromArray(arg.Fmunu + parity*arg.FmunuOffset, i, idx, arg.FmunuStride, &F[i]); 
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

      typedef cub::BlockReduce<double, blockSize> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      double aggregate = BlockReduce(temp_storage).Sum(tmpQ1);

      if (threadIdx.x == 0) atomicAdd((double *) arg.Qch, aggregate);
    }
/*
  template<typename Float, typename Gauge>
    void qChargeComputeCPU(QChargeArg<Float,Gauge> arg){
*/  /*    for(int idx=0; idx<arg.threads; ++idx){
        qChargeComputeCore(arg, idx);
      }*/
/*    }
*/

  template<typename Float>
    class QChargeCompute : Tunable {
      QChargeArg<Float> arg;
      const QudaFieldLocation location;
      GaugeField *vol;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; };//sizeof(double); };//Float); }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

//      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      QChargeCompute(QChargeArg<Float> &arg, GaugeField *vol, QudaFieldLocation location) 
        : arg(arg), vol(vol), location(location) {
	writeAuxString("threads=%d,prec=%lu",arg.threads,sizeof(Float));
	*(arg.Q_h) = 0.;
      }

      virtual ~QChargeCompute() { host_free(arg.Q_h); }

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
#if (__COMPUTE_CAPABILITY__ >= 200)
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          LAUNCH_KERNEL(qChargeComputeKernel, tp, stream, arg, Float);
	  #ifdef MULTI_GPU
	    comm_allreduce((double*) arg.Q_h);
	  #endif
#else
	  errorQuda("qChargeComputeKernel not supported on pre-Fermi architecture");
#endif
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
      long long flops() const { return 480*arg.threads; } // Cambiar
      long long bytes() const { return arg.threads*(6*18 + 72)*sizeof(Float); } // Cambiar
    };



  template<typename Float>
    void computeQCharge(GaugeField& Fmunu, QudaFieldLocation location, Float &qChg){
      QChargeArg<Float> arg(Fmunu);
      QChargeCompute<Float> qChargeCompute(arg, &Fmunu, location);
      qChargeCompute.apply(0);
      cudaDeviceSynchronize();
      checkCudaError();
      qChg = ((double *) arg.Q_h)[0];
    }

  template<typename Float>
    Float computeQCharge(GaugeField &Fmunu, QudaFieldLocation location){
      int pad = 0;
      Float res = 0.;

      computeQCharge(Fmunu, location, res);

      return res;
    }

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
    errorQuda("QCharge has not been built");
#endif

  }

} // namespace quda

