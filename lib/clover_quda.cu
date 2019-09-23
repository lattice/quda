#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  template<typename Float, typename Clover, typename Fmunu>
  struct CloverArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    Float cloverCoeff;

    Clover clover;
    Fmunu f;
    
    CloverArg(Clover &clover, Fmunu& f, const GaugeField &meta, double cloverCoeff)
      : threads(meta.VolumeCB()), cloverCoeff(cloverCoeff), clover(clover), f(f)
    { 
      for(int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
    }
  };

  /*
    Put into clover order
    Upper-left block (chirality index 0)
       /                                                                                \
       |  1 + c*I*(F[0,1] - F[2,3]) ,     c*I*(F[1,2] - F[0,3]) + c*(F[0,2] + F[1,3])   |
       |                                                                                |
       |  c*I*(F[1,2] - F[0,3]) - c*(F[0,2] + F[1,3]),   1 - c*I*(F[0,1] - F[2,3])      |
       |                                                                                |
       \                                                                                /

       /
       | 1 - c*I*(F[0] - F[5]),   -c*I*(F[2] - F[3]) - c*(F[1] + F[4])
       |
       |  -c*I*(F[2] -F[3]) + c*(F[1] + F[4]),   1 + c*I*(F[0] - F[5])
       |
       \

     Lower-right block (chirality index 1)

       /                                                                  \
       |  1 - c*I*(F[0] + F[5]),  -c*I*(F[2] + F[3]) - c*(F[1] - F[4])    |
       |                                                                  |
       |  -c*I*(F[2]+F[3]) + c*(F[1]-F[4]),     1 + c*I*(F[0] + F[5])     |
       \                                                                  /
  */
  // Core routine for constructing clover term from field strength
  template<typename Float, typename Arg>
  __device__ __host__ void cloverComputeCore(Arg &arg, int x_cb, int parity){

    constexpr int nColor = 3;
    constexpr int nSpin = 4;
    constexpr int N = nColor*nSpin/2;
    typedef complex<Float> Complex;
    typedef Matrix<Complex,nColor> Link;

    // Load the field-strength tensor from global memory
    Link F[6];
#pragma unroll
    for (int i=0; i<6; ++i) F[i] = arg.f(i, x_cb, parity);

    Complex I(0.0,1.0);
    Complex coeff(0.0,arg.cloverCoeff);
    Link block1[2], block2[2];
    block1[0] =  coeff*(F[0]-F[5]); // (18 + 6*9=) 72 floating-point ops
    block1[1] =  coeff*(F[0]+F[5]); // 72 floating-point ops
    block2[0] =  arg.cloverCoeff*(F[1]+F[4] - I*(F[2]-F[3])); // 126 floating-point ops
    block2[1] =  arg.cloverCoeff*(F[1]-F[4] - I*(F[2]+F[3])); // 126 floating-point ops

    // This uses lots of unnecessary memory
#pragma unroll
    for (int ch=0; ch<2; ++ch) {
      HMatrix<Float,N> A;
      // c = 0(1) => positive(negative) chiral block
      // Compute real diagonal elements
#pragma unroll
      for(int i=0; i<N/2; ++i){
	A(i+0,i+0) = 1.0 - block1[ch](i,i).real();
	A(i+3,i+3) = 1.0 + block1[ch](i,i).real();
      }

      // Compute off diagonal components
      // First row
      A(1,0) = -block1[ch](1,0);
      // Second row
      A(2,0) = -block1[ch](2,0);
      A(2,1) = -block1[ch](2,1);
      // Third row
      A(3,0) =  block2[ch](0,0);
      A(3,1) =  block2[ch](0,1);
      A(3,2) =  block2[ch](0,2);
      // Fourth row
      A(4,0) =  block2[ch](1,0);
      A(4,1) =  block2[ch](1,1);
      A(4,2) =  block2[ch](1,2);
      A(4,3) =  block1[ch](1,0);
      // Fifth row
      A(5,0) =  block2[ch](2,0);
      A(5,1) =  block2[ch](2,1);
      A(5,2) =  block2[ch](2,2);
      A(5,3) =  block1[ch](2,0);
      A(5,4) =  block1[ch](2,1);
      A *= static_cast<Float>(0.5);

      arg.clover(x_cb, parity, ch) = A;
    } // ch
    // 84 floating-point ops

    return;
  }


  template<typename Float, typename Clover, typename Fmunu>
  __global__ void cloverComputeKernel(CloverArg<Float,Clover,Fmunu> arg){
    int x_cb = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    if (x_cb >= arg.threads) return;
    cloverComputeCore<Float>(arg, x_cb, parity);
  }

  template<typename Float, typename Clover, typename Fmunu>
  void cloverComputeCPU(CloverArg<Float,Clover,Fmunu> arg){
    for (int parity = 0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.threads; x_cb++){
	cloverComputeCore<Float>(arg, x_cb, parity);
      }
    }
  }


  template<typename Float, typename Clover, typename Fmunu>
  class CloverCompute : TunableVectorY {
    CloverArg<Float,Clover,Fmunu> arg;
      const GaugeField &meta;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      CloverCompute(CloverArg<Float,Clover,Fmunu> &arg, const GaugeField &meta, QudaFieldLocation location) 
        : TunableVectorY(2), arg(arg), meta(meta), location(location) {
	writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,arg.clover.stride,sizeof(Float));
      }

      virtual ~CloverCompute() {}

      void apply(const qudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          cloverComputeKernel<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);  
        } else { // run the CPU code
          cloverComputeCPU(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(meta.VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return 2*arg.threads*480ll; }
      long long bytes() const { return 2*arg.threads*(6*arg.f.Bytes() + arg.clover.Bytes()); }
    };



  template<typename Float, typename Clover, typename Fmunu>
  void computeClover(Clover clover, Fmunu f, const GaugeField &meta, Float cloverCoeff, QudaFieldLocation location){
    CloverArg<Float,Clover,Fmunu> arg(clover, f, meta, cloverCoeff);
    CloverCompute<Float,Clover,Fmunu> cloverCompute(arg, meta, location);
    cloverCompute.apply(0);
    checkCudaError();
    qudaDeviceSynchronize();
  }

  template<typename Float>
  void computeClover(CloverField &clover, const GaugeField &f, Float cloverCoeff, QudaFieldLocation location){
    if (f.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (clover.isNative()) {
	typedef typename clover_mapper<Float>::type C;
	computeClover(C(clover,0), gauge::FloatNOrder<Float,18,2,18>(f), f, cloverCoeff, location);  
      } else {
	errorQuda("Clover field order %d not supported", clover.Order());
      } // clover order
    } else {
      errorQuda("Fmunu field order %d not supported", f.Precision());
    }
  }

#endif

  void computeClover(CloverField &clover, const GaugeField& f, double cloverCoeff, QudaFieldLocation location){

#ifdef GPU_CLOVER_DIRAC
    if(clover.Precision() != f.Precision()){
      errorQuda("Fmunu precision %d must match gauge precision %d", clover.Precision(), f.Precision());
    }

    if (clover.Precision() == QUDA_DOUBLE_PRECISION){
      computeClover<double>(clover, f, cloverCoeff, location);
    } else if(clover.Precision() == QUDA_SINGLE_PRECISION) {
      computeClover<float>(clover, f, cloverCoeff, location);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
    return;
#else
    errorQuda("Clover has not been built");
#endif

  }

} // namespace quda

