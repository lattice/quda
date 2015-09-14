#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

namespace CloverOrder {
  using namespace quda;
#include <clover_field_order.h>
} // CloverOrder



namespace quda {

#ifdef GPU_CLOVER_DIRAC

  template<typename Float, typename Clover, typename Fmunu>
  struct CloverArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4]; 
#endif
    double cloverCoeff;

    Clover clover;
    Fmunu f;
    
    CloverArg(Clover &clover, Fmunu& f, const GaugeField &meta, double cloverCoeff)
      : threads(meta.Volume()), 
        cloverCoeff(cloverCoeff),
        clover(clover),
	f(f)
    { 
      for(int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
      
#ifdef MULTI_GPU
      for(int dir=0; dir<4; ++dir){
	border[dir] = 2;
      }
#endif
    }
  };


  // Put into clover order 
  // Upper-left block (chirality index 0)
  //     /                                                                                \
  //     |  1 + c*I*(F[0,1] - F[2,3]) ,     c*I*(F[1,2] - F[0,3]) + c*(F[0,2] + F[1,3])   |
  //     |                                                                                |
  //     |  c*I*(F[1,2] - F[0,3]) - c*(F[0,2] + F[1,3]),   1 - c*I*(F[0,1] - F[2,3])      |
  //     |                                                                                |
  //     \                                                                                / 

  //     /
  //     | 1 - c*I*(F[0] - F[5]),   -c*I*(F[2] - F[3]) - c*(F[1] + F[4])  
  //     |
  //     |  -c*I*(F[2] -F[3]) + c*(F[1] + F[4]),   1 + c*I*(F[0] - F[5])  
  //     |
  //     \
  // 
  // Lower-right block (chirality index 1)
  //
  //     /                                                                  \
  //     |  1 - c*I*(F[0] + F[5]),  -c*I*(F[2] + F[3]) - c*(F[1] - F[4])    |
  //     |                                                                  |
  //     |  -c*I*(F[2]+F[3]) + c*(F[1]-F[4]),     1 + c*I*(F[0] + F[5])     |
  //     \                                                                  / 
  //

  // Core routine for constructing clover term from field strength
  template<typename Float, typename Clover, typename Fmunu>
    __device__ __host__
  void cloverComputeCore(CloverArg<Float,Clover,Fmunu>& arg, int idx){

      int parity = 0;  
      if(idx >= arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }
      typedef typename ComplexTypeId<Float>::Type Cmplx;


      // Load the field-strength tensor from global memory
      Matrix<Cmplx,3> F[6];
      for(int i=0; i<6; ++i){
	arg.f.load((Float*)(F[i].data), idx, i, parity);
      }

      Cmplx I; I.x = 0; I.y = 1.0;
      Cmplx coeff; coeff.x = 0; coeff.y = arg.cloverCoeff;
      Matrix<Cmplx,3> block1[2];
      Matrix<Cmplx,3> block2[2];
      block1[0] =  coeff*(F[0]-F[5]); // (18 + 6*9=) 72 floating-point ops 
      block1[1] =  coeff*(F[0]+F[5]); // 72 floating-point ops 
      block2[0] =  arg.cloverCoeff*(F[1]+F[4] - I*(F[2]-F[3])); // 126 floating-point ops
      block2[1] =  arg.cloverCoeff*(F[1]-F[4] - I*(F[2]+F[3])); // 126 floating-point ops


      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
      Float diag[6];
      Cmplx triangle[15]; 
      Float A[72];

      // This uses lots of unnecessary memory
      for(int ch=0; ch<2; ++ch){ 
        // c = 0(1) => positive(negative) chiral block
        // Compute real diagonal elements
        for(int i=0; i<3; ++i){
          diag[i]   = 1.0 - block1[ch](i,i).x;
          diag[i+3] = 1.0 + block1[ch](i,i).x;
        }

        // Compute off diagonal components
        // First row
        triangle[0]  = - block1[ch](1,0);
        // Second row
        triangle[1]  = - block1[ch](2,0);
        triangle[2]  = - block1[ch](2,1);
        // Third row
        triangle[3]  =   block2[ch](0,0);
        triangle[4]  =   block2[ch](0,1);
        triangle[5]  =   block2[ch](0,2);
        // Fourth row 
        triangle[6]  =   block2[ch](1,0);
        triangle[7]  =   block2[ch](1,1);
        triangle[8]  =   block2[ch](1,2);
        triangle[9]  =   block1[ch](1,0);
        // Fifth row
        triangle[10] =   block2[ch](2,0);
        triangle[11] =   block2[ch](2,1);
        triangle[12] =   block2[ch](2,2);
        triangle[13] =   block1[ch](2,0);
        triangle[14] =   block1[ch](2,1);


        for(int i=0; i<6; ++i){
          A[ch*36 + i] = 0.5*diag[i];
        } 
        for(int i=0; i<15; ++i){
          A[ch*36+6+2*i]     = 0.5*triangle[idtab[i]].x;
          A[ch*36+6+2*i + 1] = 0.5*triangle[idtab[i]].y;
        } 
      } // ch
      // 84 floating-point ops


      arg.clover.save(A, idx, parity);
      return;
    }


  template<typename Float, typename Clover, typename Fmunu>
    __global__
  void cloverComputeKernel(CloverArg<Float,Clover,Fmunu> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      cloverComputeCore(arg, idx);
    }

  template<typename Float, typename Clover, typename Fmunu>
  void cloverComputeCPU(CloverArg<Float,Clover,Fmunu> arg){
      for(int idx=0; idx<arg.threads; ++idx){
        cloverComputeCore(arg, idx);
      }
    }


  template<typename Float, typename Clover, typename Fmunu>
    class CloverCompute : Tunable {
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
        : arg(arg), meta(meta), location(location) {
	writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,arg.clover.stride,sizeof(Float));
      }

      virtual ~CloverCompute() {}

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
#if (__COMPUTE_CAPABILITY__ >= 200)
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          cloverComputeKernel<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);  
#else
	  errorQuda("cloverComputeKernel not supported on pre-Fermi architecture");
#endif
        } else { // run the CPU code
          cloverComputeCPU(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(meta.VolString(), typeid(*this).name(), aux);
      }

      std::string paramString(const TuneParam &param) const { // Don't print the grid dim.
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return 480*arg.threads; } 
      long long bytes() const { return arg.threads*(6*18 + 72)*sizeof(Float); } 
    };



  template<typename Float, typename Clover, typename Fmunu>
  void computeClover(Clover clover, Fmunu f, const GaugeField &meta, Float cloverCoeff, QudaFieldLocation location){
    CloverArg<Float,Clover,Fmunu> arg(clover, f, meta, cloverCoeff);
    CloverCompute<Float,Clover,Fmunu> cloverCompute(arg, meta, location);
    cloverCompute.apply(0);
    checkCudaError();
    cudaDeviceSynchronize();
  }

  template<typename Float>
  void computeClover(CloverField &clover, const GaugeField &f, Float cloverCoeff, QudaFieldLocation location){
    if (f.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (clover.isNative()) {
	typedef typename CloverOrder::quda::clover_mapper<Float>::type C;
	computeClover(C(clover,0), FloatNOrder<Float,18,2,18>(f), f, cloverCoeff, location);  
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

