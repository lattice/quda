#include <cstdio>
#include <cstdlib>

#include <tune_quda.h>
#include <quda_internal.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <dslash_quda.h>

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  namespace { // anonymous
#include <texture.h>
  }
  
  template<typename Complex, typename Output, typename InputA, typename InputB>
  struct CloverSigmaOprodArg {
    unsigned int length;
    unsigned int parity;
    InputA inA;
    InputB inB;
    Output oprod;
    typename RealTypeId<Complex>::Type coeff;
    int mu;
    int nu;
    int count;
      
    CloverSigmaOprodArg(const unsigned int parity,
			const double coeff,
			int mu,
			int nu,
			int count,
			InputA& inA,
			InputB& inB,
			Output& oprod,
			GaugeField &meta) : length(meta.VolumeCB()), parity(parity), 
					    inA(inA), inB(inB), oprod(oprod), 
					    coeff(coeff), mu(mu), nu(nu), count(count)
    {

    }
  };

  template<typename Complex, typename Output, typename InputA, typename InputB>
  __global__ void sigmaOprodKernel(CloverSigmaOprodArg<Complex, Output, InputA, InputB> arg) {
    typedef typename RealTypeId<Complex>::Type real;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    ColorSpinor<real,3,4> A, B;
    Matrix<Complex,3> result, temp;

    // workaround for code that hangs generated with CUDA 5.x
#if (CUDA_VERSION < 6000)
    if (idx >= arg.length) idx = arg.length - 1;
#else
    while(idx<arg.length){
#endif // CUDA_VERSION
      arg.inA.load(static_cast<Complex*>(A.data), idx);
      arg.inB.load(static_cast<Complex*>(B.data), idx);

      // multiply by sigma_mu_nu
      ColorSpinor<real,3,4> C = A.sigma(arg.mu,arg.nu);
      result = outerProdSpinTrace(C,B);

      if (arg.count > 0) {
	arg.oprod.load(reinterpret_cast<real*>(temp.data), idx, 0, arg.parity); 
	temp = arg.coeff*result + temp;
      } else {
	temp = arg.coeff*result;
      }
      arg.oprod.save(reinterpret_cast<real*>(temp.data), idx, 0, arg.parity); 
#if (CUDA_VERSION >= 6000)
      idx += gridDim.x*blockDim.x;
    }
#endif // CUDA_VERSION
    return;
  } // sigmaOprodKernel

  
  template<typename Complex, typename Output, typename InputA, typename InputB> 
  class CloverSigmaOprod : public Tunable {
    
  private:
    CloverSigmaOprodArg<Complex,Output,InputA,InputB> &arg;
    const GaugeField &meta;
    QudaFieldLocation location; // location of the lattice fields
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    unsigned int minThreads() const { return arg.length; }
    bool tuneGridDim() const { return false; }
    
  public:
    CloverSigmaOprod(CloverSigmaOprodArg<Complex,Output,InputA,InputB> &arg,
		     const GaugeField &meta, QudaFieldLocation location)
      : arg(arg), meta(meta), location(location) {
      writeAuxString("prec=%lu,stride=%d,mu=%d,nu=%d", 
		     sizeof(Complex)/2, arg.inA.Stride(), arg.mu, arg.nu);
      // this sets the communications pattern for the packing kernel
    } 
    
    virtual ~CloverSigmaOprod() {}
    
    void apply(const cudaStream_t &stream){
      if(location == QUDA_CUDA_FIELD_LOCATION){
	// Disable tuning for the time being
	TuneParam tp = tuneLaunch(*this,getTuning(),getVerbosity());
	sigmaOprodKernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }else{ // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply
    
    void preTune(){
      this->arg.oprod.save();
    }
    void postTune(){
      this->arg.oprod.load();
    }
  
    long long flops() const { 
      ((long long)arg.length)*(0 + 144 + 36); // spin_mu_nu + spin trace + multiply-add
    }
    long long bytes() const { 
      ((long long)arg.length)*(arg.inA.Bytes() + arg.inB.Bytes() + 2*arg.oprod.Bytes());
    }
  
    TuneKey tuneKey() const { 
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }
  }; // CloverSigmaOprod
  
  template<typename Complex, typename Output, typename InputA, typename InputB>
  void computeCloverSigmaOprodCuda(Output oprod, cudaGaugeField& out, InputA& inA, InputB& inB,
				   const unsigned int parity, const double coeff, int mu, int nu, int shift) {
    // Create the arguments 
    CloverSigmaOprodArg<Complex,Output,InputA,InputB> arg(parity, coeff, mu, nu, shift, inA, inB, oprod, out);
    CloverSigmaOprod<Complex,Output,InputA,InputB> sigma_oprod(arg, out, QUDA_CUDA_FIELD_LOCATION);
    sigma_oprod.apply(0);
  } // computeCloverSigmaOprodCuda
  
#endif // GPU_CLOVER_FORCE

  void computeCloverSigmaOprod(cudaGaugeField& oprod,
			       cudaColorSpinorField& x,  
			       cudaColorSpinorField& p,
			       const double coeff, int mu, int nu, int shift)
  {

#ifdef GPU_CLOVER_DIRAC
    if(oprod.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", oprod.Order());    

    if(x.Precision() != oprod.Precision()) errorQuda("Mixed precision not supported: %d %d\n", x.Precision(), oprod.Precision());

    for (int parity=0; parity<2; parity++) {
      cudaColorSpinorField& inA = (parity&1) ? x.Odd() : x.Even();
      cudaColorSpinorField& inB = (parity&1) ? p.Odd() : p.Even();

      if(x.Precision() == QUDA_DOUBLE_PRECISION){
	Spinor<double2, double2, double2, 12, 0, 0> spinorA(inA);
	Spinor<double2, double2, double2, 12, 0, 1> spinorB(inB);
	computeCloverSigmaOprodCuda<double2>(FloatNOrder<double, 18, 2, 18>(oprod), 
					     oprod, spinorA, spinorB, parity, coeff, mu, nu, shift);
      } else {
	errorQuda("Unsupported precision: %d\n", x.Precision());
      }
    } // parity

#else // GPU_CLOVER_DIRAC not defined
    errorQuda("Clover Dirac operator has not been built!"); 
#endif

    checkCudaError();
    return;
  } // computeCloverForce  

} // namespace quda
