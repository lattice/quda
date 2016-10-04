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

  // This is the maximum number of color spinors we can process in a single kernel
#if (CUDA_VERSION < 8000)
#define MAX_NVECTOR 1 // multi-vector code doesn't seem to work well with CUDA 7.x
#else
#define MAX_NVECTOR 9
#endif

  template<typename Float, typename Output, typename InputA, typename InputB>
  struct CloverSigmaOprodArg {
    Output oprod;
    InputA inA[MAX_NVECTOR];
    InputB inB[MAX_NVECTOR];
    Float coeff[MAX_NVECTOR][2];
    unsigned int length;
    int nvector;

    CloverSigmaOprodArg(Output &oprod, InputA *inA_, InputB *inB_,
			const std::vector<std::vector<double> > &coeff_,
			const GaugeField &meta,	int nvector)
      : oprod(oprod), inA(inA), inB(inB), length(meta.VolumeCB()), nvector(nvector)
    {
      for (int i=0; i<nvector; i++) {
	inA[i] = inA_[i];
	inB[i] = inB_[i];
	coeff[i][0] = coeff_[i][0];
	coeff[i][1] = coeff_[i][1];
      }
    }
  };

  template <typename real, int nvector, int mu, int nu, int parity, typename Arg>
  inline __device__ void sigmaOprod(Arg &arg, int idx) {
    typedef complex<real> Complex;
    Matrix<Complex,3> result;

#pragma unroll
    for (int i=0; i<nvector; i++) {
      ColorSpinor<real,3,4> A, B;

      arg.inA[i].load(static_cast<Complex*>(A.data), idx, parity);
      arg.inB[i].load(static_cast<Complex*>(B.data), idx, parity);

      // multiply by sigma_mu_nu
      ColorSpinor<real,3,4> C = A.sigma(nu,mu);
      result += arg.coeff[i][parity] * outerProdSpinTrace(C,B);
    }

    result -= conj(result);

    Matrix<Complex,3> temp;
    arg.oprod.load(reinterpret_cast<real*>(temp.data), idx, (mu-1)*mu/2 + nu, parity);
    temp = result + temp;
    arg.oprod.save(reinterpret_cast<real*>(temp.data), idx, (mu-1)*mu/2 + nu, parity);
  }

  template<int nvector, typename real, typename Output, typename InputA, typename InputB>
  __global__ void sigmaOprodKernel(CloverSigmaOprodArg<real, Output, InputA, InputB> arg) {
    typedef complex<real> Complex;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockIdx.y*blockDim.y + threadIdx.y;
    int mu_nu = blockIdx.z*blockDim.z + threadIdx.z;

    if (idx >= arg.length) return;
    if (mu_nu >= 6) return;

    switch(parity) {
    case 0:
      switch(mu_nu) {
      case 0: sigmaOprod<real, nvector, 1, 0, 0>(arg, idx); break;
      case 1: sigmaOprod<real, nvector, 2, 0, 0>(arg, idx); break;
      case 2: sigmaOprod<real, nvector, 2, 1, 0>(arg, idx); break;
      case 3: sigmaOprod<real, nvector, 3, 0, 0>(arg, idx); break;
      case 4: sigmaOprod<real, nvector, 3, 1, 0>(arg, idx); break;
      case 5: sigmaOprod<real, nvector, 3, 2, 0>(arg, idx); break;
      }
      break;
    case 1:
      switch(mu_nu) {
      case 0: sigmaOprod<real, nvector, 1, 0, 1>(arg, idx); break;
      case 1: sigmaOprod<real, nvector, 2, 0, 1>(arg, idx); break;
      case 2: sigmaOprod<real, nvector, 2, 1, 1>(arg, idx); break;
      case 3: sigmaOprod<real, nvector, 3, 0, 1>(arg, idx); break;
      case 4: sigmaOprod<real, nvector, 3, 1, 1>(arg, idx); break;
      case 5: sigmaOprod<real, nvector, 3, 2, 1>(arg, idx); break;
      }
      break;
    }

    return;
  } // sigmaOprodKernel

  
  template<typename Float, typename Output, typename InputA, typename InputB>
  class CloverSigmaOprod : public TunableVectorYZ {
    
  private:
    CloverSigmaOprodArg<Float,Output,InputA,InputB> &arg;
    const GaugeField &meta;
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    unsigned int minThreads() const { return arg.length; }
    bool tuneGridDim() const { return false; }
    
  public:
    CloverSigmaOprod(CloverSigmaOprodArg<Float,Output,InputA,InputB> &arg, const GaugeField &meta)
      : TunableVectorYZ(2,6), arg(arg), meta(meta) {
      writeAuxString("prec=%lu,stride=%d,nvector=%d", sizeof(Float), arg.inA[0].Stride(), arg.nvector);
      // this sets the communications pattern for the packing kernel
    } 

    virtual ~CloverSigmaOprod() {}
    
    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this,getTuning(),getVerbosity());
	switch(arg.nvector) {
	case  1: sigmaOprodKernel< 1><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  2: sigmaOprodKernel< 2><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  3: sigmaOprodKernel< 3><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  4: sigmaOprodKernel< 4><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  5: sigmaOprodKernel< 5><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  6: sigmaOprodKernel< 6><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  7: sigmaOprodKernel< 7><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  8: sigmaOprodKernel< 8><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	case  9: sigmaOprodKernel< 9><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
	}
      } else { // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply
    
    void preTune() { this->arg.oprod.save(); }
    void postTune() { this->arg.oprod.load(); }
  
    long long flops() const { 
      return (2*(long long)arg.length)*6*((0 + 144 + 18)*arg.nvector + 18); // spin_mu_nu + spin trace + multiply-add
    }
    long long bytes() const { 
      return (2*(long long)arg.length)*6*((arg.inA[0].Bytes() + arg.inB[0].Bytes())*arg.nvector + 2*arg.oprod.Bytes());
    }
  
    TuneKey tuneKey() const { 
      return TuneKey(meta.VolString(), "CloverSigmaOprod", aux);
    }
  }; // CloverSigmaOprod
  
  template<typename Float, typename Output, typename InputA, typename InputB>
  void computeCloverSigmaOprod(Output oprod, const GaugeField& out, InputA *inA, InputB *inB,
			       std::vector<std::vector<double> > &coeff, int nvector) {
    // Create the arguments 
    CloverSigmaOprodArg<Float,Output,InputA,InputB> arg(oprod, inA, inB, coeff, out, nvector);
    CloverSigmaOprod<Float,Output,InputA,InputB> sigma_oprod(arg, out);
    sigma_oprod.apply(0);
  } // computeCloverSigmaOprod
  
#endif // GPU_CLOVER_FORCE

  void computeCloverSigmaOprod(GaugeField& oprod,
			       std::vector<ColorSpinorField*> &x,
			       std::vector<ColorSpinorField*> &p,
			       std::vector<std::vector<double> > &coeff)
  {

#ifdef GPU_CLOVER_DIRAC
    if (x.size() > MAX_NVECTOR) {
      // divide and conquer
      std::vector<ColorSpinorField*> x0(x.begin(), x.begin()+x.size()/2);
      std::vector<ColorSpinorField*> p0(p.begin(), p.begin()+p.size()/2);
      std::vector<std::vector<double> > coeff0(coeff.begin(), coeff.begin()+coeff.size()/2);
      for (unsigned int i=0; i<coeff0.size(); i++) {
	coeff0[i].reserve(2); coeff0[i][0] = coeff[i][0]; coeff0[i][1] = coeff[i][1];
      }
      computeCloverSigmaOprod(oprod, x0, p0, coeff0);

      std::vector<ColorSpinorField*> x1(x.begin()+x.size()/2, x.end());
      std::vector<ColorSpinorField*> p1(p.begin()+p.size()/2, p.end());
      std::vector<std::vector<double> > coeff1(coeff.begin()+coeff.size()/2, coeff.end());
      for (unsigned int i=0; i<coeff1.size(); i++) {
	coeff1[i].reserve(2); coeff1[i][0] = coeff[coeff.size()/2 + i][0]; coeff1[i][1] = coeff[coeff.size()/2 + i][1];
      }
      computeCloverSigmaOprod(oprod, x1, p1, coeff1);

      return;
    }

    if(oprod.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", oprod.Order());    

    if(x[0]->Precision() != oprod.Precision())
      errorQuda("Mixed precision not supported: %d %d\n", x[0]->Precision(), oprod.Precision());

    if(oprod.Precision() == QUDA_DOUBLE_PRECISION){

      Spinor<double2, double2, 12, 0, 0> spinorA[MAX_NVECTOR];
      Spinor<double2, double2, 12, 0, 1> spinorB[MAX_NVECTOR];

      for (unsigned int i=0; i<x.size(); i++) {
	spinorA[i].set(*dynamic_cast<cudaColorSpinorField*>(x[i]));
	spinorB[i].set(*dynamic_cast<cudaColorSpinorField*>(p[i]));
      }

      computeCloverSigmaOprod<double>(gauge::FloatNOrder<double, 18, 2, 18>(oprod),
				      oprod, spinorA, spinorB, coeff, x.size());

    } else {
      errorQuda("Unsupported precision: %d\n", oprod.Precision());
    }
#else // GPU_CLOVER_DIRAC not defined
    errorQuda("Clover Dirac operator has not been built!"); 
#endif

    checkCudaError();
    return;
  } // computeCloverForce  

} // namespace quda
