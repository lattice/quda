#include <tune_quda.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

namespace quda {

  template <typename Clover>
  struct CloverInvertArg {
    const Clover clover;
    Clover inverse;
    bool computeTraceLog;
    double * const trlogA_h;
    double *trlogA_d;
    CloverInvertArg(Clover &inverse, const Clover &clover, bool computeTraceLog=0, double* const trlogA=0) :
      inverse(inverse), clover(clover), computeTraceLog(computeTraceLog), trlogA_h(trlogA) { 
      cudaHostGetDevicePointer(&trlogA_d, trlogA_h, 0); // set the matching device pointer
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

  /**
     Use a Cholesky decomposition to invert the clover matrix
     Here we use an inplace inversion which hopefully reduces register pressure
   */
  template <int blockSize, typename Float, typename Clover>
  __device__ __host__ void cloverInvertCompute(CloverInvertArg<Clover> arg, int x, int parity) {

    Float A[72];
    double trlogA = 0.0; 

    // load the clover term into memory
    arg.clover.load(A, x, parity);

    for (int ch=0; ch<2; ch++) {

      Float diag[6];
      Float tmp[6]; // temporary storage
      complex<Float> tri[15];

      // hack into the right order as MILC just to copy algorithm directly
      // FIXME use native ordering in the Cholseky 
      // factor of two is inherent to QUDA clover storage
      for (int i=0; i<6; i++) diag[i] = 2.0*A[ch*36+i];

      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
      for (int i=0; i<15; i++) tri[idtab[i]] = complex<Float>(2.0*A[ch*36+6+2*i], 2.0*A[ch*36+6+2*i+1]);

      for (int j=0; j<6; j++) {
	diag[j] = sqrt(diag[j]);
	tmp[j] = 1.0 / diag[j];

	for (int k=j+1; k<6; k++) {
	  int kj = k*(k-1)/2+j;
	  tri[kj] *= tmp[j];
	}

	for(int k=j+1;k<6;k++){
	  int kj=k*(k-1)/2+j;
	  diag[k] -= (tri[kj] * conj(tri[kj])).real();
	  for(int l=k+1;l<6;l++){
	    int lj=l*(l-1)/2+j;
	    int lk=l*(l-1)/2+k;
	    tri[lk] -= tri[lj] * conj(tri[kj]);
	  }
	}	
      }
      
      /* Accumulate trlogA */
      for (int j=0;j<6;j++) trlogA += (double)2.0*log((double)(diag[j]));

      /* Now use forward and backward substitution to construct inverse */
      complex<Float> v1[6];
      for (int k=0;k<6;k++) {
	for(int l=0;l<k;l++) v1[l] = complex<Float>(0.0, 0.0);

	/* Forward substitute */
	v1[k] = complex<Float>(tmp[k], 0.0);
	for(int l=k+1;l<6;l++){
	  complex<Float> sum = complex<Float>(0.0, 0.0);
	  for(int j=k;j<l;j++){
	    int lj=l*(l-1)/2+j;		    
	    sum -= tri[lj] * v1[j];
	  }
	  v1[l] = sum * tmp[l];
	}
	
	/* Backward substitute */
	v1[5] = v1[5] * tmp[5];
	for(int l=4;l>=k;l--){
	  complex<Float> sum = v1[l];
	  for(int j=l+1;j<6;j++){
	    int jl=j*(j-1)/2+l;
	    sum -= conj(tri[jl]) * v1[j];
	  }
	  v1[l] = sum * tmp[l];
	}
	
	/* Overwrite column k */
	diag[k] = v1[k].real();
	for(int l=k+1;l<6;l++){
	  int lk=l*(l-1)/2+k;
	  tri[lk] = v1[l];
	}
      }

      for (int i=0; i<6; i++) A[ch*36+i] = 0.5 * diag[i];
      for (int i=0; i<15; i++) {
	A[ch*36+6+2*i] = 0.5*tri[idtab[i]].real(); A[ch*36+6+2*i+1] = 0.5*tri[idtab[i]].imag();
      }
    }	     

    // save the inverted matrix
    arg.inverse.save(A, x, parity);

    if (arg.computeTraceLog) {
#ifdef __CUDA_ARCH__
      // fix me
      /*typedef cub::BlockReduce<double, blockSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	double aggregate = BlockReduce(temp_storage).Sum(trlogA);
	if (threadIdx.x == 0) atomicAdd(arg.trlogA_d+parity, aggregate);      */
      
      typedef cub::WarpReduce<double, 4> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      double aggregate = WarpReduce(temp_storage).Sum(trlogA);
      if (threadIdx.x % warpSize == 0) atomicAdd(arg.trlogA_d+parity, aggregate);
#else
      // should make this thread safe if we ever apply threads to cpu code
      arg.trlogA_h[parity] += trlogA; 
#endif
    }

  }

  template <int blockSize, typename Float, typename Clover>
  void cloverInvert(CloverInvertArg<Clover> arg) {  
    for (int parity=0; parity<2; parity++) {
      for (int x=0; x<arg.clover.volumeCB; x++) {
	cloverInvertCompute<blockSize, Float>(arg, x, parity);
      }
    }
  }

  template <int blockSize, typename Float, typename Clover>
  __global__ void cloverInvertKernel(CloverInvertArg<Clover> arg) {  
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= arg.clover.volumeCB) return;
    int parity = blockIdx.y;
    cloverInvertCompute<blockSize, Float>(arg, idx, parity);
  }

  template <typename Float, typename Clover>
  class CloverInvert : Tunable {
    CloverInvertArg<Clover> arg;
    const QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneSharedBytes() const { return false; } // Don't tune the shared memory
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.clover.volumeCB; }

  public:
    CloverInvert(CloverInvertArg<Clover> &arg, QudaFieldLocation location) 
      : arg(arg), location(location) { ; }
    virtual ~CloverInvert() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.trlogA_h[0] = 0.0; arg.trlogA_h[1] = 0.0;
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	tp.grid.y = 2; // for parity
	LAUNCH_KERNEL(cloverInvertKernel, tp, stream, arg, Float, Clover);
      } else {
	cloverInvert<1, Float, Clover>(arg);
      }
      if (arg.computeTraceLog) cudaDeviceSynchronize();
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.clover.volumeCB; 
      aux << "stride=" << arg.clover.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { return 2*arg.clover.volumeCB*(arg.inverse.Bytes() + arg.clover.Bytes()); } 
  };

  template <typename Float, typename Clover>
  void cloverInvert(Clover inverse, const Clover clover, bool computeTraceLog, 
		    double* const trlog, QudaFieldLocation location) {
    CloverInvertArg<Clover> arg(inverse, clover, computeTraceLog, trlog);
    CloverInvert<Float,Clover> invert(arg, location);
    invert.apply(0);

    cudaDeviceSynchronize();
  }

  template <typename Float>
  void cloverInvert(const CloverField &clover, bool computeTraceLog, QudaFieldLocation location) {
    if (clover.Order() == QUDA_FLOAT2_CLOVER_ORDER) {
      cloverInvert<Float>(FloatNOrder<Float,72,2>(clover, 1), 
			  FloatNOrder<Float,72,2>(clover, 0), 
			  computeTraceLog, clover.TrLog(), location);
    } else if (clover.Order() == QUDA_FLOAT4_CLOVER_ORDER) {
      cloverInvert<Float>(FloatNOrder<Float,72,4>(clover, 1), 
			  FloatNOrder<Float,72,4>(clover, 0), 
			  computeTraceLog, clover.TrLog(), location);
    } else {
      errorQuda("Clover field %d order not supported", clover.Order());
    }

  }

  // this is the function that is actually called, from here on down we instantiate all required templates
  void cloverInvert(CloverField &clover, bool computeTraceLog, QudaFieldLocation location) {
    if (clover.Precision() == QUDA_HALF_PRECISION && clover.Order() > 4) 
      errorQuda("Half precision not supported for order %d", clover.Order());

    if (clover.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverInvert<double>(clover, computeTraceLog, location);
    } else if (clover.Precision() == QUDA_SINGLE_PRECISION) {
      cloverInvert<float>(clover, computeTraceLog, location);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
  }

} // namespace quda
