#include <tune_quda.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>
#include <face_quda.h>

namespace quda {

  template <typename Clover>
  struct CloverInvertArg {
    const Clover clover;
    Clover inverse;
    bool computeTraceLog;
    double * const trlogA_h;
    double *trlogA_d;
//extra attributes for twisted mass clover
    bool twist;
    double mu2;
    CloverInvertArg(Clover &inverse, const Clover &clover, bool computeTraceLog=0, double* const trlogA=0) :
      inverse(inverse), clover(clover), computeTraceLog(computeTraceLog), trlogA_h(trlogA), twist(clover.Twisted()), mu2(clover.Mu2()){
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
  __device__ __host__ double cloverInvertCompute(CloverInvertArg<Clover> arg, int x, int parity) {

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

//Compute (T^2 + mu2) first, then invert (not optimized!):
      if(arg.twist)
      {
         complex<Float> aux[15];//hmmm, better to reuse A-regs...
         //another solution just to define (but compiler may not be happy with this, swapping everything in
         //the global buffer):
         //complex<Float>* aux = (complex<Float>*)&A[ch*36];
         //compute off-diagonal terms:
//
         aux[ 0] = tri[0]*diag[0]+diag[1]*tri[0]+conj(tri[2])*tri[1]+conj(tri[4])*tri[3]+conj(tri[7])*tri[6]+conj(tri[11])*tri[10];
//
         aux[ 1] = tri[1]*diag[0]+diag[2]*tri[1]+tri[2]*tri[0]+conj(tri[5])*tri[3]+conj(tri[8])*tri[6]+conj(tri[12])*tri[10];

         aux[ 2] = tri[2]*diag[1]+diag[2]*tri[2]+tri[1]*conj(tri[0])+conj(tri[5])*tri[4]+conj(tri[8])*tri[7]+conj(tri[12])*tri[11];
//
         aux[ 3] = tri[3]*diag[0]+diag[3]*tri[3]+tri[4]*tri[0]+tri[5]*tri[1]+conj(tri[9])*tri[6]+conj(tri[13])*tri[10];

         aux[ 4] = tri[4]*diag[1]+diag[3]*tri[4]+tri[3]*conj(tri[0])+tri[5]*tri[2]+conj(tri[9])*tri[7]+conj(tri[13])*tri[11];

         aux[ 5] = tri[5]*diag[2]+diag[3]*tri[5]+tri[3]*conj(tri[1])+tri[4]*conj(tri[2])+conj(tri[9])*tri[8]+conj(tri[13])*tri[12];
//
         aux[ 6] = tri[6]*diag[0]+diag[4]*tri[6]+tri[7]*tri[0]+tri[8]*tri[1]+tri[9]*tri[3]+conj(tri[14])*tri[10];

         aux[ 7] = tri[7]*diag[1]+diag[4]*tri[7]+tri[6]*conj(tri[0])+tri[8]*tri[2]+tri[9]*tri[4]+conj(tri[14])*tri[11];

         aux[ 8] = tri[8]*diag[2]+diag[4]*tri[8]+tri[6]*conj(tri[1])+tri[7]*conj(tri[2])+tri[9]*tri[5]+conj(tri[14])*tri[12];

         aux[ 9] = tri[9]*diag[3]+diag[4]*tri[9]+tri[6]*conj(tri[3])+tri[7]*conj(tri[4])+tri[8]*conj(tri[5])+conj(tri[14])*tri[13];
//
         aux[10] = tri[10]*diag[0]+diag[5]*tri[10]+tri[11]*tri[0]+tri[12]*tri[1]+tri[13]*tri[3]+tri[14]*tri[6];

         aux[11] = tri[11]*diag[1]+diag[5]*tri[11]+tri[10]*conj(tri[0])+tri[12]*tri[2]+tri[13]*tri[4]+tri[14]*tri[7];

         aux[12] = tri[12]*diag[2]+diag[5]*tri[12]+tri[10]*conj(tri[1])+tri[11]*conj(tri[2])+tri[13]*tri[5]+tri[14]*tri[8];

         aux[13] = tri[13]*diag[3]+diag[5]*tri[13]+tri[10]*conj(tri[3])+tri[11]*conj(tri[4])+tri[12]*conj(tri[5])+tri[14]*tri[9];

         aux[14] = tri[14]*diag[4]+diag[5]*tri[14]+tri[10]*conj(tri[6])+tri[11]*conj(tri[7])+tri[12]*conj(tri[8])+tri[13]*conj(tri[9]);


         //update diagonal elements:
         diag[0] = (Float)arg.mu2+diag[0]*diag[0]+norm(tri[ 0])+norm(tri[ 1])+norm(tri[ 3])+norm(tri[ 6])+norm(tri[10]);
         diag[1] = (Float)arg.mu2+diag[1]*diag[1]+norm(tri[ 0])+norm(tri[ 2])+norm(tri[ 4])+norm(tri[ 7])+norm(tri[11]); 
         diag[2] = (Float)arg.mu2+diag[2]*diag[2]+norm(tri[ 1])+norm(tri[ 2])+norm(tri[ 5])+norm(tri[ 8])+norm(tri[12]); 
         diag[3] = (Float)arg.mu2+diag[3]*diag[3]+norm(tri[ 3])+norm(tri[ 4])+norm(tri[ 5])+norm(tri[ 9])+norm(tri[13]); 
         diag[4] = (Float)arg.mu2+diag[4]*diag[4]+norm(tri[ 6])+norm(tri[ 7])+norm(tri[ 8])+norm(tri[ 9])+norm(tri[14]);
         diag[5] = (Float)arg.mu2+diag[5]*diag[5]+norm(tri[10])+norm(tri[11])+norm(tri[12])+norm(tri[13])+norm(tri[14]);

        //update off-diagonal elements:
         for(int i = 0; i < 15; i++) tri[i] = aux[i];
      }
//
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

    return trlogA;
  }

  template <int blockSize, typename Float, typename Clover>
  void cloverInvert(CloverInvertArg<Clover> arg) {  
    for (int parity=0; parity<2; parity++) {
      for (int x=0; x<arg.clover.volumeCB; x++) {
	// should make this thread safe if we ever apply threads to cpu code
	double trlogA = cloverInvertCompute<blockSize, Float>(arg, x, parity);
	if (arg.computeTraceLog) arg.trlogA_h[parity] += trlogA;
      }
    }
  }

  template <int blockSize, typename Float, typename Clover>
  __global__ void cloverInvertKernel(CloverInvertArg<Clover> arg) {  
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //if (idx >= arg.clover.volumeCB) return;
    int parity = blockIdx.y;
    double trlogA = 0.0;
    if (idx < arg.clover.volumeCB) trlogA = cloverInvertCompute<blockSize, Float>(arg, idx, parity);

    if (arg.computeTraceLog) {
      typedef cub::BlockReduce<double, blockSize> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      double aggregate = BlockReduce(temp_storage).Sum(trlogA);
      if (threadIdx.x == 0) atomicAdd(arg.trlogA_d+parity, aggregate);
    }

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
      if (arg.computeTraceLog) {
	cudaDeviceSynchronize();
	reduceDoubleArray(arg.trlogA_h, 2);
      }
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
