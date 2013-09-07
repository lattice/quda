#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>

namespace quda {

  template <typename Complex, typename Gauge, typename Mom>
  struct UpdateGaugeArg {
    typedef typename RealTypeId<Complex>::Type real;
    Gauge out;
    Gauge in;
    Mom momentum;
    real dt;
    int nDim;
    UpdateGaugeArg(const Gauge &out, const Gauge &in, 
		   const Mom &momentum, real dt, int nDim)
      : out(out), in(in), momentum(momentum), dt(dt), nDim(nDim) { }
  };

  template<typename Cmplx, typename Gauge, typename Mom, int N>
  __device__ __host__  void updateGaugeFieldCompute(UpdateGaugeArg<Cmplx,Gauge,Mom> &arg, 
						    int x, int parity) {
    typedef typename RealTypeId<Cmplx>::Type real;
    Matrix<Cmplx,3> link, result, mom;
    for(int dir=0; dir<arg.nDim; ++dir){
      arg.in.load((real*)(link.data), x, dir, parity);

      Cmplx tmp[5];
      arg.momentum.load((real*)(tmp), x, dir, parity);
 
      // FIXME this needs cleaned up in a function
      mom.data[0].x = 0.;
      mom.data[0].y = tmp[3].x;
      mom.data[1] = tmp[0];
      mom.data[2] = tmp[1];
      mom.data[3].x = -mom.data[1].x;
      mom.data[3].y =  mom.data[1].y;
      mom.data[4].x = 0.;
      mom.data[4].y = tmp[3].y;
      mom.data[5]   = tmp[2];
      mom.data[6].x = -mom.data[2].x;
      mom.data[6].y =  mom.data[2].y;
      mom.data[7].x = -mom.data[5].x;
      mom.data[7].y =  mom.data[5].y;
      mom.data[8].x = 0.;
      mom.data[8].y = tmp[4].x;

      result = link;

      // Nth order expansion of exponential
      for(int r=N; r>0; r--) result = (arg.dt/r)*mom*result + link;

      arg.out.save((real*)(result.data), x, dir, parity);
    } // dir

  }

  template<typename Cmplx, typename Gauge, typename Mom, int N>
  void updateGaugeField(UpdateGaugeArg<Cmplx,Gauge,Mom> arg) {

    for (unsigned int parity=0; parity<2; parity++) {
      for (unsigned int x=0; x<arg.out.volumeCB; x++) {
	updateGaugeFieldCompute<Cmplx,Gauge,Mom,N>(arg, x, parity);
      }
    }

  }

  template<typename Cmplx, typename Gauge, typename Mom, int N>
  __global__ void updateGaugeFieldKernel(UpdateGaugeArg<Cmplx,Gauge,Mom> arg) { 
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 2*arg.out.volumeCB) return;
    int parity = (idx >= arg.out.volumeCB) ? 1 : 0;
    idx -= parity*arg.out.volumeCB;

    updateGaugeFieldCompute<Cmplx,Gauge,Mom,N>(arg, idx, parity);
 }
   
  template <typename Complex, typename Gauge, typename Mom, int N>
   class UpdateGaugeField : public Tunable {
  private:
    UpdateGaugeArg<Complex,Gauge,Mom> arg;
    const int *X; // pointer to lattice dimensions
    const QudaFieldLocation location; // location of the lattice fields
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return 2*arg.in.volumeCB; }
    bool tuneGridDim() const { return false; }
    
  public:
    UpdateGaugeField(const UpdateGaugeArg<Complex,Gauge,Mom> &arg, 
		     const int *X, QudaFieldLocation location) 
      : arg(arg), X(X), location(location) {}
    virtual ~UpdateGaugeField() {}
    
    void apply(const cudaStream_t &stream){
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	updateGaugeFieldKernel<Complex,Gauge,Mom,N><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      } else { // run the CPU code
	updateGaugeField<Complex,Gauge,Mom,N>(arg);
      }
    } // apply
    
    void preTune(){}
    void postTune(){}
    
    long long flops() const { 
      const int Nc = 3;
      return arg.nDim*2*arg.in.volumeCB*N*(Nc*Nc*2 +                 // scalar-matrix multiply
					   (8*Nc*Nc*Nc - 2*Nc*Nc) +  // matrix-matrix multiply
					   Nc*Nc*2);                 // matrix-matrix addition
    }
    long long bytes() const { return arg.nDim*2*arg.in.volumeCB*
	(arg.in.Bytes() + arg.out.Bytes() + arg.momentum.Bytes()); }
    
    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << X[0] << "x";
      vol << X[1] << "x";
      vol << X[2] << "x";
      vol << X[3] << "x";
      aux << "threads=" << 2*arg.in.volumeCB << ",prec=" << sizeof(Complex)/2;
      aux << "stride=" << arg.in.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }
  };
  
  template <typename Float, typename Gauge, typename Mom>
  void updateGaugeField(Gauge &out, const Gauge &in, const Mom &mom, 
			double dt, const int *X, QudaFieldLocation location) {
    // degree of exponential expansion
    const int N = 6;

    typedef typename ComplexTypeId<Float>::Type Complex;
    UpdateGaugeArg<Complex, Gauge, Mom> arg(out, in, mom, dt, 4);
    UpdateGaugeField<Complex,Gauge,Mom,N> updateGauge(arg, X, location);
    updateGauge.apply(0); 
    if (location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();

  }

  template <typename Float, typename Gauge>
    void updateGaugeField(Gauge out, const Gauge &in, const GaugeField &mom, 
			  double dt, QudaFieldLocation location) {
    if (mom.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      updateGaugeField<Float>(out, in, FloatNOrder<Float,10,2,10>(mom), dt, mom.X(), location);
    } else if (mom.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(out, in, MILCOrder<Float,10>(mom), dt, mom.X(), location);
    } else {
      errorQuda("Gauge Field order %d not supported", mom.Order());
    }

  }

  template <typename Float>
  void updateGaugeField(GaugeField &out, const GaugeField &in, const GaugeField &mom, 
			double dt, QudaFieldLocation location) {

    const int Nc = 3;
    if (out.Ncolor() != Nc) 
      errorQuda("Ncolor=%d not supported at this time", out.Ncolor());

    if (out.Order() != in.Order() || out.Reconstruct() != in.Reconstruct()) {
      errorQuda("Input and output gauge field ordering and reconstruction must match");
    }

    if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 2, 18>(out),
				FloatNOrder<Float, Nc*Nc*2, 2, 18>(in), 
				mom, dt, location);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 2, 12>(out),
				FloatNOrder<Float, Nc*Nc*2, 2, 12>(in), 
				mom, dt, location);
      } else {
	errorQuda("Reconstruction type not supported");
      }
    } else if (out.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 4, 12>(out),
				FloatNOrder<Float, Nc*Nc*2, 4, 12>(in), 
				mom, dt, location);
      } else {
	errorQuda("Reconstruction type %d not supported", out.Order());
      }
    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(MILCOrder<Float, Nc*Nc*2>(out),
			      MILCOrder<Float, Nc*Nc*2>(in), 
			      mom, dt, location);
    } else {
      errorQuda("Gauge Field order %d not supported", out.Order());
    }

  }

  void updateGaugeField(GaugeField &out, double dt, const GaugeField& in, 
			const GaugeField& mom)
  {
    if (out.Precision() != in.Precision() || out.Precision() != mom.Precision())
      errorQuda("Gauge and momentum fields must have matching precision");

    if (out.Location() != in.Location() || out.Location() != mom.Location())
      errorQuda("Gauge and momentum fields must have matching location");

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      updateGaugeField<double>(out, in, mom, dt, out.Location());
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      updateGaugeField<float>(out, in, mom, dt, out.Location());
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    }

  }

} // namespace quda
