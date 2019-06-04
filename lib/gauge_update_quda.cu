#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <float_vector.h>
#include <complex_quda.h>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename Gauge, typename Mom>
  struct UpdateGaugeArg {
    Gauge out;
    Gauge in;
    Mom momentum;
    Float dt;
    int nDim;
    UpdateGaugeArg(const Gauge &out, const Gauge &in, 
		   const Mom &momentum, Float dt, int nDim)
      : out(out), in(in), momentum(momentum), dt(dt), nDim(nDim) { }
  };

  template<typename Float, typename Gauge, typename Mom, int N,
	   bool conj_mom, bool exact>
  __device__ __host__  void updateGaugeFieldCompute
  (UpdateGaugeArg<Float,Gauge,Mom> &arg, int x, int parity) {
    typedef complex<Float> Complex;

    Matrix<Complex,3> link, result, mom;
    for(int dir=0; dir<arg.nDim; ++dir){
      arg.in.load((Float*)(link.data), x, dir, parity);
      arg.momentum.load((Float*)(mom.data), x, dir, parity);

      Complex trace = getTrace(mom);
      mom(0,0) -= trace/static_cast<Float>(3.0);
      mom(1,1) -= trace/static_cast<Float>(3.0);
      mom(2,2) -= trace/static_cast<Float>(3.0);

      if (!exact) {
	result = link;
	
	// Nth order expansion of exponential
	if (!conj_mom) {
	  for(int r=N; r>0; r--) 
	    result = (arg.dt/r)*mom*result + link;
	} else {
	  for(int r=N; r>0; r--) 
	    result = (arg.dt/r)*conj(mom)*result + link;
	}
      } else {
	mom = arg.dt * mom;
	expsu3<Float>(mom);

	if (!conj_mom) {
	  link = mom * link;
	} else {
	  link = conj(mom) * link;
	}

	result = link;
      }

      arg.out.save((Float*)(result.data), x, dir, parity);
    } // dir

  }

  template<typename Float, typename Gauge, typename Mom, int N,
	   bool conj_mom, bool exact>
  void updateGaugeField(UpdateGaugeArg<Float,Gauge,Mom> arg) {

    for (unsigned int parity=0; parity<2; parity++) {
      for (int x=0; x<arg.out.volumeCB; x++) {
	updateGaugeFieldCompute<Float,Gauge,Mom,N,conj_mom,exact>
	  (arg, x, parity);
      }
    }
  }

  template<typename Float, typename Gauge, typename Mom, int N,
	   bool conj_mom, bool exact>
  __global__ void updateGaugeFieldKernel(UpdateGaugeArg<Float,Gauge,Mom> arg) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 2*arg.out.volumeCB) return;
    int parity = (idx >= arg.out.volumeCB) ? 1 : 0;
    idx -= parity*arg.out.volumeCB;

    updateGaugeFieldCompute<Float,Gauge,Mom,N,conj_mom,exact>(arg, idx, parity);
 }
   
  template <typename Float, typename Gauge, typename Mom, int N,
	    bool conj_mom, bool exact>
   class UpdateGaugeField : public Tunable {
  private:
    UpdateGaugeArg<Float,Gauge,Mom> arg;
    const GaugeField &meta; // meta data
    const QudaFieldLocation location; // location of the lattice fields

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return 2*arg.in.volumeCB; }
    bool tuneGridDim() const { return false; }
    
  public:
    UpdateGaugeField(const UpdateGaugeArg<Float,Gauge,Mom> &arg,
		     const GaugeField &meta, QudaFieldLocation location)
      : arg(arg), meta(meta), location(location) {
      writeAuxString("threads=%d,prec=%lu,stride=%d", 
		     2*arg.in.volumeCB, sizeof(Float), arg.in.stride);
    }
    virtual ~UpdateGaugeField() { }
    
    void apply(const cudaStream_t &stream){
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	updateGaugeFieldKernel<Float,Gauge,Mom,N,conj_mom,exact>
	  <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      } else { // run the CPU code
	updateGaugeField<Float,Gauge,Mom,N,conj_mom,exact>(arg);
      }
    } // apply
    
    long long flops() const { 
      const int Nc = 3;
      return arg.nDim*2*arg.in.volumeCB*N*(Nc*Nc*2 +                 // scalar-matrix multiply
					   (8*Nc*Nc*Nc - 2*Nc*Nc) +  // matrix-matrix multiply
					   Nc*Nc*2);                 // matrix-matrix addition
    }
    long long bytes() const { return arg.nDim*2*arg.in.volumeCB*
	(arg.in.Bytes() + arg.out.Bytes() + arg.momentum.Bytes()); }
    
    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };
  
  template <typename Float, typename Gauge, typename Mom>
  void updateGaugeField(Gauge &out, const Gauge &in, const Mom &mom, 
			double dt, const GaugeField &meta, bool conj_mom, bool exact,
			QudaFieldLocation location) {
    // degree of exponential expansion
    const int N = 8;

    if (conj_mom) {
      if (exact) {
	UpdateGaugeArg<Float, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Float,Gauge,Mom,N,true,true> updateGauge(arg, meta, location);
	updateGauge.apply(0); 
      } else {
	UpdateGaugeArg<Float, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Float,Gauge,Mom,N,true,false> updateGauge(arg, meta, location);
	updateGauge.apply(0); 
      }
    } else {
      if (exact) {
	UpdateGaugeArg<Float, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Float,Gauge,Mom,N,false,true> updateGauge(arg, meta, location);
	updateGauge.apply(0);
      } else {
	UpdateGaugeArg<Float, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Float,Gauge,Mom,N,false,false> updateGauge(arg, meta, location);
	updateGauge.apply(0); 
      }
    }

    if (location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();

  }

  template <typename Float, typename Gauge>
    void updateGaugeField(Gauge out, const Gauge &in, const GaugeField &mom, 
			  double dt, bool conj_mom, bool exact, 
			  QudaFieldLocation location) {
    if (mom.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (mom.Reconstruct() == QUDA_RECONSTRUCT_10) {
	// FIX ME - 11 is a misnomer to avoid confusion in template instantiation
	updateGaugeField<Float>(out, in, gauge::FloatNOrder<Float,18,2,11>(mom), dt, mom, conj_mom, exact, location);
      } else {
	errorQuda("Reconstruction type not supported");
      }
    } else if (mom.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(out, in, gauge::MILCOrder<Float,10>(mom), dt, mom, conj_mom, exact, location);
    } else {
      errorQuda("Gauge Field order %d not supported", mom.Order());
    }

  }

  template <typename Float>
  void updateGaugeField(GaugeField &out, const GaugeField &in, const GaugeField &mom, 
			double dt, bool conj_mom, bool exact, 
			QudaFieldLocation location) {

    const int Nc = 3;
    if (out.Ncolor() != Nc) 
      errorQuda("Ncolor=%d not supported at this time", out.Ncolor());

    if (out.Order() != in.Order() || out.Reconstruct() != in.Reconstruct()) {
      errorQuda("Input and output gauge field ordering and reconstruction must match");
    }

    if (out.isNative()) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	updateGaugeField<Float>(G(out),G(in), mom, dt, conj_mom, exact, location);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	updateGaugeField<Float>(G(out), G(in), mom, dt, conj_mom, exact, location);
      } else {
	errorQuda("Reconstruction type not supported");
      }
    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(gauge::MILCOrder<Float, Nc*Nc*2>(out),
			      gauge::MILCOrder<Float, Nc*Nc*2>(in), 
			      mom, dt, conj_mom, exact, location);
    } else {
      errorQuda("Gauge Field order %d not supported", out.Order());
    }

  }
#endif

  void updateGaugeField(GaugeField &out, double dt, const GaugeField& in, 
			const GaugeField& mom, bool conj_mom, bool exact)
  {
#ifdef GPU_GAUGE_TOOLS
    if (out.Precision() != in.Precision() || out.Precision() != mom.Precision())
      errorQuda("Gauge and momentum fields must have matching precision");

    if (out.Location() != in.Location() || out.Location() != mom.Location())
      errorQuda("Gauge and momentum fields must have matching location");

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      updateGaugeField<double>(out, in, mom, dt, conj_mom, exact, out.Location());
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      updateGaugeField<float>(out, in, mom, dt, conj_mom, exact, out.Location());
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    }
#else
  errorQuda("Gauge tools are not build");
#endif

  }

} // namespace quda
