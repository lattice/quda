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

  /**
     Direct port of the TIFR expsu3 algorithm
  */
  template <typename Float>
  __device__ __host__ void expsu3(Matrix<complex<Float>,3> &q, int x) {
    typedef complex<Float> Complex;

    Complex a2 = (q(3)*q(1)+q(7)*q(5)+q(6)*q(2) -
		  (q(0)*q(4)+(q(0)+q(4))*q(8))) / (Float)3.0 ;
    Complex a3 = q(0)*q(4)*q(8) + q(1)*q(5)*q(6) + q(2)*q(3)*q(7) -
      q(6)*q(4)*q(2) - q(3)*q(1)*q(8) - q(0)*q(7)*q(5);

    Complex sg2h3 = sqrt(a3*a3-(Float)4.*a2*a2*a2);
    Complex cp = exp( log((Float)0.5*(a3+sg2h3)) / (Float)3.0);
    Complex cm = a2/cp;

    Complex r1 = exp( Complex(0.0,1.0)*(Float)(2.0*M_PI/3.0));
    Complex r2 = exp(-Complex(0.0,1.0)*(Float)(2.0*M_PI/3.0));

    Complex w1[3];
      
    w1[0]=cm+cp;
    w1[1]=r1*cp+r2*cm;
    w1[2]=r2*cp+r1*cm;
    Complex z1=q(1)*q(6)-q(0)*q(7);
    Complex z2=q(3)*q(7)-q(4)*q(6);

    Complex al = w1[0];
    Complex wr21 = (z1+al*q(7)) / (z2+al*q(6));
    Complex wr31 = (al-q(0)-wr21*q(3))/q(6);

    al=w1[1];
    Complex wr22 = (z1+al*q(7))/(z2+al*q(6));
    Complex wr32 = (al-q(0)-wr22*q(3))/q(6);

    al=w1[2];
    Complex wr23 = (z1+al*q(7))/(z2+al*q(6));
    Complex wr33 = (al-q(0)-wr23*q(3))/q(6);

    z1=q(3)*q(2) - q(0)*q(5);
    z2=q(1)*q(5) - q(4)*q(2);

    al=w1[0];
    Complex wl21 = conj((z1+al*q(5))/(z2+al*q(2)));
    Complex wl31 = conj((al-q(0)-conj(wl21)*q(1))/q(2));

    al=w1[1];
    Complex wl22 = conj((z1+al*q(5))/(z2+al*q(2)));
    Complex wl32 = conj((al-q(0)-conj(wl22)*q(1))/q(2));

    al=w1[2];
    Complex wl23 = conj((z1+al*q(5))/(z2+al*q(2)));
    Complex wl33 = conj((al-q(0)-conj(wl23)*q(1))/q(2));

    Complex xn1 = (Float)1. + wr21*conj(wl21) + wr31*conj(wl31);
    Complex xn2 = (Float)1. + wr22*conj(wl22) + wr32*conj(wl32);
    Complex xn3 = (Float)1. + wr23*conj(wl23) + wr33*conj(wl33);

    Complex d1 = exp(w1[0]);
    Complex d2 = exp(w1[1]);
    Complex d3 = exp(w1[2]);
    Complex y11 = d1/xn1;
    Complex y12 = d2/xn2;
    Complex y13 = d3/xn3;
    Complex y21 = wr21*d1/xn1;
    Complex y22 = wr22*d2/xn2;
    Complex y23 = wr23*d3/xn3;
    Complex y31 = wr31*d1/xn1;
    Complex y32 = wr32*d2/xn2;
    Complex y33 = wr33*d3/xn3;
    q(0) = y11 + y12 + y13;
    q(1) = y21 + y22 + y23;
    q(2) = y31 + y32 + y33;
    q(3) = y11*conj(wl21) + y12*conj(wl22) + y13*conj(wl23);
    q(4) = y21*conj(wl21) + y22*conj(wl22) + y23*conj(wl23);
    q(5) = y31*conj(wl21) + y32*conj(wl22) + y33*conj(wl23);
    q(6) = y11*conj(wl31) + y12*conj(wl32) + y13*conj(wl33);
    q(7) = y21*conj(wl31) + y22*conj(wl32) + y23*conj(wl33);
    q(8) = y31*conj(wl31) + y32*conj(wl32) + y33*conj(wl33);
  }

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
	expsu3<Float>(mom, x+dir+parity);

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
