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

  /**
     Direct port of the TIFR expsu3 algorithm
  */
  template <typename complex, typename Cmplx>
  __device__ __host__ void expsu3(Matrix<Cmplx,3> &q, int x) {
    typedef typename RealTypeId<Cmplx>::Type real;

    complex a2 = (q(3)*q(1)+q(7)*q(5)+q(6)*q(2) - 
		  (q(0)*q(4)+(q(0)+q(4))*q(8))) / (real)3.0 ;
    complex a3 = q(0)*q(4)*q(8) + q(1)*q(5)*q(6) + q(2)*q(3)*q(7) - 
      q(6)*q(4)*q(2) - q(3)*q(1)*q(8) - q(0)*q(7)*q(5);

    complex sg2h3 = sqrt(a3*a3-(real)4.*a2*a2*a2);
    complex cp = exp( log((real)0.5*(a3+sg2h3)) / (real)3.0);
    complex cm = a2/cp;

    complex r1 = exp( complex(0.0,1.0)*(real)(2.0*M_PI/3.0));
    complex r2 = exp(-complex(0.0,1.0)*(real)(2.0*M_PI/3.0));

    complex w1[3];
      
    w1[0]=cm+cp;
    w1[1]=r1*cp+r2*cm;
    w1[2]=r2*cp+r1*cm;
    complex z1=q(1)*q(6)-q(0)*q(7);
    complex z2=q(3)*q(7)-q(4)*q(6);

    complex al = w1[0];
    complex wr21 = (z1+al*q(7)) / (z2+al*q(6));
    complex wr31 = (al-q(0)-wr21*q(3))/q(6);

    al=w1[1];
    complex wr22 = (z1+al*q(7))/(z2+al*q(6));
    complex wr32 = (al-q(0)-wr22*q(3))/q(6);

    al=w1[2];
    complex wr23 = (z1+al*q(7))/(z2+al*q(6));
    complex wr33 = (al-q(0)-wr23*q(3))/q(6);

    z1=q(3)*q(2) - q(0)*q(5);
    z2=q(1)*q(5) - q(4)*q(2);

    al=w1[0];
    complex wl21 = conj((z1+al*q(5))/(z2+al*q(2)));
    complex wl31 = conj((al-q(0)-conj(wl21)*q(1))/q(2));

    al=w1[1];
    complex wl22 = conj((z1+al*q(5))/(z2+al*q(2)));
    complex wl32 = conj((al-q(0)-conj(wl22)*q(1))/q(2));

    al=w1[2];
    complex wl23 = conj((z1+al*q(5))/(z2+al*q(2)));
    complex wl33 = conj((al-q(0)-conj(wl23)*q(1))/q(2));

    complex xn1 = (real)1. + wr21*conj(wl21) + wr31*conj(wl31);
    complex xn2 = (real)1. + wr22*conj(wl22) + wr32*conj(wl32);
    complex xn3 = (real)1. + wr23*conj(wl23) + wr33*conj(wl33);

    complex d1 = exp(w1[0]);
    complex d2 = exp(w1[1]);
    complex d3 = exp(w1[2]);
    complex y11 = d1/xn1;
    complex y12 = d2/xn2;
    complex y13 = d3/xn3;
    complex y21 = wr21*d1/xn1;
    complex y22 = wr22*d2/xn2;
    complex y23 = wr23*d3/xn3;
    complex y31 = wr31*d1/xn1;
    complex y32 = wr32*d2/xn2;
    complex y33 = wr33*d3/xn3;
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

  /**
     Direct port of TIFR vmcm2 routine
     Vector matrix multiply: b = (a^\dag).b
  */
  template <typename complex, typename Cmplx>
  __device__ __host__ void vmcm2(Matrix<Cmplx,3> a, Matrix<Cmplx,3> &b) {
    complex c[9];
    c[0] = conj(a(0))*b(0)+conj(a(1))*b(1) + conj(a(2))*b(2);
    c[3] = conj(a(0))*b(3)+conj(a(1))*b(4) + conj(a(2))*b(5);
    c[6] = conj(a(0))*b(6)+conj(a(1))*b(7) + conj(a(2))*b(8);
    c[1] = conj(a(3))*b(0)+conj(a(4))*b(1) + conj(a(5))*b(2);
    c[4] = conj(a(3))*b(3)+conj(a(4))*b(4) + conj(a(5))*b(5);
    c[7] = conj(a(3))*b(6)+conj(a(4))*b(7) + conj(a(5))*b(8);
    c[2] = conj(a(6))*b(0)+conj(a(7))*b(1) + conj(a(8))*b(2);
    c[5] = conj(a(6))*b(3)+conj(a(7))*b(4) + conj(a(8))*b(5);
    c[8] = conj(a(6))*b(6)+conj(a(7))*b(7) + conj(a(8))*b(8);
    for (int i=0; i<9; i++) b(i) = c[i];
  }

  template<typename Cmplx, typename Gauge, typename Mom, int N, 
	   bool conj_mom, bool exact>
  __device__ __host__  void updateGaugeFieldCompute
  (UpdateGaugeArg<Cmplx,Gauge,Mom> &arg, int x, int parity) {

    typedef typename RealTypeId<Cmplx>::Type real;
    Matrix<Cmplx,3> link, result, mom;
    for(int dir=0; dir<arg.nDim; ++dir){
      arg.in.load((real*)(link.data), x, dir, parity);
      arg.momentum.load((real*)(mom.data), x, dir, parity);

      Cmplx trace = getTrace(mom);
      mom(0,0) -= trace/3.0;
      mom(1,1) -= trace/3.0;
      mom(2,2) -= trace/3.0;

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
	expsu3<complex<real> >(mom, x+dir+parity);

	if (!conj_mom) {
	  link = mom * link;
	} else {
	  link = conj(mom) * link;
	  //vmcm2<complex<real> >(mom, link);
	}

	result = link;
      }

      arg.out.save((real*)(result.data), x, dir, parity);
    } // dir

  }

  template<typename Cmplx, typename Gauge, typename Mom, int N, 
	   bool conj_mom, bool exact>
  void updateGaugeField(UpdateGaugeArg<Cmplx,Gauge,Mom> arg) {

    for (unsigned int parity=0; parity<2; parity++) {
      for (unsigned int x=0; x<arg.out.volumeCB; x++) {
	updateGaugeFieldCompute<Cmplx,Gauge,Mom,N,conj_mom,exact>
	  (arg, x, parity);
      }
    }
  }

  template<typename Cmplx, typename Gauge, typename Mom, int N, 
	   bool conj_mom, bool exact>
  __global__ void updateGaugeFieldKernel(UpdateGaugeArg<Cmplx,Gauge,Mom> arg) { 
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 2*arg.out.volumeCB) return;
    int parity = (idx >= arg.out.volumeCB) ? 1 : 0;
    idx -= parity*arg.out.volumeCB;

    updateGaugeFieldCompute<Cmplx,Gauge,Mom,N,conj_mom,exact>
      (arg, idx, parity);
 }
   
  template <typename Complex, typename Gauge, typename Mom, int N, 
	    bool conj_mom, bool exact>
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
    UpdateGaugeField(const UpdateGaugeArg<Complex,Gauge,Mom> &arg, const int *X, QudaFieldLocation location) 
      : arg(arg), X(X), location(location) {}
    virtual ~UpdateGaugeField() {}
    
    void apply(const cudaStream_t &stream){
      if (location == QUDA_CUDA_FIELD_LOCATION) {
#if __COMPUTE_CAPABILITY__ >= 200
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	updateGaugeFieldKernel<Complex,Gauge,Mom,N,conj_mom,exact>
	  <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
	errorQuda("Not supported on pre-Fermi architecture");
#endif
      } else { // run the CPU code
	updateGaugeField<Complex,Gauge,Mom,N,conj_mom,exact>(arg);
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
			double dt, const int *X, bool conj_mom, bool exact, 
			QudaFieldLocation location) {
    // degree of exponential expansion
    const int N = 8;

    typedef typename ComplexTypeId<Float>::Type Complex;
    if (conj_mom) {
      if (exact) {
	UpdateGaugeArg<Complex, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Complex,Gauge,Mom,N,true,true> updateGauge(arg, X, location);
	updateGauge.apply(0); 
      } else {
	UpdateGaugeArg<Complex, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Complex,Gauge,Mom,N,true,false> updateGauge(arg, X, location);
	updateGauge.apply(0); 
      }
    } else {
      if (exact) {
	UpdateGaugeArg<Complex, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Complex,Gauge,Mom,N,false,true> updateGauge(arg, X, location);
	updateGauge.apply(0);
      } else {
	UpdateGaugeArg<Complex, Gauge, Mom> arg(out, in, mom, dt, 4);
	UpdateGaugeField<Complex,Gauge,Mom,N,false,false> updateGauge(arg, X, location);
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
	updateGaugeField<Float>(out, in, FloatNOrder<Float,18,2,11>(mom), dt, mom.X(), conj_mom, exact, location);
      } else {
	errorQuda("Reconstruction type not supported");
      }
    } else if (mom.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(out, in, MILCOrder<Float,10>(mom), dt, mom.X(), conj_mom, exact, location);
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

    if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 2, 18>(out),
				FloatNOrder<Float, Nc*Nc*2, 2, 18>(in), 
				mom, dt, conj_mom, exact, location);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 2, 12>(out),
				FloatNOrder<Float, Nc*Nc*2, 2, 12>(in), 
				mom, dt, conj_mom, exact, location);
      } else {
	errorQuda("Reconstruction type not supported");
      }
    } else if (out.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	updateGaugeField<Float>(FloatNOrder<Float, Nc*Nc*2, 4, 12>(out),
				FloatNOrder<Float, Nc*Nc*2, 4, 12>(in), 
				mom, dt, conj_mom, exact,  location);
      } else {
	errorQuda("Reconstruction type %d not supported", out.Order());
      }
    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
      updateGaugeField<Float>(MILCOrder<Float, Nc*Nc*2>(out),
			      MILCOrder<Float, Nc*Nc*2>(in), 
			      mom, dt, conj_mom, exact, location);
    } else {
      errorQuda("Gauge Field order %d not supported", out.Order());
    }

  }

  void updateGaugeField(GaugeField &out, double dt, const GaugeField& in, 
			const GaugeField& mom, bool conj_mom, bool exact)
  {
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

  }

} // namespace quda
