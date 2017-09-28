#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template<typename Float, typename Fmunu, typename Gauge>
    struct FmunuArg {
      int threads; // number of active threads required
      int X[4]; // grid dimensions
      int border[4]; 
      Fmunu f;
      Gauge gauge;
    
    FmunuArg(Fmunu &f, Gauge &gauge, const GaugeField &meta, const GaugeField &meta_ex)
      : threads(meta.VolumeCB()), f(f), gauge(gauge) {
      for (int dir=0; dir<4; ++dir) {
	X[dir] = meta.X()[dir];
	border[dir] = (meta_ex.X()[dir] - X[dir])/2;
      }
    }
  };

  template <int mu, int nu, typename Float, typename Arg>
  __device__ __forceinline__ void computeFmunuCore(Arg &arg, int idx, int parity) {

      typedef Matrix<complex<Float>,3> Link;

      int x[4];
      auto &X = arg.X;

      getCoords(x, idx, X, parity);
      for (int dir=0; dir<4; ++dir) {
	x[dir] += arg.border[dir];
	X[dir] += 2*arg.border[dir];
      }

      Link F;
      { // U(x,mu) U(x+mu,nu) U[dagger](x+nu,mu) U[dagger](x,nu)

	// load U(x)_(+mu)
	int dx[4] = {0, 0, 0, 0};
	Link U1 = arg.gauge(mu, linkIndexShift(x,dx,X), parity);

	// load U(x+mu)_(+nu)
	dx[mu]++;
	Link U2 = arg.gauge(nu, linkIndexShift(x,dx,X), 1-parity);
	dx[mu]--;

	// load U(x+nu)_(+mu)
	dx[nu]++;
	Link U3 = arg.gauge(mu, linkIndexShift(x,dx,X), 1-parity);
	dx[nu]--;

	// load U(x)_(+nu)
	Link U4 = arg.gauge(nu, linkIndexShift(x,dx,X), parity);

	// compute plaquette
	F = U1 * U2 * conj(U3) * conj(U4);
      }

      { // U(x,nu) U[dagger](x+nu-mu,mu) U[dagger](x-mu,nu) U(x-mu, mu)

	// load U(x)_(+nu)
	int dx[4] = {0, 0, 0, 0};
	Link U1 = arg.gauge(nu, linkIndexShift(x,dx,X), parity);

	// load U(x+nu)_(-mu) = U(x+nu-mu)_(+mu)
	dx[nu]++;
	dx[mu]--;
	Link U2 = arg.gauge(mu, linkIndexShift(x,dx,X), parity);
	dx[mu]++;
	dx[nu]--;

	// load U(x-mu)_nu
	dx[mu]--;
	Link U3 = arg.gauge(nu, linkIndexShift(x,dx,X), 1-parity);
	dx[mu]++;

	// load U(x)_(-mu) = U(x-mu)_(+mu)
	dx[mu]--;
	Link U4 = arg.gauge(mu, linkIndexShift(x,dx,X),1-parity);
	dx[mu]++;

	// sum this contribution to Fmunu
	F += U1 * conj(U2) * conj(U3) * U4;
      }

      { // U[dagger](x-nu,nu) U(x-nu,mu) U(x+mu-nu,nu) U[dagger](x,mu)

	// load U(x)_(-nu)
	int dx[4] = {0, 0, 0, 0};
	dx[nu]--;
	Link U1 = arg.gauge(nu, linkIndexShift(x,dx,X), 1-parity);
	dx[nu]++;

	// load U(x-nu)_(+mu)
	dx[nu]--;
	Link U2 = arg.gauge(mu, linkIndexShift(x,dx,X), 1-parity);
	dx[nu]++;

	// load U(x+mu-nu)_(+nu)
	dx[mu]++;
	dx[nu]--;
	Link U3 = arg.gauge(nu, linkIndexShift(x,dx,X), parity);
	dx[nu]++;
	dx[mu]--;

	// load U(x)_(+mu)
	Link U4 = arg.gauge(mu, linkIndexShift(x,dx,X), parity);

	// sum this contribution to Fmunu
	F += conj(U1) * U2 * U3 * conj(U4);
      }

      { // U[dagger](x-mu,mu) U[dagger](x-mu-nu,nu) U(x-mu-nu,mu) U(x-nu,nu)

	// load U(x)_(-mu)
	int dx[4] = {0, 0, 0, 0};
	dx[mu]--;
	Link U1 = arg.gauge(mu, linkIndexShift(x,dx,X), 1-parity);
	dx[mu]++;

	// load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
	dx[mu]--;
	dx[nu]--;
	Link U2 = arg.gauge(nu, linkIndexShift(x,dx,X), parity);
	dx[nu]++;
	dx[mu]++;

	// load U(x-nu)_mu
	dx[mu]--;
	dx[nu]--;
	Link U3 = arg.gauge(mu, linkIndexShift(x,dx,X), parity);
	dx[nu]++;
	dx[mu]++;

	// load U(x)_(-nu) = U(x-nu)_(+nu)
	dx[nu]--;
	Link U4 = arg.gauge(nu, linkIndexShift(x,dx,X), 1-parity);
	dx[nu]++;

	// sum this contribution to Fmunu
	F += conj(U1) * conj(U2) * U3 * U4;
      }
      // 3 matrix additions, 12 matrix-matrix multiplications, 8 matrix conjugations
      // Each matrix conjugation involves 9 unary minus operations but these ar not included in the operation count
      // Each matrix addition involves 18 real additions
      // Each matrix-matrix multiplication involves 9*3 complex multiplications and 9*2 complex additions
      // = 9*3*6 + 9*2*2 = 198 floating-point ops
      // => Total number of floating point ops per site above is
      // 3*18 + 12*198 =  54 + 2376 = 2430
      {
	F -= conj(F); // 18 real subtractions + one matrix conjugation
	F *= static_cast<Float>(0.125); // 18 real multiplications
	// 36 floating point operations here
      }

      constexpr int munu_idx = (mu*(mu-1))/2 + nu; // lower-triangular indexing
      arg.f(munu_idx, idx, parity) = F;
   }


  template<typename Float, typename Arg>
  __global__ void computeFmunuKernel(Arg arg){
    int x_cb = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int mu_nu = threadIdx.z + blockIdx.z*blockDim.z;
    if (x_cb >= arg.threads) return;
    if (mu_nu >= 6) return;

    switch(mu_nu) { // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
    case 0: computeFmunuCore<1,0,Float>(arg, x_cb, parity); break;
    case 1: computeFmunuCore<2,0,Float>(arg, x_cb, parity); break;
    case 2: computeFmunuCore<2,1,Float>(arg, x_cb, parity); break;
    case 3: computeFmunuCore<3,0,Float>(arg, x_cb, parity); break;
    case 4: computeFmunuCore<3,1,Float>(arg, x_cb, parity); break;
    case 5: computeFmunuCore<3,2,Float>(arg, x_cb, parity); break;
    }
  }
  
  template<typename Float, typename Arg>
  void computeFmunuCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.threads; x_cb++) {
	for (int mu=0; mu<4; mu++) {
	  for (int nu=0; nu<mu; nu++) {
	    int mu_nu = (mu*(mu-1))/2 + nu;
	    switch(mu_nu) { // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
	    case 0: computeFmunuCore<1,0,Float>(arg, x_cb, parity); break;
	    case 1: computeFmunuCore<2,0,Float>(arg, x_cb, parity); break;
	    case 2: computeFmunuCore<2,1,Float>(arg, x_cb, parity); break;
	    case 3: computeFmunuCore<3,0,Float>(arg, x_cb, parity); break;
	    case 4: computeFmunuCore<3,1,Float>(arg, x_cb, parity); break;
	    case 5: computeFmunuCore<3,2,Float>(arg, x_cb, parity); break;
	    }
	  }
	}
      }
    }
  }


  template<typename Float, typename Arg>
    class FmunuCompute : TunableVectorYZ {
      Arg &arg;
      const GaugeField &meta;
      const QudaFieldLocation location;

    private:
      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      FmunuCompute(Arg &arg, const GaugeField &meta, QudaFieldLocation location)
        : TunableVectorYZ(2,6), arg(arg), meta(meta), location(location) {
	writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,meta.Stride(),sizeof(Float));
      }
      virtual ~FmunuCompute() {}

      void apply(const cudaStream_t &stream){
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          computeFmunuKernel<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
        } else {
          computeFmunuCPU<Float>(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(meta.VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return (2430 + 36)*6*2*(long long)arg.threads; }
      long long bytes() const { return ((16*arg.gauge.Bytes() + arg.f.Bytes())*6*2*arg.threads); } //  Ignores link reconstruction

    }; // FmunuCompute


  template<typename Float, typename Fmunu, typename Gauge>
  void computeFmunu(Fmunu f_munu, Gauge gauge, const GaugeField &meta, const GaugeField &meta_ex, QudaFieldLocation location) {
    FmunuArg<Float,Fmunu,Gauge> arg(f_munu, gauge, meta, meta_ex);
    FmunuCompute<Float,FmunuArg<Float,Fmunu,Gauge> > fmunuCompute(arg, meta, location);
    fmunuCompute.apply(0);
    cudaDeviceSynchronize();
    checkCudaError();
  }

  template<typename Float>
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge, QudaFieldLocation location) {
    if (Fmunu.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (gauge.isNative()) {
	typedef gauge::FloatNOrder<Float, 18, 2, 18> F;

	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_8) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else {
	  errorQuda("Reconstruction type %d not supported", gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge field order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Fmunu field order %d not supported", Fmunu.Order());
    }
    
  }

#endif // GPU_GAUGE_TOOLS

  void computeFmunu(GaugeField &Fmunu, const GaugeField& gauge, QudaFieldLocation location){

#ifdef GPU_GAUGE_TOOLS
    if (Fmunu.Precision() != gauge.Precision()) {
      errorQuda("Fmunu precision %d must match gauge precision %d", Fmunu.Precision(), gauge.Precision());
    }
    
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION){
      computeFmunu<double>(Fmunu, gauge, location);
    } else if(gauge.Precision() == QUDA_SINGLE_PRECISION) {
      computeFmunu<float>(Fmunu, gauge, location);
    } else {
      errorQuda("Precision %d not supported", gauge.Precision());
    }
    return;
#else
    errorQuda("Fmunu has not been built");
#endif // GPU_GAUGE_TOOLS

  }

} // namespace quda

