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
#ifdef MULTI_GPU
      int border[4]; 
#endif
      Fmunu f;
      Gauge gauge;
    
    FmunuArg(Fmunu& f, Gauge &gauge, const GaugeField &meta)
      : threads(meta.Volume()), f(f), gauge(gauge) { 
      for(int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
      
#ifdef MULTI_GPU
      for(int dir=0; dir<4; ++dir){
	border[dir] = 2;
      }
#endif
    }
  };

  template <typename Float, typename Fmunu, typename GaugeOrder>
    __host__ __device__ void computeFmunuCore(FmunuArg<Float,Fmunu,GaugeOrder>& arg, int idx) {

      // compute spacetime dimensions and parity
      int parity = 0;
      if(idx >= arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }

      int X[4]; 
      for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

      int x[4];
      getCoords(x, idx, X, parity);
#ifdef MULTI_GPU
      for(int dir=0; dir<4; ++dir){
           x[dir] += arg.border[dir];
           X[dir] += 2*arg.border[dir];
      }
#endif

      typedef typename ComplexTypeId<Float>::Type Cmplx;

      for (int mu=0; mu<4; mu++) {
        for (int nu=0; nu<mu; nu++) {
          Matrix<Cmplx,3> F;
          setZero(&F);
          { // U(x,mu) U(x+mu,nu) U[dagger](x+nu,mu) U[dagger](x,nu)

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            arg.gauge.load((Float*)(U1.data),linkIndexShift(x,dx,X), mu, parity); 
            // load U(x+mu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]++;
            arg.gauge.load((Float*)(U2.data),linkIndexShift(x,dx,X), nu, 1-parity); 
            dx[mu]--;
   

            Matrix<Cmplx,3> Ftmp = U1 * U2;

            // load U(x+nu)_(+mu)
            Matrix<Cmplx,3> U3;
            dx[nu]++;
            arg.gauge.load((Float*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity); 
            dx[nu]--;

            Ftmp = Ftmp * conj(U3) ;

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U4;
            arg.gauge.load((Float*)(U4.data),linkIndexShift(x,dx,X), nu, parity); 

            // complete the plaquette
            F = Ftmp * conj(U4);
          }


          { // U(x,nu) U[dagger](x+nu-mu,mu) U[dagger](x-mu,nu) U(x-mu, mu)

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            arg.gauge.load((Float*)(U1.data), linkIndexShift(x,dx,X), nu, parity);

            // load U(x+nu)_(-mu) = U(x+nu-mu)_(+mu)
            Matrix<Cmplx,3> U2;
            dx[nu]++;
            dx[mu]--;
            arg.gauge.load((Float*)(U2.data), linkIndexShift(x,dx,X), mu, parity);
            dx[mu]++;
            dx[nu]--;

            Matrix<Cmplx,3> Ftmp =  U1 * conj(U2);

            // load U(x-mu)_nu
            Matrix<Cmplx,3> U3;
            dx[mu]--;
            arg.gauge.load((Float*)(U3.data), linkIndexShift(x,dx,X), nu, 1-parity);
            dx[mu]++;

            Ftmp =  Ftmp * conj(U3);

            // load U(x)_(-mu) = U(x-mu)_(+mu)
            Matrix<Cmplx,3> U4;
            dx[mu]--;
            arg.gauge.load((Float*)(U4.data), linkIndexShift(x,dx,X), mu, 1-parity);
            dx[mu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp;
          }

          { // U[dagger](x-nu,nu) U(x-nu,mu) U(x+mu-nu,nu) U[dagger](x,mu)


            // load U(x)_(-nu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[nu]--;
            arg.gauge.load((Float*)(U1.data), linkIndexShift(x,dx,X), nu, 1-parity);
            dx[nu]++;

            // load U(x-nu)_(+mu)
            Matrix<Cmplx,3> U2;
            dx[nu]--;
            arg.gauge.load((Float*)(U2.data), linkIndexShift(x,dx,X), mu, 1-parity);
            dx[nu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * U2;

            // load U(x+mu-nu)_(+nu)
            Matrix<Cmplx,3> U3;
            dx[mu]++;
            dx[nu]--;
            arg.gauge.load((Float*)(U3.data), linkIndexShift(x,dx,X), nu, parity);
            dx[nu]++;
            dx[mu]--;

            Ftmp = Ftmp * U3;

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U4;
            arg.gauge.load((Float*)(U4.data), linkIndexShift(x,dx,X), mu, parity);

            Ftmp = Ftmp * conj(U4);

            // sum this contribution to Fmunu
            F += Ftmp;
          }

          { // U[dagger](x-mu,mu) U[dagger](x-mu-nu,nu) U(x-mu-nu,mu) U(x-nu,nu)


            // load U(x)_(-mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[mu]--;
            arg.gauge.load((Float*)(U1.data), linkIndexShift(x,dx,X), mu, 1-parity);
            dx[mu]++;



            // load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]--;
            dx[nu]--;
            arg.gauge.load((Float*)(U2.data), linkIndexShift(x,dx,X), nu, parity);
            dx[nu]++;
            dx[mu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * conj(U2);

            // load U(x-nu)_mu
            Matrix<Cmplx,3> U3;
            dx[mu]--;
            dx[nu]--;
            arg.gauge.load((Float*)(U3.data), linkIndexShift(x,dx,X), mu, parity);
            dx[nu]++;
            dx[mu]++;

            Ftmp = Ftmp * U3;

            // load U(x)_(-nu) = U(x-nu)_(+nu)
            Matrix<Cmplx,3> U4;
            dx[nu]--;
            arg.gauge.load((Float*)(U4.data), linkIndexShift(x,dx,X), nu, 1-parity);
            dx[nu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp;

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
            F *= 1.0/8.0; // 18 real multiplications
            // 36 floating point operations here
          }
          

          int munu_idx = (mu*(mu-1))/2 + nu; // lower-triangular indexing
	  arg.f.save((Float*)(F.data), idx, munu_idx, parity);
        } // nu < mu
      } // mu
      // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
      return;
    }


  template<typename Float, typename Fmunu, typename Gauge>
  __global__ void computeFmunuKernel(FmunuArg<Float,Fmunu,Gauge> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= arg.threads) return;
    computeFmunuCore(arg,idx);
  }
  
  template<typename Float, typename Fmunu, typename Gauge>
  void computeFmunuCPU(FmunuArg<Float,Fmunu,Gauge>& arg){
    for(int idx=0; idx<arg.threads; idx++){
      computeFmunuCore(arg,idx);
    }
  }


  template<typename Float, typename Fmunu, typename Gauge>
    class FmunuCompute : Tunable {
      FmunuArg<Float,Fmunu,Gauge> arg;
      const GaugeField &meta;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      FmunuCompute(FmunuArg<Float,Fmunu,Gauge> &arg, const GaugeField &meta, QudaFieldLocation location)
        : arg(arg), meta(meta), location(location) {
	writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,sizeof(Float));
      }
      virtual ~FmunuCompute() {}

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
#if (__COMPUTE_CAPABILITY__ >= 200)
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          computeFmunuKernel<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);  
#else
	  errorQuda("computeFmunuKernel not supported on pre-Fermi architecture");
#endif
        }else{
          computeFmunuCPU(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(meta.VolString(), typeid(*this).name(), aux);
      }


      std::string paramString(const TuneParam &param) const {
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return (2430 + 36)*6*arg.threads; }
      long long bytes() const { return (4*4*18 + 18)*6*arg.threads*sizeof(Float); } //  Ignores link reconstruction

    }; // FmunuCompute



  template<typename Float, typename Fmunu, typename Gauge>
  void computeFmunu(Fmunu f_munu, Gauge gauge, const GaugeField &meta, QudaFieldLocation location) {
    FmunuArg<Float,Fmunu,Gauge> arg(f_munu, gauge, meta);
    FmunuCompute<Float,Fmunu,Gauge> fmunuCompute(arg, meta, location);
    fmunuCompute.apply(0);
    cudaDeviceSynchronize();
    checkCudaError();
  }

  template<typename Float>
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge, QudaFieldLocation location) {
    if (Fmunu.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (gauge.isNative()) {
	typedef FloatNOrder<Float, 18, 2, 18> F;

	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, location);  
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, location);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_8) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, location);
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

