#include <gauge_field_order.h>
#include <comm_quda.h>
#include <complex_quda.h>
#include <index_helper.cuh>

/**
   This code has not been checked.  In particular, I suspect it is
 erroneous in multi-GPU since it looks like the halo ghost region
 isn't being treated here.
 */

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename Order>
  struct GaugePhaseArg {
    Order order;
    int X[4];
    int threads;
    Float tBoundary;
    Float i_mu;
    complex<Float> i_mu_phase;
    GaugePhaseArg(const Order &order, const GaugeField &u) 
      : order(order), threads(u.VolumeCB()), i_mu(u.iMu())
    {
      // if staggered phases are applied, then we are removing them
      // else we are applying them
      Float dir = u.StaggeredPhaseApplied() ? -1.0 : 1.0;

      i_mu_phase = complex<Float>( cos(M_PI * u.iMu() / (u.X()[3]*comm_dim(3)) ), 
				   dir * sin(M_PI * u.iMu() / (u.X()[3]*comm_dim(3))) );

      for (int d=0; d<4; d++) X[d] = u.X()[d];

      // only set the boundary condition on the last time slice of nodes
#ifdef MULTI_GPU
      bool last_node_in_t = (commCoords(3) == commDim(3)-1);
#else
      bool last_node_in_t = true;
#endif
      tBoundary = (Float)(last_node_in_t ? u.TBoundary() : QUDA_PERIODIC_T);
    }
    GaugePhaseArg(const GaugePhaseArg &arg) 
      : order(arg.order), tBoundary(arg.tBoundary), threads(arg.threads), 
	i_mu(arg.i_mu), i_mu_phase(arg.i_mu_phase) {
      for (int d=0; d<4; d++) X[d] = arg.X[d];
    }
  };

  
  
  // FIXME need to check this with odd local volumes
  template <int dim, typename Float, QudaStaggeredPhase phaseType, typename Arg>
  __device__ __host__ Float getPhase(int x, int y, int z, int t, Arg &arg) {
    Float phase = 1.0;
    if (phaseType == QUDA_MILC_STAGGERED_PHASE) {
      if (dim==0) {
	phase = (1.0 - 2.0 * (t % 2) );		
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((t + x) % 2) );	
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((t + x + y) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } if (phaseType == QUDA_TIFR_STAGGERED_PHASE) {
      if (dim==0) {
	phase = (1.0 - 2.0 * ((3 + t + z + y) % 2) );		
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((2 + t + z) % 2) );	
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((1 + t) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } else if (phaseType == QUDA_CPS_STAGGERED_PHASE) {
      if (dim==0) {
	phase = 1.0;
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((1 + x) % 2) );	
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((1 + x + y) % 2) );		
      } else if (dim == 3) { // also apply boundary condition
	phase = ((t == arg.X[3]-1) ? arg.tBoundary : 1.0) * 
	  (1.0 - 2 * ((1 + x + y + z) % 2) );
      }
    } 
    return phase;
  }

  template <typename Float, int length, QudaStaggeredPhase phaseType, int dim, typename Arg>
  __device__ __host__ void gaugePhase(int indexCB, int parity, Arg &arg) {
    typedef typename mapper<Float>::type RegType;

    int x[4];
    getCoords(x, indexCB, arg.X, parity);

    RegType phase = getPhase<dim,Float,phaseType>(x[0], x[1], x[2], x[3], arg);
    RegType u[length];
    arg.order.load(u, indexCB, dim, parity);
    for (int i=0; i<length; i++) u[i] *= phase;

    // apply imaginary chemical potential if needed
    if (dim==3 && arg.i_mu != 0.0) {
      complex<RegType>* v = reinterpret_cast<complex<RegType>*>(u);
      for (int i=0; i<length/2; i++) v[i] *= arg.i_mu_phase;
    }

    arg.order.save(u, indexCB, dim, parity);
  }

  /**
     Generic CPU staggered phase application
  */
  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  void gaugePhase(Arg &arg) {  
    for (int parity=0; parity<2; parity++) {
      for (int indexCB=0; indexCB < arg.threads; indexCB++) {
	gaugePhase<Float,length,phaseType,0>(indexCB, parity, arg);
	gaugePhase<Float,length,phaseType,1>(indexCB, parity, arg);
	gaugePhase<Float,length,phaseType,2>(indexCB, parity, arg);
	gaugePhase<Float,length,phaseType,3>(indexCB, parity, arg);
      }
    }
  }

  /**
     Generic GPU staggered phase application
  */
  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  __global__ void gaugePhaseKernel(Arg arg) {  
    int indexCB = blockIdx.x * blockDim.x + threadIdx.x; 	
    if (indexCB >= arg.threads) return;
    int parity = blockIdx.y;

    gaugePhase<Float,length,phaseType,0>(indexCB, parity, arg);
    gaugePhase<Float,length,phaseType,1>(indexCB, parity, arg);
    gaugePhase<Float,length,phaseType,2>(indexCB, parity, arg);
    gaugePhase<Float,length,phaseType,3>(indexCB, parity, arg);
  }

  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  class GaugePhase : Tunable {
    Arg &arg;
    const GaugeField &meta; // used for meta data only
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugePhase(Arg &arg, const GaugeField &meta, QudaFieldLocation location) 
      : arg(arg), meta(meta), location(location) { 
      writeAuxString("stride=%d,prec=%lu",arg.order.stride,sizeof(Float));
    }
    virtual ~GaugePhase() { ; }
  
    bool advanceBlockDim(TuneParam &param) const {
      bool rtn = Tunable::advanceBlockDim(param);
      param.grid.y = 2;
      return rtn;
    }
    
    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid.y = 2;
    }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	tp.grid.y = 2; // parity is the y grid dimension
	gaugePhaseKernel<Float, length, phaseType, Arg> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      } else {
	gaugePhase<Float, length, phaseType, Arg>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    void preTune() { arg.order.save(); }
    void postTune() { arg.order.load(); }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { return 2 * arg.threads * 2 * arg.order.Bytes(); } // parity * e/o volume * i/o * vec size
  };


  template <typename Float, int length, typename Order>
  void gaugePhase(Order order, const GaugeField &u,  QudaFieldLocation location) {
    if (u.StaggeredPhase() == QUDA_MILC_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, u);
      GaugePhase<Float,length,QUDA_MILC_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, u, location);
      phase.apply(0);
    } else if (u.StaggeredPhase() == QUDA_CPS_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, u);
      GaugePhase<Float,length,QUDA_CPS_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, u, location);
      phase.apply(0);
    } else if (u.StaggeredPhase() == QUDA_TIFR_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, u);
      GaugePhase<Float,length,QUDA_TIFR_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, u, location);
      phase.apply(0);
    } else {
      errorQuda("Undefined phase type");
    }

    if (location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  /** This is the template driver for gaugePhase */
  template <typename Float>
  void gaugePhase(GaugeField &u) {
    const int length = 18;

    QudaFieldLocation location = 
      (typeid(u)==typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	gaugePhase<Float,length>(G(u), u, location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	gaugePhase<Float,length>(G(u), u, location);
      } else {
	errorQuda("Unsupported recsontruction type");
      }
    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      gaugePhase<Float,length>(TIFROrder<Float,length>(u), u, location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

#endif

  void applyGaugePhase(GaugeField &u) {

#ifdef GPU_GAUGE_TOOLS
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      gaugePhase<double>(u);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      gaugePhase<float>(u);
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
#else
    errorQuda("Gauge tools are not build");
#endif

  }

} // namespace quda
