#include <gauge_field_order.h>
#include <comm_quda.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <tune_quda.h>

/**
   This code has not been checked.  In particular, I suspect it is
 erroneous in multi-GPU since it looks like the halo ghost region
 isn't being treated here.
 */

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, int Nc, typename Order>
  struct GaugePhaseArg {
    static constexpr int nColor = Nc;
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
    if (phaseType == QUDA_STAGGERED_PHASE_MILC) {
      if (dim==0) {
	phase = (1.0 - 2.0 * (t % 2) );		
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((t + x) % 2) );	
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((t + x + y) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } else if (phaseType == QUDA_STAGGERED_PHASE_TIFR) {
      if (dim==0) {
	phase = (1.0 - 2.0 * ((3 + t + z + y) % 2) );		
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((2 + t + z) % 2) );	
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((1 + t) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } else if (phaseType == QUDA_STAGGERED_PHASE_CPS) {
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

  template <typename Float, QudaStaggeredPhase phaseType, int dim, typename Arg>
  __device__ __host__ void gaugePhase(int indexCB, int parity, Arg &arg) {
    typedef typename mapper<Float>::type real;

    int x[4];
    getCoords(x, indexCB, arg.X, parity);

    real phase = getPhase<dim,Float,phaseType>(x[0], x[1], x[2], x[3], arg);
    Matrix<complex<real>,Arg::nColor> u = arg.order(dim, indexCB, parity);
    u *= phase;

    // apply imaginary chemical potential if needed
    if (dim==3 && arg.i_mu != 0.0) u *= arg.i_mu_phase;

    arg.order(dim, indexCB, parity) = u;
  }

  /**
     Generic CPU staggered phase application
  */
  template <typename Float, QudaStaggeredPhase phaseType, typename Arg>
  void gaugePhase(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int indexCB=0; indexCB < arg.threads; indexCB++) {
	gaugePhase<Float,phaseType,0>(indexCB, parity, arg);
	gaugePhase<Float,phaseType,1>(indexCB, parity, arg);
	gaugePhase<Float,phaseType,2>(indexCB, parity, arg);
	gaugePhase<Float,phaseType,3>(indexCB, parity, arg);
      }
    }
  }

  /**
     Generic GPU staggered phase application
  */
  template <typename Float, QudaStaggeredPhase phaseType, typename Arg>
  __global__ void gaugePhaseKernel(Arg arg) {
    int indexCB = blockIdx.x * blockDim.x + threadIdx.x; 	
    if (indexCB >= arg.threads) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    gaugePhase<Float,phaseType,0>(indexCB, parity, arg);
    gaugePhase<Float,phaseType,1>(indexCB, parity, arg);
    gaugePhase<Float,phaseType,2>(indexCB, parity, arg);
    gaugePhase<Float,phaseType,3>(indexCB, parity, arg);
  }

  template <typename Float, QudaStaggeredPhase phaseType, typename Arg>
  class GaugePhase : TunableVectorY {
    Arg &arg;
    const GaugeField &meta; // used for meta data only

  private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugePhase(Arg &arg, const GaugeField &meta)
      : TunableVectorY(2), arg(arg), meta(meta) {
      writeAuxString("stride=%d,prec=%lu",arg.order.stride,sizeof(Float));
    }
    virtual ~GaugePhase() { ; }
  
    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	gaugePhaseKernel<Float, phaseType, Arg>
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      } else {
	gaugePhase<Float, phaseType, Arg>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    void preTune() { arg.order.save(); }
    void postTune() { arg.order.load(); }

    long long flops() const { return 0; } 
    long long bytes() const { return 2 * arg.threads * 2 * arg.order.Bytes(); } // parity * e/o volume * i/o * vec size
  };


  template <typename Float, int Nc, typename Order>
  void gaugePhase(Order order, const GaugeField &u) {
    if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
      GaugePhaseArg<Float,Nc,Order> arg(order, u);
      GaugePhase<Float,QUDA_STAGGERED_PHASE_MILC,
		 GaugePhaseArg<Float,Nc,Order> > phase(arg, u);
      phase.apply(0);
    } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_CPS) {
      GaugePhaseArg<Float,Nc,Order> arg(order, u);
      GaugePhase<Float,QUDA_STAGGERED_PHASE_CPS,
		 GaugePhaseArg<Float,Nc,Order> > phase(arg, u);
      phase.apply(0);
    } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
      GaugePhaseArg<Float,Nc,Order> arg(order, u);
      GaugePhase<Float,QUDA_STAGGERED_PHASE_TIFR,
		 GaugePhaseArg<Float,Nc,Order> > phase(arg, u);
      phase.apply(0);
    } else {
      errorQuda("Undefined phase type");
    }

    if (u.Location() == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  /** This is the template driver for gaugePhase */
  template <typename Float>
  void gaugePhase(GaugeField &u) {
    if (u.Ncolor() != 3) errorQuda("Unsupported number of colors %d", u.Ncolor());
    constexpr int Nc = 3;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	gaugePhase<Float,Nc>(G(u), u);
      } else {
	errorQuda("Unsupported reconstruction type");
      }
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

    if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) {
      // ensure that ghosts are updated if needed
      u.exchangeGhost();
    }

  }

} // namespace quda
