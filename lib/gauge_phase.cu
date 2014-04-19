#include <gauge_field_order.h>
#include <comm_quda.h>

namespace quda {
  template <typename Float, typename Order>
  struct GaugePhaseArg {
    Order order;
    int X[4];
    int volume;
    Float tBoundary;
    GaugePhaseArg(const Order &order, const int *X_, QudaTboundary tBoundary_) 
      : order(order) {
      volume = 1;
      for (int d=0; d<4; d++) {
	X[d] = X_[d];
	volume *= X[d];
      }

      // only set the boundary condition on the last time slice of nodes
#ifdef MULTI_GPU
      bool last_node_in_t = (commCoords(3) == commDim(3)-1);
#else
      bool last_node_in_t = true;
#endif
      tBoundary = (Float)(last_node_in_t ? tBoundary_ : QUDA_PERIODIC_T);
      printf("node=%d Tboundary = %e\n", comm_rank(), tBoundary);
    }
    GaugePhaseArg(const GaugePhaseArg &arg) 
      : order(arg.order), tBoundary(arg.tBoundary), volume(arg.volume) {
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
  __device__ __host__ void gaugePhase(int xh, int y, int z, int t, int parity, Arg &arg) {
    typedef typename mapper<Float>::type RegType;
    int indexCB = ((t*arg.X[2] + z)*arg.X[1] + y)*(arg.X[0]>>1) + xh;
    int x = 2*xh + parity;
    Float phase = getPhase<dim,Float,phaseType>(x, y, z, t, arg);
    //printf("dim=%d xh=%d y=%d z=%d t=%d parity = %d phase = %e\n",
    //	   dim, xh, y, z, t, parity, phase);
    RegType u[length];
    arg.order.load(u, indexCB, dim, parity);
    for (int i=0; i<length; i++) u[i] *= phase;
    arg.order.save(u, indexCB, dim, parity);
  }

  /**
     Generic CPU staggered phase application
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  void gaugePhase(Arg &arg) {  
    for (int parity=0; parity<2; parity++) {
      for (int t=0; t<arg.X[3]; t++) {
	for (int z=0; z<arg.X[2]; z++) {
	  for (int y=0; y<arg.X[1]; y++) {
	    for (int xh=0; xh<arg.X[0]>>1; xh++) {
	      gaugePhase<Float,length,phaseType,0>(xh, y, z, t, parity, arg);
	      gaugePhase<Float,length,phaseType,1>(xh, y, z, t, parity, arg);
	      gaugePhase<Float,length,phaseType,2>(xh, y, z, t, parity, arg);
	      gaugePhase<Float,length,phaseType,3>(xh, y, z, t, parity, arg);
	    }
	  }
	}
      }
    } // parity
  }

  /**
     Generic GPU staggered phase application
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  __global__ void gaugePhaseKernel(Arg arg) {  
    int X = blockIdx.x * blockDim.x + threadIdx.x; 	
    if (X >= (arg.volume>>1)) return;
    int parity = blockIdx.y;

    int tzy = X   / (arg.X[0]>>1);
    int xh  = X   - tzy*(arg.X[0]>>1);
    int tz  = tzy / arg.X[1];
    int y   = tzy - tz*arg.X[1];
    int t   = tz  / arg.X[2];
    int z   = tz  - t * arg.X[2];
    gaugePhase<Float,length,phaseType,0>(xh, y, z, t, parity, arg);
    gaugePhase<Float,length,phaseType,1>(xh, y, z, t, parity, arg);
    gaugePhase<Float,length,phaseType,2>(xh, y, z, t, parity, arg);
    gaugePhase<Float,length,phaseType,3>(xh, y, z, t, parity, arg);
  }

  template <typename Float, int length, QudaStaggeredPhase phaseType, typename Arg>
  class GaugePhase : Tunable {
    Arg &arg;
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volume>>1; }

  public:
    GaugePhase(Arg &arg, QudaFieldLocation location) : arg(arg), location(location) { }
    virtual ~GaugePhase() { ; }
  
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
      std::stringstream vol, aux;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3];    
      aux << "stride=" << arg.order.stride << ",prec=" << sizeof(Float);
      aux << "stride=" << arg.order.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { return arg.volume * 2 * arg.order.Bytes(); } // volume * i/o * vec size
  };


  template <typename Float, int length, typename Order>
  void gaugePhase(Order order, QudaStaggeredPhase phaseType, QudaTboundary tBoundary,
		  const int *X, QudaFieldLocation location) {  
    if (phaseType == QUDA_MILC_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, X, tBoundary);
      GaugePhase<Float,length,QUDA_MILC_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, location);
      phase.apply(0);
    } else if (phaseType == QUDA_CPS_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, X, tBoundary);
      GaugePhase<Float,length,QUDA_CPS_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, location);
      phase.apply(0);
    } else if (phaseType == QUDA_TIFR_STAGGERED_PHASE) {
      GaugePhaseArg<Float,Order> arg(order, X, tBoundary);
      GaugePhase<Float,length,QUDA_TIFR_STAGGERED_PHASE,
		 GaugePhaseArg<Float,Order> > phase(arg, location);
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

    if (u.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  gaugePhase<Float,length>(FloatNOrder<Float,length,2,19>(u), u.StaggeredPhase(), 
				   u.TBoundary(), u.X(), location);
	} else {
	  gaugePhase<Float,length>(FloatNOrder<Float,length,2,18>(u), u.StaggeredPhase(), 
				   u.TBoundary(), u.X(), location);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	gaugePhase<Float,length>(FloatNOrder<Float,length,2,12>(u), u.StaggeredPhase(), 
				 u.TBoundary(), u.X(), location);
      } else {
	errorQuda("Unsupported recsontruction type");
      }
    } else if (u.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  gaugePhase<Float,length>(FloatNOrder<Float,length,1,19>(u), u.StaggeredPhase(), 
				   u.TBoundary(), u.X(), location);
	} else {
	  gaugePhase<Float,length>(FloatNOrder<Float,length,1,18>(u), u.StaggeredPhase(), 
				   u.TBoundary(), u.X(), location);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	gaugePhase<Float,length>(FloatNOrder<Float,length,4,12>(u), u.StaggeredPhase(), 
				 u.TBoundary(), u.X(), location);
      } else {
	errorQuda("Unsupported recsontruction type");
      }
    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      gaugePhase<Float,length>(TIFROrder<Float,length>(u), u.StaggeredPhase(), 
			       u.TBoundary(), u.X(), location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

  void applyGaugePhase(GaugeField &u) {
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      gaugePhase<double>(u);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      gaugePhase<float>(u);
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
  }

} // namespace quda
