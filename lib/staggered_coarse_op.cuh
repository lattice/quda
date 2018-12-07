#include <tune_quda.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_coarse_op_kernel.cuh>

#define max_color_per_block 8

namespace quda {

  
  enum ComputeType {
    COMPUTE_VUV,
    COMPUTE_MASS,
    COMPUTE_CONVERT,
    COMPUTE_RESCALE,
    COMPUTE_INVALID
  };

  template <typename Float,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredY : public TunableVectorYZ {

  protected:
    Arg &arg;
    const GaugeField &meta;
    GaugeField &Y;
    GaugeField &X;
    GaugeField &Y_atomic;
    GaugeField &X_atomic;

    int dim;
    QudaDirection dir;
    ComputeType type;

    // NEED TO UPDATE
    long long flops() const
    {
      long long flops_ = 0;
      switch (type) {
      case COMPUTE_VUV:
        flops_ = 0; // permutation
        break;
      case COMPUTE_MASS:
        flops_ = 2l * arg.coarseVolumeCB*coarseSpin*coarseColor;
        break;
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
        // no floating point operations
        flops_ = 0;
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      // 2 from parity, 8 from complex
      return flops_;
    }

    long long bytes() const
    {
      long long bytes_ = 0;
      switch (type) {
      case COMPUTE_VUV:
        bytes_ = 2*2*arg.U.Bytes(); // only needs to read and write non-zero elements
        break; 
      case COMPUTE_MASS:
        bytes_ = 2*2*arg.X.Bytes(); // 2 from i/o, 2 from parity
        break;
      case COMPUTE_CONVERT:
        bytes_ = 2*(arg.X.Bytes() + arg.X_atomic.Bytes() + 8*(arg.Y.Bytes() + arg.Y_atomic.Bytes()));
        bytes_ = dim == 4 ? 2*(arg.X.Bytes() + arg.X_atomic.Bytes()) : 2*(arg.Y.Bytes() + arg.Y_atomic.Bytes());
        break;
      case COMPUTE_RESCALE:
        bytes_ = 2*2*arg.Y.Bytes(); // 2 from i/o, 2 from parity
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      return bytes_;
    }

    unsigned int minThreads() const {
      unsigned int threads = 0;
      switch (type) {
      case COMPUTE_VUV:
        threads = arg.fineVolumeCB;
        break;
      case COMPUTE_MASS:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
        threads = arg.coarseVolumeCB;
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      return threads;
    }

    bool tuneGridDim() const { return false; } // don't tune the grid dimension
    bool tuneAuxDim() const { return false; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return TunableVectorYZ::sharedBytesPerBlock(param);
    }

  public:
    CalculateStaggeredY(Arg &arg, const GaugeField &meta, GaugeField &Y, GaugeField &X, GaugeField &Y_atomic, GaugeField &X_atomic)
      : TunableVectorYZ(2,1), arg(arg), type(COMPUTE_INVALID),
        meta(meta), Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic), dim(0), dir(QUDA_BACKWARDS)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/staggered_coarse_op_kernel.cuh");
#endif
      }
      strcpy(aux, compile_type_str(meta));
      strcpy(aux, meta.AuxString());
      strcat(aux,comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }
    virtual ~CalculateStaggeredY() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

      	if (type == COMPUTE_VUV) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }

      	} else if (type == COMPUTE_MASS) {

      	  AddCoarseStaggeredMassCPU<Float,coarseSpin,coarseColor>(arg);

      	} else if (type == COMPUTE_CONVERT) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;
      	  ConvertStaggeredCPU<Float,coarseSpin,coarseColor>(arg);

        } else if (type == COMPUTE_RESCALE) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;
          RescaleStaggeredYCPU<Float,coarseSpin,coarseColor>(arg);

      	} else {
      	  errorQuda("Undefined compute type %d", type);
      	}
      } else {

      	if (type == COMPUTE_VUV) {

#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::ComputeStaggeredVUVGPU")
            .instantiate(Type<Float>(),dim,dir,fineColor,coarseSpin,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
          if (dir == QUDA_BACKWARDS) {
            if      (dim==0) ComputeStaggeredVUVGPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==1) ComputeStaggeredVUVGPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==2) ComputeStaggeredVUVGPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==3) ComputeStaggeredVUVGPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else if (dir == QUDA_FORWARDS) {
            if      (dim==0) ComputeStaggeredVUVGPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==1) ComputeStaggeredVUVGPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==2) ComputeStaggeredVUVGPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (dim==3) ComputeStaggeredVUVGPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else {
            errorQuda("Undefined direction %d", dir);
          }
#endif // JITIFY

      	} else if (type == COMPUTE_MASS) {

#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::AddCoarseStaggeredMassGPU")
            .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
      	  AddCoarseStaggeredMassGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif

      	} else if (type == COMPUTE_CONVERT) {

      	  arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;
#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::ConvertStaggeredGPU")
            .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
      	  ConvertStaggeredGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif

      	} else if (type == COMPUTE_RESCALE) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;
#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::RescaleYGPU")
            .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
          RescaleStaggeredYGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif

        } else {
      	  errorQuda("Undefined compute type %d", type);
        }
      }
    }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDirection(QudaDirection dir_) { dir = dir_; }

    /**
       Set which computation we are doing
     */
    void setComputeType(ComputeType type_) {
      type = type_;
      switch(type) {
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
        resizeVector(2*coarseColor,coarseColor);
        break;
      case COMPUTE_VUV:
        resizeVector(2,fineColor*fineColor);
        break;
      default:
      	resizeVector(2,1);
      	break;
      }

      resizeStep(1,1);
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
       
      else return false;
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == COMPUTE_VUV)                strcat(Aux,",computeStaggeredVUV");
      else if (type == COMPUTE_MASS)               strcat(Aux,",computeStaggeredMass");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeStaggeredConvert");
      else if (type == COMPUTE_RESCALE)            strcat(Aux,",ComputeStaggeredRescale");
      else errorQuda("Unknown type=%d\n", type);

      if (type == COMPUTE_VUV) {
      	if      (dim == 0) strcat(Aux,",dim=0");
      	else if (dim == 1) strcat(Aux,",dim=1");
      	else if (dim == 2) strcat(Aux,",dim=2");
      	else if (dim == 3) strcat(Aux,",dim=3");

      	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
      	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");
      }

      const char *vol_str = (type == COMPUTE_MASS ||
			     type == COMPUTE_CONVERT || type == COMPUTE_RESCALE) ? X.VolString () : meta.VolString();

      if (type == COMPUTE_VUV) {
      	strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
                meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      	strcat(Aux,"coarse_vol=");
      	strcat(Aux,X.VolString());
      } else {
        strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped" :
                meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device" : ",CPU");
      }

      return TuneKey(vol_str, typeid(*this).name(), Aux);
    }

    void preTune() {
      switch (type) {
      case COMPUTE_VUV:
        Y_atomic.backup();
      case COMPUTE_MASS:
        X_atomic.backup();
        break;
      case COMPUTE_CONVERT:
        if (Y_atomic.Gauge_p() == Y.Gauge_p()) Y.backup();
        if (X_atomic.Gauge_p() == X.Gauge_p()) X.backup();
        break;
      case COMPUTE_RESCALE:
        Y.backup();
        break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }

    void postTune() {
      switch (type) {
      case COMPUTE_VUV:
        Y_atomic.restore();
      case COMPUTE_MASS:
        X_atomic.restore();
        break;
      case COMPUTE_CONVERT:
        if (Y_atomic.Gauge_p() == Y.Gauge_p()) Y.restore();
        if (X_atomic.Gauge_p() == X.Gauge_p()) X.restore();
        break;
      case COMPUTE_RESCALE:
        Y.restore();
        break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }
  };



  /**
     @brief Calculate the coarse-link field, including the coarse clover field.

     @param Y[out] Coarse (fat-)link field accessor
     @param X[out] Coarse clover field accessor
     @param G[in] Fine grid link / gauge field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param G_[in] Fine gauge field 
     @param mass[in] Kappa parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
  template<typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge>
  void calculateStaggeredY(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  fineGauge &G, GaugeField &Y_, GaugeField &X_, GaugeField &Y_atomic_, GaugeField &X_atomic_,
      const GaugeField &G_, double mass, QudaDiracType dirac, QudaMatPCType matpc,
		  const int *fine_to_coarse, const int *coarse_to_fine) {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    // This is the last time we use fineSpin, since this file only coarsens
    // staggered-type ops, not wilson-type AND coarse-type.
    if (fineSpin != 1)
      errorQuda("Input Dirac operator %d should have nSpin=1, not nSpin=%d\n", dirac, fineSpin);
    if (fineColor != 3)
      errorQuda("Input Dirac operator %d should have nColor=3, not nColor=%d\n", dirac, fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) x_size[i] = G_.X()[i];
    x_size[4] = 1;

    int xc_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM] = { };
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = 0; // 0 -> spin-less types.

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateStaggeredYArg<Float,coarseSpin,fineColor,coarseColor,coarseGauge,coarseGaugeAtomic,fineGauge> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, G, mass, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine);
    CalculateStaggeredY<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, G_, Y_, X_, Y_atomic_, X_atomic_);

    QudaFieldLocation location = checkLocation(Y_, X_, G_);
    printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of all of the (fat) links.
    double max_scale = 1e-12;
    for (int d = 0; d < nDim; d++) {
      double U_max = arg.U.abs_max(d);
      if (U_max > max_scale) max_scale = U_max;
    }
    printfQuda("Global U_max = %e\n", max_scale);

    if (coarseGaugeAtomic::fixedPoint()) {
      arg.Y_atomic.resetScale(max_scale);
      arg.X_atomic.resetScale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
    }

    // zero the atomic fields before summing to them
    Y_atomic_.zero();
    X_atomic_.zero();

    // We can technically do a uni-directional build, but becauase
    // the coarse link builds are just permutations plus lots of zeros,
    // it's faster to skip the flip!

    // First compute the coarse forward links 
    for (int d = 0; d < nDim; d++) {
    	y.setDimension(d);
    	y.setDirection(QUDA_FORWARDS);
    	printfQuda("Computing forward %d VUV\n", d);

    	// if we are writing to a temporary, we need to zero it first
      if (Y_atomic.Geometry() == 1) Y_atomic_.zero();
      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
    	y.apply(0);
    	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] (atomic) = %e\n", 4+d, arg.Y_atomic.norm2( (4+d) % arg.Y_atomic.geometry ));
      // now convert from atomic to application computation format if necessary for Y[d]
      if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint()) {
        if (coarseGauge::fixedPoint()) {
          double y_max = arg.Y_atomic.abs_max( (4+d) % arg.Y_atomic.geometry );
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Y[%d] (atomic) max = %e Y[%d] scale = %e\n", 4+d, y_max, 4+d, Y_.Scale());
          Y_.Scale(max_scale);
          arg.Y.resetScale(Y_.Scale());
        }
        y.setComputeType(COMPUTE_CONVERT);
        y.apply(0);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", 4+d, arg.Y.norm2( 4+d ));
      }

    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      printfQuda("Computing backward %d VUV\n", d);

      // if we are writing to a temporary, we need to zero it first
      if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] (atomic) = %e\n", d, arg.Y_atomic.norm2( d%arg.Y_atomic.geometry ));
      // now convert from atomic to application computation format if necessary for Y[d]
      if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {
        if (coarseGauge::fixedPoint()) {
          double y_max = arg.Y_atomic.abs_max( (4+d) % arg.Y_atomic.geometry );
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Y[%d] (atomic) max = %e Y[%d] scale = %e\n", 4+d, y_max, 4+d, Y_.Scale());
          Y_.Scale(max_scale);
          arg.Y.resetScale(Y_.Scale());
        }

        y.setComputeType(COMPUTE_CONVERT);
        y.apply(0);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", d, arg.Y.norm2( d ));
      }

    }

    printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // Add the mass.
    printfQuda("Adding diagonal mass contribution to coarse clover\n");
    y.setComputeType(COMPUTE_MASS);
    y.apply(0);

    // now convert from atomic to application computation format if necessary for X field
    if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {
      // dim=4 corresponds to X field
      y.setDimension(8);
      y.setDirection(QUDA_BACKWARDS);
      if (coarseGauge::fixedPoint()) {
        double x_max = arg.X_atomic.abs_max(0);
        X_.Scale(x_max);
        arg.X.resetScale(x_max);
      }

      y.setComputeType(COMPUTE_CONVERT);
      y.apply(0);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", arg.X.norm2(0));

  }


} // namespace quda
