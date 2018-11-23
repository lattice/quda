#include <tune_quda.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_coarse_op_kernel.cuh>

#define max_color_per_block 8

namespace quda {

  // All staggered operators are un-preconditioned, so we use uni-directional
  // coarsening. For debugging, though, we can force bi-directional coarsening.
  static bool bidirectional_debug = true; //false;


  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_VUV,
    COMPUTE_REVERSE_Y,
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
    const ColorSpinorField &meta;
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
      case COMPUTE_UV:
        flops_ = 0; // permutation
        break;
      case COMPUTE_VUV:
        // when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
        flops_ = 2l * arg.fineVolumeCB * 8 * coarseColor * coarseColor * fineColor / coarseSpin;
        break;
      case COMPUTE_REVERSE_Y:
        // no floating point operations
        flops_ = 0;
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
      case COMPUTE_UV:
        // 2 for complex, 2 for read/write
        bytes_ = arg.UV.Bytes() + arg.U.Bytes();
        break;
      case COMPUTE_VUV:
        {
          // formula for shared-atomic variant assuming parity_flip = true
          int writes = 4;
          // we use a (coarseColor * coarseColor) matrix of threads so each load is input element is loaded coarseColor times
          // we ignore the multiple loads of spin since these are per thread (and should be cached?)
          bytes_ = 2*writes*arg.Y.Bytes() + (arg.bidirectional ? 1 : 2) * 2*writes*arg.X.Bytes() + coarseColor*(arg.UV.Bytes() + arg.V.Bytes());
          break;
        }
      case COMPUTE_REVERSE_Y:
        bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
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
      case COMPUTE_UV:
      case COMPUTE_VUV:
        threads = arg.fineVolumeCB;
        break;
      case COMPUTE_REVERSE_Y:
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
    bool tuneAuxDim() const { return type != COMPUTE_VUV ? false : true; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      if (arg.shared_atomic && type == COMPUTE_VUV) return 4*sizeof(storeType)*max_color_per_block*max_color_per_block*4*coarseSpin*coarseSpin;
      return TunableVectorYZ::sharedBytesPerBlock(param);
    }

  public:
    CalculateStaggeredY(Arg &arg, const ColorSpinorField &meta, GaugeField &Y, GaugeField &X, GaugeField &Y_atomic, GaugeField &X_atomic)
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

      	if (type == COMPUTE_UV) {

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredUVCPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredUVCPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredUVCPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredUVCPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredUVCPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredUVCPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredUVCPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredUVCPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }


      	} else if (type == COMPUTE_VUV) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }

      	} else if (type == COMPUTE_REVERSE_Y) {

      	  ComputeStaggeredYReverseCPU<Float,coarseSpin,coarseColor>(arg);

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

      	if (type == COMPUTE_UV) {

          if (dir != QUDA_BACKWARDS && dir != QUDA_FORWARDS) errorQuda("Undefined direction %d", dir);
#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::ComputeStaggeredUVGPU")
            .instantiate(Type<Float>(),dim,dir,fineColor,coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // no JITIFY
      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredUVGPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredUVGPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredUVGPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredUVGPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredUVGPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredUVGPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredUVGPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredUVGPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } 
#endif // no JITIFY

      	} else if (type == COMPUTE_VUV) {

          arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;

          // need to resize the grid since we don't tune over the entire coarseColor dimension
          // factor of two comes from parity onto different blocks (e.g. in the grid)
          tp.grid.y = (2*coarseColor + tp.block.y - 1) / tp.block.y;
          tp.grid.z = (coarseColor + tp.block.z - 1) / tp.block.z;

          arg.shared_atomic = tp.aux.y;
          arg.parity_flip = tp.aux.z;

          if (arg.shared_atomic) {
            // check we have a valid problem size for shared atomics
            // constrint is due to how shared memory initialization and global store are done
            int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
            if (block_size/2 < coarseSpin*coarseSpin)
              errorQuda("Block size %d not supported in shared-memory atomic coarsening", block_size);

            arg.aggregates_per_block = tp.aux.x;
            tp.block.x *= tp.aux.x;
            tp.grid.x /= tp.aux.x;
          }

          if (arg.coarse_color_wave) {
            // swap x and y grids
            std::swap(tp.grid.y,tp.grid.x);
            // augment x grid with coarseColor row grid (z grid)
            arg.grid_z = tp.grid.z;
            arg.coarse_color_grid_z = coarseColor*tp.grid.z;
            tp.grid.x *= tp.grid.z;
            tp.grid.z = 1;
          }

          tp.shared_bytes -= sharedBytesPerBlock(tp); // shared memory is static so don't include it in launch

#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::ComputeStaggeredVUVGPU")
            .instantiate(arg.shared_atomic,arg.parity_flip,Type<Float>(),dim,dir,fineColor,coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
          if (arg.shared_atomic) {
            if (arg.parity_flip != true) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
            constexpr bool parity_flip = true;
            if (dir == QUDA_BACKWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<true,parity_flip,Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<true,parity_flip,Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<true,parity_flip,Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<true,parity_flip,Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else if (dir == QUDA_FORWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<true,parity_flip,Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<true,parity_flip,Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<true,parity_flip,Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<true,parity_flip,Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else {
              errorQuda("Undefined direction %d", dir);
            }
          } else {
            if (arg.parity_flip != false) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
            constexpr bool parity_flip = false;
            if (dir == QUDA_BACKWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<false,parity_flip,Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<false,parity_flip,Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<false,parity_flip,Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<false,parity_flip,Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else if (dir == QUDA_FORWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<false,parity_flip,Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<false,parity_flip,Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<false,parity_flip,Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<false,parity_flip,Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else {
              errorQuda("Undefined direction %d", dir);
            }
          }
#endif // JITIFY
          tp.shared_bytes += sharedBytesPerBlock(tp); // restore shared memory

          if (arg.coarse_color_wave) {
            // revert the grids
            tp.grid.z = arg.grid_z;
            tp.grid.x /= tp.grid.z;
            std::swap(tp.grid.x,tp.grid.y);
          }
          if (arg.shared_atomic) {
            tp.block.x /= tp.aux.x;
            tp.grid.x *= tp.aux.x;
          }

      	} else if (type == COMPUTE_REVERSE_Y) {

#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::ComputeStaggeredYReverseGPU")
            .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
      	  ComputeStaggeredYReverseGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
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
      case COMPUTE_VUV:
        arg.shared_atomic = false;
        arg.parity_flip = false;
        if (arg.shared_atomic) {
          // if not parity flip then we need to force parity within the block (hence factor of 2)
          resizeVector( (arg.parity_flip ? 1 : 2) * max_color_per_block,max_color_per_block);
        } else {
          resizeVector(2*max_color_per_block,max_color_per_block);
        }
      break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
        resizeVector(2*coarseColor,coarseColor);
        break;
      case COMPUTE_UV:
        resizeVector(2,fineColor*fineColor);
        break;
      default:
      	resizeVector(2,1);
      	break;
      }

      resizeStep(1,1);
      if (arg.shared_atomic && type == COMPUTE_VUV && !arg.parity_flip) resizeStep(2,1);

      // do not tune spatial block size for VUV
      tune_block_x = (type == COMPUTE_VUV) ? false : true;
    }

    bool advanceAux(TuneParam &param) const
    {
      if (type != COMPUTE_VUV) return false;

      // exhausted the global-atomic search space so switch to
      // shared-atomic space
      if (param.aux.y == 0) {
        // pre-Maxwell does not support shared-memory atomics natively so no point in trying
        if ( __COMPUTE_CAPABILITY__ < 500 ) return false;
        // before advancing, check we can use shared-memory atomics
        int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
        if (block_size/2 < coarseSpin*coarseSpin) return false;

        arg.shared_atomic = true;
        arg.parity_flip = true; // this is usually optimal for shared atomics

        resizeVector( (arg.parity_flip ? 1 : 2) * max_color_per_block,max_color_per_block);
        if (!arg.parity_flip) resizeStep(2,1);

        // need to reset since we're switching to shared-memory atomics
        initTuneParam(param);

        return true;
      } else {
        // already doing shared-memory atomics but can tune number of
        // coarse grid points per block
        if (param.aux.x < 4) {
          param.aux.x *= 2;
          return true;
        } else {
          param.aux.x = 1;
          // completed all shared-memory tuning so reset to global atomics
          arg.shared_atomic = false;
          arg.parity_flip = false; // this is usually optimal for global atomics
          initTuneParam(param);
          return false;
        }
      }
    }

    bool advanceSharedBytes(TuneParam &param) const {
      return (!arg.shared_atomic && type == COMPUTE_VUV) ? false : Tunable::advanceSharedBytes(param);
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
       
      else return false;
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      param.aux.x = 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip; // not actually tuned over at present

      // with shared-atomic VUV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && type == COMPUTE_VUV) {
      	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
      	param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);

      param.aux.x = 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip; // not actually tuned over at present

      // with shared-atomic VUV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && type == COMPUTE_VUV) {
        param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
        param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == COMPUTE_UV)                 strcat(Aux,",computeStaggeredUV");
      else if (type == COMPUTE_VUV)                strcat(Aux,",computeStaggeredVUV");
      else if (type == COMPUTE_REVERSE_Y)          strcat(Aux,",computeStaggeredYreverse");
      else if (type == COMPUTE_MASS)               strcat(Aux,",computeStaggeredMass");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeStaggeredConvert");
      else if (type == COMPUTE_RESCALE)            strcat(Aux,",ComputeStaggeredRescale");
      else errorQuda("Unknown type=%d\n", type);

      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
      	if      (dim == 0) strcat(Aux,",dim=0");
      	else if (dim == 1) strcat(Aux,",dim=1");
      	else if (dim == 2) strcat(Aux,",dim=2");
      	else if (dim == 3) strcat(Aux,",dim=3");

      	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
      	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");

        if (arg.bidirectional && type == COMPUTE_VUV) strcat(Aux,",bidirectional");
      }

      const char *vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_MASS ||
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
      case COMPUTE_UV:
      case COMPUTE_REVERSE_Y:
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
      case COMPUTE_UV:
      case COMPUTE_REVERSE_Y:
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
     @param UV[out] Temporary accessor used to store fine link field * null space vectors
     @param V[in] Packed null-space vector accessor
     @param G[in] Fine grid link / gauge field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param mass[in] Kappa parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
  template<typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   typename Ftmp, typename Vt, typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge>
  void calculateStaggeredY(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  Ftmp &UV, Vt &V, fineGauge &G,
		  GaugeField &Y_, GaugeField &X_, GaugeField &Y_atomic_, GaugeField &X_atomic_,
      ColorSpinorField &uv, const ColorSpinorField &v,
		  double mass, QudaDiracType dirac, QudaMatPCType matpc,
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
    for (int i=0; i<4; i++) x_size[i] = v.X(i);
    x_size[4] = 1;

    int xc_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM] = { };
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse(); // may need to check this. I believe 0 was meant for spin-less types.

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = bidirectional_debug;
    if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
    else printfQuda("Doing uni-directional link coarsening\n");

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateStaggeredYArg<Float,coarseSpin,fineColor,coarseColor,coarseGauge,coarseGaugeAtomic,fineGauge,Ftmp,Vt> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, G, V, mass, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    CalculateStaggeredY<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, v, Y_, X_, Y_atomic_, X_atomic_);

    QudaFieldLocation location = checkLocation(Y_, X_, v);
    printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v, v.Ghost());  // point the accessor to the correct ghost buffer
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    printfQuda("V2 = %e\n", arg.V.norm2());

    // work out what to set the scales to
    // ESW hack
    if (coarseGaugeAtomic::fixedPoint()) {
      double max = 10.0; // FIXME - I can probably put a bound on fat link magnitudes
      arg.Y_atomic.resetScale(max);
      arg.X_atomic.resetScale(max);
    }

    // zero the atomic fields before summing to them
    Y_atomic_.zero();
    X_atomic_.zero();

    bool set_scale = false; // records where the scale has been set already or not

    // First compute the coarse forward links if needed
    if (bidirectional_links) {
      for (int d = 0; d < nDim; d++) {
      	y.setDimension(d);
      	y.setDirection(QUDA_FORWARDS);
      	printfQuda("Computing forward %d UV and VUV\n", d);

      	if (uv.Precision() == QUDA_HALF_PRECISION) {
      	  double U_max = 3.0*arg.U.abs_max(d);
      	  double uv_max = U_max * v.Scale();
      	  uv.Scale(uv_max);
      	  arg.UV.resetScale(uv_max);

      	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
      	}

      	y.setComputeType(COMPUTE_UV);  // compute U*V product
      	y.apply(0);
      	printfQuda("UV2[%d] = %e\n", d, arg.UV.norm2());

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
            if (!set_scale) {
              Y_.Scale(1.1*y_max); // slightly oversize to avoid unnecessary rescaling
              arg.Y.resetScale(Y_.Scale());
              set_scale = true;
            } else if (y_max > Y_.Scale()) {
              // we have exceeded the maximum used before so we need to reset the maximum and rescale the elements
              arg.rescale = Y_.Scale() / y_max; // how much we need to shrink the elements by
              y.setComputeType(COMPUTE_RESCALE);
              for (int d_=0; d_<d; d_++) {
                y.setDimension(d_);
                y.apply(0);
              }
              y.setDimension(d);
              Y_.Scale(y_max);
              arg.Y.resetScale(Y_.Scale());
            }
          }
          y.setComputeType(COMPUTE_CONVERT);
          y.apply(0);
          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", 4+d, arg.Y.norm2( 4+d ));
        }

      }
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      printfQuda("Computing backward %d UV and VUV\n", d);

      if (uv.Precision() == QUDA_HALF_PRECISION) {
      	double U_max = 3.0*arg.U.abs_max(d);
      	double uv_max = U_max * v.Scale();
      	uv.Scale(uv_max);
      	arg.UV.resetScale(uv_max);

      	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
      }

      y.setComputeType(COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      printfQuda("UV2[%d] = %e\n", d+4, arg.UV.norm2());

      // if we are writing to a temporary, we need to zero it first
      if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
            if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] (atomic) = %e\n", d, arg.Y_atomic.norm2( d%arg.Y_atomic.geometry ));
      // now convert from atomic to application computation format if necessary for Y[d]
      if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {
        if (coarseGauge::fixedPoint()) {
          double y_max = arg.Y_atomic.abs_max( d % arg.Y_atomic.geometry );
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Y[%d] (atomic) max = %e Y[%d] scale = %e\n", d, y_max, d, Y_.Scale());
          if (!set_scale) {
            Y_.Scale(1.1*y_max); // slightly oversize to avoid unnecessary rescaling
            arg.Y.resetScale(Y_.Scale());
            set_scale = true;
          } else if (y_max > Y_.Scale()) {
            // we have exceeded the maximum used before so we need to reset the maximum and rescale the elements
            arg.rescale = Y_.Scale() / y_max; // how much we need to shrink the elements by
            y.setComputeType(COMPUTE_RESCALE);
            // update all prior compute Y links
            if (bidirectional_links) {
              y.setDirection(QUDA_FORWARDS);
              for (int d_=0; d_<4; d_++) {
                y.setDimension(d_);
                y.apply(0);
              }
            }
            y.setDirection(QUDA_BACKWARDS);
            for (int d_=0; d_<d; d_++) {
              y.setDimension(d_);
              y.apply(0);
            }
            y.setDimension(d);
            Y_.Scale(y_max);
            arg.Y.resetScale(Y_.Scale());
          }
        }
        y.setComputeType(COMPUTE_CONVERT);
        y.apply(0);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", d, arg.Y.norm2( d ));
      }

    }

    printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      printfQuda("Reversing links\n");
      y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(0);
    }

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
