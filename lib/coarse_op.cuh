#include <tune_quda.h>
#include <jitify_helper.cuh>
#include <kernels/coarse_op_kernel.cuh>
#include <uint_to_char.h>

namespace quda {

  // For coarsening un-preconditioned operators we use uni-directional
  // coarsening to reduce the set up code.  For debugging we can force
  // bi-directional coarsening.
  static bool bidirectional_debug = false;

  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_AV,
    COMPUTE_TMAV,
    COMPUTE_TMCAV,
    COMPUTE_CLOVER_INV_MAX,
    COMPUTE_TWISTED_CLOVER_INV_MAX,
    COMPUTE_VUV,
    COMPUTE_COARSE_CLOVER,
    COMPUTE_REVERSE_Y,
    COMPUTE_DIAGONAL,
    COMPUTE_TMDIAGONAL,
    COMPUTE_CONVERT,
    COMPUTE_RESCALE,
    COMPUTE_INVALID
  };

  /**
     @brief Launcher for CPU instantiations of coarse-link construction
   */
  template <QudaFieldLocation location, bool from_coarse, typename Float, int fineSpin,
            int fineColor, int coarseSpin, int coarseColor, typename Arg> struct Launch {
    Launch(Arg &arg, CUresult &error, TuneParam &tp, ComputeType type, const qudaStream_t &stream)
    {
      if (type == COMPUTE_UV) {
        if (arg.dir == QUDA_BACKWARDS) {
          if      (arg.dim==0) ComputeUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==1) ComputeUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==2) ComputeUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==3) ComputeUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
        } else if (arg.dir == QUDA_FORWARDS) {
          if      (arg.dim==0) ComputeUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==1) ComputeUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==2) ComputeUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==3) ComputeUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
        } else {
          errorQuda("Undefined direction %d", arg.dir);
        }
      } else if (type == COMPUTE_AV) {
        if (from_coarse) errorQuda("ComputeAV should only be called from the fine grid");

#if defined(GPU_CLOVER_DIRAC) && !defined(COARSECOARSE)
        ComputeAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);
#else
        errorQuda("Clover dslash has not been built");
#endif

      } else if (type == COMPUTE_TMAV) {
        if (from_coarse) errorQuda("ComputeTMAV should only be called from the fine grid");

#if defined(GPU_TWISTED_MASS_DIRAC) && !defined(COARSECOARSE)
        ComputeTMAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);
#else
        errorQuda("Twisted mass dslash has not been built");
#endif

      } else if (type == COMPUTE_TMCAV) {
        if (from_coarse) errorQuda("ComputeTMCAV should only be called from the fine grid");

#if defined(GPU_TWISTED_CLOVER_DIRAC) && !defined(COARSECOARSE)
        ComputeTMCAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);
#else
        errorQuda("Twisted clover dslash has not been built");
#endif

      } else if (type == COMPUTE_CLOVER_INV_MAX) {
        if (from_coarse) errorQuda("ComputeInvCloverMax should only be called from the fine grid");

#if defined(DYNAMIC_CLOVER) && !defined(COARSECOARSE)
        ComputeCloverInvMaxCPU<Float, false>(arg);
        double max = arg.max_h;
        comm_allreduce_max(&max);
        arg.max_h = max;
#else
        errorQuda("ComputeInvCloverMax only enabled with dynamic clover");
#endif

      } else if (type == COMPUTE_TWISTED_CLOVER_INV_MAX) {
        if (from_coarse) errorQuda("ComputeInvCloverMax should only be called from the fine grid");

#if defined(DYNAMIC_CLOVER) && !defined(COARSECOARSE)
        ComputeCloverInvMaxCPU<Float, true>(arg);
        double max = arg.max_h;
        comm_allreduce_max(&max);
        arg.max_h = max;
#else
        errorQuda("ComputeInvCloverMax only enabled with dynamic clover");
#endif

      } else if (type == COMPUTE_VUV) {
        if (arg.dir == QUDA_BACKWARDS) {
          if      (arg.dim==0) ComputeVUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==1) ComputeVUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==2) ComputeVUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==3) ComputeVUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,coarseSpin>(arg);
        } else if (arg.dir == QUDA_FORWARDS) {
          if      (arg.dim==0) ComputeVUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==1) ComputeVUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==2) ComputeVUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
          else if (arg.dim==3) ComputeVUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,coarseSpin>(arg);
        } else {
          errorQuda("Undefined direction %d", arg.dir);
        }
      } else if (type == COMPUTE_COARSE_CLOVER) {
        ComputeCoarseCloverCPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg);
      } else if (type == COMPUTE_REVERSE_Y) {
        ComputeYReverseCPU<Float,coarseSpin,coarseColor>(arg);
      } else if (type == COMPUTE_DIAGONAL) {
        AddCoarseDiagonalCPU<Float,coarseSpin,coarseColor>(arg);
      } else if (type == COMPUTE_TMDIAGONAL) {
        AddCoarseTmDiagonalCPU<Float,coarseSpin,coarseColor>(arg);
      } else if (type == COMPUTE_CONVERT) {
        ConvertCPU<Float,coarseSpin,coarseColor>(arg);
      } else if (type == COMPUTE_RESCALE) {
        RescaleYCPU<Float,coarseSpin,coarseColor>(arg);
      } else {
        errorQuda("Undefined compute type %d", type);
      }
    }
  };

  /**
     @brief Launcher for GPU instantiations of coarse-link construction
  */
  template <bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  struct Launch<QUDA_CUDA_FIELD_LOCATION, from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> {
    Launch(Arg &arg, CUresult &error, TuneParam &tp, ComputeType type, const qudaStream_t &stream)
    {
#ifdef JITIFY
      using namespace jitify::reflection;
#endif
      if (type == COMPUTE_UV) {

        if (arg.dir != QUDA_BACKWARDS && arg.dir != QUDA_FORWARDS) errorQuda("Undefined direction %d", arg.dir);
#ifdef JITIFY
        error = program->kernel("quda::ComputeUVGPU")
          .instantiate(from_coarse,Type<Float>(),arg.dim,arg.dir,fineSpin,coarseSpin,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        if (arg.dir == QUDA_BACKWARDS) {
          if      (arg.dim==0) ComputeUVGPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==1) ComputeUVGPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==2) ComputeUVGPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==3) ComputeUVGPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
        } else if (arg.dir == QUDA_FORWARDS) {
          if      (arg.dim==0) ComputeUVGPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==1) ComputeUVGPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==2) ComputeUVGPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          else if (arg.dim==3) ComputeUVGPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
        }
#endif

      } else if (type == COMPUTE_AV) {

        if (from_coarse) errorQuda("ComputeAV should only be called from the fine grid");
#ifdef JITIFY
        error = program->kernel("quda::ComputeAVGPU")
          .instantiate(Type<Float>(),fineSpin,fineColor,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
#if defined(GPU_CLOVER_DIRAC) && !defined(COARSECOARSE)
          ComputeAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
          errorQuda("Clover dslash has not been built");
#endif
#endif

      } else if (type == COMPUTE_TMAV) {

        if (from_coarse) errorQuda("ComputeTMAV should only be called from the fine grid");
#ifdef JITIFY
        error = program->kernel("quda::ComputeTMAVGPU")
          .instantiate(Type<Float>(),fineSpin,fineColor,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
#if defined(GPU_TWISTED_MASS_DIRAC) && !defined(COARSECOARSE)
        ComputeTMAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
        errorQuda("Twisted mass dslash has not been built");
#endif
#endif

      } else if (type == COMPUTE_TMCAV) {

        if (from_coarse) errorQuda("ComputeTMCAV should only be called from the fine grid");
#ifdef JITIFY
        error = program->kernel("quda::ComputeTMCAVGPU")
          .instantiate(Type<Float>(),fineSpin,fineColor,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
#if defined(GPU_TWISTED_CLOVER_DIRAC) && !defined(COARSECOARSE)
        ComputeTMCAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
        errorQuda("Twisted clover dslash has not been built");
#endif
#endif

      } else if (type == COMPUTE_CLOVER_INV_MAX) {

        if (from_coarse) errorQuda("ComputeCloverInvMax should only be called from the fine grid");
        arg.max_d = static_cast<Float*>(pool_device_malloc(2 * arg.fineVolumeCB *sizeof(Float)));

#ifdef JITIFY
        error = program->kernel("quda::ComputeCloverInvMaxGPU")
          .instantiate(Type<Float>(), false, Type<Arg>())
          .configure(tp.grid, tp.block, tp.shared_bytes, stream)
          .launch(arg);
#else
#if defined(DYNAMIC_CLOVER) && !defined(COARSECOARSE)
        ComputeCloverInvMaxGPU<Float, false><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#else
        errorQuda("ComputeCloverInvMax only enabled with dynamic clover");
#endif
#endif

        if (!activeTuning()) { // only do reduction once tuning is done else we have nested tuning
          double max = reduce(QUDA_CUDA_FIELD_LOCATION, arg.max_d, 2 * arg.fineVolumeCB,
                             static_cast<Float>(0.0), maximum<Float>());
          comm_allreduce_max(&max);
          arg.max_h = max;
        }
        pool_device_free(arg.max_d);

      } else if (type == COMPUTE_TWISTED_CLOVER_INV_MAX) {

        if (from_coarse) errorQuda("ComputeCloverInvMax should only be called from the fine grid");
        arg.max_d = static_cast<Float *>(pool_device_malloc(2 * arg.fineVolumeCB * sizeof(Float)));

#ifdef JITIFY
        error = program->kernel("quda::ComputeCloverInvMaxGPU")
          .instantiate(Type<Float>(), true, Type<Arg>())
          .configure(tp.grid, tp.block, tp.shared_bytes, stream)
          .launch(arg);
#else
#if defined(DYNAMIC_CLOVER) && !defined(COARSECOARSE)
        ComputeCloverInvMaxGPU<Float, true><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#else
        errorQuda("ComputeCloverInvMax only enabled with dynamic clover");
#endif
#endif

        if (!activeTuning()) { // only do reduction once tuning is done else we have nested tuning
          double max = reduce(QUDA_CUDA_FIELD_LOCATION, arg.max_d, 2 * arg.fineVolumeCB,
                              static_cast<Float>(0.0), maximum<Float>());
          comm_allreduce_max(&max);
          arg.max_h = max;
        }
        pool_device_free(arg.max_d);

      } else if (type == COMPUTE_VUV) {

        // need to resize the grid since we don't tune over the entire coarseColor dimension
        // factor of two comes from parity onto different blocks (e.g. in the grid)
        tp.grid.y = (2*arg.vuvTile.M_tiles + tp.block.y - 1) / tp.block.y;
        tp.grid.z = (arg.vuvTile.N_tiles + tp.block.z - 1) / tp.block.z;

        arg.shared_atomic = tp.aux.y;
        arg.parity_flip = tp.aux.z;

        if (arg.shared_atomic) {
          // check we have a valid problem size for shared atomics
          // constraint is due to how shared memory initialization and global store are done
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
          arg.coarse_color_grid_z = arg.vuvTile.M_tiles*tp.grid.z;
          tp.grid.x *= tp.grid.z;
          tp.grid.z = 1;
        }

#ifdef JITIFY
        error = program->kernel("quda::ComputeVUVGPU")
          .instantiate(arg.shared_atomic,arg.parity_flip,from_coarse,Type<Float>(),arg.dim,arg.dir,fineSpin,coarseSpin,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        if (arg.shared_atomic) {
          if (arg.parity_flip != true) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
          constexpr bool parity_flip = true;

          if (arg.dir == QUDA_BACKWARDS) {
            if      (arg.dim==0) ComputeVUVGPU<true,parity_flip,from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==1) ComputeVUVGPU<true,parity_flip,from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==2) ComputeVUVGPU<true,parity_flip,from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==3) ComputeVUVGPU<true,parity_flip,from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else if (arg.dir == QUDA_FORWARDS) {
            if      (arg.dim==0) ComputeVUVGPU<true,parity_flip,from_coarse,Float,0,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==1) ComputeVUVGPU<true,parity_flip,from_coarse,Float,1,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==2) ComputeVUVGPU<true,parity_flip,from_coarse,Float,2,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==3) ComputeVUVGPU<true,parity_flip,from_coarse,Float,3,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else {
            errorQuda("Undefined direction %d", arg.dir);
          }
        } else {
          if (arg.parity_flip != false) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
          constexpr bool parity_flip = false;

          if (arg.dir == QUDA_BACKWARDS) {
            if      (arg.dim==0) ComputeVUVGPU<false,parity_flip,from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==1) ComputeVUVGPU<false,parity_flip,from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==2) ComputeVUVGPU<false,parity_flip,from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==3) ComputeVUVGPU<false,parity_flip,from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else if (arg.dir == QUDA_FORWARDS) {
            if      (arg.dim==0) ComputeVUVGPU<false,parity_flip,from_coarse,Float,0,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==1) ComputeVUVGPU<false,parity_flip,from_coarse,Float,1,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==2) ComputeVUVGPU<false,parity_flip,from_coarse,Float,2,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            else if (arg.dim==3) ComputeVUVGPU<false,parity_flip,from_coarse,Float,3,QUDA_FORWARDS,fineSpin,coarseSpin><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          } else {
            errorQuda("Undefined direction %d", arg.dir);
          }
        }
#endif

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

      } else if (type == COMPUTE_COARSE_CLOVER) {

#ifdef JITIFY
        error = program->kernel("quda::ComputeCoarseCloverGPU")
          .instantiate(from_coarse,Type<Float>(),fineSpin,coarseSpin,fineColor,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        ComputeCoarseCloverGPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>
          <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif

      } else if (type == COMPUTE_REVERSE_Y) {

#ifdef JITIFY
        error = program->kernel("quda::ComputeYReverseGPU")
          .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        ComputeYReverseGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
      } else if (type == COMPUTE_DIAGONAL) {

#ifdef JITIFY
        error = program->kernel("quda::AddCoarseDiagonalGPU")
          .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        AddCoarseDiagonalGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
      } else if (type == COMPUTE_TMDIAGONAL) {

#ifdef JITIFY
        error = program->kernel("quda::AddCoarseTmDiagonalGPU")
          .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        AddCoarseTmDiagonalGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
      } else if (type == COMPUTE_CONVERT) {

#ifdef JITIFY
        error = program->kernel("quda::ConvertGPU")
          .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        ConvertGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
      } else if (type == COMPUTE_RESCALE) {

#ifdef JITIFY
        error = program->kernel("quda::RescaleYGPU")
          .instantiate(Type<Float>(),coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        RescaleYGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif

      } else {
        errorQuda("Undefined compute type %d", type);
      }
    }
  };

  template <QudaFieldLocation location, bool from_coarse, typename Float, int fineSpin,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateY : public TunableVectorYZ {
  public:

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

    long long flops() const
    {
      long long flops_ = 0;
      switch (type) {
      case COMPUTE_UV:
	// when fine operator is coarse take into account that the link matrix has spin dependence
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * coarseColor * fineColor * fineColor * (!from_coarse ? 1 : fineSpin);
	break;
      case COMPUTE_AV:
      case COMPUTE_TMAV:
	// # chiral blocks * size of chiral block * number of null space vectors
	flops_ = 2l * arg.fineVolumeCB * 8 * (fineSpin/2) * (fineSpin/2) * (fineSpin/2) * fineColor * fineColor * coarseColor;
	break;
      case COMPUTE_TMCAV:
	// # Twice chiral blocks * size of chiral block * number of null space vectors
	flops_ = 4l * arg.fineVolumeCB * 8 * (fineSpin/2) * (fineSpin/2) * (fineSpin/2) * fineColor * fineColor * coarseColor;
	break;
      case COMPUTE_VUV:
	// when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case COMPUTE_COARSE_CLOVER:
	// when the fine operator is truly fine the clover multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
      case COMPUTE_CLOVER_INV_MAX: // FIXME
      case COMPUTE_TWISTED_CLOVER_INV_MAX: // FIXME
        // no floating point operations
        flops_ = 0;
	break;
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
	// read addition on the diagonal
	flops_ = 2l * arg.coarseVolumeCB*coarseSpin*coarseColor;
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
	bytes_ = arg.UV.Bytes() + arg.V.Bytes() + 2*arg.U.Bytes()*coarseColor;
	break;
      case COMPUTE_AV:
	bytes_ = arg.AV.Bytes() + arg.V.Bytes() + 2*arg.C.Bytes()*coarseColor;
	break;
      case COMPUTE_TMAV:
	bytes_ = arg.AV.Bytes() + arg.V.Bytes();
	break;
      case COMPUTE_TMCAV:
#ifdef DYNAMIC_CLOVER
	bytes_ = arg.AV.Bytes() + arg.V.Bytes() + 2*arg.C.Bytes()*coarseColor; // A single clover field
#else
	bytes_ = arg.AV.Bytes() + arg.V.Bytes() + 4*arg.C.Bytes()*coarseColor; // Both clover and its inverse
#endif
	break;
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_TWISTED_CLOVER_INV_MAX:
        bytes_ = 2*arg.C.Bytes(); // read both parities of the clover field
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
      case COMPUTE_COARSE_CLOVER:
	bytes_ = 2*arg.X.Bytes() + 2*arg.C.Bytes() + arg.V.Bytes(); // 2 from parity
	break;
      case COMPUTE_REVERSE_Y:
	bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
	bytes_ = 2*2*arg.X.Bytes(); // 2 from i/o, 2 from parity
	break;
      case COMPUTE_CONVERT:
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
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_TWISTED_CLOVER_INV_MAX:
      case COMPUTE_VUV:
      case COMPUTE_COARSE_CLOVER:
	threads = arg.fineVolumeCB;
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
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
      if (arg.shared_atomic && type == COMPUTE_VUV)
        return 4*sizeof(storeType)*arg.max_color_height_per_block*arg.max_color_width_per_block*4*coarseSpin*coarseSpin;
      return TunableVectorYZ::sharedBytesPerBlock(param);
    }

  public:
    CalculateY(Arg &arg, const ColorSpinorField &meta, GaugeField &Y, GaugeField &X, GaugeField &Y_atomic, GaugeField &X_atomic)
      : TunableVectorYZ(2,1), arg(arg), type(COMPUTE_INVALID),
	meta(meta), Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic), dim(0), dir(QUDA_BACKWARDS)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/coarse_op_kernel.cuh");
#endif
      }
      strcpy(aux, compile_type_str(meta));
      strcat(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.dim = dim;
      arg.dir = dir;
      if (type == COMPUTE_VUV || type == COMPUTE_CONVERT || type == COMPUTE_RESCALE) arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;

      if (type == COMPUTE_VUV) tp.shared_bytes -= sharedBytesPerBlock(tp); // shared memory is static so don't include it in launch
      Launch<location, from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg>(arg, jitify_error, tp, type, stream);
      if (type == COMPUTE_VUV) tp.shared_bytes += sharedBytesPerBlock(tp); // restore shared memory
    };

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which direction we are working on (where applicable)
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
          resizeVector((arg.parity_flip ? 1 : 2) * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
        } else {
          resizeVector(2 * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
        }
	break;
      case COMPUTE_COARSE_CLOVER: // no shared atomic version so keep separate from above
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
	resizeVector(2*coarseColor,coarseColor);
        break;
      case COMPUTE_UV: resizeVector(2 * arg.uvTile.M_tiles, arg.uvTile.N_tiles); break;
      case COMPUTE_TMAV: resizeVector(2, coarseColor); break;
      case COMPUTE_AV:
      case COMPUTE_TMCAV: resizeVector(4, coarseColor); break; // y dimension is chirality and parity
      default: resizeVector(2, 1); break;
      }

      resizeStep(1,1);
      if (arg.shared_atomic && type == COMPUTE_VUV && !arg.parity_flip) resizeStep(2,1);

      // do not tune spatial block size for VUV or COARSE_CLOVER
      tune_block_x = (type == COMPUTE_VUV || type == COMPUTE_COARSE_CLOVER) ? false : true;
    }

    bool advanceAux(TuneParam &param) const
    {
      if (type != COMPUTE_VUV) return false;

      // exhausted the global-atomic search space so switch to
      // shared-atomic space
      if (param.aux.y == 0) {

        // pre-Maxwell does not support shared-memory atomics natively so no point in trying
        if (deviceProp.major < 5) return false;

        // before advancing, check we can use shared-memory atomics
        int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
        if (block_size/2 < coarseSpin*coarseSpin) return false;

        arg.shared_atomic = true;
        arg.parity_flip = true; // this is usually optimal for shared atomics

        resizeVector( (arg.parity_flip ? 1 : 2) * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
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
      return ( (!arg.shared_atomic && !from_coarse && type == COMPUTE_VUV) || type == COMPUTE_COARSE_CLOVER) ? false : Tunable::advanceSharedBytes(param);
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

      if      (type == COMPUTE_UV)                 strcat(Aux,",computeUV");
      else if (type == COMPUTE_AV)                 strcat(Aux,",computeAV");
      else if (type == COMPUTE_TMAV)               strcat(Aux,",computeTmAV");
      else if (type == COMPUTE_TMCAV)              strcat(Aux,",computeTmcAV");
      else if (type == COMPUTE_CLOVER_INV_MAX)
        strcat(Aux, ",computeCloverInverseMax");
      else if (type == COMPUTE_TWISTED_CLOVER_INV_MAX)
        strcat(Aux, ",computeTwistedCloverInverseMax");
      else if (type == COMPUTE_VUV)                strcat(Aux,",computeVUV");
      else if (type == COMPUTE_COARSE_CLOVER)      strcat(Aux,",computeCoarseClover");
      else if (type == COMPUTE_REVERSE_Y)          strcat(Aux,",computeYreverse");
      else if (type == COMPUTE_DIAGONAL)           strcat(Aux,",computeCoarseDiagonal");
      else if (type == COMPUTE_TMDIAGONAL)         strcat(Aux,",computeCoarseTmDiagonal");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeConvert");
      else if (type == COMPUTE_RESCALE)            strcat(Aux,",computeRescale");
      else errorQuda("Unknown type=%d\n", type);

#ifdef DYNAMIC_CLOVER
      if (type == COMPUTE_AV || type == COMPUTE_CLOVER_INV_MAX || // ensure separate tuning for dynamic
          type == COMPUTE_TMCAV || type == COMPUTE_TWISTED_CLOVER_INV_MAX)
        strcat(Aux, ",Dynamic");
#endif
      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
        strcat(Aux, ",tile_size=");
        char tile[16];
        u32toa(tile, type == COMPUTE_UV ? arg.uvTile.M : arg.vuvTile.M);
        strcat(Aux, tile);
        strcat(Aux,"x");
        u32toa(tile, type == COMPUTE_UV ? arg.uvTile.N : arg.vuvTile.N);
        strcat(Aux, tile);
        strcat(Aux,"x");
        u32toa(tile, type == COMPUTE_UV ? arg.uvTile.K : arg.vuvTile.K);
        strcat(Aux, tile);
      }

      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
        if      (dim == 0) strcat(Aux, ",dim=0");
        else if (dim == 1) strcat(Aux, ",dim=1");
        else if (dim == 2) strcat(Aux, ",dim=2");
        else if (dim == 3) strcat(Aux, ",dim=3");

	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");

        if (arg.bidirectional && type == COMPUTE_VUV) strcat(Aux,",bidirectional");
      }

      const char *vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_DIAGONAL || type == COMPUTE_TMDIAGONAL ||
                             type == COMPUTE_CONVERT || type == COMPUTE_RESCALE) ? X.VolString () : meta.VolString();

      if (type == COMPUTE_VUV || type == COMPUTE_COARSE_CLOVER) {
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
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
      case COMPUTE_COARSE_CLOVER:
	X_atomic.backup();
        break;
      case COMPUTE_CONVERT:
	if (Y_atomic.Gauge_p() == Y.Gauge_p()) Y.backup();
	if (X_atomic.Gauge_p() == X.Gauge_p()) X.backup();
        break;
      case COMPUTE_RESCALE:
        Y.backup();
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_TWISTED_CLOVER_INV_MAX:
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
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
      case COMPUTE_COARSE_CLOVER:
	X_atomic.restore();
        break;
      case COMPUTE_CONVERT:
	if (Y_atomic.Gauge_p() == Y.Gauge_p()) Y.restore();
	if (X_atomic.Gauge_p() == X.Gauge_p()) X.restore();
        break;
      case COMPUTE_RESCALE:
        Y.restore();
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_TWISTED_CLOVER_INV_MAX:
      case COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }
  };

  /**
     @brief Calculate the coarse-link field, including the coarse clover field.

     @param Y[out] Coarse link field accessor
     @param X[out] Coarse clover field accessor
     @param UV[out] Temporary accessor used to store fine link field * null space vectors
     @param AV[out] Temporary accessor use to store fine clover inverse * null
     space vectors (only applicable when fine-grid operator is the
     preconditioned clover operator else in general this just aliases V
     @param V[in] Packed null-space vector accessor
     @param G[in] Fine grid link / gauge field accessor
     @param C[in] Fine grid clover field accessor
     @param Cinv[in] Fine grid clover inverse field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param kappa[in] Kappa parameter
     @param mu[in] Twisted-mass parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
     @param need_bidirectional[in] If we need to force bi-directional build or not. Required
     if some previous level was preconditioned, even if this one isn't
   */
  template<QudaFieldLocation location, bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename F,
	   typename Ftmp, typename Vt, typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  Ftmp &UV, F &AV, Vt &V, fineGauge &G, fineClover &C, fineClover &Cinv,
		  GaugeField &Y_, GaugeField &X_, GaugeField &Y_atomic_, GaugeField &X_atomic_,
                  ColorSpinorField &uv, ColorSpinorField &av, const ColorSpinorField &v,
		  const GaugeField &G_, const CloverField &C_,
                  double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
		  bool need_bidirectional, const int *fine_to_coarse, const int *coarse_to_fine) {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    bool is_dirac_coarse = (dirac == QUDA_COARSE_DIRAC || dirac == QUDA_COARSEPC_DIRAC) ? true : false;
    if (is_dirac_coarse && fineSpin != 2)
      errorQuda("Input Dirac operator %d should have nSpin=2, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineSpin != 4)
      errorQuda("Input Dirac operator %d should have nSpin=4, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineColor != 3)
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
    int spin_bs = V.Nspin()/Y.NspinCoarse();

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_COARSEPC_DIRAC || bidirectional_debug ||
				dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC || need_bidirectional);

    if (getVerbosity() >= QUDA_VERBOSE) {
      if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
      else printfQuda("Doing uni-directional link coarsening\n");
    }

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateYArg<Float,fineSpin,coarseSpin,fineColor,coarseColor,coarseGauge,coarseGaugeAtomic,fineGauge,F,Ftmp,Vt,fineClover> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, AV, G, V, C, Cinv, kappa,
	    mu, mu_factor, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    CalculateY<location, from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> y(arg, v, Y_, X_, Y_atomic_, X_atomic_);

    QudaFieldLocation location_ = checkLocation(Y_, X_, av, v);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Running link coarsening on the %s\n", location_ == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v, v.Ghost());  // point the accessor to the correct ghost buffer
    if (&v == &av) arg.AV.resetGhost(av, av.Ghost());
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("V2 = %e\n", arg.V.norm2());

    // If doing preconditioned clover then we first multiply the
    // null-space vectors by the clover inverse matrix, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing AV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
#ifdef DYNAMIC_CLOVER
        y.setComputeType(COMPUTE_CLOVER_INV_MAX);
        y.apply(0);
        double max = 6 * arg.max_h;
#else
        double max = 6*C_.abs_max(true);
#endif
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("clover max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_AV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // If doing preconditioned twisted-mass then we first multiply the
    // null-space vectors by the inverse twist, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_MASSPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing TMAV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
	// this is just a trivial rescaling kernel, find the maximum
	complex<Float> fp(1./(1.+arg.mu*arg.mu),-arg.mu/(1.+arg.mu*arg.mu));
	complex<Float> fm(1./(1.+arg.mu*arg.mu),+arg.mu/(1.+arg.mu*arg.mu));
	double max = std::max(abs(fp), abs(fm));
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("tm max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_TMAV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // If doing preconditioned twisted-clover then we first multiply the
    // null-space vectors by the inverse of the squared clover matrix plus
    // mu^2, and then we multiply the result by the clover matrix. This is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing TMCAV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
#ifdef DYNAMIC_CLOVER
        y.setComputeType(COMPUTE_TWISTED_CLOVER_INV_MAX);
        y.apply(0);
	double max = 6*sqrt(arg.max_h);
#else
	double max = 6*sqrt(C_.abs_max(true));
#endif
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("tmc max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_TMCAV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // work out what to set the scales to
    if (coarseGaugeAtomic::fixedPoint()) {
      double max = 500.0; // Should be more than sufficient
      Y_atomic_.Scale(max);
      arg.Y_atomic.resetScale(max);
      X_atomic_.Scale(max);
      arg.X_atomic.resetScale(max);
    }

    // zero the atomic fields before we start summing to them
    Y_atomic_.zero();
    X_atomic_.zero();

    bool set_scale = false; // records where the scale has been set already or not

    // First compute the coarse forward links if needed
    if (bidirectional_links) {
      for (int d = 0; d < nDim; d++) {
        y.setDimension(d);
	y.setDirection(QUDA_FORWARDS);
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing forward %d UV and VUV\n", d);

	if (uv.Precision() == QUDA_HALF_PRECISION) {
	  double U_max = 3.0*G_.abs_max(from_coarse ? d+4 : d);
	  double uv_max = U_max * v.Scale();
	  uv.Scale(uv_max);
	  arg.UV.resetScale(uv_max);

	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
	}

	y.setComputeType(COMPUTE_UV);  // compute U*V product
	y.apply(0);
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("UV2[%d] = %e\n", d, arg.UV.norm2());

        // if we are writing to a temporary, we need to zero it before each computation
        if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

        y.setComputeType(COMPUTE_VUV); // compute Y += VUV
        y.apply(0);
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("Y2[%d] (atomic) = %e\n", 4+d, Y_atomic_.norm2((4+d) % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint()));

        // now convert from atomic to application computation format if necessary for Y[d]
        if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint()) {

          if (coarseGauge::fixedPoint()) {
            double y_max = Y_atomic_.abs_max((4+d) % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint());

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

          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", 4+d, Y_.norm2( 4+d ));
        }

      }
    }

    if ( (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC) &&
	 (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      av.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
      arg.AV.resetGhost(av, av.Ghost());  // make sure we point to the correct pointer in the accessor
      LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing backward %d UV and VUV\n", d);

      if (uv.Precision() == QUDA_HALF_PRECISION) {
	double U_max = 3.0*G_.abs_max(d);
	double uv_max = U_max * av.Scale();
	uv.Scale(uv_max);
	arg.UV.resetScale(uv_max);

	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e av_max = %e uv_max = %e\n", d, U_max, av.Scale(), uv_max);
      }

      y.setComputeType(COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("UAV2[%d] = %e\n", d, arg.UV.norm2());

      // if we are writing to a temporary, we need to zero it before each computation
      if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      if (getVerbosity() >= QUDA_VERBOSE)
        printfQuda("Y2[%d] (atomic) = %e\n", d, Y_atomic_.norm2(d%arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint()));

      // now convert from atomic to application computation format if necessary for Y[d]
      if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {

        if (coarseGauge::fixedPoint()) {
          double y_max = Y_atomic_.abs_max(d % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint());
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

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", d, Y_.norm2( d ));
      }

    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_atomic_.norm2(0, coarseGaugeAtomic::fixedPoint()));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Reversing links\n");
      y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(0);
    }

    // Check if we have a clover term that needs to be coarsened
    if (dirac == QUDA_CLOVER_DIRAC || dirac == QUDA_COARSE_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing fine->coarse clover term\n");
      y.setComputeType(COMPUTE_COARSE_CLOVER);
      y.apply(0);
    } else {  //Otherwise, we just have to add the identity matrix
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Summing diagonal contribution to coarse clover\n");
      y.setComputeType(COMPUTE_DIAGONAL);
      y.apply(0);
    }

    if (arg.mu*arg.mu_factor!=0 || dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      if (dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC)
	arg.mu_factor += 1.;
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Adding mu = %e\n",arg.mu*arg.mu_factor);
      y.setComputeType(COMPUTE_TMDIAGONAL);
      y.apply(0);
    }

    // now convert from atomic to application computation format if necessary for X field
    if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {
      // dim=4 corresponds to X field
      y.setDimension(8);
      y.setDirection(QUDA_BACKWARDS);

      if (coarseGauge::fixedPoint()) {
        double x_max = X_atomic_.abs_max(0, coarseGaugeAtomic::fixedPoint());
        X_.Scale(x_max);
        arg.X.resetScale(x_max);
      }

      y.setComputeType(COMPUTE_CONVERT);
      y.apply(0);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_.norm2(0));
  }

} // namespace quda
