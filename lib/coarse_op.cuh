#pragma once

#include <kernels/coarse_op_kernel.cuh>
#include <uint_to_char.h>
#include <coarse_op_mma_launch.h>
#include <tunable_nd.h>

namespace quda {

  // For coarsening un-preconditioned operators we use uni-directional
  // coarsening to reduce the set up code.  For debugging we can force
  // bi-directional coarsening.
  static bool bidirectional_debug = false;

  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_LV,
    COMPUTE_AV,
    COMPUTE_TMAV,
    COMPUTE_TMCAV,
    COMPUTE_KV,
    COMPUTE_VUV,
    COMPUTE_VLV,
    COMPUTE_COARSE_CLOVER,
    COMPUTE_REVERSE_Y,
    COMPUTE_DIAGONAL,
    COMPUTE_STAGGEREDMASS,
    COMPUTE_TMDIAGONAL,
    COMPUTE_CONVERT,
    COMPUTE_RESCALE,
    COMPUTE_INVALID
  };

  template <bool use_mma, QudaFieldLocation location_template, typename Arg>
  class CalculateY : public TunableKernel3D {
  public:
    using Float = typename Arg::Float;
    static constexpr bool from_coarse = Arg::from_coarse;
    static constexpr bool from_kd_op = Arg::from_kd_op;
    static constexpr int fineSpin = Arg::fineSpin;
    static constexpr int coarseSpin = Arg::coarseSpin;
    static constexpr int fineColor = Arg::fineColor;
    static constexpr int coarseColor = Arg::coarseColor;

  protected:
    Arg &arg;
    const ColorSpinorField &V;
    const ColorSpinorField &UV;
    const ColorSpinorField &AV;
    GaugeField &Y;
    GaugeField &X;
    GaugeField &Y_atomic;
    GaugeField &X_atomic;

    int dim;
    QudaDirection dir;
    ComputeType type;
    bool kd_dagger; /** Whether or not we're applying KD dagger or KD in compute_kv */
    bool compute_max;
    int nFace; /** for staggered vs asqtad UV, VUV */

    long long flops() const override
    {
      long long flops_ = 0;
      switch (type) {
      case COMPUTE_UV:
      case COMPUTE_LV:
        {
          // fine volume of "fine" link matrices times coarseColor
          flops_ = 2l * arg.fineVolumeCB * 8 * (fineColor * fineColor * coarseColor);

          // Multiplicative factors as a function of fine operator
          if (from_coarse) {
            // when fine operator is coarse take into account that the "fine" link matrix has spin dependence
            flops_ *= fineSpin * fineSpin;
          } else if (fineSpin == 4 || (fineSpin == 1 && !from_kd_op)) {
            // when the fine operator is Wilson-type or a non-KD staggered operator, we need to acount for each near-null vector
            // having fineSpin spin components.
            flops_ *= fineSpin;
          } else if (fineSpin == 1 && from_kd_op) {
            if (dir == QUDA_FORWARDS) {
              // only loading from V, so there's only one "spin"
              flops_ *= Arg::fineSpinorV::nSpin;
            } else if (dir == QUDA_BACKWARDS) {
             // loading from AV, so we need to keep track of two spins == coarse chiralities
              flops_ *= Arg::fineSpinorAV::nSpin;
            } else {
              errorQuda("Unexpected direction %d", dir);
            }
          } else {
            errorQuda("Invalid operator combination for COMPUTE_UV/COMPUTE_LV");
          }
        }
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
      case COMPUTE_KV:
      // similar to a single KD apply, but there are separate accumulations for even source and odd source
      flops_ = 2l * arg.fineVolumeCB * fineColor * ( 8ll * fineColor * 8ll - 2ll ) * coarseColor;
      break;
      case COMPUTE_VUV:
      case COMPUTE_VLV:
      {
        // fine volume of contractions over fineColor
        flops_ = 2l * arg.fineVolumeCB * 8 * (coarseColor * fineColor * coarseColor);

        // Multiplicative factors as a function of the fine operator
        if (from_coarse) {
          // when the "fine" operator is a coarse op we're also contracting over fineSpin
          flops_ *= fineSpin * fineSpin;
        } else if (fineSpin == 4) {
          // when the fine operator is Wilson-type, there's still a contraction over fineSpin,
          // but the VUV multiplication is block sparse which halves the number of operations
          flops_ *= fineSpin * fineSpin / coarseSpin;
        } else if (fineSpin == 1 && !from_kd_op) {
          // trivial contraction over fineSpin
          flops_ *= fineSpin * fineSpin;
        } else if (fineSpin == 1 && from_kd_op) {
          // trivial contraction over fineSpin, times the two source parities baked into in AV (or UAV)
          flops_ *= fineSpin * fineSpin * 2;
        } else {
          errorQuda("Invalid operator combination for COMPUTE_VUV/COMPUTE_VLV");
        }
      }
	break;
      case COMPUTE_COARSE_CLOVER:
	// when the fine operator is truly fine the clover multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
      case COMPUTE_DIAGONAL:
      case COMPUTE_STAGGEREDMASS:
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

    long long bytes() const override
    {
      long long bytes_ = 0;
      switch (type) {
      case COMPUTE_UV:
      case COMPUTE_LV:
        {
          // Loading the fine gauge field coarseColor times
          bytes_ = 2 * ((type == COMPUTE_UV) ? arg.U.Bytes() : arg.L.Bytes()) * coarseColor;

          // Loading V/AV: this is only relevant for the KD op, otherwise these two have the same size
          bytes_ += (dir == QUDA_BACKWARDS) ? AV.Bytes() : V.Bytes();

          // Storing to UV: only special for the KD op
          bytes_ += compute_max * ((from_kd_op && dir == QUDA_FORWARDS) ? (UV.Bytes() / 2) : UV.Bytes());
        }
	break;
      case COMPUTE_AV:
	bytes_ = compute_max * AV.Bytes() + V.Bytes() + 2*arg.C.Bytes()*coarseColor;
	break;
      case COMPUTE_TMAV:
	bytes_ = AV.Bytes() + V.Bytes();
	break;
      case COMPUTE_TMCAV:
	bytes_ = compute_max * AV.Bytes() + V.Bytes() + (2 + 2 * !clover::dynamic_inverse()) * arg.C.Bytes()*coarseColor;
	break;
      case COMPUTE_KV:
        bytes_ = AV.Bytes() + V.Bytes() + arg.K.Bytes() * coarseColor;
        break;
      case COMPUTE_VUV:
      case COMPUTE_VLV:
        {
          // formula for shared-atomic variant assuming parity_flip = true
          int writes = 4;
          // we use a (coarseColor * coarseColor) matrix of threads so each load is input element is loaded coarseColor times
          // we ignore the multiple loads of spin since these are per thread (and should be cached?)
          bytes_ = 2*writes*arg.Y.Bytes() + (arg.bidirectional ? 1 : 2) * 2*writes*arg.X.Bytes() + coarseColor*(UV.Bytes() + V.Bytes());
          break;
        }
      case COMPUTE_COARSE_CLOVER:
	bytes_ = 2 * arg.X.Bytes() + 2 * arg.C.Bytes() + (1 + coarseColor) * V.Bytes(); // 2 from parity
	break;
      case COMPUTE_REVERSE_Y:
	bytes_ = 4 * 2 * 2 * arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
	bytes_ = 2 * 2 * arg.X.Bytes(); // 2 from i/o, 2 from parity
	break;
      case COMPUTE_DIAGONAL:
      case COMPUTE_STAGGEREDMASS:
      case COMPUTE_TMDIAGONAL:
	bytes_ = 2 * 2 * arg.X.Bytes() / (coarseSpin*coarseColor); // 2 from i/o, 2 from parity, division because of diagonal
	break;
      case COMPUTE_CONVERT:
	bytes_ = dim == 4 ? 2 * (arg.X.Bytes() + arg.X_atomic.Bytes()) : 2 * (arg.Y.Bytes() + arg.Y_atomic.Bytes());
	break;
      case COMPUTE_RESCALE:
	bytes_ = 2 * 2 * arg.Y.Bytes(); // 2 from i/o, 2 from parity
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      return bytes_;
    }

    unsigned int minThreads() const override
    {
      unsigned int threads = 0;
      switch (type) {
      case COMPUTE_UV:
      case COMPUTE_LV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_KV:
      case COMPUTE_VUV:
      case COMPUTE_VLV:
      case COMPUTE_COARSE_CLOVER:
	threads = arg.fineVolumeCB;
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_DIAGONAL:
      case COMPUTE_STAGGEREDMASS:
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

    bool tuneGridDim() const override { return false; } // don't tune the grid dimension
    bool tuneAuxDim() const override { return (type != COMPUTE_VUV && type != COMPUTE_VLV) ? false : true; }

    int candidate_iter() const override { return 1; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const override
    {
      if (type == COMPUTE_VUV || type == COMPUTE_VLV)
        return 4*sizeof(storeType)*arg.max_color_height_per_block*arg.max_color_width_per_block*4*coarseSpin*coarseSpin;
      return TunableKernel3D::sharedBytesPerBlock(param);
    }

  public:
    CalculateY(Arg &arg, const ColorSpinorField &V, const ColorSpinorField &UV, const ColorSpinorField &AV,
               GaugeField &Y, GaugeField &X, GaugeField &Y_atomic, GaugeField &X_atomic, int nFace) :
      TunableKernel3D(V, 2, 1),
      arg(arg),
      V(V),
      UV(UV),
      AV(AV),
      Y(Y),
      X(X),
      Y_atomic(Y_atomic),
      X_atomic(X_atomic),
      dim(0),
      dir(QUDA_BACKWARDS),
      type(COMPUTE_INVALID),
      kd_dagger(false),
      compute_max(false),
      nFace(nFace)
    {
      strcat(aux, comm_dim_partitioned_string());
    }

    /**
       @brief Launcher for CPU instantiations of coarse-link construction
    */
    template <QudaFieldLocation location_> std::enable_if_t<location_ == QUDA_CPU_FIELD_LOCATION>
    Launch(Arg &arg, TuneParam &tp, ComputeType type, const qudaStream_t &stream)
    {
      if (compute_max) {
        memset(arg.max_h, 0, sizeof(typename Arg::Float));
        arg.max = arg.max_h;
      }

      if (type == COMPUTE_UV) {
        if (compute_max) launch_host<compute_uv>(tp, stream, ArgMax<Arg>(arg));
        else launch_host<compute_uv>(tp, stream, arg);
      } else if (type == COMPUTE_LV) {
        if (fineSpin != 1) errorQuda("compute_lv should only be called for a staggered operator");

#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
        if (compute_max) launch_host<compute_lv>(tp, stream, ArgMax<Arg>(arg));
        else launch_host<compute_lv>(tp, stream, arg);
#else
        errorQuda("Staggered dslash has not been built");
#endif
      } else if (type == COMPUTE_AV) {
        if (from_coarse) errorQuda("compute_av should only be called from the fine grid");

#if defined(GPU_CLOVER_DIRAC) && defined(WILSONCOARSE)
        if (compute_max) launch_host<compute_av>(tp, stream, ArgMax<Arg>(arg));
        else launch_host<compute_av>(tp, stream, arg);
#else
        errorQuda("Clover dslash has not been built");
#endif

      } else if (type == COMPUTE_TMAV) {
        if (from_coarse) errorQuda("compute_tmav should only be called from the fine grid");

#if defined(GPU_TWISTED_MASS_DIRAC) && defined(WILSONCOARSE)
        launch_host<compute_tmav>(tp, stream, arg);
#else
        errorQuda("Twisted mass dslash has not been built");
#endif

      } else if (type == COMPUTE_TMCAV) {
        if (from_coarse) errorQuda("compute_tmcav should only be called from the fine grid");

#if defined(GPU_TWISTED_CLOVER_DIRAC) && defined(WILSONCOARSE)
        if (compute_max) launch_host<compute_tmcav>(tp, stream, ArgMax<Arg>(arg));
        else launch_host<compute_tmcav>(tp, stream, arg);
#else
        errorQuda("Twisted clover dslash has not been built");
#endif

      } else if (type == COMPUTE_KV) {
        if (fineSpin != 1) errorQuda("compute_kv should only be called for a staggered operator");
        if (!from_kd_op) errorQuda("compute_kv should only be called for a Kahler-Dirac operator");
        if (from_coarse) errorQuda("compute_kv should only be called from the fine grid");

#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
        if (compute_max) launch_host<compute_kv>(tp, stream, ArgMax<Arg>(arg));
        else launch_host<compute_kv>(tp, stream, arg);
#else
        errorQuda("Staggered dslash has not been built");
#endif
      } else if (type == COMPUTE_VUV) {
        launch_host<compute_vuv>(tp, stream, arg);
      } else if (type == COMPUTE_VLV) {
        if (fineSpin != 1) errorQuda("compute_vlv should only be called for a staggered operator");

#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
        else launch_host<compute_vlv>(tp, stream, arg);
#else
        errorQuda("Staggered dslash has not been built");
#endif
      } else if (type == COMPUTE_COARSE_CLOVER) {
#if defined(WILSONCOARSE)
        launch_host<compute_coarse_clover>(tp, stream, arg);
#else
        errorQuda("compute_coarse_clover not enabled for non-Wilson coarsenings");
#endif
      } else if (type == COMPUTE_REVERSE_Y) {
        launch_host<reverse>(tp, stream, arg);
      } else if (type == COMPUTE_DIAGONAL) {
#if defined(WILSONCOARSE) || defined(COARSECOARSE)
        launch_host<add_coarse_diagonal>(tp, stream, arg);
#else
        errorQuda("add_coarse_diagonal not enabled for staggered coarsenings");
#endif
      } else if (type == COMPUTE_STAGGEREDMASS) {
#if defined(STAGGEREDCOARSE)
        launch_host<add_coarse_staggered_mass>(tp, stream, arg);
#else
        errorQuda("add_coarse_staggered_mass not enabled for non-staggered coarsenings");
#endif
      } else if (type == COMPUTE_TMDIAGONAL) {
#if defined(WILSONCOARSE) || defined(COARSECOARSE)
        launch_host<add_coarse_tm>(tp, stream, arg);
#else
        errorQuda("add_coarse_tm not enabled for non-wilson coarsenings");
#endif
      } else if (type == COMPUTE_CONVERT) {
        launch_host<convert>(tp, stream, arg);
      } else if (type == COMPUTE_RESCALE) {
        launch_host<rescale>(tp, stream, arg);
      } else {
        errorQuda("Undefined compute type %d", type);
      }

      if (compute_max) comm_allreduce_max(*arg.max_h);
    }

    /**
       @brief Launcher for GPU instantiations of coarse-link construction
    */
    template <QudaFieldLocation location_> std::enable_if_t<location_ == QUDA_CUDA_FIELD_LOCATION>
    Launch(Arg &arg, TuneParam &tp, ComputeType type, const qudaStream_t &stream)
    {
      if (compute_max) {
        memset(arg.max_h, 0, sizeof(typename Arg::Float));
        if (!activeTuning()) qudaMemsetAsync(arg.max_d, 0, sizeof(typename Arg::Float), stream);
        arg.max = arg.max_d;
      }

      if (type == COMPUTE_UV) {

        if constexpr (use_mma) {
          if (compute_max) mma::launch_compute_uv_kernel(tp, ArgMax<Arg>(arg), arg.fineVolumeCB, stream, *this);
          else mma::launch_compute_uv_kernel(tp, arg, arg.fineVolumeCB, stream, *this);
        } else {
          if (compute_max) launch_device<compute_uv>(tp, stream, ArgMax<Arg>(arg));
          else launch_device<compute_uv>(tp, stream, arg);
        }

      } else if (type == COMPUTE_LV) {
        if (fineSpin != 1) errorQuda("compute_lv should only be called for a staggered operator");

#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
        if (compute_max) launch_device<compute_lv>(tp, stream, ArgMax<Arg>(arg));
        else launch_device<compute_lv>(tp, stream, arg);
#else
        errorQuda("Staggered dslash has not been built");
#endif
      } else if (type == COMPUTE_AV) {

        if (from_coarse) errorQuda("compute_av should only be called from the fine grid");
#if defined(GPU_CLOVER_DIRAC) && defined(WILSONCOARSE)
        if (compute_max) launch_device<compute_av>(tp, stream, ArgMax<Arg>(arg));
        else launch_device<compute_av>(tp, stream, arg);
#else
        errorQuda("Clover dslash has not been built");
#endif

      } else if (type == COMPUTE_TMAV) {

        if (from_coarse) errorQuda("compute_tmav should only be called from the fine grid");
#if defined(GPU_TWISTED_MASS_DIRAC) && defined(WILSONCOARSE)
        launch_device<compute_tmav>(tp, stream, arg);
#else
        errorQuda("Twisted mass dslash has not been built");
#endif

      } else if (type == COMPUTE_TMCAV) {

        if (from_coarse) errorQuda("compute_tmcav should only be called from the fine grid");
#if defined(GPU_TWISTED_CLOVER_DIRAC) && defined(WILSONCOARSE)
        if (compute_max) launch_device<compute_tmcav>(tp, stream, ArgMax<Arg>(arg));
        else launch_device<compute_tmcav>(tp, stream, arg);
#else
        errorQuda("Twisted clover dslash has not been built");
#endif

      } else if (type == COMPUTE_KV) {
        if (fineSpin != 1) errorQuda("compute_kv should only be called for a staggered operator");
        if (!from_kd_op) errorQuda("compute_kv should only be called for a Kahler-Dirac operator");
        if (from_coarse) errorQuda("compute_kv should only be called from the fine grid");

#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
        if (compute_max) launch_device<compute_kv>(tp, stream, ArgMax<Arg>(arg));
        else launch_device<compute_kv>(tp, stream, arg);
#else
        errorQuda("Staggered dslash has not been built");
#endif
      } else if (type == COMPUTE_VUV || type == COMPUTE_VLV) {

        if (type == COMPUTE_VLV && fineSpin != 1)
          errorQuda("compute_vlv should only be called for a staggered operator");

        if constexpr (use_mma) {

          if (type == COMPUTE_VUV)
            mma::launch_compute_vuv_kernel(tp, arg, arg.fineVolumeCB, stream, *this);


        } else {

          // need to resize the grid since we don't tune over the entire coarseColor dimension
          // factor of two comes from parity onto different blocks (e.g. in the grid)
          tp.grid.y = (2 * arg.vuvTile.M_tiles + tp.block.y - 1) / tp.block.y;
          tp.grid.z = (arg.vuvTile.N_tiles + tp.block.z - 1) / tp.block.z;

          arg.shared_atomic = tp.aux.y;
          arg.parity_flip = tp.aux.z;
          arg.coarse_color_wave = !tp.aux.w;

          if (arg.shared_atomic) {
            // check we have a valid problem size for shared atomics
            // constraint is due to how shared memory initialization and global store are done
            int block_size = arg.fineVolumeCB / arg.coarseVolumeCB;
            if (block_size / 2 < Arg::coarseSpin * Arg::coarseSpin)
              errorQuda("Block size %d not supported in shared-memory atomic coarsening", block_size);

            arg.aggregates_per_block = tp.aux.x;
            tp.block.x *= tp.aux.x;
            tp.grid.x /= tp.aux.x;
          }

          if (arg.coarse_color_wave) {
            // swap x and y grids
            std::swap(tp.grid.y, tp.grid.x);
            // augment x grid with coarseColor row grid (z grid)
            arg.grid_z = tp.grid.z;
            arg.coarse_color_grid_z = arg.vuvTile.M_tiles * tp.grid.z;
            tp.grid.x *= tp.grid.z;
            tp.grid.z = 1;
          }

          // this will ensure we pass the generic kernel bounds check
          arg.threads.x = tp.grid.x * tp.block.x;
          resizeVector(tp.grid.y * tp.block.y, tp.grid.z * tp.block.z);
          if (type == COMPUTE_VUV)
            launch_device<compute_vuv>(tp, stream, arg);
#if defined(GPU_STAGGERED_DIRAC) && defined(STAGGEREDCOARSE)
          else if (type == COMPUTE_VLV)
            launch_device<compute_vlv>(tp, stream, arg);
#else
          else errorQuda("Staggered dslash has not been built");
#endif
          arg.threads.x = minThreads();
          resizeVector((arg.parity_flip ? 1 : 2) * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);

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

        } // if use_mma

      } else if (type == COMPUTE_COARSE_CLOVER) {

#if defined(WILSONCOARSE)
        launch_device<compute_coarse_clover>(tp, stream, arg);
#else
        errorQuda("compute_coarse_clover not enabled for non-wilson coarsenings");
#endif

      } else if (type == COMPUTE_REVERSE_Y) {
        launch_device<reverse>(tp, stream, arg);
      } else if (type == COMPUTE_DIAGONAL) {
#if defined(WILSONCOARSE) || defined(COARSECOARSE)
        launch_device<add_coarse_diagonal>(tp, stream, arg);
#else
        errorQuda("add_coarse_diagonal not enabled for staggered coarsenings");
#endif
      } else if (type == COMPUTE_STAGGEREDMASS) {
#if defined(STAGGEREDCOARSE)
        launch_device<add_coarse_staggered_mass>(tp, stream, arg);
#else
        errorQuda("add_coarse_staggered_mass not enabled for non-staggered coarsenings");
#endif
      } else if (type == COMPUTE_TMDIAGONAL) {
#if defined(WILSONCOARSE) || defined(COARSECOARSE)
        launch_device<add_coarse_tm>(tp, stream, arg);
#else
        errorQuda("add_coarse_tm not enabled for non-wilson coarsenings");
#endif
      } else if (type == COMPUTE_CONVERT) {
        launch_device<convert>(tp, stream, arg);
      } else if (type == COMPUTE_RESCALE) {
        launch_device<rescale>(tp, stream, arg);
      } else {
        errorQuda("Undefined compute type %d", type);
      }

      if (compute_max && !activeTuning()) {
        qudaMemcpyAsync(arg.max_h, arg.max_d, sizeof(typename Arg::Float), qudaMemcpyDeviceToHost, stream);
        qudaStreamSynchronize(const_cast<qudaStream_t&>(stream));
        comm_allreduce_max(*arg.max_h);
      }
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.threads.x = minThreads();
      arg.dim = dim;
      arg.dir = dir;
      if (type == COMPUTE_VUV || type == COMPUTE_VLV || type == COMPUTE_CONVERT || type == COMPUTE_RESCALE) arg.dim_index = 4*(dir==QUDA_BACKWARDS ? 0 : 1) + dim;
      arg.kd_dagger = kd_dagger;

      if (type == COMPUTE_VUV || type == COMPUTE_VLV) tp.shared_bytes -= sharedBytesPerBlock(tp); // shared memory is static so don't include it in launch
      Launch<location_template>(arg, tp, type, stream);
      if (type == COMPUTE_VUV || type == COMPUTE_VLV) tp.shared_bytes += sharedBytesPerBlock(tp); // restore shared memory
    };

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which direction we are working on (where applicable)
    */
    void setDirection(QudaDirection dir_)
    {
      if (dir_ != QUDA_BACKWARDS && dir_ != QUDA_FORWARDS && dir_ != QUDA_IN_PLACE) errorQuda("Undefined direction %d", dir_);
      dir = dir_;
    }

    /**
       Set whether or not we're applying the dagger of the kd inverse (where applicable)
    */
    void setKDagger(bool kd_dagger_) { kd_dagger = kd_dagger_; }

    /**
       Set which computation we are doing
     */
    void setComputeType(ComputeType type_)
    {
      type = type_;
      switch(type) {
      case COMPUTE_VUV:
      case COMPUTE_VLV:
        arg.shared_atomic = false;
        arg.parity_flip = false;
        arg.coarse_color_wave = false;
        // if not parity flip then we need to force parity within the block (hence factor of 2)
        resizeVector((arg.parity_flip ? 1 : 2) * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
	break;
      case COMPUTE_COARSE_CLOVER: // no shared atomic version so keep separate from above
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_RESCALE:
	resizeVector(2 * coarseColor, coarseColor);
        break;
      case COMPUTE_UV:
      case COMPUTE_LV:
        resizeVector(2 * arg.uvTile.M_tiles, arg.uvTile.N_tiles); break;
      case COMPUTE_AV:
      case COMPUTE_TMCAV:
        resizeVector(4, coarseColor); break; // y dimension is chirality and parity
      case COMPUTE_TMAV:
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL: resizeVector(2, coarseColor); break;
      case COMPUTE_KV: resizeVector(4, coarseColor); break; // y dimension is [chirality (clover), source parity (staggered)] and parity
      default: resizeVector(2, 1); break;
      }

      resizeStep(1,1);
      if ((type == COMPUTE_VUV || type == COMPUTE_VLV) && !arg.parity_flip) resizeStep(2,1);

      // do not tune spatial block size for VUV or COARSE_CLOVER
      tune_block_x = (type == COMPUTE_VUV || type == COMPUTE_VLV || type == COMPUTE_COARSE_CLOVER) ? false : true;
    }

    void setComputeMax(bool compute_max_) { compute_max = compute_max_; }

    bool advanceSwizzle(TuneParam &param) const
    {
      if (param.aux.w == 0) {
        param.aux.w = 1;
        arg.coarse_color_wave = true;
        return true;
      } else {
        param.aux.w = 0;
        arg.coarse_color_wave = false;
        return false;
      }
    }

    bool advanceParityFlip(TuneParam &param) const
    {
      if (param.aux.z == 0) {
        arg.parity_flip = true;
        resizeVector(arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
        resizeStep(1, 1);
        initTuneParam(param);
        return true;
      } else {
        arg.parity_flip = false;
        resizeVector(2 * arg.max_height_tiles_per_block, arg.max_width_tiles_per_block);
        resizeStep(2, 1);

        initTuneParam(param);
        return false;
      }
    }

    bool advanceAtomic(TuneParam &param) const
    {
      if (param.aux.y == 0) {
        // exhausted the global-atomic search space so switch to
        // shared-atomic space

        // pre-Maxwell does not support shared-memory atomics natively so no point in trying
        if (!device::shared_memory_atomic_supported()) return false;

        // before advancing, check we can use shared-memory atomics
        int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
        if (block_size/2 < coarseSpin*coarseSpin) return false;

        arg.shared_atomic = true;

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
          // completed all shared-memory tuning so reset to global atomics
          arg.shared_atomic = false;
          initTuneParam(param);
          return false;
        }
      }
    }

    bool advanceAux(TuneParam &param) const override
    {
      return ((type == COMPUTE_VUV || type == COMPUTE_VLV) ? (advanceAtomic(param) || advanceParityFlip(param) || advanceSwizzle(param)) : false);
    }

    bool advanceSharedBytes(TuneParam &param) const override
    {
      return ( (!arg.shared_atomic && !from_coarse && (type == COMPUTE_VUV || type == COMPUTE_VLV)) || type == COMPUTE_COARSE_CLOVER) ? false : Tunable::advanceSharedBytes(param);
    }

    bool advanceTuneParam(TuneParam &param) const override
    {
      if constexpr (use_mma) {
        // note: for now there aren't MMA versions of COMPUTE_LV or COMPUTE_VLV, this is just forward thinking if we find it makes sense one day
        if (type == COMPUTE_UV || type == COMPUTE_LV || type == COMPUTE_VUV || type == COMPUTE_VLV) {
          constexpr bool query_max = true;
          int max = 0;
          if (type == COMPUTE_UV) {
            max = mma::launch_compute_uv_kernel<query_max>(param, arg, 1, device::get_default_stream(), *this);
          } else if (type == COMPUTE_VUV) {
            max = mma::launch_compute_vuv_kernel<query_max>(param, arg, 1, device::get_default_stream(), *this);
          }

          if (param.aux.x < max) {
            param.aux.x++;
            return true;
          } else {
            return false;
          }
        }
      }

      // only do autotuning if we have device fields
      if (location == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    void initTuneParam(TuneParam &param) const override
    {
      TunableKernel3D::initTuneParam(param);
      // note: for now there aren't MMA versions of COMPUTE_LV or COMPUTE_VLV, this is just forward thinking if we find it makes sense one day
      param.aux.x = ((type == COMPUTE_VUV || type == COMPUTE_VLV || type == COMPUTE_UV || type == COMPUTE_LV) && use_mma) ? 0 : 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip;
      param.aux.w = arg.coarse_color_wave;

      // with shared-atomic VUV, VLV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && (type == COMPUTE_VUV || type == COMPUTE_VLV)) {
	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
	param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const override
    {
      TunableKernel3D::defaultTuneParam(param);
      param.aux.x = ((type == COMPUTE_VUV || type == COMPUTE_VLV || type == COMPUTE_UV || type == COMPUTE_LV) && use_mma) ? 0 : 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip;
      param.aux.w = arg.coarse_color_wave;

      // with shared-atomic VUV, VLV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && (type == COMPUTE_VUV || type == COMPUTE_VLV)) {
	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
	param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    TuneKey tuneKey() const override
    {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if (type == COMPUTE_UV) {
        strcat(Aux, ",computeUV");
        if constexpr (use_mma) {
          strcat(Aux, ",mma");
#ifdef QUDA_MMA_AVAILABLE
          strcat(Aux, mma::mg_mma_dispatch_t<Float>::type::get_type_name().c_str());
#endif
        }
      }
      else if (type == COMPUTE_LV) strcat(Aux, ",computeLV");
      else if (type == COMPUTE_AV)
        strcat(Aux, ",computeAV");
      else if (type == COMPUTE_TMAV)               strcat(Aux,",computeTmAV");
      else if (type == COMPUTE_TMCAV)              strcat(Aux,",computeTmcAV");
      else if (type == COMPUTE_KV)                 strcat(Aux, ",computeKV");
      else if (type == COMPUTE_VUV) {
        strcat(Aux, ",computeVUV");
        if constexpr (use_mma) {
          strcat(Aux, ",mma");
#ifdef QUDA_MMA_AVAILABLE
          strcat(Aux, mma::mg_mma_dispatch_t<Float>::type::get_type_name().c_str());
#endif
        }
      }
      else if (type == COMPUTE_VLV) strcat(Aux, ",computeVLV");
      else if (type == COMPUTE_COARSE_CLOVER)
        strcat(Aux, ",computeCoarseClover");
      else if (type == COMPUTE_REVERSE_Y)          strcat(Aux,",computeYreverse");
      else if (type == COMPUTE_DIAGONAL)           strcat(Aux,",computeCoarseDiagonal");
      else if (type == COMPUTE_STAGGEREDMASS)      strcat(Aux,",computeCoarseStaggeredMass");
      else if (type == COMPUTE_TMDIAGONAL)         strcat(Aux,",computeCoarseTmDiagonal");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeConvert");
      else if (type == COMPUTE_RESCALE)            strcat(Aux,",computeRescale");
      else errorQuda("Unknown type=%d\n", type);

      if (compute_max) strcat(Aux, ",compute_max");

      if ((type == COMPUTE_AV || type == COMPUTE_TMCAV) && clover::dynamic_inverse()) strcat(Aux, ",Dynamic");
      if (!use_mma && (type == COMPUTE_UV || type == COMPUTE_LV || type == COMPUTE_VUV || type == COMPUTE_VLV)) {
        strcat(Aux, ",tile_size=");
        char tile[16];
        u32toa(tile, (type == COMPUTE_UV || type == COMPUTE_LV) ? arg.uvTile.M : arg.vuvTile.M);
        strcat(Aux, tile);
        strcat(Aux,"x");
        u32toa(tile, (type == COMPUTE_UV || type == COMPUTE_LV) ? arg.uvTile.N : arg.vuvTile.N);
        strcat(Aux, tile);
        strcat(Aux,"x");
        u32toa(tile, (type == COMPUTE_UV || type == COMPUTE_LV) ? arg.uvTile.K : arg.vuvTile.K);
        strcat(Aux, tile);
      }

      if (type == COMPUTE_UV || type == COMPUTE_LV || type == COMPUTE_VUV || type == COMPUTE_VLV) {
        if      (dim == 0) strcat(Aux, ",dim=0");
        else if (dim == 1) strcat(Aux, ",dim=1");
        else if (dim == 2) strcat(Aux, ",dim=2");
        else if (dim == 3) strcat(Aux, ",dim=3");

	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");
        else if (dir == QUDA_IN_PLACE) strcat(Aux,",dir=clover");

        if (type == COMPUTE_UV || type == COMPUTE_LV || type == COMPUTE_VUV || type == COMPUTE_VLV) {
          strcat(Aux, ",nFace=");
          u32toa(Aux + strlen(Aux), nFace);
        }

        // needed to break the degeneracy from staggered KD and non-KD
        if (arg.from_kd_op) strcat(Aux, ",fromkd");

        if (arg.bidirectional && (type == COMPUTE_VUV || type == COMPUTE_VLV)) strcat(Aux,",bidirectional");
      }

      auto vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_DIAGONAL || type == COMPUTE_STAGGEREDMASS || type == COMPUTE_TMDIAGONAL ||
                      type == COMPUTE_CONVERT || type == COMPUTE_RESCALE) ? X.VolString() : V.VolString();

      if (type == COMPUTE_VUV || type == COMPUTE_VLV || type == COMPUTE_COARSE_CLOVER) {
	strcat(Aux, (location == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
               location == QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
	strcat(Aux,"coarse_vol=");
	strcat(Aux, X.VolString().c_str());
      } else {
	strcat(Aux, (location == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped" :
               location == QUDA_CUDA_FIELD_LOCATION ? ",GPU-device" : ",CPU");
      }

      return TuneKey(vol_str.c_str(), typeid(*this).name(), Aux);
    }

    void preTune() override
    {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_VLV:
        Y_atomic.backup();
	X_atomic.backup();
        break;
      case COMPUTE_DIAGONAL:
      case COMPUTE_STAGGEREDMASS:
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
        break;
      case COMPUTE_UV:
      case COMPUTE_LV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_KV:
      case COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }

    void postTune() override
    {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_VLV:
	Y_atomic.restore();
	X_atomic.restore();
        break;
      case COMPUTE_DIAGONAL:
      case COMPUTE_STAGGEREDMASS:
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
        break;
      case COMPUTE_UV:
      case COMPUTE_LV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_KV:
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
     @param L[in] Fine grid long link / gauge field accessor (aliases G for non-improved staggered)
     @param K[in] Fine grid KD gauge field accessor (aliases G for non-improved staggered)
     @param C[in] Fine grid clover field accessor, or Xinv accessor for the KD operator
     @param Cinv[in] Fine grid clover inverse field accessor, or Xinv accessor for the KD operator
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param kappa[in] Kappa parameter
     @param mass[in] mass parameter
     @param mu[in] Twisted-mass parameter
     @param allow_truncation[in] whether or not we let MG coarsening drop improvements, here dropping the long links for small aggregation dimensions
     @param matpc[in] The type of preconditioning of the source fine-grid operator
     @param need_bidirectional[in] If we need to force bi-directional build or not. Required
     if some previous level was preconditioned, even if this one isn't
   */
  template <bool use_mma, QudaFieldLocation location, bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin,
            int coarseColor, typename avSpinor, typename uvSpinor, typename vSpinor, typename coarseGauge, typename coarseGaugeAtomic,
            typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X, coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic, uvSpinor &UV,
                  avSpinor &AV, vSpinor &V, fineGauge &G, fineGauge &L, fineGauge &K, fineClover &C, fineClover &Cinv, GaugeField &Y_, GaugeField &X_,
                  GaugeField &Y_atomic_, GaugeField &X_atomic_, ColorSpinorField &uv, ColorSpinorField &av,
                  const ColorSpinorField &v, double kappa, double mass, double mu,
                  double mu_factor, bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional,
                  const int *fine_to_coarse, const int *coarse_to_fine)
  {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    bool is_dirac_coarse = (dirac == QUDA_COARSE_DIRAC || dirac == QUDA_COARSEPC_DIRAC) ? true : false;
    bool is_dirac_staggered = (dirac == QUDA_STAGGERED_DIRAC || dirac == QUDA_STAGGEREDPC_DIRAC ||
                               dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC);

    bool is_dirac_wilson = !is_dirac_coarse && !is_dirac_staggered;
    if (is_dirac_coarse && fineSpin != 2)
      errorQuda("Input Dirac operator %d should have nSpin=2, not nSpin=%d\n", dirac, fineSpin);
    if (is_dirac_wilson && fineSpin != 4)
      errorQuda("Input Dirac operator %d should have nSpin=4, not nSpin=%d\n", dirac, fineSpin);
    if (is_dirac_staggered && fineSpin != 1)
      errorQuda("Input Dirac operator %d should have nSpin=1, not nSpin=%d\n", dirac, fineSpin);
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

    // needed to detect if we can safely coarsen long links
    int geo_bs[QUDA_MAX_DIM] = { };
    for (int d = 0; d < 4; d++) geo_bs[d] = x_size[d] / xc_size[d];

    // check if we can safely coarsen the KD op
    if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC)
      for (int d = 0; d < 4; d++)
        if (geo_bs[d] % 2 != 0)
          errorQuda("Invalid aggregation size geo_bs[%d] = %d for KD operator, aggregation size must be even", d, geo_bs[d]);

    int spin_bs = V.Nspin()/Y.NspinCoarse();

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_COARSEPC_DIRAC || bidirectional_debug ||
				dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC || dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC || need_bidirectional);

    if (getVerbosity() >= QUDA_VERBOSE) {
      if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
      else printfQuda("Doing uni-directional link coarsening\n");
    }

    // Figure out nFace
    const int nFace = (dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) ? 3 : 1;

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    using Arg = CalculateYArg<from_coarse, Float,fineSpin,coarseSpin,fineColor,coarseColor,coarseGauge,coarseGaugeAtomic,fineGauge,avSpinor,uvSpinor,vSpinor,fineClover>;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, AV, G, L, K, V, C, Cinv, v, kappa, mass,
	    mu, mu_factor, x_size, xc_size, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    arg.max_h = static_cast<Float*>(pool_pinned_malloc(sizeof(Float)));
    arg.max_d = static_cast<Float*>(pool_device_malloc(sizeof(Float)));

    CalculateY<use_mma, location, Arg> y(arg, v, uv, av, Y_, X_, Y_atomic_, X_atomic_, nFace);

    QudaFieldLocation location_ = checkLocation(Y_, X_, av, v);
    logQuda(QUDA_VERBOSE, "Running link coarsening on the %s\n", location_ == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors; 3 for long-link operators
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v.Ghost());  // point the accessor to the correct ghost buffer
    if (&v == &av) arg.AV.resetGhost(av.Ghost());
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    logQuda(QUDA_VERBOSE, "V2 = %e\n", arg.V.norm2(v));

    // If doing preconditioned clover then we first multiply the
    // null-space vectors by the clover inverse matrix, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      logQuda(QUDA_VERBOSE, "Computing AV\n");
      y.setComputeType(COMPUTE_AV);

      if (av.Precision() == QUDA_HALF_PRECISION) {
        y.setComputeMax(true);
        y.apply(device::get_default_stream());
        y.setComputeMax(false);
        auto av_max = *arg.max_h;
        logQuda(QUDA_DEBUG_VERBOSE, "av_max %e\n", av_max);
	av.Scale(av_max);
	arg.AV.resetScale(av_max);
      }

      y.apply(device::get_default_stream());
      logQuda(QUDA_VERBOSE, "AV2 = %e\n", arg.AV.norm2(av));
    }

    // If doing preconditioned twisted-mass then we first multiply the
    // null-space vectors by the inverse twist, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_MASSPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      logQuda(QUDA_VERBOSE, "Computing TMAV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
	// this is just a trivial rescaling kernel, find the maximum
	complex<Float> fp(1./(1.+arg.mu*arg.mu),-arg.mu/(1.+arg.mu*arg.mu));
	complex<Float> fm(1./(1.+arg.mu*arg.mu),+arg.mu/(1.+arg.mu*arg.mu));
	double max = std::max({abs(fp.real()), abs(fp.imag()), abs(fm.real()), abs(fm.imag())}) * v.Scale();
	logQuda(QUDA_DEBUG_VERBOSE, "tm max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_TMAV);
      y.apply(device::get_default_stream());

      logQuda(QUDA_VERBOSE, "AV2 = %e\n", arg.AV.norm2(av));
    }

    // If doing preconditioned twisted-clover then we first multiply the
    // null-space vectors by the inverse of the squared clover matrix plus
    // mu^2, and then we multiply the result by the clover matrix. This is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      logQuda(QUDA_VERBOSE, "Computing TMCAV\n");
      y.setComputeType(COMPUTE_TMCAV);

      if (av.Precision() == QUDA_HALF_PRECISION) {
        y.setComputeMax(true);
        y.apply(device::get_default_stream());
        y.setComputeMax(false);
        auto av_max = *arg.max_h;
	logQuda(QUDA_DEBUG_VERBOSE, "av_max %e\n", av_max);
	av.Scale(av_max);
	arg.AV.resetScale(av_max);
      }

      y.apply(device::get_default_stream());
      logQuda(QUDA_VERBOSE, "AV2 = %e\n", arg.AV.norm2(av));
    }

    // If doing the staggered or ASQTAD KD op we first multiply the null-space vectors by
    // the dagger of the KD term. This is needed for the coarse link computation
    if ( dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC ) {
      logQuda(QUDA_VERBOSE, "Computing KAV\n");
      y.setKDagger(true);
      y.setComputeType(COMPUTE_KV);

      if (av.Precision() == QUDA_HALF_PRECISION) {
        // each component of AV is the product of 8 ("KD" links times near-null component)
        y.setComputeMax(true);
        y.apply(device::get_default_stream());
        y.setComputeMax(false);
        auto kv_max = *arg.max_h;
        logQuda(QUDA_DEBUG_VERBOSE, "kv_max %e\n", kv_max);
        av.Scale(kv_max);
        arg.AV.resetScale(kv_max);
      }

      y.apply(device::get_default_stream());
      logQuda(QUDA_VERBOSE, "KV2 = %e\n", arg.AV.norm2(av));
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
        logQuda(QUDA_VERBOSE, "Computing forward %d UV and VUV\n", d);
        y.setDimension(d);
      	y.setDirection(QUDA_FORWARDS);
        y.setComputeType(COMPUTE_UV);  // compute U*V product

        if (uv.Precision() == QUDA_HALF_PRECISION) {
          y.setComputeMax(true);
          y.apply(device::get_default_stream());
          y.setComputeMax(false);
          auto uv_max = *arg.max_h;
          uv.Scale(uv_max);
          arg.UV.resetScale(uv_max);
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE)  printfQuda("%d uv_max = %e\n", from_coarse ? d+4 : d, uv_max);
        }

        y.apply(device::get_default_stream());
        logQuda(QUDA_VERBOSE, "UV2[%d] = %e\n", d, arg.UV.norm2(uv));

        // if we are writing to a temporary, we need to zero it before each computation
        if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

        y.setComputeType(COMPUTE_VUV); // compute Y += VUV
        y.apply(device::get_default_stream());

        // long link coarsen, if necessary
        if (dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) {
          // required to make sure coarsened three-hop links stay in X and Y (as opposed to needing a coarse two-hop term)
          if (geo_bs[d] >= 3) {
            y.setComputeType(COMPUTE_LV); // compute L*V product

            if (uv.Precision() == QUDA_HALF_PRECISION) {
              y.setComputeMax(true);
              y.apply(device::get_default_stream());
              y.setComputeMax(false);
              auto lv_max = *arg.max_h;
              // we still reuse uv
              uv.Scale(lv_max);
              arg.UV.resetScale(lv_max);
              if (getVerbosity() >= QUDA_DEBUG_VERBOSE)  printfQuda("%d lv_max = %e\n", d, lv_max);
            }

            y.apply(device::get_default_stream());
            logQuda(QUDA_VERBOSE, "LV2[%d] = %e\n", d, arg.UV.norm2(uv));

            // *do not* zero out X and Y --- X;Y = VUV + VLV
            y.setComputeType(COMPUTE_VLV); // compute Y += VLV
            y.apply(device::get_default_stream());

          } else if (allow_truncation) {
            logQuda(QUDA_VERBOSE, "Skipping long link coarsening because geo_bs[%d] = %d is too small\n", d, geo_bs[d]);
          } else {
            errorQuda("Aggregate size geo_bs[%d] == %d is too small for long link coarsening", d, geo_bs[d]);
          }
        }

        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("Y2[%d] (atomic) = %e\n", 4+d, Y_atomic_.norm2((4+d) % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint()));

        // now convert from atomic to application computation format if necessary for Y[d]
        if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint()) {

          if (coarseGauge::fixedPoint()) {
            double y_max = Y_atomic_.abs_max((4+d) % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint());

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
                y.apply(device::get_default_stream());
              }

              y.setDimension(d);
              Y_.Scale(y_max);
              arg.Y.resetScale(Y_.Scale());
            }
            logQuda(QUDA_DEBUG_VERBOSE, "Y[%d] (atomic) max = %e Y[%d] scale = %e\n", 4+d, y_max, 4+d, Y_.Scale());
          }

          y.setComputeType(COMPUTE_CONVERT);
          y.apply(device::get_default_stream());
        }

        logQuda(QUDA_VERBOSE, "Y2[%d] = %e\n", 4+d, Y_.norm2( 4+d ));
      }
    }

    if ( ((dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC) &&
	 (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD)) || (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) ) {
      av.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
      arg.AV.resetGhost(av.Ghost());  // make sure we point to the correct pointer in the accessor
      LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      logQuda(QUDA_VERBOSE, "Computing backward %d UV and VUV\n", d);
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      y.setComputeType(COMPUTE_UV);  // compute U*A*V product

      if (uv.Precision() == QUDA_HALF_PRECISION) {
        y.setComputeMax(true);
        y.apply(device::get_default_stream());
        y.setComputeMax(false);
        auto uv_max = *arg.max_h;
        uv.Scale(uv_max);
        arg.UV.resetScale(uv_max);
        logQuda(QUDA_DEBUG_VERBOSE, "%d uv_max = %e\n", d, uv_max);
      }

      y.apply(device::get_default_stream());
      logQuda(QUDA_VERBOSE, "UAV2[%d] = %e\n", d, arg.UV.norm2(uv));

      // if we are writing to a temporary, we need to zero it before each computation
      if (Y_atomic.Geometry() == 1) Y_atomic_.zero();

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(device::get_default_stream());

      // long link coarsen, if necessary
      if (dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) {
        // required to make sure coarsened three-hop links stay in X and Y (as opposed to needing a coarse two-hop term)
        if (geo_bs[d] >= 3) {
          y.setComputeType(COMPUTE_LV); // compute L*V product

          if (uv.Precision() == QUDA_HALF_PRECISION) {
            y.setComputeMax(true);
            y.apply(device::get_default_stream());
            y.setComputeMax(false);
            auto lv_max = *arg.max_h;
            // we still reuse uv
            uv.Scale(lv_max);
            arg.UV.resetScale(lv_max);
            if (getVerbosity() >= QUDA_DEBUG_VERBOSE)  printfQuda("%d lv_max = %e\n", d, lv_max);
          }

          y.apply(device::get_default_stream());
          logQuda(QUDA_VERBOSE, "LAV2[%d] = %e\n", d, arg.UV.norm2(uv));

          // *do not* zero out X and Y --- X;Y = VUV + VLV
          y.setComputeType(COMPUTE_VLV); // compute Y += VLV
          y.apply(device::get_default_stream());

        } else if (allow_truncation) {
          logQuda(QUDA_VERBOSE, "Skipping long link coarsening because geo_bs[%d] = %d is too small\n", d, geo_bs[d]);
        } else {
          errorQuda("Aggregate size geo_bs[%d] == %d is too small for long link coarsening", d, geo_bs[d]);
        }
      }

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Y2[%d] (atomic) = %e\n", d, Y_atomic_.norm2(d%arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint()));

      // now convert from atomic to application computation format if necessary for Y[d]
      if (coarseGaugeAtomic::fixedPoint() || coarseGauge::fixedPoint() ) {

        if (coarseGauge::fixedPoint()) {
          double y_max = Y_atomic_.abs_max(d % arg.Y_atomic.geometry, coarseGaugeAtomic::fixedPoint());

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
                y.apply(device::get_default_stream());
              }
            }

            y.setDirection(QUDA_BACKWARDS);
            for (int d_=0; d_<d; d_++) {
              y.setDimension(d_);
              y.apply(device::get_default_stream());
            }

            y.setDimension(d);
            Y_.Scale(y_max);
            arg.Y.resetScale(Y_.Scale());
          }
          logQuda(QUDA_DEBUG_VERBOSE, "Y[%d] (atomic) max = %e Y[%d] scale = %e\n", d, y_max, d, Y_.Scale());
        }

        y.setComputeType(COMPUTE_CONVERT);
        y.apply(device::get_default_stream());
      }

      logQuda(QUDA_VERBOSE, "Y2[%d] = %e\n", d, Y_.norm2( d ));
    }

    logQuda(QUDA_VERBOSE, "X2 = %e\n", X_atomic_.norm2(0, coarseGaugeAtomic::fixedPoint()));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      logQuda(QUDA_VERBOSE, "Reversing links\n");
      y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(device::get_default_stream());
    }

    // Check if we have a fine or coarse clover term that needs to be coarsened
    if (dirac == QUDA_CLOVER_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      logQuda(QUDA_VERBOSE, "Computing fine->coarse clover term\n");
      y.setComputeType(COMPUTE_COARSE_CLOVER);
      y.apply(device::get_default_stream());
    } else if (dirac == QUDA_COARSE_DIRAC) {
      logQuda(QUDA_VERBOSE, "Computing coarse CV and VCV via UV and VUV\n");

      // We can write coarsening the coarse clover as a UV, VUV sequence where `U` is replaced with `C`
      y.setDimension(-1);
      y.setDirection(QUDA_IN_PLACE);
      y.setComputeType(COMPUTE_UV);  // compute C*V product

      if (uv.Precision() == QUDA_HALF_PRECISION) {
        y.setComputeMax(true);
        y.apply(device::get_default_stream());
        y.setComputeMax(false);
        auto uv_max = *arg.max_h;
        uv.Scale(uv_max);
        arg.UV.resetScale(uv_max);
        logQuda(QUDA_DEBUG_VERBOSE, "cv_max = %e\n", uv_max);
      }

      y.apply(device::get_default_stream());
      logQuda(QUDA_VERBOSE, "CV2 = %e\n", arg.UV.norm2(uv));

      y.setComputeType(COMPUTE_VUV); // compute X += VCV
      y.apply(device::get_default_stream());
      if (getVerbosity() >= QUDA_VERBOSE)
        printfQuda("X2 (atomic) = %e\n", X_atomic_.norm2(0, coarseGaugeAtomic::fixedPoint()));

    } else if (dirac == QUDA_STAGGERED_DIRAC || dirac == QUDA_STAGGEREDPC_DIRAC || dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC) {
      logQuda(QUDA_VERBOSE, "Summing staggered mass contribution to coarse clover\n");
      y.setComputeType(COMPUTE_STAGGEREDMASS);
      y.apply(device::get_default_stream());
    } else if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) {
      if (arg.mass != static_cast<Float>(0.)) {
        // We can write coarsening the coarse KD op as ( K^dag V) . (V)
        // ( K^dag V) is already contained in AV
        y.setDimension(-1);
        y.setDirection(QUDA_IN_PLACE);

        logQuda(QUDA_VERBOSE, "Computing staggered mass times KD inverse contribution to coarse clover\n");
        y.setComputeType(COMPUTE_VUV);
        y.apply(device::get_default_stream());
      }
    } else {  //Otherwise, we just have to add the identity matrix
      logQuda(QUDA_VERBOSE, "Summing diagonal contribution to coarse clover\n");
      y.setComputeType(COMPUTE_DIAGONAL);
      y.apply(device::get_default_stream());
    }

    if (arg.mu*arg.mu_factor!=0 || dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      if (dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC)
	arg.mu_factor += 1.;
      logQuda(QUDA_VERBOSE, "Adding mu = %e\n",arg.mu*arg.mu_factor);
      y.setComputeType(COMPUTE_TMDIAGONAL);
      y.apply(device::get_default_stream());
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
      y.apply(device::get_default_stream());
    }

    logQuda(QUDA_VERBOSE, "X2 = %e\n", X_.norm2(0));

    pool_device_free(arg.max_d);
    pool_pinned_free(arg.max_h);
  }

} // namespace quda
