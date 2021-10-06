#pragma once

#include <tunable_kernel.h>
#include <lattice_field.h>
#include <block_reduction_kernel.h>
#include <block_reduction_kernel_host.h>

namespace quda
{

  /**
     @brief This derived tunable class is for block reduction kernels,
     and partners the BlockReduction2D kernel.  Each reduction block
     is mapped to x grid, with each block mapping to a thread block.
     Each thread block is potentially two dimensional, with the y
     dimension can be mapped to vector of computations, similar to
     TunableKernel2D.  Due to the exact mapping of reduction block to
     the thread blocks, no block-size tuning is performed in the x
     dimension.
   */
  class TunableBlock2D : public TunableKernel
  {
  protected:
    const LatticeField &field;
    mutable unsigned int vector_length_y;
    mutable unsigned int step_y;
    const unsigned int max_block_y;
    bool tune_block_x;

    static constexpr bool grid_stride = false;

    /**
       Block reduction kernels do not use grid-size tuning, so disable this, and
       we mark as final to prevent a derived class from accidentally
       switching it on.
    */
    bool tuneGridDim() const final { return grid_stride; }

    template <int idx, typename Block, template <typename> class Functor, typename FunctorArg>
    std::enable_if_t<idx != 0, void> launch_device(const FunctorArg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == Block::block[idx]) {
        using Arg = BlockKernelArg<Block::block[idx], FunctorArg>;
        TunableKernel::launch_device<Functor, grid_stride>(KERNEL(BlockKernel2D), tp, stream, Arg(arg));
      } else {
        launch_device<idx - 1, Block, Functor>(arg, tp, stream);
      }
    }

    template <int idx, typename Block, template <typename> class Functor, typename FunctorArg>
    std::enable_if_t<idx == 0, void> launch_device(const FunctorArg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == Block::block[idx]) {
        using Arg = BlockKernelArg<Block::block[idx], FunctorArg>;
        TunableKernel::launch_device<Functor, grid_stride>(KERNEL(BlockKernel2D), tp, stream, Arg(arg));
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
      }
    }

    template <template <typename> class Functor, typename Block, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      const_cast<Arg &>(arg).grid_dim = tp.grid;
      const_cast<Arg &>(arg).block_dim = tp.block;
      launch_device<Block::block.size() - 1, Block, Functor>(arg, tp, stream);
    }

    template <int idx, typename Block, template <typename> class Functor, typename Arg>
    std::enable_if_t<idx != 0, void> launch_host(const TuneParam &tp, const Arg &arg)
    {
      if (tp.block.x == Block::block[idx])
        BlockKernel2D_host<Functor>(BlockKernelArg<Block::block[idx], Arg>(arg));
      else
        launch_host<idx - 1, Block, Functor>(tp, arg);
    }

    template <int idx, typename Block, template <typename> class Functor, typename Arg>
    std::enable_if_t<idx == 0, void> launch_host(const TuneParam &tp, const Arg &arg)
    {
      if (tp.block.x == Block::block[idx])
        BlockKernel2D_host<Functor>(BlockKernelArg<Block::block[idx], Arg>(arg));
      else
        errorQuda("Unexpected block size %d", tp.block.x);
    }

    template <template <typename> class Functor, typename Block, typename Arg>
    void launch_host(const TuneParam &tp, const qudaStream_t &, const Arg &arg)
    {
      const_cast<Arg &>(arg).grid_dim = tp.grid;
      const_cast<Arg &>(arg).block_dim = tp.block;
      launch_host<Block::block.size() - 1, Block, Functor>(tp, arg);
    }

    /**
       @brief Launch function for BlockReductionKernel2D.
       @tparam Functor Class which performs any pre-reduction
       transformation (defined as ternary operator) as well as a store
       method for writing out the result.
       @tparam Block Class that must contain a static std::array of
       block sizes "block" we wish to instantiate
       @param[in] tp Kernel launch parameters
       @param[in] stream Stream in which to execute
       @param[in,out] arg Algorithm meta data
     */
    template <template <typename> class Functor, typename Block, bool enable_host = false, typename Arg>
    std::enable_if_t<!enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Block>(tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, typename Block, bool enable_host = false, typename Arg>
    std::enable_if_t<enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Block>(tp, stream, arg);
      } else {
        launch_host<Functor, Block>(tp, stream, arg);
      }
    }

  public:
    TunableBlock2D(const LatticeField &field, unsigned int vector_length_y, unsigned int max_block_y = 0,
                   QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location()),
      field(field),
      vector_length_y(vector_length_y),
      step_y(1),
      max_block_y(max_block_y == 0 ? vector_length_y : max_block_y),
      tune_block_x(false)
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
      if (location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux, field.AuxString());
#ifdef QUDA_FAST_COMPILE_REDUCE
      strcat(aux, ",fast_compile");
#endif
    }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = tune_block_x ? Tunable::advanceBlockDim(param) : false;
      param.block.y = block.y;
      param.grid.y = grid.y;

      if (ret) {
        return true;
      } else { // block.x (spacetime) was reset

        // we can advance spin/block-color since this is valid
        if (param.block.y < vector_length_y && param.block.y < device::max_threads_per_block_dim(1)
            && param.block.x * (param.block.y + step_y) * param.block.z <= device::max_threads_per_block()
            && ((param.block.y + step_y) <= max_block_y)) {
          param.block.y += step_y;
          param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
          return true;
        } else { // we have run off the end so let's reset
          param.block.y = step_y;
          param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
          return false;
        }
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    void resizeVector(int y) const { vector_length_y = y; }
    void resizeStep(int y) const { step_y = y; }
  };

} // namespace quda
