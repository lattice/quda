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
       @brief Block reduction kernels do not use grid-size tuning, so
       disable this, and we mark as final to prevent a derived class
       from accidentally switching it on.
    */
    bool tuneGridDim() const final { return grid_stride; }

    /**
       @brief Launch the block reduction kernel with a given block
       size on the device performing the block reduction defined in
       the functor.  We recursively iterate over the length of
       instantiated block sizes until we succeed, or error out.
       @tparam Functor Class which performs any pre-reduction
       transformation (defined as ternary operator) as well as a store
       method for writing out the result.
       @tparam Block Class that must contain a static array of block
       sizes "block" we wish to instantiate
       @tparam idx Index of the block-size array we are instantiating
       (default = -1 which initiates a recursion over the length of
       the array)
       @param[in] arg Kernel argument struct
       @param[in] tp The launch parameters
       @param[in] stream The stream on which the execution is done
     */
    template <template <typename> class Functor, typename Block, int idx = -1, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      constexpr int block_idx = idx == -1 ? Block::block.size() - 1 : idx;
      if (tp.block.x == Block::block[block_idx]) {
        const_cast<Arg &>(arg).grid_dim = tp.grid;
        const_cast<Arg &>(arg).block_dim = tp.block;
        // derive a BlockKernelArg from the kernel argument to allow for block-size knowledge in the kernel
        using BlockArg = BlockKernelArg<Block::block[block_idx], Arg>;
        TunableKernel::launch_device<Functor, grid_stride>(KERNEL(BlockKernel2D), tp, stream, BlockArg(arg));
      } else if constexpr(block_idx != 0) {
        launch_device<block_idx - 1, Block, Functor>(arg, tp, stream);
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
      }
    }

    /**
       @brief Launch the block reduction kernel with a given block
       size on the host performing the block reduction defined in the
       functor.  We recursively iterate over the length of
       instantiated block sizes until we succeed, or error out.
       @tparam Functor Class which performs any pre-reduction
       transformation (defined as ternary operator) as well as a store
       method for writing out the result.
       @tparam Block Class that must contain a static array of block
       sizes "block" we wish to instantiate
       @tparam idx Index of the block-size array we are instantiating
       (default = -1 which initiates a recursion over the length of
       the array)
       @param[in] arg Kernel argument struct
       @param[in] tp The launch parameters
     */
    template <template <typename> class Functor, typename Block, int idx = -1, typename Arg>
    void launch_host(const TuneParam &tp, const qudaStream_t &, const Arg &arg)
    {
      constexpr int block_idx = idx == -1 ? Block::block.size() - 1 : idx;
      if (tp.block.x == Block::block[block_idx]) {
        const_cast<Arg &>(arg).grid_dim = tp.grid;
        const_cast<Arg &>(arg).block_dim = tp.block;
        BlockKernel2D_host<Functor>(BlockKernelArg<Block::block[block_idx], Arg>(arg));
      } else if constexpr(block_idx != 0) {
        launch_host<block_idx - 1, Block, Functor>(tp, arg);
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
      }
    }

    /**
       @brief Launch the block reduction kernel on the set loation
       with a given block size on the device performing the block
       reduction defined in the functor.  We recursively iterate over
       the length of instantiated block sizes until we succeed, or
       error out.
       @tparam Functor Class which performs any pre-reduction
       transformation (defined as ternary operator) as well as a store
       method for writing out the result.
       @tparam Block Class that must contain a static array of block
       sizes "block" we wish to instantiate
       @tparam enable_host Whether to enable host compilation (default is not to)
       @param[in] arg Kernel argument struct
       @param[in] tp The launch parameters
       @param[in] stream The stream on which the execution is done
     */
    template <template <typename> class Functor, typename Block, bool enable_host = false, typename Arg>
    void launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Block>(tp, stream, arg);
      } else if constexpr (enable_host) {
        launch_host<Functor, Block>(tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

  public:
    /**
       @brief Constructor for kernels that use a lattice field
       @param[in] field A lattice field instance used for metadata
       @param[in] vector_length_y Batch size for the block-reduction in the y-dimension
       @param[in] max_block_y Maximum batch size per block (maximum y-dimension block size)
       @param[in] location Optional overload for the location where the calculation will take place
     */
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

    /**
       @brief Derived specialization for autotuning the batch size
       dimension
       @param[in,out] param TuneParam object passed during autotuning
     */
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

    /**
       @brief Overload that sets ensures the y-dimension block size is set appropriately
       @param[in,out] param TuneParam object passed during autotuning
     */
    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    /**
       @brief Overload that sets ensures the y-dimension block size is set appropriately
       @param[in,out] param TuneParam object passed during autotuning
     */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    /**
       @brief Resize the problem size in the y dimension
       @brief[in] y New problem size
    */
    void resizeVector(int y) const { vector_length_y = y; }

    /**
       @brief Resize the autotuning step size in the y dimension
       @brief[in] y New step size
    */
    void resizeStep(int y) const { step_y = y; }
  };

} // namespace quda
