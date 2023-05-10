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
    const bool tune_block_x;
    mutable unsigned int vector_length_z;
    mutable unsigned int step_z;
    const unsigned int max_block_z;

    static constexpr bool grid_stride = false;

    /**
       @brief Block reduction kernels do not use grid-size tuning, so
       disable this.
    */
    bool tuneGridDim() const { return grid_stride; }

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
       (default = 0 which initiates a recursion over the length of
       the array)
       @param[in] arg Kernel argument struct
       @param[in] tp The launch parameters
       @param[in] stream The stream on which the execution is done
     */
    template <template <typename> class Functor, typename Block, unsigned int idx = 0, typename FunctorArg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const FunctorArg &arg)
    {
      // in block == 0, then we aren't templating on block size
      if (tp.block.x == Block::block[idx] || Block::block[idx] == 1) {
        const_cast<FunctorArg &>(arg).grid_dim = tp.grid;
        const_cast<FunctorArg &>(arg).block_dim = tp.block;
        // derive a BlockKernelArg from the kernel argument to allow for block-size knowledge in the kernel
        using Arg = BlockKernelArg<Block::block[idx], FunctorArg>;
        TunableKernel::launch_device<Functor, grid_stride>(KERNEL(BlockKernel2D), tp, stream, Arg(arg));
      } else if constexpr (idx < Block::block.size() - 1) {
        launch_device<Functor, Block, idx + 1>(tp, stream, arg);
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
       (default = 0 which initiates a recursion over the length of
       the array)
       @param[in] tp The launch parameters
       @param[in] stream The stream on which the execution is done (unused on the host)
       @param[in] arg Kernel argument struct
     */
    template <template <typename> class Functor, typename Block, unsigned int idx = 0, typename Arg>
    void launch_host(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (tp.block.x == Block::block[idx]) {
        const_cast<Arg &>(arg).grid_dim = tp.grid;
        const_cast<Arg &>(arg).block_dim = tp.block;
        BlockKernel2D_host<Functor>(BlockKernelArg<Block::block[idx], Arg>(arg));
      } else if constexpr (idx < Block::block.size() - 1) {
        launch_host<Functor, Block, idx + 1>(tp, stream, arg);
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
        launch_host<Functor, Block>(tp, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

  public:
    /**
       @brief Constructor for kernels that use a lattice field
       @param[in] field A lattice field instance used for metadata
       @param[in] vector_length_z Batch size for the block-reduction in the z-dimension
       @param[in] max_block_z Maximum batch size per block (maximum z-dimension block size)
       @param[in] location Optional overload for the location where the calculation will take place
     */
    TunableBlock2D(const LatticeField &field, bool tune_block_x, unsigned int vector_length_z,
                   unsigned int max_block_z = 0, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel(field, location),
      field(field),
      tune_block_x(tune_block_x),
      vector_length_z(vector_length_z),
      step_z(1),
      max_block_z(max_block_z == 0 ? vector_length_z : max_block_z) {}

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
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (ret) {
        return true;
      } else { // block.x (spacetime) was reset

        // we can advance spin/block-color since this is valid
        if (param.block.z < vector_length_z && param.block.z < device::max_threads_per_block_dim(2)
            && param.block.x * param.block.y * (param.block.z + step_z) <= device::max_threads_per_block()
            && ((param.block.z + step_z) <= max_block_z)) {
          param.block.z += step_z;
          param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
          return true;
        } else { // we have run off the end so let's reset
          param.block.z = step_z;
          param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
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
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    /**
       @brief Overload that sets ensures the y-dimension block size is set appropriately
       @param[in,out] param TuneParam object passed during autotuning
     */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    /**
       @brief Resize the problem size in the y dimension
       @brief[in] y New problem size
    */
    void resizeVector(int z) const { vector_length_z = z; }

    /**
       @brief Resize the autotuning step size in the z dimension
       @brief[in] z New step size
    */
    void resizeStep(int z) const { step_z = z; }
  };

} // namespace quda
