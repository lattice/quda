#pragma once

#include <tune_quda.h>
#include <device.h>
#include <kernel_helper.h>
#include <reduce_helper.h>
#include <block_reduction_kernel.h>

#ifdef JITIFY
#include <jitify_helper.cuh>
#endif

namespace quda {

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
  class TunableBlockReduction2D : public Tunable
  {
  protected:
    const LatticeField &field;
    QudaFieldLocation location;
    mutable unsigned int vector_length_y;
    mutable unsigned int step_y;
    const unsigned int max_block_y;
    bool tune_block_x;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    /**
       Block reduction kernels do not use grid-size tuning, so disable this, and
       we mark as final to prevent a derived class from accidentally
       switching it on.
    */
    bool tuneGridDim() const final { return false; }

    template <int idx, typename Block, template <typename> class Transformer, typename Arg>
    typename std::enable_if<idx != 0, void>::type launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == Block::block[idx]) qudaLaunchKernel(BlockReductionKernel2D<Block::block[idx], Transformer, Arg>, tp, stream, arg);
      else launch<idx - 1, Block, Transformer>(arg, tp, stream);
    }

    template <int idx, typename Block, template <typename> class Transformer, typename Arg>
    typename std::enable_if<idx == 0, void>::type launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == Block::block[idx]) qudaLaunchKernel(BlockReductionKernel2D<Block::block[idx], Transformer, Arg>, tp, stream, arg);
      else errorQuda("Unexpected block size %d\n", tp.block.x);
    }

    template <template <typename> class Transformer, typename Block, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
#ifdef JITIFY
      jitify_error = launch_jitify_block("quda::BlockReductionKernel2D", tp, stream, arg);
#else
      launch<Block::block.size() - 1, Block, Transformer>(arg, tp, stream);
#endif
    }

    template <template <typename> class Transformer, typename Block, typename Arg>
    void launch_host(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      Transformer<Arg> t(arg);

      for (int block = 0; block < arg.n_block; block++) {
        for (int j = 0; j < (int)arg.threads.y; j++) {
          reduce_t value;
          for (int i = 0; i < (int)arg.threads.x; i++) value += t(block, i, j);

          t.store(value, block, j);
        }
      }
    }

    /**
       @brief Launch function for BlockReductionKernel2D.
       @tparam Transformer Class which performs any pre-reduction
       transformation (defined as ternary operator) as well as a store
       method for writing out the result.
       @tparam Block Class that must contain a static std::array of
       block sizes "block" we wish to instantiate
       @param[in] tp Kernel launch parameters
       @param[in] stream Stream in which to execute
       @param[in,out] arg Algorithm meta data
     */
    template <template <typename> class Transformer, typename Block, bool enable_host = false, typename Arg>
    typename std::enable_if<!enable_host, void>::type launch(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer, Block>(tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Transformer, typename Block, bool enable_host = false, typename Arg>
    typename std::enable_if<enable_host, void>::type launch(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer, Block>(tp, stream ,arg);
      } else {
        launch_host<Transformer, Block>(tp, stream, arg);
      }
    }

  public:
    TunableBlockReduction2D(const LatticeField &field, unsigned int vector_length_y,
                            unsigned int max_block_y = 0, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      field(field),
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location()),
      vector_length_y(vector_length_y),
      step_y(1),
      max_block_y(max_block_y == 0 ? vector_length_y : max_block_y),
      tune_block_x(false)
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
      strcat(aux, field.AuxString());
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
	if (param.block.y < vector_length_y && param.block.y < device::max_threads_per_block_dim(1) &&
	    param.block.x*(param.block.y+step_y)*param.block.z <= device::max_threads_per_block() &&
            ((param.block.y + step_y) <= max_block_y)) {
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

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

}
