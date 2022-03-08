#pragma once

#include <tunable_kernel.h>
#include <lattice_field.h>
#include <register_traits.h>
#include <reduction_kernel.h>
#include <reduction_kernel_host.h>

namespace quda
{

  /**
     @brief This derived tunable class is for reduction kernels, and
     partners the Reduction2D kernel.  The x threads will
     typically correspond to the checkerboarded volume.  Each thread
     block is two dimensional, with the y dimension typically equal to
     two and is the parity dimension.
   */
  template <int block_size_y = 2> class TunableReduction2D : public TunableKernel
  {
  protected:
    static constexpr bool grid_stride = true;

    /**
       Reduction kernels require grid-size tuning, so enable this, and
       we mark as final to prevent a derived class from accidentally
       switching it off.
    */
    bool tuneGridDim() const final { return grid_stride; }

    virtual unsigned int minGridSize() const { return Tunable::minGridSize(); }
    virtual int gridStep() const { return minGridSize(); }

    /**
       The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
     */
    virtual unsigned int maxBlockSize(const TuneParam &) const { return device::max_reduce_block_size<block_size_y>(); }

    template <int block_size_x, template <typename> class Functor, typename FunctorArg>
    qudaError_t launch(FunctorArg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Reduction2D), tp, stream, Arg(arg));
      } else if constexpr (block_size_x != device::warp_size()) {
        return launch<block_size_x - device::warp_size(), Functor>(arg, tp, stream);
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
        return QUDA_ERROR;
      }
    }

    template <template <typename> class Functor, typename T, typename Arg>
    void launch_device(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      arg.launch_error = launch<device::max_reduce_block_size<block_size_y>(), Functor>(arg, tp, stream);

      if (!commAsyncReduction()) {
        std::vector<T> result_(1);
        arg.complete(result_, stream);
        if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result_);
        result = result_[0];
      }
    }

    template <template <typename> class Functor, typename T, typename Arg>
    void launch_host(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      result_[0] = Reduction2D_host<Functor, Arg>(arg);
      if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result_);
      result = result_[0];
    }

    /**
       @brief Launch reduction kernel
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which we the reduction is being done
       @param[in] arg Kernel argument struct
       @param[in] param Constant kernel meta data
     */
    template <template <typename> class Functor, bool enable_host = false, typename T, typename Arg>
    void launch(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor>(result, tp, stream, arg);
      } else if constexpr (enable_host) {
        launch_host<Functor>(result, tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableReduction2D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
#ifdef QUDA_FAST_COMPILE_REDUCE
      strcat(aux, "fast_compile,");
#endif
      if (commAsyncReduction()) strcat(aux, "async,");
      strcat(aux, field.SiteSubset() == QUDA_FULL_SITE_SUBSET ? "nParity=2," : "nParity=1,");
      strcat(aux, field.AuxString());
    }

    TunableReduction2D(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
#ifdef QUDA_FAST_COMPILE_REDUCE
      strcat(aux, "fast_compile,");
#endif
      if (commAsyncReduction()) strcat(aux, "async,");
    }

    virtual bool advanceBlockDim(TuneParam &param) const
    {
      bool rtn = Tunable::advanceBlockDim(param);
      param.block.y = block_size_y;
      return rtn;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = block_size_y;
    }

    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = block_size_y;
    }
  };

  /**
     @brief This derived tunable class is for multi-reduction kernels,
     and partners the MultiReduction kernel.  Each thread block is
     three dimensional.  The x threads will typically correspond to
     the checkerboarded volume.  The y thread dimension is constrained
     to remain inside the thread block and this dimension is
     contracted in the reduction.  The z thread dimension is a batch
     dimension that is not contracted in the reduction.
   */
  template <int block_size_y = 1> class TunableMultiReduction : public TunableReduction2D<block_size_y>
  {
    using Tunable::apply;
    using TunableReduction2D<block_size_y>::location;
    using TunableReduction2D<block_size_y>::grid_stride;

  protected:
    const unsigned int n_batch;           /** Reduction batch size */
    const unsigned int n_batch_block_max; /** Maximum reduction batch per thread block */

    /**
       @brief we don't want to inherit TunableReduction2D behaviour
       here which is catered for non-block / non-batch reductions, so
       inherit from the "grandfather"
    */
    virtual unsigned int minGridSize() const { return Tunable::minGridSize(); }

    /**
       @brief we don't want to inherit TunableReduction2D behaviour
       here which is catered for non-block / non-batch reductions, so
       inherit from the "grandfather"
    */
    virtual int gridStep() const { return Tunable::gridStep(); }

    /**
       The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
     */
    unsigned int maxBlockSize(const TuneParam &) const { return device::max_multi_reduce_block_size<block_size_y>(); }

    template <int block_size_x, template <typename> class Functor, typename FunctorArg>
    qudaError_t launch(FunctorArg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(MultiReduction), tp, stream, Arg(arg));
      } else if constexpr (block_size_x != device::warp_size()) {
        return launch<block_size_x / 2, Functor>(arg, tp, stream);
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
        return QUDA_ERROR;
      }
    }

    template <template <typename> class Functor, typename T, typename Arg>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (n_batch_block_max > Arg::max_n_batch_block)
        errorQuda("n_batch_block_max = %u greater than maximum supported %u", n_batch_block_max, Arg::max_n_batch_block);
      arg.launch_error = launch<device::max_multi_reduce_block_size<block_size_y>(), Functor>(arg, tp, stream);

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result);
      }
    }

    template <template <typename> class Functor, typename T, typename Arg>
    void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      if (n_batch_block_max > Arg::max_n_batch_block)
        errorQuda("n_batch_block_max = %u greater than maximum supported %u", n_batch_block_max, Arg::max_n_batch_block);
      using reduce_t = typename Functor<Arg>::reduce_t;

      int input_size = vec_length<reduce_t>::value;
      int output_size = vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      auto value = MultiReduction_host<Functor, Arg>(arg);

      for (int j = 0; j < (int)arg.threads.z; j++) result[j] = value[j];
      if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result);
    }

    template <template <typename> class Functor, bool enable_host = false, typename T, typename Arg>
    void launch(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor>(result, tp, stream, arg);
      } else if constexpr (enable_host) {
        launch_host<Functor>(result, tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableMultiReduction(const LatticeField &field, unsigned int n_batch, unsigned int n_batch_block_max = 1u,
                          QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(field, location), n_batch(n_batch), n_batch_block_max(n_batch_block_max)
    {
    }

    TunableMultiReduction(size_t n_items, unsigned int n_batch, unsigned int n_batch_block_max = 1u,
                          QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(n_items, location), n_batch(n_batch), n_batch_block_max(n_batch_block_max)
    {
    }

    template <typename T> bool is_power2(T x) const { return (x != 0) && ((x & (x - 1)) == 0); }

    /**
       @brief Custom variant that only selects power of two block
       sizes.  We restrict in this way to limit instantiations.
     */
    bool advanceBlockDim(TuneParam &param) const
    {
      bool rtn;
      do {
        rtn = Tunable::advanceBlockDim(param);
      } while (rtn && !is_power2(param.block.x));

      if (rtn) {
        return true;
      } else {
        if (param.block.z < n_batch && param.block.z < device::max_threads_per_block_dim(2)
            && param.block.x * param.block.y * (param.block.z + 1) <= device::max_threads_per_block()
            && param.block.z < n_batch_block_max) {
          param.block.z++;
          param.grid.z = (n_batch + param.block.z - 1) / param.block.z;
          return true;
        } else { // we have run off the end so let's reset
          param.block.z = 1;
          param.grid.z = (n_batch + param.block.z - 1) / param.block.z;
          return false;
        }
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block = {param.block.x, 1, 1};
      param.grid = {param.grid.x, 1, (n_batch + param.block.z - 1) / param.block.z};
    }

    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block = {param.block.x, 1, 1};
      param.grid = {param.grid.x, 1, (n_batch + param.block.z - 1) / param.block.z};
    }
  };

} // namespace quda
