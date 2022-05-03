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
  class TunableReduction2D : public TunableKernel
  {
  protected:
    static constexpr bool grid_stride = true;

    /**
       @brief Reduction kernels require grid-size tuning, so enable this, and
       we mark as final to prevent a derived class from accidentally
       switching it off.
    */
    bool tuneGridDim() const final { return grid_stride; }

    virtual unsigned int minGridSize() const { return Tunable::minGridSize(); }
    virtual int gridStep() const { return minGridSize(); }

    /**
       @brief The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
       @return Maximum block size
     */
    virtual unsigned int maxBlockSize(const TuneParam &) const { return device::max_block_size(); }

    /**
       @brief Launch reduction kernel on the device performing the
       reduction defined in the functor.  After the local computation
       has completed, the comm_reduce function that is defined in the
       functor class will be used to perform the inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which we the reduction is being done
       @param[in] arg Kernel argument struct
     */
    template <template <typename> class Functor, typename T, typename FunctorArg>
    void launch_device(T &result, const TuneParam &tp, const qudaStream_t &stream, FunctorArg &arg)
    {
      using Arg = ReduceKernelArg<FunctorArg>;
      arg.launch_error = TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Reduction2D), tp, stream, Arg(arg));

      if (!commAsyncReduction()) {
        std::vector<T> result_(1);
        arg.complete(result_, stream);
        if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result_);
        result = result_[0];
      }
    }

    /**
       @brief Launch reduction kernel on the host performing the reduction defined
       in the functor.  After the local computation has completed, the
       comm_reduce function that is defined in the functor class will
       be used to perform the inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @param[out] result The reduction result is copied here
       @param[in] arg Kernel argument struct
     */
    template <template <typename> class Functor, typename T, typename Arg>
    void launch_host(T &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      std::vector<T> result_(1);
      result_[0] = Reduction2D_host<Functor, Arg>(arg);
      if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result_);
      result = result_[0];
    }

    /**
       @brief Launch reduction kernel on the set location performing
       the reduction defined in the functor.  After the local
       computation has completed, the comm_reduce function that is
       defined in the functor class will be used to perform the
       inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @tparam enable_host Whether to enable host compilation (default is not to)
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which the execution is done
       @param[in] arg Kernel argument struct
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
    /**
       @brief Constructor for kernels that use a lattice field
       @param[in] field A lattice field instance used for metadata
       @param[in] location Optional overload for the location where the calculation will take place
     */
    TunableReduction2D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
      if (commAsyncReduction()) strcat(aux, "async,");
      strcat(aux, field.SiteSubset() == QUDA_FULL_SITE_SUBSET ? "nParity=2," : "nParity=1,");
      strcat(aux, field.AuxString());
    }

    /**
       @brief Constructor for kernels that have a problem size only
       @param[in] n_items Number of items being reduced
       @param[in] location Location where the calculation will take place
     */
    TunableReduction2D(size_t n_items, QudaFieldLocation location) : TunableKernel(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
      if (commAsyncReduction()) strcat(aux, "async,");
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
  class TunableMultiReduction : public TunableReduction2D
  {
    using Tunable::apply;

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
       @brief The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
       @return Maximum block size
     */
    unsigned int maxBlockSize(const TuneParam &) const { return device::max_block_size(); }

    /**
       @brief Launch multi-reduction kernel on the device performing
       the reduction defined in the functor.  After the local
       computation has completed, the comm_reduce function that is
       defined in the functor class will be used to perform the
       inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which we the reduction is being done
       @param[in] arg Kernel argument struct
     */
    template <template <typename> class Functor, typename T, typename FunctorArg>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, FunctorArg &arg)
    {
      if (n_batch_block_max > FunctorArg::max_n_batch_block)
        errorQuda("n_batch_block_max = %u greater than maximum supported %u", n_batch_block_max, FunctorArg::max_n_batch_block);
      using Arg = ReduceKernelArg<FunctorArg>;
      arg.launch_error = TunableKernel::launch_device<Functor, grid_stride>(KERNEL(MultiReduction), tp, stream, Arg(arg));

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result);
      }
    }

    /**
       @brief Launch multi-reduction kernel on the host performing the
       reduction defined in the functor.  After the local computation
       has completed, the comm_reduce function that is defined in the
       functor class will be used to perform the inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @param[out] result The reduction result is copied here
       @param[in] arg Kernel argument struct
     */
    template <template <typename> class Functor, typename T, typename Arg>
    void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      if (n_batch_block_max > Arg::max_n_batch_block)
        errorQuda("n_batch_block_max = %u greater than maximum supported %u", n_batch_block_max, Arg::max_n_batch_block);

      auto value = MultiReduction_host<Functor, Arg>(arg);
      for (int j = 0; j < (int)arg.threads.z; j++) result[j] = value[j];
      if (!activeTuning() && commGlobalReduction()) Functor<Arg>::comm_reduce(result);
    }

    /**
       @brief Launch multi-reduction kernel on the set location
       performing the reduction defined in the functor.  After the
       local computation has completed, the comm_reduce function that
       is defined in the functor class will be used to perform the
       inter-comm reduction.
       @tparam Functor The functor that defined the reduction operation
       @tparam enable_host Whether to enable host compilation (default is not to)
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which we the reduction is being done
       @param[in] arg Kernel argument struct
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
    /**
       @brief Constructor for kernels that use a lattice field
       @param[in] n_batch The batch size for the multi-reduction (number of reductions we are performing)
       @param[in] n_batch_block_max Maximum batch size per block (maximum z-dimension block size)
       @param[in] field A lattice field instance used for metadata
       @param[in] location Optional overload for the location where the calculation will take place
     */
    TunableMultiReduction(const LatticeField &field, unsigned int n_batch, unsigned int n_batch_block_max = 1u,
                          QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D(field, location), n_batch(n_batch), n_batch_block_max(n_batch_block_max)
    {
    }

    /**
       @brief Constructor for kernels that have a problem size only
       @param[in] n_items Number of items being reduced
       @param[in] n_batch The batch size for the multi-reduction (number of reductions we are performing)
       @param[in] n_batch_block_max Maximum batch size per block (maximum z-dimension block size)
       @param[in] location Location where the calculation will take place
     */
    TunableMultiReduction(size_t n_items, unsigned int n_batch, unsigned int n_batch_block_max,
                          QudaFieldLocation location) :
      TunableReduction2D(n_items, location), n_batch(n_batch), n_batch_block_max(n_batch_block_max)
    {
    }

    template <typename T> bool is_power2(T x) const { return (x != 0) && ((x & (x - 1)) == 0); }

    /**
       @brief Overload that only selects power of two block
       sizes.  We restrict in this way to limit instantiations.
       @param[in,out] param TuneParam object passed during autotuning
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

    /**
       @brief Overload that sets ensures the z-dimension block size is set appropriately
       @param[in,out] param TuneParam object passed during autotuning
     */
    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block = {param.block.x, 1, 1};
      param.grid = {param.grid.x, 1, (n_batch + param.block.z - 1) / param.block.z};
    }

    /**
       @brief Overload that sets ensures the z-dimension block size is set appropriately
       @param[in,out] param TuneParam object passed during autotuning
     */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block = {param.block.x, 1, 1};
      param.grid = {param.grid.x, 1, (n_batch + param.block.z - 1) / param.block.z};
    }
  };

} // namespace quda
