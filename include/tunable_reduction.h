#pragma once

#include <tunable_kernel.h>
#include <lattice_field.h>
#include <reduction_kernel.h>
#include <reduction_kernel_host.h>

namespace quda
{

  /** Dummy comm reducer where no inter-process reduction is done */
  template <typename T> struct comm_reduce_null {
    void operator()(std::vector<T> &) { }
  };

  /** comm reducer for doing summation inter-process reduction */
  template <typename T> struct comm_reduce_sum {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v)
    {
      comm_allreduce_array(reinterpret_cast<double *>(v.data()), v.size() * sizeof(T) / sizeof(double));
    }
  };

  /** comm reducer for doing max inter-process reduction */
  template <typename T> struct comm_reduce_max {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v)
    {
      comm_allreduce_max_array(reinterpret_cast<double *>(v.data()), v.size() * sizeof(T) / sizeof(double));
    }
  };

  /** comm reducer for doing min inter-process reduction */
  template <typename T> struct comm_reduce_min {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v)
    {
      comm_allreduce_min_array(reinterpret_cast<double *>(v.data()), v.size() * sizeof(T) / sizeof(double));
    }
  };

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
    std::enable_if_t<block_size_x != device::warp_size(), qudaError_t> launch(FunctorArg &arg, const TuneParam &tp,
                                                                              const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Reduction2D), tp, stream, Arg(arg));
      } else {
        return launch<block_size_x - device::warp_size(), Functor>(arg, tp, stream);
      }
    }

    template <int block_size_x, template <typename> class Functor, typename FunctorArg>
    std::enable_if_t<block_size_x == device::warp_size(), qudaError_t> launch(FunctorArg &arg, const TuneParam &tp,
                                                                              const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Reduction2D), tp, stream, Arg(arg));
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
        return QUDA_ERROR;
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      arg.launch_error = launch<device::max_reduce_block_size<block_size_y>(), Functor>(arg, tp, stream);

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) CommReducer()(result);
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_device(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch_device<Functor, T, CommReducer>(result_, tp, stream, arg);
      result = result_[0];
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      using reduce_t = typename Functor<Arg>::reduce_t;
      reduce_t value = Reduction2D_host<Functor, Arg>(arg);

      int input_size = vec_length<reduce_t>::value;
      int output_size = result.size() * vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      // copy element by element to output vector
      for (int i = 0; i < output_size; i++) {
        reinterpret_cast<typename scalar<T>::type *>(result.data())[i]
          = reinterpret_cast<typename scalar<reduce_t>::type *>(&value)[i];
      }

      if (!activeTuning() && commGlobalReduction()) CommReducer()(result);
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_host(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch_host<Functor, T, CommReducer>(result_, tp, stream, arg);
      result = result_[0];
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>,
              bool enable_host = false, typename Arg>
    std::enable_if_t<!enable_host, void> launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream,
                                                Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, T, CommReducer>(result, tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>,
              bool enable_host = false, typename Arg>
    std::enable_if_t<enable_host, void> launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream,
                                               Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, T, CommReducer>(result, tp, stream, arg);
      } else {
        launch_host<Functor, T, CommReducer>(result, tp, stream, arg);
      }
    }

    /**
       @brief Overload providing a simple reference interface
       @param[out] result The reduction result is copied here
       @param[in] tp The launch parameters
       @param[in] stream The stream on which we the reduction is being done
       @param[in] arg Kernel argument struct
       @param[in] param Constant kernel meta data
     */
    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch<Functor, T, CommReducer>(result_, tp, stream, arg);
      result = result_[0];
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
     @brief This derived tunable class is for reduction kernels, and
     partners the Reduction2D kernel.  The x threads will
     typically correspond to the checkerboarded volume.  Each thread
     block is two dimensional, with the y dimension typically equal to
     two and is the parity dimension.
   */
  template <int block_size_y = 1> class TunableMultiReduction : public TunableReduction2D<block_size_y>
  {
    // for now we do not support anything other than block_size_y = 1
    static_assert(block_size_y == 1, "only block_size_y = 1 supported");
    using Tunable::apply;
    using TunableReduction2D<block_size_y>::location;
    using TunableReduction2D<block_size_y>::grid_stride;

  protected:
    const int n_batch;

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
    std::enable_if_t<block_size_x != device::warp_size(), qudaError_t> launch(FunctorArg &arg, const TuneParam &tp,
                                                                              const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(MultiReduction), tp, stream, Arg(arg));
      } else {
        return launch<block_size_x / 2, Functor>(arg, tp, stream);
      }
    }

    template <int block_size_x, template <typename> class Functor, typename FunctorArg>
    std::enable_if_t<block_size_x == device::warp_size(), qudaError_t> launch(FunctorArg &arg, const TuneParam &tp,
                                                                              const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) {
        using Arg = ReduceKernelArg<block_size_x, block_size_y, FunctorArg>;
        return TunableKernel::launch_device<Functor, grid_stride>(KERNEL(MultiReduction), tp, stream, Arg(arg));
      } else {
        errorQuda("Unexpected block size %d", tp.block.x);
        return QUDA_ERROR;
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      arg.launch_error = launch<device::max_multi_reduce_block_size<block_size_y>(), Functor>(arg, tp, stream);

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) CommReducer()(result);
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>, typename Arg>
    void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      using reduce_t = typename Functor<Arg>::reduce_t;

      int input_size = vec_length<reduce_t>::value;
      int output_size = vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      auto value = MultiReduction_host<Functor, Arg>(arg);
      for (int j = 0; j < (int)arg.threads.y; j++) {
        // copy element by element to output vector
        for (int i = 0; i < output_size; i++) {
          reinterpret_cast<typename scalar<T>::type *>(&result[j])[i]
            = reinterpret_cast<typename scalar<reduce_t>::type *>(&value[j])[i];
        }
      }

      if (!activeTuning() && commGlobalReduction()) CommReducer()(result);
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>,
              bool enable_host = false, typename Arg>
    std::enable_if_t<!enable_host, void> launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream,
                                                Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, T, CommReducer>(result, tp, stream, arg);
      } else {
        errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, typename T, typename CommReducer = comm_reduce_sum<T>,
              bool enable_host, typename Arg>
    std::enable_if_t<enable_host, void> launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream,
                                               Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, T, CommReducer>(result, tp, stream, arg);
      } else {
        launch_host<Functor, T, CommReducer>(result, tp, stream, arg);
      }
    }

  public:
    TunableMultiReduction(const LatticeField &field, int n_batch,
                          QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(field, location), n_batch(n_batch)
    {
    }

    TunableMultiReduction(size_t n_items, int n_batch, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(n_items, location), n_batch(n_batch)
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
      return rtn;
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = n_batch;
    }

    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = 1;
      param.grid.y = n_batch;
    }
  };

} // namespace quda
