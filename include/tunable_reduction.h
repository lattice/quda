#pragma once

#include <tune_quda.h>
#include <lattice_field.h>
#include <device.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <reduce_helper.h>
#include <reduction_kernel.h>
#include <register_traits.h>

#ifdef JITIFY
#include <jitify_helper2.cuh>
#endif

namespace quda {

  /**
     @brief This derived tunable class is for reduction kernels, and
     partners the Reduction2D kernel.  The x threads will
     typically correspond to the checkerboarded volume.  Each thread
     block is two dimensional, with the y dimension typically equal to
     two and is the parity dimension.
   */
  template <int block_size_y = 2>
  class TunableReduction2D : public Tunable
  {
  protected:
    QudaFieldLocation location;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    /**
       Reduction kernels require grid-size tuning, so enable this, and
       we mark as final to prevent a derived class from accidentally
       switching it off.
    */
    bool tuneGridDim() const final { return true; }

    virtual unsigned int minGridSize() const { return Tunable::minGridSize(); }
    virtual int gridStep() const { return minGridSize(); }

    /**
       The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
     */
    virtual unsigned int maxBlockSize(const TuneParam &) const { return device::max_reduce_block_size<block_size_y>(); }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
      launch_device(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      return qudaLaunchKernel(Reduction2D<block_size_x, block_size_y, Transformer, Arg>, tp, stream, arg);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
      launch_device(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
      qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
      return qudaLaunchKernel(Reduction2D<block_size_x, block_size_y, Transformer, Arg>, tp, stream, arg);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<block_size_x != device::warp_size(), qudaError_t>
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) return launch_device<block_size_x, Transformer, Arg>(arg, tp, stream);
      else return launch<block_size_x - device::warp_size(), Transformer>(arg, tp, stream);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<block_size_x == device::warp_size(), qudaError_t>
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) return launch_device<block_size_x, Transformer, Arg>(arg, tp, stream);
      else errorQuda("Unexpected block size %d\n", tp.block.x);
      return QUDA_ERROR;
    }

    template <template <typename> class Transformer, typename Arg, typename T>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
#ifdef JITIFY
      arg.launch_error = launch_jitify<Transformer, true, Arg, true>("Reduction2D", tp, stream, arg);
#else
      arg.launch_error = launch<device::max_reduce_block_size<block_size_y>(), Transformer>(arg, tp, stream);
#endif

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) {
          // FIXME - this will break when we have non-summation reductions (MG fixed point will break and so will force monitoring)
          // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
          comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
        }
      }
    }

    template <template <typename> class Transformer, typename Arg, typename T>
    void launch_device(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch_device<Transformer>(result_, tp, stream, arg);
      result = result_[0];
    }

    template <template <typename> class Transformer, typename Arg, typename T>
     void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      Transformer<Arg> t(arg);

      reduce_t value = arg.init();

      for (int j = 0; j < (int)arg.threads.y; j++) {
        for (int i = 0; i < (int)arg.threads.x; i++) {
          value = t(value, i, j);
        }
      }

      int input_size = vec_length<reduce_t>::value;
      int output_size = result.size() * vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      // copy element by element to output vector
      for (int i = 0; i < output_size; i++) {
        reinterpret_cast<typename scalar<T>::type*>(result.data())[i] =
          reinterpret_cast<typename scalar<reduce_t>::type*>(&value)[i];
      }

      if (!activeTuning() && commGlobalReduction()) {
        // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
        comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
      }
    }

    template <template <typename> class Transformer, typename Arg, typename T>
    void launch_host(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch_host<Transformer>(result_, tp, stream, arg);
      result = result_[0];
    }

    template <template <typename> class Transformer, bool enable_host = false, typename T, typename Arg>
    std::enable_if_t<!enable_host, void>
      launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer>(result, tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Transformer, bool enable_host = false, typename T, typename Arg>
    std::enable_if_t<enable_host, void>
      launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer>(result, tp, stream, arg);
      } else {
        launch_host<Transformer>(result, tp, stream, arg);
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
    template <template <typename> class Transformer, typename T, typename Arg>
    void launch(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      std::vector<T> result_(1);
      launch<Transformer>(result_, tp, stream, arg);
      result = result_[0];
    }

  public:
    TunableReduction2D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
      strcat(aux, field.AuxString());
#ifdef QUDA_FAST_COMPILE_REDUCE
      strcat(aux, ",fast_compile");
#endif
    }

    TunableReduction2D(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
    }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    virtual bool advanceBlockDim(TuneParam &param) const {
      param.block.y = block_size_y;
      bool rtn = Tunable::advanceBlockDim(param);
      return rtn;
    }

    virtual void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.block.y = block_size_y;
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.block.y = block_size_y;
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

  /**
     @brief This derived tunable class is for reduction kernels, and
     partners the Reduction2D kernel.  The x threads will
     typically correspond to the checkerboarded volume.  Each thread
     block is two dimensional, with the y dimension typically equal to
     two and is the parity dimension.
   */
  template <int block_size_y = 1>
  class TunableMultiReduction : public TunableReduction2D<block_size_y>
  {
    // for now we do not support anything other than block_size_y = 1
    static_assert(block_size_y == 1, "only block_size_y = 1 supported");
    using Tunable::apply;
    using TunableReduction2D<block_size_y>::location;

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
    unsigned int maxBlockSize(const TuneParam &) const { return device::max_multi_reduce_block_size(); }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
      launch_device(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      return qudaLaunchKernel(MultiReduction<block_size_x, block_size_y, Transformer, Arg>, tp, stream, arg);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
      launch_device(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
      qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
      return qudaLaunchKernel(MultiReduction<block_size_x, block_size_y, Transformer, Arg>, tp, stream, arg);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<block_size_x != device::warp_size(), qudaError_t>
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) return launch_device<block_size_x, Transformer, Arg>(arg, tp, stream);
      else return launch<block_size_x - device::warp_size(), Transformer>(arg, tp, stream);
    }

    template <int block_size_x, template <typename> class Transformer, typename Arg>
    std::enable_if_t<block_size_x == device::warp_size(), qudaError_t>
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x) return launch_device<block_size_x, Transformer, Arg>(arg, tp, stream);
      else errorQuda("Unexpected block size %d\n", tp.block.x);
      return QUDA_ERROR;
    }

    template <template <typename> class Transformer, typename Arg, typename T>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
#ifdef JITIFY
      arg.launch_error = launch_jitify<Transformer, true, Arg, true>("MultiReduction", tp, stream, arg);
#else
      arg.launch_error = launch<device::max_multi_reduce_block_size(), Transformer>(arg, tp, stream);
#endif

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
#if 0 // WAR for now since otherwise we do a double reduction in
      // multi_reduce_quda and the accessors which do not necsssarily
      // use summation and do their global reduction elsewhere
        if (!activeTuning() && commGlobalReduction()) {
          // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
          comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
        }
#endif
      }
    }

    template <template <typename> class Transformer, typename Arg, typename T>
    void launch_host(std::vector<T> &result, const TuneParam &, const qudaStream_t &, Arg &arg)
    {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      Transformer<Arg> t(arg);

      int input_size = vec_length<reduce_t>::value;
      int output_size = vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      for (int j = 0; j < (int)arg.threads.y; j++) {
        reduce_t value = arg.init();

        for (int k = 0; k < (int)arg.threads.z; k++) {
          for (int i = 0; i < (int)arg.threads.x; i++) {
            value = t(value, i, j, k);
          }
        }

        // copy element by element to output vector
        for (int i = 0; i < output_size; i++) {
          reinterpret_cast<typename scalar<T>::type*>(&result[j])[i] =
            reinterpret_cast<typename scalar<reduce_t>::type*>(&value)[i];
        }
      }


#if 0 // WAR for now to avoid double and/or wrong reduction type
      if (!activeTuning() && commGlobalReduction()) {
        // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
        comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
      }
#endif
    }

    template <template <typename> class Transformer, bool enable_host = false, typename T, typename Arg>
    std::enable_if_t<!enable_host, void>
    launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer>(result, tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Transformer, bool enable_host, typename T, typename Arg>
    std::enable_if_t<enable_host, void>
    launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer>(result, tp, stream, arg);
      } else {
	launch_host<Transformer>(result, tp, stream, arg);
      }
    }

  public:
    TunableMultiReduction(const LatticeField &field, int n_batch, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(field, location),
      n_batch(n_batch)
    { }

    TunableMultiReduction(size_t n_items, int n_batch, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableReduction2D<block_size_y>(n_items, location),
      n_batch(n_batch)
    { }

    bool advanceBlockDim(TuneParam &param) const { return Tunable::advanceBlockDim(param); }

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

}
