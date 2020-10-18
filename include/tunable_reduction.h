#pragma once

#include <tune_quda.h>
#include <device.h>
#include <reduce_helper.h>
#include <reduction_kernel.h>
#include <register_traits.h>

#ifdef JITIFY
#include <jitify_helper.cuh>
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
    const LatticeField &field;
    QudaFieldLocation location;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    /**
       Reduction kernels require grid-size tuning, so enable this, and
       we mark as final to prevent a derived class from accidentally
       switching it off.
    */
    bool tuneGridDim() const final { return true; }

    unsigned int minGridSize() const { return maxGridSize() / 8; }
    int gridStep() const { return minGridSize(); }

    /**
       The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension.  Since
       parity is local to the thread block in the y dimension, half
       the max threads in the x dimension.
     */
    unsigned int maxBlockSize(const TuneParam &param) const { return device::max_reduce_block_size<block_size_y>(); }

    template <int block_size_x, template <typename> class Transformer, template <typename> class Reducer, typename Arg>
    typename std::enable_if<block_size_x != device::warp_size(), qudaError_t>::type
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size_x)
        return qudaLaunchKernel(Reduction2D<block_size_x, block_size_y, Transformer, Reducer, Arg>, tp, stream, arg);
      else
        return launch<block_size_x - device::warp_size(), Transformer, Reducer>(arg, tp, stream);
    }

    template <int block_size_x, template <typename> class Transformer, template <typename> class Reducer, typename Arg>
    typename std::enable_if<block_size_x == device::warp_size(), qudaError_t>::type
      launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      return qudaLaunchKernel(Reduction2D<block_size_x, block_size_y, Transformer, Reducer, Arg>, tp, stream, arg);
    }

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename Arg, typename T>
    void launch_device(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                       const std::vector<constant_param_t> &param = dummy_param)
    {
      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

#ifdef JITIFY
      std::string kernel_file(std::string("kernels/") + Transformer<Arg>::filename());
      create_jitify_program(kernel_file);
      using namespace jitify::reflection;

      // we need this hackery to get the naked unbound template class parameters
      auto Transformer_instance = reflect<Transformer<Arg>>();
      auto Transformer_naked = Transformer_instance.substr(0, Transformer_instance.find("<"));
      auto Reducer_instance = reflect<Reducer<Arg>>();
      auto Reducer_naked = Reducer_instance.substr(0, Reducer_instance.find("<"));

      auto instance = program->kernel("quda::Reduction2D")
      .instantiate({reflect((int)tp.block.x), reflect((int)tp.block.y),
            Transformer_naked, Reducer_naked, reflect<Arg>()});

      for (unsigned int i=0; i < param.size(); i++) {
        auto device_ptr = instance.get_constant_ptr(param[i].device_name);
        qudaMemcpyAsync((void*)device_ptr, param[i].host, param[i].bytes, cudaMemcpyHostToDevice, stream);
      }

      jitify_error = instance.configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
      arg.launch_error = jitify_error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
#else
      for (unsigned int i = 0; i < param.size(); i++)
        qudaMemcpyAsync(param[i].device_ptr, param[i].host, param[i].bytes, cudaMemcpyHostToDevice, stream);
      arg.launch_error = launch<device::max_reduce_block_size<block_size_y>(), Transformer, Reducer>(arg, tp, stream);
#endif

      if (!commAsyncReduction()) {
        arg.complete(result, stream);
        if (!activeTuning() && commGlobalReduction()) {
          // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
          comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
        }
      }
    }

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename Arg, typename T>
    void launch_device(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                       const std::vector<constant_param_t> &param = dummy_param)
    {
      std::vector<T> result_(1);
      launch_device<Transformer, Reducer>(result_, tp, stream, arg, param);
      result = result_[0];
    }

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename Arg, typename T>
    void launch_host(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                     const std::vector<constant_param_t> &param = dummy_param)
    {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      Transformer<Arg> t(arg);
      Reducer<reduce_t> r;

      reduce_t reduced_value = arg.init();

      for (int j = 0; j < (int)arg.threads.y; j++) {
        for (int i = 0; i < (int)arg.threads.x; i++) {
          auto value = t(i, j);
          reduced_value = r(value, reduced_value);
        }
      }

      int input_size = vec_length<reduce_t>::value;
      int output_size = result.size() * vec_length<T>::value;
      if (output_size != input_size) errorQuda("Input %d and output %d length do not match", input_size, output_size);

      // copy element by element to output vector
      for (int i = 0; i < output_size; i++) {
        reinterpret_cast<typename scalar<T>::type*>(result.data())[i] =
          reinterpret_cast<typename scalar<reduce_t>::type*>(&reduced_value)[i];
      }

      if (!activeTuning() && commGlobalReduction()) {
        // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
        comm_allreduce_array((double*)result.data(), result.size() * sizeof(T) / sizeof(double));
      }
    }

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename Arg, typename T>
    void launch_host(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                     const std::vector<constant_param_t> &param = dummy_param)
    {
      std::vector<T> result_(1);
      launch_host<Transformer, Reducer>(result_, tp, stream, arg, param);
      result = result_[0];
    }

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename T, typename Arg>
    void launch(std::vector<T> &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                const std::vector<constant_param_t> &param = dummy_param)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Transformer, Reducer>(result, tp, stream, arg, param);
      } else {
	errorQuda("CPU not supported yet");
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
    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename T, typename Arg>
    void launch(T &result, const TuneParam &tp, const qudaStream_t &stream, Arg &arg,
                const std::vector<constant_param_t> &param = dummy_param)
    {
      std::vector<T> result_(1);
      launch<Transformer, Reducer>(result_, tp, stream, arg, param);
      result = result_[0];
    }

  public:
    TunableReduction2D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      field(field),
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(aux, compile_type_str(field));
      strcat(aux, field.AuxString());
      strcat(aux, location == QUDA_CPU_FIELD_LOCATION ? ",CPU" : ",GPU");
    }

    bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    bool advanceBlockDim(TuneParam &param) const {
      bool rtn = Tunable::advanceBlockDim(param);
      param.block.y = block_size_y;
      return rtn;
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.block.y = block_size_y;
    }

    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.block.y = block_size_y;
    }

    TuneKey tuneKey() const { return TuneKey(field.VolString(), typeid(*this).name(), aux); }
  };

}
