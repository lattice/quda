#pragma once

#include <tune_quda.h>
#include <device.h>
#include <reduce_helper.h>
#include <reduction_kernel.h>

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
    const LatticeField &field;

  protected:
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
    unsigned int maxBlockSize(const TuneParam &param) const { return device::max_block_size<block_size_y>(); }

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

    template <template <typename> class Transformer, template <typename> class Reducer = plus, typename Arg>
    void launch(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      if (field.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        std::string kernel_file(std::string("kernels/") + Transformer<Arg>::filename());
        if (field.Location() == QUDA_CUDA_FIELD_LOCATION) create_jitify_program(kernel_file);
        using namespace jitify::reflection;

        // we need this hackery to get the naked unbound template class parameters        
        auto Transformer_instance = reflect<Transformer<Arg>>();
        auto Transformer_naked = Transformer_instance.substr(0, Transformer_instance.find("<"));
        auto Reducer_instance = reflect<Reducer<Arg>>();
        auto Reducer_naked = Reducer_instance.substr(0, Reducer_instance.find("<"));

        jitify_error = program->kernel("quda::Reduction2D")
        .instantiate({reflect((int)tp.block.x), reflect((int)tp.block.y),
              Transformer_naked, Reducer_naked, reflect<Arg>()})
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
        arg.launch_error = jitify_error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
#else
        arg.launch_error = launch<device::max_block_size<block_size_y>(), Transformer, Reducer>(arg, tp, stream);
#endif
      } else {
	errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableReduction2D(const LatticeField &field) :
      field(field)
    {
      strcpy(aux, compile_type_str(field));
      strcat(aux, field.AuxString());
      strcat(aux, field.Location() == QUDA_CPU_FIELD_LOCATION ? ",CPU" : ",GPU");
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
