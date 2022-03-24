#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <kernel_helper.h>
#include <kernel.h>

#ifdef JITIFY
#include <jitify_helper.h>
#endif

#include <task_graph.h>

namespace quda
{

  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] arg Host address of argument struct
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, const qudaStream_t &stream, const void *arg);

  class TunableKernel : public Tunable
  {
  protected:
    QudaFieldLocation location;

    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    qudaError_t launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
#ifdef JITIFY
      // note we do the copy to constant memory after the kernel has been compiled in launch_jitify (if needed)
      launch_error = launch_jitify<Functor, grid_stride, Arg>(kernel.name, tp, stream, arg);
#else
      // only use graph launch if we're not tuning
      if (arg.use_graph && use_task_graph() && !activeTuning()) {
        launch_error = task_graph::launch_kernel(kernel, tp, stream, &arg, sizeof(Arg), device::use_kernel_arg<Arg>(), device::get_constant_buffer<Arg>());
      } else {
        if constexpr (!device::use_kernel_arg<Arg>()) {
          static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
          qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
        }
        launch_error = qudaLaunchKernel(kernel.func, tp, stream, static_cast<const void *>(&arg));
      }
#endif
      return launch_error;
    }

  public:
    /**
       @brief Special kernel launcher used for raw CUDA kernels with no
       assumption made about shape of parallelism.  Kernels launched
       using this must take responsibility of bounds checking and
       assignment of threads.
     */
    template <template <typename> class Functor, typename Arg>
    void launch_cuda(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg) const
    {
      constexpr bool grid_stride = false;
      const_cast<TunableKernel *>(this)->launch_device<Functor, grid_stride>(KERNEL(raw_kernel), tp, stream, arg);
    }

    TunableKernel(QudaFieldLocation location) : location(location)
    {
      if (location == QUDA_INVALID_FIELD_LOCATION) errorQuda("Location invalid");
    }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

} // namespace quda
