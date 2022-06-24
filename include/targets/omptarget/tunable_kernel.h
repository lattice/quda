#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <kernel_helper.h>
#include <kernel.h>

namespace quda {

  namespace target {
    namespace omptarget {
      // defined in ../../../lib/targets/omptarget/quda_api.cpp:/qudaSetupLaunchParameter
      int qudaSetupLaunchParameter(const TuneParam &);
      void set_runtime_error(int error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error = false);
    }
  }

  template <typename Arg>
  inline bool acceptThreads(const TuneParam &tp, const Arg &arg)
  {
    bool r = tp.block.x*tp.block.y*tp.block.z<=device::max_block_size() &&
       (arg.threads.x<tp.block.x ||    // trivial cases where arg.threads.x == 1 or few
        arg.threads.y<tp.block.y ||
        arg.threads.z<tp.block.z ||
        (arg.threads.x%tp.block.x==0 &&
         arg.threads.y%tp.block.y==0 &&
         arg.threads.z%tp.block.z==0));
    if(!r && getVerbosity() >= QUDA_VERBOSE)
      ompwip("WARNING: rejecting threads setup arg %d %d %d tp (%d %d %d)x(%d %d %d)",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
    return r;
  }

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    qudaError_t launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (acceptThreads(tp, arg) && 0==target::omptarget::qudaSetupLaunchParameter(tp)) {
        if constexpr (device::use_kernel_arg<Arg>()) {
          reinterpret_cast<void(*)(Arg)>(const_cast<void*>(kernel.func))(arg);
        } else {
          static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
          qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
          reinterpret_cast<void(*)(void)>(const_cast<void*>(kernel.func))();
        }
        launch_error = QUDA_SUCCESS;
      } else {
        target::omptarget::set_runtime_error(QUDA_ERROR, __func__, __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
        launch_error = QUDA_ERROR;
      }
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
      const_cast<TunableKernel*>(this)->launch_device<Functor, grid_stride>(KERNEL(raw_kernel), tp, stream, arg);
    }

    TunableKernel(QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) : location(location) { } 

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

}
