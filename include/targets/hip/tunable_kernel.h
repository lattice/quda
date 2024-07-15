#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <lattice_field.h>
#include <kernel_helper.h>
#include <kernel.h>
#include <kernel_ops_target.h>
#include <quda_hip_api.h>

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

  /**
     @brief This helper function indicates if the present
     compilation unit has explicit constant memory usage enabled.
  */
  static constexpr bool use_constant_memory()
  {
#ifdef QUDA_USE_CONSTANT_MEMORY
    return true;
#else
    return false;
#endif
  }

  template <class Arg, size_t size> constexpr bool check_arg()
  {
    if constexpr (device::use_kernel_arg<Arg>()) {
      // using kernel arguments
      static_assert(size <= device::max_kernel_arg_size(), "Paramter struct is greater than max kernel size");
      return size <= device::max_kernel_arg_size();
    } else if constexpr (!device::use_kernel_arg<Arg>() && use_constant_memory()) {
      // not using kernel arguments and constant memory enabled
      static_assert(size <= device::max_constant_size(), "Parameter struct is greater than max constant size");
      return size <= device::max_constant_size();
    } else {
      // not using kernel arguments but constant memory not enabled
      static_assert(size < 0, "Invalid parameter struct");
      return false;
    }
  }

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    qudaError_t launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      checkSharedBytes(tp);
      if constexpr (check_arg<Arg, sizeof(Arg)>()) {
        if constexpr (!device::use_kernel_arg<Arg>() && use_constant_memory()) {
          qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
        }
        launch_error = qudaLaunchKernel(kernel.func, tp, stream, static_cast<const void *>(&arg));
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
      checkSharedBytes(tp);
      constexpr bool grid_stride = false;
      const_cast<TunableKernel *>(this)->launch_device<Functor, grid_stride>(KERNEL(raw_kernel), tp, stream, arg);
    }

    TunableKernel(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString().c_str());
      strcpy(aux, compile_type_str(field, location));
      if (this->location == QUDA_CUDA_FIELD_LOCATION && use_constant_memory()) strcat(aux, "cmem,");
      if (this->location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux, field.AuxString().c_str());
    }

    TunableKernel(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) : location(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
      if (location == QUDA_CUDA_FIELD_LOCATION && use_constant_memory()) strcat(aux, "cmem,");
      if (this->location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }

    virtual bool advanceTuneParam(TuneParam &param) const override
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const override { return TuneKey(vol, typeid(*this).name(), aux); }
  };

} // namespace quda
