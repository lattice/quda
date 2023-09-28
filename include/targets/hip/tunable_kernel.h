#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <lattice_field.h>
#include <kernel_helper.h>
#include <kernel.h>
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
  static bool use_constant_memory()
  {
#ifdef QUDA_USE_CONSTANT_MEMORY
    return true;
#else
    return false;
#endif
  }

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
    launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      launch_error = qudaLaunchKernel(kernel.func, tp, stream, static_cast<const void *>(&arg));
      return launch_error;
    }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
    launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
      qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
      launch_error = qudaLaunchKernel(kernel.func, tp, stream, static_cast<const void *>(&arg));
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
