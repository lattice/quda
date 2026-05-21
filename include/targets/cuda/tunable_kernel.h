#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <lattice_field.h>
#include <kernel_helper.h>
#include <kernel.h>
#include <kernel_ops_target.h>

#ifdef JITIFY
#include <jitify_helper.h>
#endif

namespace quda
{

  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] arg Host address of argument struct
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const void *arg);

  /**
     @brief Wrapper around cudaOccupancyMaxActiveBlocks
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
  */
  int qudaOccupancyMaxActiveBlocks(const kernel_t &kernel, const TuneParam &tp);

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    /**
       @brief Set the maximum number of blocks that can reside on an
       SM.  This is called when we are autotuning to allow us to work
       out how many different shared memory over allocations we should
       use to minimally cover all occupancy variations.
     */
    void setMaxActiveBlocks(const kernel_t &kernel, const TuneParam &tp) const
    {
      if (activeTuningWarmup() && tuneSharedBytes()) {
        auto tp2 = tp;
        setSharedBytes(tp2);
        // only compute max number blocks when we have no shared memory over subscription
        if (tp.shared_bytes == tp2.shared_bytes) max_active_blocks = qudaOccupancyMaxActiveBlocks(kernel, tp);
      }
    }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
    launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      checkSharedBytes<Functor>(tp, arg);
      const_cast<Arg &>(arg).block_size = tp.block.x * tp.block.y * tp.block.z;
      if constexpr (Arg::is_dslash) const_cast<Arg &>(arg).arg.block_size = arg.block_size;
#ifdef JITIFY
      launch_error = launch_jitify<Functor, grid_stride, Arg>(kernel.name, tp, stream, arg);
#else
      setMaxActiveBlocks(kernel, tp);
      launch_error = qudaLaunchKernel(kernel, tp, stream, static_cast<const void *>(&arg));
#endif
      return launch_error;
    }

    template <typename Arg, size_t arg_size = sizeof(Arg)> void check_arg_size(Arg &)
    {
      static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
    }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
    launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      checkSharedBytes<Functor>(tp, arg);
      const_cast<Arg &>(arg).block_size = tp.block.x * tp.block.y * tp.block.z;
      if constexpr (Arg::is_dslash) const_cast<Arg &>(arg).arg.block_size = arg.block_size;
#ifdef JITIFY
      // note we do the copy to constant memory after the kernel has been compiled in launch_jitify
      launch_error = launch_jitify<Functor, grid_stride, Arg>(kernel.name, tp, stream, arg);
#else
      check_arg_size(arg);
      qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
      setMaxActiveBlocks(kernel, tp);
      launch_error = qudaLaunchKernel(kernel, tp, stream, static_cast<const void *>(&arg));
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
      checkSharedBytes<Functor>(tp, arg);
      const_cast<Arg &>(arg).block_size = tp.block.x * tp.block.y * tp.block.z;
      if constexpr (Arg::is_dslash) const_cast<Arg &>(arg).arg.block_size = arg.block_size;
      constexpr bool grid_stride = false;
      const_cast<TunableKernel *>(this)->launch_device<Functor, grid_stride>(KERNEL(raw_kernel), tp, stream, arg);
    }

    TunableKernel(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString().c_str());
      strcpy(aux, compile_type_str(field, location));
      if (tuneSharedCarveOut()) strcat(aux, getSharedCarveOutStr().c_str());
      if (this->location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef QUDA_LARGE_KERNEL_ARG
        strcat(aux, "large_kernel_arg,");
#else
        strcat(aux, "kernel_arg_threshold=");
        i32toa(aux + strlen(aux), device::max_kernel_arg_size());
        strcat(aux, ",");
#endif
      }
      if (this->location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux, field.AuxString().c_str());
    }

    TunableKernel(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) : location(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
      if (tuneSharedCarveOut()) strcat(aux, getSharedCarveOutStr().c_str());
      if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef QUDA_LARGE_KERNEL_ARG
        strcat(aux, "large_kernel_arg,");
#else
        strcat(aux, "kernel_arg_threshold=");
        i32toa(aux + strlen(aux), device::max_kernel_arg_size());
        strcat(aux, ",");
#endif
      }
      if (this->location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }

    virtual bool advanceTuneParam(TuneParam &param) const override
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const override { return TuneKey(vol, typeid(*this).name(), aux); }
  };

} // namespace quda
