#pragma once

#include <constant_kernel_arg.h>
#include <tune_quda.h>
#include <target_device.h>
#include <lattice_field.h>
#include <kernel_helper.h>
#include <kernel.h>
#include <kernel_ops_target.h>
#include <quda_omptarget_api.h>

namespace quda
{

  template <typename Arg>
  concept announce_threads_sync = requires { Arg::requires_threads_sync; };

  template <typename Arg> inline bool acceptThreads(const TuneParam &tp, const Arg &arg)
  {
    if (tp.block.x * tp.block.y * tp.block.z > device::max_block_size()) return false;

    // Preserve the synchronization constraints of the OpenMP reduction and block kernels.
    if constexpr (announce_threads_sync<Arg>) {
      if (((Arg::requires_threads_sync & ThreadsSyncX) && arg.threads.x % tp.block.x)
          || ((Arg::requires_threads_sync & ThreadsSyncY) && arg.threads.y % tp.block.y)
          || ((Arg::requires_threads_sync & ThreadsSyncZ) && arg.threads.z % tp.block.z))
        return false;
    }
    return true;
  }

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    // The OpenMP shared arena has a fixed size and cannot throttle occupancy.
    bool advanceSharedBytes(TuneParam &) const override { return false; }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    qudaError_t launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      checkSharedBytes<Functor>(tp, arg);
      const_cast<Arg &>(arg).block_size = tp.block.x * tp.block.y * tp.block.z;
      const_cast<Arg &>(arg).x_batch_stride = grid_stride ? tp.grid.x * tp.block.x : 0u;
      if constexpr (Arg::is_dslash) const_cast<Arg &>(arg).arg.block_size = arg.block_size;
      launch_error = QUDA_SUCCESS;
      if (acceptThreads(tp, arg) && 0 == target::omptarget::qudaSetupLaunchParameter(tp)) {
        if constexpr (device::use_kernel_arg<Arg>()) {
          reinterpret_cast<void (*)(Arg)>(const_cast<void *>(kernel.func))(arg);
        } else {
          static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
          Arg *argp = reinterpret_cast<Arg *>(device::get_constant_buffer<Arg>());
          memcpy(argp, &arg, sizeof(Arg));
          reinterpret_cast<void (*)(Arg *)>(const_cast<void *>(kernel.func))(argp);
        }
        launch_error = qudaGetLastError();
      } else {
        launch_error = QUDA_ERROR;
      }
      target::omptarget::set_runtime_error(launch_error, __func__, kernel.name.c_str(), __FILE__,
                                           __STRINGIFY__(__LINE__), activeTuning());
      return launch_error;
    }

  public:
    TunableKernel(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString().c_str());
      strcpy(aux, compile_type_str(field, location));
      if (this->location == QUDA_CUDA_FIELD_LOCATION) {
        strcat(aux, "kernel_arg_threshold=");
        i32toa(aux + strlen(aux), device::max_kernel_arg_size());
        strcat(aux, ",");
      }
      if (this->location == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux, field.AuxString().c_str());
    }

    TunableKernel(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) : location(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
      if (this->location == QUDA_CUDA_FIELD_LOCATION) {
        strcat(aux, "kernel_arg_threshold=");
        i32toa(aux + strlen(aux), device::max_kernel_arg_size());
        strcat(aux, ",");
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
