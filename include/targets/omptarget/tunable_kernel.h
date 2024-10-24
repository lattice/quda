#pragma once

#include <constant_kernel_arg.h>
#include <tune_quda.h>
#include <target_device.h>
#include <lattice_field.h>
#include <kernel_helper.h>
#include <kernel.h>
#include <kernel_ops_target.h>
#include <quda_omptarget_api.h>

namespace quda {

  template <typename Arg>
  concept announce_threads_sync = requires {
    Arg::requires_threads_sync;
  };

  template <typename Arg>
  inline bool acceptThreads(const TuneParam &tp, const Arg &arg)
  {
    bool fit = tp.block.x*tp.block.y*tp.block.z<=device::max_block_size();
    bool xrem = arg.threads.x % tp.block.x > 0;
    bool yrem = arg.threads.y % tp.block.y > 0;
    bool zrem = arg.threads.z % tp.block.z > 0;
    if (!fit) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        warningQuda("rejecting threads setup with a large block size with arg %d %d %d tp grid %d %d %d block %d %d %d\n",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
      return false;
    }
    if (xrem || yrem || zrem) {
      if constexpr(announce_threads_sync<Arg>){
        if (((Arg::requires_threads_sync & ThreadsSyncX) && xrem) ||
            ((Arg::requires_threads_sync & ThreadsSyncY) && yrem) ||
            ((Arg::requires_threads_sync & ThreadsSyncZ) && zrem)) {
          if(getVerbosity() >= QUDA_DEBUG_VERBOSE)
            warningQuda("rejecting threads setup with a non-divisible block size with arg %d %d %d tp grid %d %d %d block %d %d %d\n",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
          return false;
        } else {
          if(getVerbosity() >= QUDA_DEBUG_VERBOSE)
            warningQuda("accepting threads setup with a non-divisible block size with arg %d %d %d tp grid %d %d %d block %d %d %d\n",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
          return true;
        }
      } else {  // ! announce_threads_sync<Arg>
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
          bool cont = true;
          std::string reply;
            ompwip("threads setup with a non-divisible block size with arg %d %d %d tp grid %d %d %d block %d %d %d, yes to stop?",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
          std::getline(std::cin, reply);
          if (reply[0] == 'y' || reply[0] == 'Y') {
            cont = false;
          }
          return cont;
        } else {
          if (getVerbosity() >= QUDA_VERBOSE)
            ompwip("accepting threads setup with a non-divisible block size with arg %d %d %d tp grid %d %d %d block %d %d %d",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
          return true;
        }
      }
    } else {
      if(getVerbosity() >= QUDA_DEBUG_VERBOSE)
        warningQuda("threads setup with arg %d %d %d tp grid %d %d %d block %d %d %d\n",arg.threads.x,arg.threads.y,arg.threads.z,tp.grid.x,tp.grid.y,tp.grid.z,tp.block.x,tp.block.y,tp.block.z);
    }
    return true;
  }

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
    qudaError_t launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      checkSharedBytes(tp);
      launch_error = QUDA_SUCCESS;
      if (acceptThreads(tp, arg) && 0==target::omptarget::qudaSetupLaunchParameter(tp)) {
        if constexpr (device::use_kernel_arg<Arg>()) {
          reinterpret_cast<void(*)(Arg)>(const_cast<void*>(kernel.func))(arg);
        } else {
          static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
          Arg *argp = reinterpret_cast<Arg*>(device::get_constant_buffer<Arg>());
          memcpy(argp, &arg, sizeof(Arg));
          reinterpret_cast<void(*)(Arg*)>(const_cast<void*>(kernel.func))(argp);
        }
        launch_error = qudaGetLastError();
      } else {
        launch_error = QUDA_ERROR;
      }
      target::omptarget::set_runtime_error(launch_error, __func__, __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
      return launch_error;
    }

  public:
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

}
