#pragma once

#include <tune_quda.h>
#include <lattice_field.h>
#include <device.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <kernel.h>

#ifdef JITIFY
#include <jitify_helper2.cuh>
#endif

namespace quda {

  class TunableKernel : public Tunable
  {

  protected:
    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    template <template <typename> class Functor, bool grid_stride = false, typename Arg>
    std::enable_if_t<device::use_kernel_arg<Arg>(), void>
      launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
#ifdef JITIFY
      launch_error = launch_jitify<Functor, grid_stride, Arg, false>(kernel.name, tp, stream, arg);
#else
      launch_error = qudaLaunchKernel(kernel.func, tp, stream, arg);
#endif
    }

    template <template <typename> class Functor, bool grid_stride = false, typename Arg>
    std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
      launch_device(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
#ifdef JITIFY
      launch_error = launch_jitify<Functor, grid_stride, Arg, false>(kernel.name, tp, stream, arg);
#else
      static_assert(sizeof(Arg) <= device::max_constant_size(), "Parameter struct is greater than max constant size");
      qudaMemcpyAsync(device::get_constant_buffer<Arg>(), &arg, sizeof(Arg), qudaMemcpyHostToDevice, stream);
      launch_error = qudaLaunchKernel(kernel.func, tp, stream, arg);
#endif
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

  };

  template <bool grid_stride>
  class TunableKernel1D_base : public TunableKernel
  {
  protected:
    QudaFieldLocation location;

    /**
       Return whether we are grid-size tuning.  Marked as virtual in
       case we need to override this, despite the grid_stride template
       (Dslash NVSHMEM does this).
    */
    virtual bool tuneGridDim() const { return grid_stride; }

    template <template <typename> class Functor, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Kernel1D), tp, stream, arg);
    }

    template <template <typename> class Functor, typename Arg>
    void launch_host(const TuneParam &, const qudaStream_t &, const Arg &arg)
    {
      Functor<Arg> f(const_cast<Arg &>(arg));
      for (int i = 0; i < (int)arg.threads.x; i++) {
        f(i);
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
      std::enable_if_t<!enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
      std::enable_if_t<enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
	launch_host<Functor, Arg>(tp, stream, arg);
      }
    }

  public:
    TunableKernel1D_base(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(vol, field.VolString());
      strcpy(aux, compile_type_str(field, location));
      strcat(aux, field.AuxString());
    }

    TunableKernel1D_base(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      location(location)
    {
      u64toa(vol, n_items);
      strcpy(aux, compile_type_str(location));
    }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

  class TunableKernel1D : public TunableKernel1D_base<false> {
  protected:
    using Tunable::aux;

    /**
       Since we are not grid-size tuning, we require any derivations
       to specify the minimum thread count.
     */
    virtual unsigned int minThreads() const = 0;

  public:
    TunableKernel1D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<false>(field, location) {}

    TunableKernel1D(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<false>(n_items, location) {}
  };

  class TunableGridStrideKernel1D : public TunableKernel1D_base<true> {
  protected:
    using Tunable::aux;

  public:
    TunableGridStrideKernel1D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<true>(field, location) {}

    TunableGridStrideKernel1D(size_t n_items, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<true>(n_items, location) {}
  };

  /**
     @brief This derived class is for algorithms that deploy a vector
     of computations across the y dimension of both the threads block
     and grid.  For example this could be parity in the y dimension
     and checkerboarded volume in x.
   */
  template <bool grid_stride = false>
  class TunableKernel2D_base : public TunableKernel1D_base<grid_stride>
  {
  protected:
    mutable unsigned int vector_length_y;
    mutable unsigned int step_y;
    bool tune_block_x;

    template <template <typename> class Functor, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Kernel2D), tp, stream, arg);
    }

    template <template <typename> class Functor, typename Arg>
    void launch_host(const TuneParam &, const qudaStream_t &, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      Functor<Arg> f(const_cast<Arg &>(arg));
      for (int i = 0; i < (int)arg.threads.x; i++) {
        for (int j = 0; j < (int)arg.threads.y; j++) {
          f(i, j);
        }
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
    std::enable_if_t<!enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (TunableKernel1D_base<grid_stride>::location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
    std::enable_if_t<enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (TunableKernel1D_base<grid_stride>::location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
        launch_host<Functor, Arg>(tp, stream, arg);
      }
    }

  public:
    TunableKernel2D_base(const LatticeField &field, unsigned int vector_length_y,
                         QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<grid_stride>(field, location),
      vector_length_y(vector_length_y),
      step_y(1),
      tune_block_x(true)
    { }

    TunableKernel2D_base(size_t n_items, unsigned int vector_length_y,
                         QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D_base<grid_stride>(n_items, location),
      vector_length_y(vector_length_y),
      step_y(1),
      tune_block_x(true)
    { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      param.block.y = block.y;
      param.grid.y = grid.y;
      bool ret = tune_block_x ? Tunable::advanceBlockDim(param) : false;

      if (ret) {
	return true;
      } else { // block.x (spacetime) was reset

        auto next = param;
        next.block.y += step_y;
        auto shared_bytes = std::max(this->sharedBytesPerThread() * next.block.x * next.block.y * next.block.z, this->sharedBytesPerBlock(next));

	// we can advance spin/block-color since this is valid
	if (param.block.y < vector_length_y && param.block.y < device::max_threads_per_block_dim(1) &&
	    param.block.x*(param.block.y+step_y)*param.block.z <= device::max_threads_per_block() &&
            shared_bytes <= this->maxSharedBytesPerBlock()) {
	  param.block.y += step_y;
	  param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
          param.shared_bytes = shared_bytes;
          return true;
	} else { // we have run off the end so let's reset
	  param.block.y = step_y;
	  param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;

          return false;
	}
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
      param.shared_bytes = std::max(this->sharedBytesPerThread() * param.block.x * param.block.y * param.block.z, this->sharedBytesPerBlock(param));
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
      param.shared_bytes = std::max(this->sharedBytesPerThread() * param.block.x * param.block.y * param.block.z, this->sharedBytesPerBlock(param));
    }

    void resizeVector(int y) const { vector_length_y = y; }
    void resizeStep(int y) const { step_y = y; }
  };

  class TunableKernel2D : public TunableKernel2D_base<false> {
  protected:
    using Tunable::aux;

    /**
       Since we are not grid-size tuning, we require any derivations
       to specify the minimum thread count.
     */
    virtual unsigned int minThreads() const = 0;

  public:
    TunableKernel2D(const LatticeField &field, unsigned int vector_length_y,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<false>(field, vector_length_y, location) {}

    TunableKernel2D(size_t n_items, unsigned int vector_length_y,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<false>(n_items, vector_length_y, location) {}
  };

  class TunableGridStrideKernel2D : public TunableKernel2D_base<true> {
  protected:
    using Tunable::aux;

  public:
    TunableGridStrideKernel2D(const LatticeField &field, unsigned int vector_length_y,
                              QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<true>(field, vector_length_y, location) {}

    TunableGridStrideKernel2D(size_t n_items, unsigned int vector_length_y,
                              QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<true>(n_items, vector_length_y, location) {}
  };

  /**
     @brief This derived class is for algorithms that deploy a vector
     of computations across the y and z dimensions of both the threads
     block and grid.  For example this could be parity in the y
     dimension, direction in the z dimension and checkerboarded volume
     in x.
   */
  template <bool grid_stride = false>
  class TunableKernel3D_base : public TunableKernel2D_base<grid_stride>
  {
  protected:
    using TunableKernel2D_base<grid_stride>::vector_length_y;
    mutable unsigned vector_length_z;
    mutable unsigned step_z;
    bool tune_block_y;

    template <template <typename> class Functor, typename Arg>
    void launch_device(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      const_cast<Arg &>(arg).threads.z = vector_length_z;
      TunableKernel::launch_device<Functor, grid_stride>(KERNEL(Kernel3D), tp, stream, arg);
    }

    template <template <typename> class Functor, typename Arg>
    void launch_host(const TuneParam &, const qudaStream_t &, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      const_cast<Arg &>(arg).threads.z = vector_length_z;
      Functor<Arg> f(const_cast<Arg &>(arg));
      for (int i = 0; i < (int)arg.threads.x; i++) {
        for (int j = 0; j < (int)arg.threads.y; j++) {
          for (int k = 0; k < (int)arg.threads.z; k++) {
            f(i, j, k);
          }
        }
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
  std::enable_if_t<!enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (TunableKernel2D_base<grid_stride>::location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
	errorQuda("CPU not supported yet");
      }
    }

    template <template <typename> class Functor, bool enable_host = false, typename Arg>
  std::enable_if_t<enable_host, void> launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (TunableKernel2D_base<grid_stride>::location == QUDA_CUDA_FIELD_LOCATION) {
        launch_device<Functor, Arg>(tp, stream, arg);
      } else {
        launch_host<Functor, Arg>(tp, stream, arg);
      }
    }

  public:
    TunableKernel3D_base(const LatticeField &field, unsigned int vector_length_y, unsigned int vector_length_z,
                         QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<grid_stride>(field, vector_length_y, location),
      vector_length_z(vector_length_z),
      step_z(1),
      tune_block_y(true) { }

    TunableKernel3D_base(size_t n_items, unsigned int vector_length_y, unsigned int vector_length_z,
                         QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D_base<grid_stride>(n_items, vector_length_y, location),
      vector_length_z(vector_length_z),
      step_z(1),
      tune_block_y(true) { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      param.block.z = block.z;
      param.grid.z = grid.z;
      bool ret = tune_block_y ? TunableKernel2D_base<grid_stride>::advanceBlockDim(param) :
      TunableKernel2D_base<grid_stride>::tune_block_x ? Tunable::advanceBlockDim(param) : false;

      if (ret) {
	// we advanced the block.x / block.y so we're done
	return true;
      } else { // block.x/block.y (spacetime) was reset

        auto next = param;
        next.block.z += step_z;
        auto shared_bytes = std::max(this->sharedBytesPerThread() * next.block.x * next.block.y * next.block.z, this->sharedBytesPerBlock(next));

        // we can advance spin/block-color since this is valid
	if (param.block.z < vector_length_z && param.block.z < device::max_threads_per_block_dim(2) &&
	    param.block.x*param.block.y*(param.block.z+step_z) <= device::max_threads_per_block() &&
            shared_bytes <= this->maxSharedBytesPerBlock()) {
	  param.block.z += step_z;
	  param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
          param.shared_bytes = shared_bytes;
	  return true;
	} else { // we have run off the end so let's reset
	  param.block.z = step_z;
	  param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
	  return false;
	}
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableKernel2D_base<grid_stride>::initTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
      param.shared_bytes = std::max(this->sharedBytesPerThread() * param.block.x * param.block.y * param.block.z, this->sharedBytesPerBlock(param));
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableKernel2D_base<grid_stride>::defaultTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
      param.shared_bytes = std::max(this->sharedBytesPerThread() * param.block.x * param.block.y * param.block.z, this->sharedBytesPerBlock(param));
    }

    void resizeVector(int y, int z) const { vector_length_z = z;  TunableKernel2D_base<grid_stride>::resizeVector(y); }
    void resizeStep(int y, int z) const { step_z = z;  TunableKernel2D_base<grid_stride>::resizeStep(y); }
  };

  class TunableKernel3D : public TunableKernel3D_base<false> {
  protected:
    using Tunable::aux;

    /**
       Since we are not grid-size tuning, we require any derivations
       to specify the minimum thread count.
     */
    virtual unsigned int minThreads() const = 0;

  public:
    TunableKernel3D(const LatticeField &field, unsigned int vector_length_y, unsigned int vector_length_z,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel3D_base<false>(field, vector_length_y, vector_length_z, location) {}

    TunableKernel3D(size_t n_items, unsigned int vector_length_y, unsigned int vector_length_z,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel3D_base<false>(n_items, vector_length_y, vector_length_z, location) {}
  };

  class TunableGridStrideKernel3D : public TunableKernel3D_base<true> {
  protected:
    using Tunable::aux;

  public:
    TunableGridStrideKernel3D(const LatticeField &field, unsigned int vector_length_y, unsigned int vector_length_z,
                              QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel3D_base<true>(field, vector_length_y, vector_length_z, location) {}

    TunableGridStrideKernel3D(size_t n_items, unsigned int vector_length_y, unsigned int vector_length_z,
                              QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel3D_base<true>(n_items, vector_length_y, vector_length_z, location) {}
  };

}
