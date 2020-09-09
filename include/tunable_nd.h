#pragma once

#include <tune_quda.h>
#include <lattice_field.h>
#include <device.h>
#include <kernel.h>

#ifdef JITIFY
#include <jitify_helper.cuh>
#endif

namespace quda {

  class TunableKernel1D : public Tunable
  {
  protected:
    const LatticeField &field;
    QudaFieldLocation location;

    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    /**
       Kernel1D (and its derivations Kernel2D and Kernel3D) do not
       use grid-size tuning, so disable this, and we mark as final to
       prevent a derived class from accidentally switching it on.
    */
    bool tuneGridDim() const final { return false; }

    /**
       Since we are not grid-size tuning, we require any derivations
       to specify the minimum thread count.
     */
    unsigned int minThreads() const = 0;

    template <template <typename> class Functor, typename Arg> void launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        std::string kernel_file(std::string("kernels/") + Functor<Arg>::filename());
        create_jitify_program(kernel_file);
        using namespace jitify::reflection;

        // we need this hackery to get the naked unbound template class parameters        
        auto Functor_instance = reflect<Functor<Arg>>();
        auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));

        jitify_error = program->kernel("quda::Kernel1D")
        .instantiate({Functor_naked, reflect<Arg>()})
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        qudaLaunchKernel(Kernel1D<Functor, Arg>, tp, stream, arg);
#endif
      } else {
	errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableKernel1D(const LatticeField &field, QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      field(field),
      location(location != QUDA_INVALID_FIELD_LOCATION ? location : field.Location())
    {
      strcpy(aux, compile_type_str(field));
      strcat(aux, field.AuxString());
      strcat(aux, field.Location() == QUDA_CPU_FIELD_LOCATION ? ",CPU" : ",GPU");
    }

    TuneKey tuneKey() const { return TuneKey(field.VolString(), typeid(*this).name(), aux); }
  };

  /**
     @brief This derived class is for algorithms that deploy a vector
     of computations across the y dimension of both the threads block
     and grid.  For example this could be parity in the y dimension
     and checkerboarded volume in x.
   */
  class TunableKernel2D : public TunableKernel1D
  {
  protected:
    mutable unsigned int vector_length_y;
    mutable unsigned int step_y;
    bool tune_block_x;

    template <template <typename> class Functor, typename Arg> void launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        std::string kernel_file(std::string("kernels/") + Functor<Arg>::filename());
        create_jitify_program(kernel_file);
        using namespace jitify::reflection;

        // we need this hackery to get the naked unbound template class parameters        
        auto Functor_instance = reflect<Functor<Arg>>();
        auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));

        jitify_error = program->kernel("quda::Kernel2D")
        .instantiate({Functor_naked, reflect<Arg>()})
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        qudaLaunchKernel(Kernel2D<Functor, Arg>, tp, stream, arg);
#endif
      } else {
	errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableKernel2D(const LatticeField &field, unsigned int vector_length_y,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel1D(field, location),
      vector_length_y(vector_length_y),
      step_y(1),
      tune_block_x(true)
    { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = tune_block_x ? Tunable::advanceBlockDim(param) : false;
      param.block.y = block.y;
      param.grid.y = grid.y;

      if (ret) {
	return true;
      } else { // block.x (spacetime) was reset

	// we can advance spin/block-color since this is valid
	if (param.block.y < vector_length_y && param.block.y < (unsigned int)deviceProp.maxThreadsDim[1] &&
	    param.block.x*(param.block.y+step_y)*param.block.z <= (unsigned int)deviceProp.maxThreadsPerBlock) {
	  param.block.y += step_y;
	  param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
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
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    void resizeVector(int y) const { vector_length_y = y; }
    void resizeStep(int y) const { step_y = y; }
  };

  /**
     @brief This derived class is for algorithms that deploy a vector
     of computations across the y and z dimensions of both the threads
     block and grid.  For example this could be parity in the y
     dimension, direction in the z dimension and checkerboarded volume
     in x.
   */
  class TunableKernel3D : public TunableKernel2D
  {
    mutable unsigned vector_length_z;
    mutable unsigned step_z;
    bool tune_block_y;

  protected:
    template <template <typename> class Functor, typename Arg> void launch(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
    {
      const_cast<Arg &>(arg).threads.y = vector_length_y;
      const_cast<Arg &>(arg).threads.z = vector_length_z;
      if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        std::string kernel_file(std::string("kernels/") + Functor<Arg>::filename());
        create_jitify_program(kernel_file);
        using namespace jitify::reflection;

        // we need this hackery to get the naked unbound template class parameters        
        auto Functor_instance = reflect<Functor<Arg>>();
        auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));

        jitify_error = program->kernel("quda::Kernel3D")
        .instantiate({Functor_naked, reflect<Arg>()})
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        qudaLaunchKernel(Kernel3D<Functor, Arg>, tp, stream, arg);
#endif
      } else {
	errorQuda("CPU not supported yet");
      }
    }

  public:
    TunableKernel3D(const LatticeField &field, unsigned int vector_length_y, unsigned int vector_length_z,
                    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) :
      TunableKernel2D(field, vector_length_y, location),
      vector_length_z(vector_length_z),
      step_z(1),
      tune_block_y(true) { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = tune_block_y ? TunableKernel2D::advanceBlockDim(param) : TunableKernel2D::tune_block_x ? Tunable::advanceBlockDim(param) : false;
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (ret) {
	// we advanced the block.x / block.y so we're done
	return true;
      } else { // block.x/block.y (spacetime) was reset

	// we can advance spin/block-color since this is valid
	if (param.block.z < vector_length_z && param.block.z < (unsigned int)deviceProp.maxThreadsDim[2] &&
	    param.block.x*param.block.y*(param.block.z+step_z) <= (unsigned int)deviceProp.maxThreadsPerBlock) {
	  param.block.z += step_z;
	  param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
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
      TunableKernel2D::initTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableKernel2D::defaultTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    void resizeVector(int y, int z) const { vector_length_z = z;  TunableKernel2D::resizeVector(y); }
    void resizeStep(int y, int z) const { step_z = z;  TunableKernel2D::resizeStep(y); }
  };


}
