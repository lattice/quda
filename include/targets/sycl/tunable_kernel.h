#pragma once

#include <tune_quda.h>
#include <target_device.h>
#include <kernel_helper.h>
//#include <kernel.h>
#include <quda_sycl.h>

namespace quda {

  class TunableKernel : public Tunable
  {

  protected:
    QudaFieldLocation location;

    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    template <template <typename> class Functor, bool grid_stride, typename Arg>
    qudaError_t
    launch_device(const kernel_t &kernel, const TuneParam &tp,
		  const qudaStream_t &stream, const Arg &arg)
    {
      using launcher_t = qudaError_t(*)(const TuneParam &, const qudaStream_t &,
					const Arg &);
      auto f = reinterpret_cast<launcher_t>(const_cast<void *>(kernel.func));
      launch_error = f(tp, stream, arg);
      return launch_error;
    }

  public:
    TunableKernel(QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION) : location(location) { }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }
  };

  // kernel helpers
  inline sycl::range<3> globalRange(const TuneParam &tp) {
    sycl::range<3> r(1,1,1);
    r[RANGE_X] = tp.grid.x*tp.block.x;
    r[RANGE_Y] = tp.grid.y*tp.block.y;
    r[RANGE_Z] = tp.grid.z*tp.block.z;
    return r;
  }
  inline sycl::range<3> localRange(const TuneParam &tp) {
    sycl::range<3> r(1,1,1);
    r[RANGE_X] = tp.block.x;
    r[RANGE_Y] = tp.block.y;
    r[RANGE_Z] = tp.block.z;
    return r;
  }

}
