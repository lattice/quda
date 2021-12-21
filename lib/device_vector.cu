#include <device_vector.h>
#include <kernels/device_vector_axpby.cuh>

namespace quda
{

  template <class T> struct axpby_wrapper : TunableKernel1D {

    AxpbyArg<T> arg;

    device_vector<T> _backup_vector;
    T *_backup_ptr = nullptr;

    axpby_wrapper(T *out, T a, const T *x, T b, const T *y, int size) :
      TunableKernel1D(size, QUDA_CUDA_FIELD_LOCATION), arg(out, a, x, b, y, size)
    {
      strcat(aux, ",axpby");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Axpby>(tp, stream, arg);
    }

    void preTune()
    {
      _backup_vector.resize(arg.threads.x);
      _backup_vector.from_device(arg.out);
      _backup_ptr = arg.out;
      arg.out = _backup_vector.data();
    }

    void postTune() { arg.out = _backup_ptr; }

    unsigned int minThreads() const { return arg.threads.x; }
    long long flops() const { return 4 * arg.threads.x; }
    long long bytes() const { return 3 * sizeof(T) * arg.threads.x; }
  };

  void axpby(device_vector<float> &out, float a, const device_vector<float> &x, float b, const device_vector<float> &y)
  {
    axpby_wrapper<float> w(out.data(), a, x.data(), b, y.data(), out.size());
  }

} // namespace quda
