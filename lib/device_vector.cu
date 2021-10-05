#include <device_vector.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>

namespace quda
{

  template <class T>
    struct AxpbyArg: kernel_param<> {
      T *out;

      T a;
      const T *x;

      T b;
      const T *y;

      AxpbyArg(T *out, T a, const T *x, T b, const T *y, int size):
        kernel_param(size), out(out), a(a), x(x), b(b), y(y) { }
    };

  template <class Arg>
    struct Axpby {
      const Arg &arg;
      constexpr Axpby(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline void operator()(int thread_idx) {
        arg.out[thread_idx] += arg.a * arg.x[thread_idx] + arg.b * arg.y[thread_idx];
      }
    };

  template <class T>
    struct axpby_wrapper: TunableKernel1D {

      const AxpbyArg<T> arg;

      axpby_wrapper(T *out, T a, const T *x, T b, const T *y, int size): TunableKernel1D(size, QUDA_CUDA_FIELD_LOCATION), arg(out, a, x, b, y, size)
      { 
        strcat(aux, ",axpby");
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<Axpby>(tp, stream, arg);
      }

      unsigned int minThreads() const { return arg.threads.x; }
      long long flops() const { return 4 * arg.threads.x; }
      long long bytes() const { return 3 * sizeof(T) * arg.threads.x; }

    };

  void axpby(device_vector<float> &out, float a, const device_vector<float> &x, float b,
      const device_vector<float> &y)
  {
    axpby_wrapper<float> w(out.data(), a, x.data(), b, y.data(), out.size());
  }

}
