#include <tunable_nd.h>

/**
  @file The following contains the argument and kernel for applying axpby to device vectors.
*/

namespace quda
{

  template <class T> struct AxpbyArg : kernel_param<> {
    T *out;
    T a;
    const T *x;
    T b;
    const T *y;

    AxpbyArg(T *out, T a, const T *x, T b, const T *y, int size) : kernel_param(size), out(out), a(a), x(x), b(b), y(y)
    {
    }
  };

  template <class Arg> struct Axpby {
    const Arg &arg;
    constexpr Axpby(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int thread_idx)
    {
      arg.out[thread_idx] += arg.a * arg.x[thread_idx] + arg.b * arg.y[thread_idx];
    }
  };

} // namespace quda
