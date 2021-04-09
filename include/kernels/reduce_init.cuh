#include <reduce_helper.h>
#include <kernel.h>

namespace quda {

  namespace reducer {

    template <typename T_> struct init_arg : kernel_param<> {
      using T = T_;
      T *count;
      init_arg(T *count) :
        kernel_param(dim3(max_n_reduce(), 1, 1)),
        count(count) { }
    };

    template <typename Arg> struct init_count {
      Arg &arg;
      static constexpr const char *filename() { return KERNEL_FILE; }
      constexpr init_count(Arg &arg) : arg(arg) {}
      __device__ void operator()(int i) { new (arg.count + i) typename Arg::T {0}; }
    };

  }
}
