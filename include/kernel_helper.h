#pragma once

namespace quda {

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) :
      func(func),
      name(name) {}
  };

  template <bool use_kernel_arg_ = true>
  struct kernel_param {
    static constexpr bool use_kernel_arg = use_kernel_arg_;
    dim3 threads; /** number of active threads required */

    constexpr kernel_param() = default;

    constexpr kernel_param(dim3 threads) :
      threads(threads)
    { }
  };

#ifdef JITIFY
#define KERNEL(kernel) kernel_t(nullptr, #kernel)
#else
#define KERNEL(kernel) kernel_t(reinterpret_cast<const void *>(kernel<Functor, Arg, grid_stride>), #kernel)
#endif

}
