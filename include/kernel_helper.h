#pragma once

namespace quda {

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) :
      func(func),
      name(name) {}
  };

#ifdef JITIFY
#define KERNEL(kernel) kernel_t(nullptr, #kernel)
#else
#define KERNEL(kernel) kernel_t(reinterpret_cast<const void *>(kernel<Functor, Arg, grid_stride>), #kernel)
#endif

}
