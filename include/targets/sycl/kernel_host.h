#pragma once

namespace quda
{

  template <template <typename> class Functor, typename Arg> void Kernel1D_host(const Arg &arg)
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < static_cast<int>(arg.threads.x); i++) { f(i); }
  }

  template <template <typename> class Functor, typename Arg> void Kernel2D_host(const Arg &arg)
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < static_cast<int>(arg.threads.x); i++) {
      for (int j = 0; j < static_cast<int>(arg.threads.y); j++) { f(i, j); }
    }
  }

  template <template <typename> class Functor, typename Arg> void Kernel3D_host(const Arg &arg)
  {
    if constexpr (needsSharedMem<getKernelOps<Functor<Arg>>>) {
      const int maxsmemsize = 64 * 1024;
      auto smemsize = sharedMemSize<getKernelOps<Functor<Arg>>>(dim3(1, 1, 1), arg);
      if (smemsize > maxsmemsize) errorQuda("Smem size (%d) > Max Smem size (%d)", smemsize, maxsmemsize);
      char smem[maxsmemsize];
      Functor<Arg> f {arg, &smem[0]};
      for (int i = 0; i < static_cast<int>(arg.threads.x); i++) {
        for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
          for (int k = 0; k < static_cast<int>(arg.threads.z); k++) { f(i, j, k); }
        }
      }
    } else {
      Functor<Arg> f(const_cast<Arg &>(arg));
      for (int i = 0; i < static_cast<int>(arg.threads.x); i++) {
        for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
          for (int k = 0; k < static_cast<int>(arg.threads.z); k++) { f(i, j, k); }
        }
      }
    }
  }

} // namespace quda
