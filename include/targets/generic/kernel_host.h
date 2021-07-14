#pragma once

namespace quda {

  template <template <typename> class Functor, typename Arg>
  void Kernel1D_host(const Arg &arg)
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < (int)arg.threads.x; i++) {
      f(i);
    }
  }

  template <template <typename> class Functor, typename Arg>
  void Kernel2D_host(const Arg &arg)
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < (int)arg.threads.x; i++) {
      for (int j = 0; j < (int)arg.threads.y; j++) {
        f(i, j);
      }
    }
  }

  template <template <typename> class Functor, typename Arg>
  void Kernel3D_host(const Arg &arg)
  {
    Functor<Arg> f(const_cast<Arg &>(arg));
    for (int i = 0; i < (int)arg.threads.x; i++) {
      for (int j = 0; j < (int)arg.threads.y; j++) {
        for (int k = 0; k < (int)arg.threads.z; k++) {
          f(i, j, k);
        }
      }
    }
  }

}
