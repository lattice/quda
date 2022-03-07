#pragma once

#include <vector>

namespace quda
{

  template <template <typename> class Functor, typename Arg> auto Reduction2D_host(const Arg &arg)
  {
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    reduce_t value = t.init();

    for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
      for (int i = 0; i < static_cast<int>(arg.threads.x); i++) { value = t(value, i, j); }
    }

    return value;
  }

  template <template <typename> class Functor, typename Arg> auto MultiReduction_host(const Arg &arg)
  {
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    std::vector<reduce_t> value(arg.threads.z);
    for (int k = 0; k < static_cast<int>(arg.threads.z); k++) {
      value[k] = t.init();

      for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
        for (int i = 0; i < static_cast<int>(arg.threads.x); i++) { value[j] = t(value[j], i, j, k); }
      }
    }

    return value;
  }

} // namespace quda
