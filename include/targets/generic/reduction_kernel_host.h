#pragma once

#include <vector>

namespace quda
{

  template <template <typename> class Functor, typename Arg> auto Reduction2D_host(const Arg &arg)
  {
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    reduce_t value = t.init();
#pragma omp parallel for collapse(2) reduction(Functor <Arg>::apply : value)
    for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
      for (int i = 0; i < static_cast<int>(arg.threads.x); i++) { value = t(value, i, j); }
    }

    return value;
  }

  template <template <typename> class Functor, typename Arg> auto MultiReduction_host(const Arg &arg)
  {
#pragma omp declare reduction(multi_reduce                                                                             \
                              : typename Functor <Arg>::reduce_t                                                       \
                              : omp_out = Functor <Arg>::apply(omp_out, omp_in))                                       \
  initializer(omp_priv = Functor <Arg>::init())

    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    std::vector<reduce_t> value(arg.threads.z, t.init());
    for (int k = 0; k < static_cast<int>(arg.threads.z); k++) {
      auto val = t.init();

#pragma omp parallel for collapse(2) reduction(multi_reduce : val)
      for (int j = 0; j < static_cast<int>(arg.threads.y); j++) {
        for (int i = 0; i < static_cast<int>(arg.threads.x); i++) { val = t(val, i, j, k); }
      }

      value[k] = val;
    }

    return value;
  }

} // namespace quda
