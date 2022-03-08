#pragma once

#include <color_spinor_field.h>

namespace quda
{

  template <int...> struct IntList {
  };

  template <class F, class... Args>
  void instantiateLsList(F, ColorSpinorField &out, IntList<>, Args &&...)
  {
    errorQuda("Ls = %d has not been instantiated", out.X(4));
  }

  template <class F, int Ls, int... N, class... Args>
  void instantiateLsList(F f, ColorSpinorField &out, IntList<Ls, N...>, Args &&...args)
  {
    if (out.X(4) == Ls) {
      f.template operator()<Ls>(out, args...);
    } else {
      instantiateLsList(f, out, IntList<N...>(), args...);
    }
  }

}



