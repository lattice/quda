#pragma once

namespace quda
{

  /**
     @brief An implementation of a constexpr evaluated for loop.  This
     allow for the loop index to be used in template expressions for
     example.

     @tparam Start The starting index
     @tparam End The end index
     @tparam Inc The loop increment
     @tparam F The functor template type to apply each iteration
     @param[in,out] The functor instance to apply each iteration
   */
  template <auto Start, auto End, auto Inc, class F> constexpr void constexpr_for(F &&f)
  {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      constexpr_for<Start + Inc, End, Inc>(f);
    }
  }

} // namespace quda
