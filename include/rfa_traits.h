#pragma once

namespace quda
{

  /** @brief Trait for reproducible floating-point accumulator types. */
  template <class T, class Enable = void> struct is_rfa {
    static constexpr bool value = false;
  };

  namespace reducer
  {
#if defined(QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE)
    // Non-template: each TU has its own __constant__ bin_device_buffer.
    // Call this from a real function in that .cu (not only ReduceArg's
    // header ctor, which nvcc can emit in another TU).
    static void init_rfa_device_bins_impl();
#endif

    /** Upload RFA bin tables to this TU's device constant memory when needed. */
    template <typename T> static inline void init_rfa_device_bins()
    {
#if defined(QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE)
      if constexpr (is_rfa<get_scalar_t<T>>::value) init_rfa_device_bins_impl();
#endif
    }
  } // namespace reducer

} // namespace quda
