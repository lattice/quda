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
    template <typename T> void init_rfa_device_bins_impl();
#endif

    /** Upload RFA bin tables to device constant memory when needed. */
    template <typename T> inline void init_rfa_device_bins()
    {
#if defined(QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE)
      init_rfa_device_bins_impl<T>();
#endif
    }
  } // namespace reducer

} // namespace quda
