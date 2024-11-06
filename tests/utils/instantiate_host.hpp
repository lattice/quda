#pragma once

#include <quda.h>

/**
 *  @brief Helper function for returning if a given precision is enabled
 *  @tparam precision The precision requested
 *  @return True if enabled, false if not
 */
constexpr bool is_enabled(QudaPrecision precision)
{
  switch (precision) {
  case QUDA_DOUBLE_PRECISION: return (QUDA_PRECISION & 8) ? true : false;
  case QUDA_SINGLE_PRECISION: return (QUDA_PRECISION & 4) ? true : false;
  case QUDA_HALF_PRECISION: return (QUDA_PRECISION & 2) ? true : false;
  case QUDA_QUARTER_PRECISION: return (QUDA_PRECISION & 1) ? true : false;
  default: return false;
  }
}

/**
 * @brief This instantiate function helps with casting void* fields to double or float
 *
 * @tparam Apply Type of structure with a constructor that is being instantiated
 * @tparam Args Variadic arguments passed along to the instantiation struct constructors
 * @param[in] precision Floating-point precision for the computation
 * @param[in,out] args Any additional arguments required for the computation at hand
 */
template <template <typename> class Apply, typename... Args>
constexpr void instantiate_host(QudaPrecision precision, Args &&...args)
{
  // always instantiate double precision
  if (precision == QUDA_DOUBLE_PRECISION) {
    Apply<double>()(args...);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
      Apply<float>()(args...);
    else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
  } else {
    errorQuda("Unsupported precision %d\n", precision);
  }
}

/**
 * @brief This instantiate function helps with casting void* fields to double or float
 *        in kernels also requiring a reduction
 *
 * @tparam Apply Type of structure with a constructor that is being instantiated
 * @tparam return_t Type of return value
 * @tparam Args Variadic arguments passed along to the instantiation struct constructors
 * @param[in] precision Floating-point precision for the computation
 * @param[in,out] args Any additional arguments required for the computation at hand
 * @return Result of reduction
 */
template <template <typename> class Apply, typename return_t, typename... Args>
constexpr return_t instantiate_host_reduce(QudaPrecision precision, Args &&...args)
{
  // always instantiate double precision
  if (precision == QUDA_DOUBLE_PRECISION) {
    return Apply<double>()(args...);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
      return Apply<float>()(args...);
    else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
  } else {
    errorQuda("Unsupported precision %d\n", precision);
  }
  return return_t();
}
