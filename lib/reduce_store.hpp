#pragma once

#include <type_traits>
#include <instantiate.h>
#include <float_vector.h>

namespace quda
{

  namespace blas
  {

    template <typename store_t> constexpr QudaPrecision store_prec_v = QUDA_INVALID_PRECISION;
    // Explicit specializations are not implicitly inline; clang emits a strong
    // symbol per TU (gcc merges them). Both reduce_quda.cpp and
    // multi_reduce_quda.cpp include this header.
    template <> inline constexpr QudaPrecision store_prec_v<double> = QUDA_DOUBLE_PRECISION;
    template <> inline constexpr QudaPrecision store_prec_v<float> = QUDA_SINGLE_PRECISION;
    template <> inline constexpr QudaPrecision store_prec_v<short> = QUDA_HALF_PRECISION;
    template <> inline constexpr QudaPrecision store_prec_v<int8_t> = QUDA_QUARTER_PRECISION;

    // Double stays instantiable for host fields even when GPU double is off
    // (same convention as instantiate<> in blas_helper.cuh).
    template <typename store_t> constexpr bool reduce_prec_enabled()
    {
      if constexpr (std::is_same_v<store_t, double>) return true;
      return is_enabled(store_prec_v<store_t>);
    }

    // Unlike reduce_prec_enabled, multi-reduce has no CPU-field fallback
    // (see the `errorQuda("Only implemented for GPU fields")` in
    // MultiReduce::compute), so there is no reason to keep double
    // artificially "enabled" here: doing so only forces
    // multi_reduce_quda_double.cu to redundantly compile the same device
    // kernels already built by whichever precision is actually enabled.
    template <typename store_t> constexpr bool multi_reduce_prec_enabled() { return is_enabled(store_prec_v<store_t>); }

    template <typename Fn> auto dispatch_reduce_prec(QudaPrecision prec, Fn &&fn)
    {
      if (!is_enabled(prec) && prec != QUDA_DOUBLE_PRECISION)
        errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, prec);

      switch (prec) {
      case QUDA_DOUBLE_PRECISION: return fn.template operator()<double>();
      case QUDA_SINGLE_PRECISION:
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) return fn.template operator()<float>();
        break;
      case QUDA_HALF_PRECISION:
        if constexpr (is_enabled(QUDA_HALF_PRECISION)) return fn.template operator()<short>();
        break;
      case QUDA_QUARTER_PRECISION:
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) return fn.template operator()<int8_t>();
        break;
      default: errorQuda("Unsupported precision %d", prec);
      }
      errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, prec);
      using Ret = decltype(fn.template operator()<double>());
      if constexpr (!std::is_void_v<Ret>) return Ret {};
    }

  } // namespace blas

} // namespace quda
