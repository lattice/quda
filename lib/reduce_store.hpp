#pragma once

#include <type_traits>
#include <instantiate.h>
#include <float_vector.h>

namespace quda
{

  namespace blas
  {

    template <typename store_t> constexpr QudaPrecision store_prec_v = QUDA_INVALID_PRECISION;
    template <> constexpr QudaPrecision store_prec_v<double> = QUDA_DOUBLE_PRECISION;
    template <> constexpr QudaPrecision store_prec_v<float> = QUDA_SINGLE_PRECISION;
    template <> constexpr QudaPrecision store_prec_v<short> = QUDA_HALF_PRECISION;
    template <> constexpr QudaPrecision store_prec_v<int8_t> = QUDA_QUARTER_PRECISION;

    // Double stays instantiable for host fields even when GPU double is off
    // (same convention as instantiate<> in blas_helper.cuh).
    template <typename store_t> constexpr bool reduce_prec_enabled()
    {
      if constexpr (std::is_same_v<store_t, double>) return true;
      return is_enabled(store_prec_v<store_t>);
    }

    // Call only from an `if constexpr (reduce_prec_enabled<store_t>())` branch.
    // Do not wrap *_impl in a lambda passed to a helper: decltype(fn())
    // instantiates the lambda body and emits kernels for disabled precisions.
    template <typename store_t> void init_reduce_store() { reducer::init_rfa_device_bins<device_reduce_t>(); }

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
