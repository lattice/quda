#pragma once

#include "float128_t.h"

#ifdef QUDA_USE_QUAD_SCALAR

#include <cstddef>

#if defined(__SIZEOF_FLOAT128__)
#include <quadmath.h>
#else
#include <cmath>
#include <cstdlib>
#endif

namespace quda
{
  namespace fp128
  {

#if defined(__SIZEOF_FLOAT128__)
#define QUDA_FP128_UNARY(name, gnu, iec)                                                                               \
  inline float128_t name(float128_t x) { return gnu(x); }
#define QUDA_FP128_BINARY(name, gnu, iec)                                                                              \
  inline float128_t name(float128_t x, float128_t y) { return gnu(x, y); }
#define QUDA_FP128_PRED(name, gnu, iec)                                                                                \
  inline bool name(float128_t x) { return gnu(x); }
#else
#define QUDA_FP128_UNARY(name, gnu, iec)                                                                               \
  inline float128_t name(float128_t x) { return iec(x); }
#define QUDA_FP128_BINARY(name, gnu, iec)                                                                              \
  inline float128_t name(float128_t x, float128_t y) { return iec(x, y); }
#define QUDA_FP128_PRED(name, gnu, iec)                                                                                \
  inline bool name(float128_t x) { return iec(x); }
#endif

    QUDA_FP128_UNARY(sqrt, sqrtq, __builtin_sqrtf128)
    QUDA_FP128_UNARY(fabs, fabsq, __builtin_fabsf128)
    QUDA_FP128_UNARY(cbrt, cbrtq, __builtin_cbrtf128)
    QUDA_FP128_UNARY(sin, sinq, __builtin_sinf128)
    QUDA_FP128_UNARY(cos, cosq, __builtin_cosf128)
    QUDA_FP128_UNARY(sinh, sinhq, __builtin_sinhf128)
    QUDA_FP128_UNARY(cosh, coshq, __builtin_coshf128)
    QUDA_FP128_UNARY(acos, acosq, __builtin_acosf128)
    QUDA_FP128_UNARY(asinh, asinhq, __builtin_asinhf128)
    QUDA_FP128_UNARY(acosh, acoshq, __builtin_acoshf128)
    QUDA_FP128_UNARY(exp, expq, __builtin_expf128)
    QUDA_FP128_UNARY(log, logq, __builtin_logf128)
    QUDA_FP128_BINARY(pow, powq, __builtin_powf128)
    QUDA_FP128_BINARY(fmod, fmodq, __builtin_fmodf128)
    QUDA_FP128_BINARY(hypot, hypotq, __builtin_hypotf128)
    QUDA_FP128_PRED(isinf, isinfq, __builtin_isinff128)
    QUDA_FP128_PRED(isnan, isnanq, __builtin_isnanf128)

#undef QUDA_FP128_UNARY
#undef QUDA_FP128_BINARY
#undef QUDA_FP128_PRED

    inline float128_t abs(float128_t x) { return fabs(x); }
    inline bool isfinite(float128_t x) { return !isnan(x) && !isinf(x); }

    inline constexpr float128_t huge_val()
    {
#if defined(__SIZEOF_FLOAT128__)
      return __builtin_huge_valq();
#else
      return __builtin_huge_valf128();
#endif
    }

    inline float128_t nan()
    {
#if defined(__SIZEOF_FLOAT128__)
      return __builtin_nanq("");
#else
      return __builtin_nanf128("");
#endif
    }

    inline float128_t nans()
    {
#if defined(__SIZEOF_FLOAT128__)
      return __builtin_nansq("");
#else
      return __builtin_nansf128("");
#endif
    }

    inline int snprintf(char *buf, std::size_t n, float128_t x)
    {
#if defined(__SIZEOF_FLOAT128__)
      return quadmath_snprintf(buf, n, "%.40Qe", x);
#else
      return strfromf128(buf, n, "%.40e", x);
#endif
    }

  } // namespace fp128
} // namespace quda

#endif
