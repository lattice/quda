#pragma once

#include "float128_t.h"

#ifdef QUDA_USE_QUAD_SCALAR

#include <cstddef>

#if defined(__SIZEOF_FLOAT128__)
#include <quadmath.h>
#else
// Declare libm TS 18661-3 entry points. Do not rely on math.h exposing them;
// cmath may already have been included without the IEC feature-test macro.
extern "C" {
_Float128 sqrtf128(_Float128);
_Float128 fabsf128(_Float128);
_Float128 cbrtf128(_Float128);
_Float128 sinf128(_Float128);
_Float128 cosf128(_Float128);
_Float128 sinhf128(_Float128);
_Float128 coshf128(_Float128);
_Float128 acosf128(_Float128);
_Float128 asinhf128(_Float128);
_Float128 acoshf128(_Float128);
_Float128 expf128(_Float128);
_Float128 logf128(_Float128);
_Float128 powf128(_Float128, _Float128);
_Float128 fmodf128(_Float128, _Float128);
_Float128 hypotf128(_Float128, _Float128);
int isinff128(_Float128);
int isnanf128(_Float128);
int strfromf128(char *, std::size_t, const char *, _Float128);
}
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

    QUDA_FP128_UNARY(sqrt, sqrtq, sqrtf128)
    QUDA_FP128_UNARY(fabs, fabsq, fabsf128)
    QUDA_FP128_UNARY(cbrt, cbrtq, cbrtf128)
    QUDA_FP128_UNARY(sin, sinq, sinf128)
    QUDA_FP128_UNARY(cos, cosq, cosf128)
    QUDA_FP128_UNARY(sinh, sinhq, sinhf128)
    QUDA_FP128_UNARY(cosh, coshq, coshf128)
    QUDA_FP128_UNARY(acos, acosq, acosf128)
    QUDA_FP128_UNARY(asinh, asinhq, asinhf128)
    QUDA_FP128_UNARY(acosh, acoshq, acoshf128)
    QUDA_FP128_UNARY(exp, expq, expf128)
    QUDA_FP128_UNARY(log, logq, logf128)
    QUDA_FP128_BINARY(pow, powq, powf128)
    QUDA_FP128_BINARY(fmod, fmodq, fmodf128)
    QUDA_FP128_BINARY(hypot, hypotq, hypotf128)
    QUDA_FP128_PRED(isinf, isinfq, isinff128)
    QUDA_FP128_PRED(isnan, isnanq, isnanf128)

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
