#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include "float128_t.h"
#include <cmath>
#include <iostream>
#include <limits>
// std overloads for host .cpp (nvcc host pass uses math_helper / nvvm-latest)
#if !defined(__CUDACC__)

#include <quadmath.h>

namespace std
{

  inline quda::float128_t sqrt(quda::float128_t x) { return sqrtq(x); }
  inline quda::float128_t abs(quda::float128_t x) { return fabsq(x); }
  inline quda::float128_t cbrt(quda::float128_t x) { return cbrtq(x); }
  inline quda::float128_t sin(quda::float128_t x) { return sinq(x); }
  inline quda::float128_t cos(quda::float128_t x) { return cosq(x); }
  inline quda::float128_t sinh(quda::float128_t x) { return sinhq(x); }
  inline quda::float128_t cosh(quda::float128_t x) { return coshq(x); }
  inline quda::float128_t acos(quda::float128_t x) { return acosq(x); }
  inline quda::float128_t asinh(quda::float128_t x) { return asinhq(x); }
  inline quda::float128_t acosh(quda::float128_t x) { return acoshq(x); }
  inline quda::float128_t fabs(quda::float128_t x) { return fabsq(x); }
  inline quda::float128_t exp(quda::float128_t x) { return expq(x); }
  inline quda::float128_t log(quda::float128_t x) { return logq(x); }
  inline quda::float128_t pow(quda::float128_t x, quda::float128_t y) { return powq(x, y); }
  inline quda::float128_t fmod(quda::float128_t x, quda::float128_t y) { return fmodq(x, y); }
  inline bool isinf(quda::float128_t x) { return isinfq(x); }
  inline bool isnan(quda::float128_t x) { return isnanq(x); }
  inline bool isfinite(quda::float128_t x) { return !isnanq(x) && !isinfq(x); }

  inline std::ostream &operator<<(std::ostream &os, quda::float128_t x);

  // libstdc++ < 14 leaves numeric_limits for binary128 unspecialized (infinity() == 0).
  // GCC 14+ already provides this specialization; redefining it is an error.
#if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE < 14
  template <> class numeric_limits<quda::float128_t>
  {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = __FLT128_MANT_DIG__;
    static constexpr int digits10 = __FLT128_DIG__;
    static constexpr int max_digits10 = 36;
    static constexpr int radix = 2;
    static constexpr int min_exponent = __FLT128_MIN_EXP__;
    static constexpr int min_exponent10 = __FLT128_MIN_10_EXP__;
    static constexpr int max_exponent = __FLT128_MAX_EXP__;
    static constexpr int max_exponent10 = __FLT128_MAX_10_EXP__;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    // Use compiler builtins, not FLT128_* / Q-suffix literals (those need
    // non-constexpr operator""Q from quadmath.h).
    static constexpr quda::float128_t min() { return __FLT128_MIN__; }
    static constexpr quda::float128_t max() { return __FLT128_MAX__; }
    static constexpr quda::float128_t lowest() { return -__FLT128_MAX__; }
    static constexpr quda::float128_t epsilon() { return __FLT128_EPSILON__; }
    static constexpr quda::float128_t round_error() { return quda::float128_t(1) / quda::float128_t(2); }
    static constexpr quda::float128_t infinity() { return __builtin_huge_valq(); }
    static quda::float128_t quiet_NaN() { return __builtin_nanq(""); }
    static quda::float128_t signaling_NaN() { return __builtin_nansq(""); }
    static constexpr quda::float128_t denorm_min() { return __FLT128_DENORM_MIN__; }
  };
#endif

} // namespace std

#endif

namespace quda
{

  // Host .cpp only; .cu host pass uses math_helper fabs(float128_t) dispatch
#if !defined(__CUDACC__)
  inline float128_t fabs(float128_t x) { return fabsq(x); }
#endif

  inline double to_double(float128_t x) { return static_cast<double>(x); }

  inline float128_t real_cast(double x) { return static_cast<float128_t>(x); }

  template <typename T> inline float128_t max(const float128_t &a, const T &b)
  {
    float128_t y = static_cast<float128_t>(b);
    return a > y ? a : y;
  }

  template <typename T> inline float128_t max(const T &a, const float128_t &b) { return max(b, a); }

  template <typename T> inline float128_t min(const float128_t &a, const T &b)
  {
    float128_t y = static_cast<float128_t>(b);
    return a < y ? a : y;
  }

  template <typename T> inline float128_t min(const T &a, const float128_t &b) { return min(b, a); }

} // namespace quda

#if !defined(__CUDACC__)
inline std::ostream &std::operator<<(std::ostream &os, quda::float128_t x)
{
  return os << quda::to_double(x);
}
#endif

#endif // QUDA_USE_QUAD_SCALAR
