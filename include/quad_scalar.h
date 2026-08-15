#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include <cmath>
#include <iostream>
#include <limits>
// std overloads for host .cpp (nvcc host pass uses math_helper / nvvm-latest)
#if !defined(__CUDACC__)

#include <quadmath.h>

namespace std
{

  inline __float128 sqrt(__float128 x) { return sqrtq(x); }
  inline __float128 abs(__float128 x) { return fabsq(x); }
  inline __float128 cbrt(__float128 x) { return cbrtq(x); }
  inline __float128 sin(__float128 x) { return sinq(x); }
  inline __float128 cos(__float128 x) { return cosq(x); }
  inline __float128 sinh(__float128 x) { return sinhq(x); }
  inline __float128 cosh(__float128 x) { return coshq(x); }
  inline __float128 acos(__float128 x) { return acosq(x); }
  inline __float128 asinh(__float128 x) { return asinhq(x); }
  inline __float128 acosh(__float128 x) { return acoshq(x); }
  inline __float128 fabs(__float128 x) { return fabsq(x); }
  inline __float128 exp(__float128 x) { return expq(x); }
  inline __float128 log(__float128 x) { return logq(x); }
  inline __float128 pow(__float128 x, __float128 y) { return powq(x, y); }
  inline __float128 fmod(__float128 x, __float128 y) { return fmodq(x, y); }
  inline bool isinf(__float128 x) { return isinfq(x); }
  inline bool isnan(__float128 x) { return isnanq(x); }
  inline bool isfinite(__float128 x) { return !isnanq(x) && !isinfq(x); }

  inline std::ostream &operator<<(std::ostream &os, __float128 x);

  // libstdc++ < 14 leaves numeric_limits<__float128> unspecialized (infinity() == 0).
  // GCC 14+ already provides this specialization; redefining it is an error.
#if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE < 14
  template <> class numeric_limits<__float128>
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
    static constexpr __float128 min() { return __FLT128_MIN__; }
    static constexpr __float128 max() { return __FLT128_MAX__; }
    static constexpr __float128 lowest() { return -__FLT128_MAX__; }
    static constexpr __float128 epsilon() { return __FLT128_EPSILON__; }
    static constexpr __float128 round_error() { return __float128(1) / __float128(2); }
    static constexpr __float128 infinity() { return __builtin_huge_valq(); }
    static __float128 quiet_NaN() { return __builtin_nanq(""); }
    static __float128 signaling_NaN() { return __builtin_nansq(""); }
    static constexpr __float128 denorm_min() { return __FLT128_DENORM_MIN__; }
  };
#endif

} // namespace std

#endif

namespace quda
{

  // Host .cpp only; .cu host pass uses math_helper fabs(__float128) dispatch
#if !defined(__CUDACC__)
  inline __float128 fabs(__float128 x) { return fabsq(x); }
#endif

  inline double to_double(__float128 x) { return static_cast<double>(x); }

  inline __float128 real_cast(double x) { return static_cast<__float128>(x); }

  template <typename T> inline __float128 max(const __float128 &a, const T &b)
  {
    __float128 y = static_cast<__float128>(b);
    return a > y ? a : y;
  }

  template <typename T> inline __float128 max(const T &a, const __float128 &b) { return max(b, a); }

  template <typename T> inline __float128 min(const __float128 &a, const T &b)
  {
    __float128 y = static_cast<__float128>(b);
    return a < y ? a : y;
  }

  template <typename T> inline __float128 min(const T &a, const __float128 &b) { return min(b, a); }

} // namespace quda

#if !defined(__CUDACC__)
inline std::ostream &std::operator<<(std::ostream &os, __float128 x)
{
  return os << quda::to_double(x);
}
#endif

#endif // QUDA_USE_QUAD_SCALAR
