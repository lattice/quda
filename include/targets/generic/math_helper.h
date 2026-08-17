#pragma once

#include <cmath>

namespace quda
{

  inline float abs(const float a) { return ::fabsf(a); }
  inline double abs(const double a) { return ::fabs(a); }
  inline float fabs(float a) { return ::fabsf(a); }
  inline double fabs(double a) { return ::fabs(a); }
  template <typename T> inline T sqrt(const T a) { return ::sqrt(a); }
  template <typename T> inline T exp(const T a) { return ::exp(a); }
  template <typename T> inline T log(const T a) { return ::log(a); }
  template <typename T> inline T sin(const T a) { return ::sin(a); }
  template <typename T> inline T cos(const T a) { return ::cos(a); }
  template <typename T> inline T sinh(const T a) { return ::sinh(a); }
  template <typename T> inline T cosh(const T a) { return ::cosh(a); }
  template <typename T> inline T acos(const T a) { return ::acos(a); }
  template <typename T> inline T acosh(const T a) { return ::acosh(a); }
  template <typename T> inline T asinh(const T a) { return ::asinh(a); }
  template <typename T> inline T cbrt(const T a) { return ::cbrt(a); }
  inline bool isnan(float a) { return std::isnan(a); }
  inline bool isnan(double a) { return std::isnan(a); }
  template <typename T> inline T max(const T a, const T b) { return a > b ? a : b; }
  template <typename T> inline T min(const T a, const T b) { return a < b ? a : b; }
  template <typename T> inline void sincos(const T a, T *s, T *c) { ::sincos(a, s, c); }
  template <typename T> inline void sincospi(const T a, T *s, T *c) { ::sincos(a * static_cast<T>(M_PI), s, c); }
  template <typename T> inline T sinpi(const T a) { return ::sin(a * static_cast<float>(M_PI)); }
  template <typename T> inline T cospi(const T a) { return ::cos(a * static_cast<float>(M_PI)); }
  template <typename T> inline T rsqrt(const T a) { return static_cast<T>(1.0) / ::sqrt(a); }
  template <typename T> inline T pow(const T a, const T b) { return ::pow(a, b); }
  template <typename T> inline T pow(const T a, const int b) { return ::pow(a, b); }
  template <typename T> inline T fpow(const T a, const int b) { return ::pow(a, b); }
  template <typename T> inline T fmod(const T a, const T b) { return ::fmod(a, b); }
  inline float fdivide(float a, float b) { return a / b; }
  inline double fdivide(double a, double b) { return a / b; }

  /**
     @brief IEEE double precision multiplication (host)
  */
  inline double dmul_rn(double a, double b) { return a * b; }

  /**
     @brief IEEE double precision addition (host)
  */
  inline double dadd_rn(double a, double b) { return a + b; }

  /**
     @brief IEEE double precision fused multiply add (host)
  */
  inline double fma_rn(double a, double b, double c) { return std::fma(a, b, c); }

  inline float2 fma2(float2 a, float2 b, float2 c) { return {std::fmaf(a.x, b.x, c.x), std::fmaf(a.y, b.y, c.y)}; }
  inline double2 fma2(double2 a, double2 b, double2 c) { return {std::fma(a.x, b.x, c.x), std::fma(a.y, b.y, c.y)}; }

  inline float2 mul2(float2 a, float2 b) { return {a.x * b.x, a.y * b.y}; }
  inline double2 mul2(double2 a, double2 b) { return {a.x * b.x, a.y * b.y}; }

  inline float2 add2(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
  inline double2 add2(double2 a, double2 b) { return {a.x + b.x, a.y + b.y}; }

#ifdef QUDA_USE_QUAD_SCALAR
#include <float128_t.h>
#include <quadmath.h>

  inline float128_t sqrt(float128_t a) { return sqrtq(a); }
  inline float128_t cbrt(float128_t a) { return cbrtq(a); }
  inline float128_t cos(float128_t a) { return cosq(a); }
  inline float128_t acos(float128_t a) { return acosq(a); }
  inline float128_t cosh(float128_t a) { return coshq(a); }
  inline float128_t acosh(float128_t a) { return acoshq(a); }
  inline float128_t sinh(float128_t a) { return sinhq(a); }
  inline float128_t asinh(float128_t a) { return asinhq(a); }
  inline bool isnan(float128_t a) { return isnanq(a); }

#endif

} // namespace quda
