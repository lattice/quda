#pragma once

#include <cmath>

namespace quda
{

  inline float abs(const float a) { return fabs(a); }
  inline double abs(const double a) { return fabs(a); }
  template <typename T> inline T sqrt(const T a) { return ::sqrt(a); }
  template <typename T> inline T exp(const T a) { return ::exp(a); }
  template <typename T> inline T log(const T a) { return ::log(a); }
  template <typename T> inline T sin(const T a) { return ::sin(a); }
  template <typename T> inline T cos(const T a) { return ::cos(a); }
  template <typename T> inline T sinh(const T a) { return ::sinh(a); }
  template <typename T> inline T cosh(const T a) { return ::cosh(a); }
  template <typename T> inline T max(const T a, const T b) { return a > b ? a : b; }
  template <typename T> inline T min(const T a, const T b) { return a < b ? a : b; }
  template <typename T> inline void sincos(const T a, T *s, T *c) { ::sincos(a, s, c); }
  inline void sincos(float a, float *s, float *c) { *s = std::sin(a); *c = std::cos(a); }
  template <typename T> inline void sincospi(const T a, T *s, T *c) { ::sincos(a * static_cast<T>(M_PI), s, c); }
  inline void sincospi(float a, float *s, float *c) { sincos(a * static_cast<float>(M_PI), s, c); }
  template <typename T> inline T sinpi(const T a) { return ::sin(a * static_cast<float>(M_PI)); }
  template <typename T> inline T cospi(const T a) { return ::cos(a * static_cast<float>(M_PI)); }
  template <typename T> inline T rsqrt(const T a) { return static_cast<T>(1.0) / ::sqrt(a); }
  template <typename T> inline T pow(const T a, const T b) { return ::pow(a, b); }
  template <typename T> inline T pow(const T a, const int b) { return ::pow(a, b); }
  template <typename T> inline T fpow(const T a, const int b) { return ::pow(a, b); }
  template <typename T> inline T fmod(const T a, const T b) { return ::fmod(a, b); }
  inline float fdividef(const float a, const float b) { return a / b; }

  inline float2 fma2(float2 a, float2 b, float2 c) { return {a.x * b.x + c.x, a.y * b.y + c.y}; }
  inline double2 fma2(double2 a, double2 b, double2 c) { return {a.x * b.x + c.x, a.y * b.y + c.y}; }

  inline float2 mul2(float2 a, float2 b) { return {a.x * b.x, a.y * b.y}; }
  inline double2 mul2(double2 a, double2 b) { return {a.x * b.x, a.y * b.y}; }

  inline float2 add2(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
  inline double2 add2(double2 a, double2 b) { return {a.x + b.x, a.y + b.y}; }

} // namespace quda
