#pragma once

#include <cmath>
#include <target_device.h>

namespace quda {

  /**
   * @brief Maximum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline T max(const T &a, const T &b) { return a > b ? a : b; }

  /**
   * @brief Minimum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline T min(const T &a, const T &b) { return a < b ? a : b; }


  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline void sincos(const T& a, T* s, T* c) { *s = std::sin(a); *c = std::cos(a); }

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template <typename T> inline void sincospi(const T& a, T *s, T *c) { quda::sincos(a * static_cast<T>(M_PI), s, c); }

  /**
   * @brief Sine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the sin(a * pi)
   */
  template <typename T> inline T sinpi(T a) { return sin(a * static_cast<T>(M_PI)); }

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the cos(a * pi)
   */
  template <typename T> inline T cospi(T a) { return cos(a * static_cast<T>(M_PI)); }

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   */
  template<typename T> inline T rsqrt(T a) { return static_cast<T>(1.0) / std::sqrt(a); }

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real fpow(real a, int b) { return std::pow(a, b); }

  /**
     @brief Optimized division routine on the device
  */
  __device__ __host__ inline float fdividef(float a, float b) { return a / b; }

}
