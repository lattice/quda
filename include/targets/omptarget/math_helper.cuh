#pragma once

#include <cmath>
#include <target_device.h>

namespace quda {

  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline __host__ __device__ void sincos(const T& a, T* s, T* c) { *s = std::sin(a); *c = std::cos(a); }

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   *
   * some math libraries provide a fast inverse sqrt() function.
   * this implementation trusts the compiler.
   */
  template<typename T> inline __host__ __device__ T rsqrt(T a) { return static_cast<T>(1.0) / std::sqrt(a); }

    /**
     Generic wrapper for Trig functions -- used in gauge field order 
    */
  template <bool isFixed, typename T>
  struct Trig {
    __device__ __host__ static T Atan2( const T &a, const T &b) { return std::atan2(a,b); }
    __device__ __host__ static T Sin( const T &a ) { return std::sin(a); }
    __device__ __host__ static T Cos( const T &a ) { return std::cos(a); }
    __device__ __host__ static void SinCos(const T &a, T *s, T *c) { sincos(a, s, c); }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return std::atan2(a,b)/M_PI; }
    __device__ __host__ static float Sin(const float &a) { return std::sin(a * static_cast<float>(M_PI)); }
    __device__ __host__ static float Cos(const float &a) { return std::cos(a * static_cast<float>(M_PI)); }
    __device__ __host__ static void SinCos(const float &a, float *s, float *c) { sincos(a * static_cast<float>(M_PI), s, c); }
  };

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
