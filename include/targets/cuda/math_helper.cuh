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
  inline __host__ __device__ void sincos(const T& a, T* s, T* c) { ::sincos(a,s,c); }

  template <bool is_device> struct sincosf_impl {
    inline void operator()(const float& a, float * s, float *c) { ::sincosf(a, s, c); }
  };

  template <> struct sincosf_impl<true> {
    __device__ inline void operator()(const float& a, float * s, float *c) { __sincosf(a, s, c); }
  };

  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments. Device function calls CUDA intrinsic
   */
  template<>
  inline __host__ __device__ void sincos(const float& a, float * s, float *c) { target::dispatch<sincosf_impl>(a, s, c); }


  template <bool is_device> struct rsqrt_impl {
    template <typename T> inline T operator()(T a) { return static_cast<T>(1.0) / sqrt(a); }
  };

  template <> struct rsqrt_impl<true> {
    template <typename T> __device__ inline T operator()(T a) { return ::rsqrt(a); }
  };

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   *
   * some math libraries provide a fast inverse sqrt() function.
   * this implementation uses the CUDA builtins
   */
  template<typename T> inline __host__ __device__ T rsqrt(T a) { return target::dispatch<rsqrt_impl>(a); }

  /**
     Generic wrapper for Trig functions -- used in gauge field order
  */
  template <bool isFixed, typename T>
  struct Trig {
    __device__ __host__ static T Atan2( const T &a, const T &b) { return ::atan2(a,b); }
    __device__ __host__ static T Sin( const T &a ) { return ::sin(a); }
    __device__ __host__ static T Cos( const T &a ) { return ::cos(a); }
    __device__ __host__ static void SinCos(const T &a, T *s, T *c) { sincos(a, s, c); }
  };

  template <bool is_device> struct sinf_impl { inline float operator()(const float& a) { return ::sinf(a); } };
  template <> struct sinf_impl<true> { __device__ inline float operator()(const float& a) { return __sinf(a); } };

  template <bool is_device> struct cosf_impl { inline float operator()(const float& a) { return ::cosf(a); } };
  template <> struct cosf_impl<true> { __device__ inline float operator()(const float& a) { return __cosf(a); } };

  /**
     Specialization of Trig functions using floats
   */
  template <>
    struct Trig<false,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return ::atan2f(a,b); }
    __device__ __host__ static float Sin(const float &a) { return target::dispatch<sinf_impl>(a); }
    __device__ __host__ static float Cos(const float &a) { return target::dispatch<cosf_impl>(a); }
    __device__ __host__ static void SinCos(const float &a, float *s, float *c) { target::dispatch<sincosf_impl>(a, s, c); }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return ::atan2f(a,b) / static_cast<float>(M_PI); }
    __device__ __host__ static float Sin(const float &a) { return target::dispatch<sinf_impl>(a * static_cast<float>(M_PI)); }
    __device__ __host__ static float Cos(const float &a) { return target::dispatch<cosf_impl>(a * static_cast<float>(M_PI)); }
    __device__ __host__ static void SinCos(const float &a, float *s, float *c) { target::dispatch<sincosf_impl>(a * static_cast<float>(M_PI), s, c); }
  };

  template <bool is_device> struct fpow_impl { template <typename real> inline real operator()(real a, int b) { return std::pow(a, b); } };

  template <> struct fpow_impl<true> {
    __device__ inline double operator()(double a, int b) { return ::pow(a, b); }

    __device__ inline float operator()(float a, int b)
    {
      float sign = signbit(a) ? -1.0f : 1.0f;
      float power = __powf(fabsf(a), b);
      return b & 1 ? sign * power : power;
    }
  };

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real fpow(real a, int b) { return target::dispatch<fpow_impl>(a, b); }

  template <bool is_device> struct fdividef_impl { inline float operator()(float a, float b) { return a / b; } };
  template <> struct fdividef_impl<true> { __device__ inline float operator()(float a, float b) { return __fdividef(a, b); } };

  /**
     @brief Optimized division routine on the device
  */
  __device__ __host__ inline float fdividef(float a, float b) { return target::dispatch<fdividef_impl>(a, b); }

}
