#pragma once

#include <cmath>
#include <target_device.h>
#include <hip/math_functions.h>

namespace quda
{

  /**
   * @brief Maximum of two numbers
   * @param a first number
   * @param b second number
   */
  template <typename T> inline __host__ __device__ T max(const T &a, const T &b) { return a > b ? a : b; }

  /**
   * @brief Minimum of two numbers
   * @param a first number
   * @param b second number
   */
  template <typename T> inline __host__ __device__ T min(const T &a, const T &b) { return a < b ? a : b; }

  /**
   * @brief Combined sin and cos colculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template <typename T> inline __host__ __device__ void sincos(const T &a, T *s, T *c)
  {
    // Hip Does not have ::sincos(a,s,c);
    // Just on device sincosf
    // Need to do this as 2 stepper?
    *s = ::sin(a);
    *c = ::cos(a);
  }

  // Impl for float type host
  template <bool is_device> struct sincosf_impl {
    inline void operator()(const float &a, float *s, float *c)
    {
      *s = ::sinf(a);
      *c = ::cosf(a);
    }
  };

  // Impl for float device
  template <> struct sincosf_impl<true> {
    __device__ inline void operator()(const float &a, float *s, float *c) { ::sincosf(a, s, c); }
  };

  // impl for double type host
  template <bool is_device> struct sincos_impl {
    inline void operator()(const double &a, double *s, double *c)
    {
      *s = ::sin(a);
      *c = ::cos(a);
    }
  };

  // impl for double type on device
  template <> struct sincos_impl<true> {
    __device__ inline void operator()(const double &a, double *s, double *c) { ::sincos(a, s, c); }
  };

  /**
   * @brief Combined sin and cos colculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments.
   */
  template <> inline __host__ __device__ void sincos(const float &a, float *s, float *c)
  {
    target::dispatch<sincosf_impl>(a, s, c);
  }

  template <> inline __host__ __device__ void sincos(const double &a, double *s, double *c)
  {
    target::dispatch<sincos_impl>(a, s, c);
  }

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template <typename T> inline __host__ __device__ void sincospi(const T& a, T *s, T *c) { quda::sincos(a * static_cast<T>(M_PI), s, c); }

  /**
   * @brief Sine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the sin(a * pi)
   *
   * Specialization to float.
   */
  template <typename T> inline __host__ __device__ T sinpi(T a) { return sin(a * static_cast<T>(M_PI)); }

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the cos(a * pi)
   *
   * Specialization to float.
   */
  template <typename T> inline __host__ __device__ T cospi(T a) { return cos(a * static_cast<T>(M_PI)); }

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   *
   * some math libraries provide a fast inverse sqrt() function.
   * this implementation assumes rsqrt() doesn't exist in general
   * but specializes to use __rsqrtf() on the device.
   */
  template <typename T> inline __host__ __device__ T rsqrt(const T &a) { return 1.0 / ::sqrt(a); }

  // Impl for float type host
  template <bool is_device> struct rsqrtf_impl {
    inline float operator()(const float &a) { return 1.0 / ::sqrtf(a); }
  };

  // Impl for double type host
  template <bool is_device> struct rsqrt_impl {
    inline double operator()(const double &a) { return 1.0 / ::sqrt(a); }
  };

  template <> struct rsqrtf_impl<true> {
    __device__ inline float operator()(const float &a) { return ::rsqrtf(a); }
  };

  template <> struct rsqrt_impl<true> {
    __device__ inline double operator()(const double &a) { return ::rsqrt(a); }
  };

  template <> inline __host__ __device__ float rsqrt(const float &a) { return target::dispatch<rsqrtf_impl>(a); }
  template <> inline __host__ __device__ double rsqrt(const double &a) { return target::dispatch<rsqrt_impl>(a); }


  template <bool is_device> struct fpow_impl {
    template <typename real> inline real operator()(real a, int b) { return ::pow(a, b); }
  };

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
  template <typename real> __device__ __host__ inline real fpow(real a, int b)
  {
    return target::dispatch<fpow_impl>(a, b);
  }

  template <bool is_device> struct fdividef_impl {
    inline float operator()(float a, float b) { return a / b; }
  };
  template <> struct fdividef_impl<true> {
    __device__ inline float operator()(float a, float b) { return __fdividef(a, b); }
  };

  /**
     @brief Optimized division routine on the device
  */
  __device__ __host__ inline float fdividef(float a, float b) { return target::dispatch<fdividef_impl>(a, b); }

} // namespace quda
