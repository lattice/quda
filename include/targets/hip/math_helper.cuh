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
   * Specialization to float arguments. Device function calls CUDA intrinsic
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

  /**
    Generic wrapper for Trig functions -- used in gauge field order
   */
  template <bool isFixed, typename T> struct Trig {
    __device__ __host__ static T Atan2(const T &a, const T &b) { return ::atan2(a, b); }
    __device__ __host__ static T Sin(const T &a) { return ::sin(a); }
    __device__ __host__ static T Cos(const T &a) { return ::cos(a); }
    __device__ __host__ static void SinCos(const T &a, T *s, T *c)
    {
      *s = ::sin(a);
      *c = ::cos(a);
    }
  };

  template <bool is_device> struct sinf_impl {
    inline float operator()(const float &a) { return ::sinf(a); }
  };
  template <> struct sinf_impl<true> {
    __device__ inline float operator()(const float &a) { return ::sinf(a); }
  };

  template <bool is_device> struct cosf_impl {
    inline float operator()(const float &a) { return ::cosf(a); }
  };
  template <> struct cosf_impl<true> {
    __device__ inline float operator()(const float &a) { return ::cosf(a); }
  };

  template <bool is_device> struct sin_impl {
    inline double operator()(const double &a) { return ::sin(a); }
  };
  template <> struct sin_impl<true> {
    __device__ inline double operator()(const double &a) { return ::sin(a); }
  };

  template <bool is_device> struct cos_impl {
    inline double operator()(const double &a) { return ::cos(a); }
  };
  template <> struct cos_impl<true> {
    __device__ inline double operator()(const double &a) { return ::cos(a); }
  };

  /**
     Specialization of Trig functions using floats
   */
  template <> struct Trig<false, float> {
    __device__ __host__ static float Atan2(const float &a, const float &b) { return ::atan2f(a, b); }
    __device__ __host__ static float Sin(const float &a) { return target::dispatch<sinf_impl>(a); }
    __device__ __host__ static float Cos(const float &a) { return target::dispatch<cosf_impl>(a); }
    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
      target::dispatch<sincosf_impl>(a, s, c);
    }
  };

  template <> struct Trig<true, float> {
    __device__ __host__ static float Atan2(const float &a, const float &b) { return ::atan2f(a, b) / M_PI; }
    __device__ __host__ static float Sin(const float &a)
    {
      return target::dispatch<sinf_impl>(a * static_cast<float>(M_PI));
    }
    __device__ __host__ static float Cos(const float &a)
    {
      return target::dispatch<cosf_impl>(a * static_cast<float>(M_PI));
    }
    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
      auto ampi = a * static_cast<float>(M_PI);
      target::dispatch<sincosf_impl>(ampi, s, c);
    }
  };

  /**
     Specialization of Trig functions using doubles
   */
  template <> struct Trig<false, double> {
    __device__ __host__ static double Atan2(const double &a, const double &b) { return ::atan2(a, b); }
    __device__ __host__ static double Sin(const double &a) { return target::dispatch<sin_impl>(a); }
    __device__ __host__ static double Cos(const double &a) { return target::dispatch<cos_impl>(a); }
    __device__ __host__ static void SinCos(const double &a, double *s, double *c)
    {
      target::dispatch<sincos_impl>(a, s, c);
    }
  };

  template <> struct Trig<true, double> {
    __device__ __host__ static double Atan2(const double &a, const double &b) { return ::atan2(a, b) / M_PI; }
    __device__ __host__ static double Sin(const double &a)
    {
      return target::dispatch<sin_impl>(a * static_cast<double>(M_PI));
    }
    __device__ __host__ static double Cos(const double &a)
    {
      return target::dispatch<cos_impl>(a * static_cast<double>(M_PI));
    }
    __device__ __host__ static void SinCos(const double &a, double *s, double *c)
    {
      auto ampi = a * static_cast<double>(M_PI);
      target::dispatch<sincos_impl>(ampi, s, c);
    }
  };

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
