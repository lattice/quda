#pragma once

#include <cmath>
#include <target_device.h>

#if (CUDA_VERSION >= 11070) && !defined(_NVHPC_CUDA)
#define BUILTIN_ASSUME(x) \
  bool p = x;             \
  __builtin_assume(p);
#else
#define BUILTIN_ASSUME(x)
#endif

namespace quda {

  /**
   * @brief Maximum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T max(const T &a, const T &b) { return a > b ? a : b; }

  /**
   * @brief Minimum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T min(const T &a, const T &b) { return a < b ? a : b; }


  template <bool is_device> struct sincos_impl {
    template <typename T> inline void operator()(const T& a, T *s, T *c) { ::sincos(a, s, c); }
  };

  template <> struct sincos_impl<true> {
    template <typename T> __device__ inline void operator()(const T& a, T *s, T *c)
    {
      BUILTIN_ASSUME(fabs(a) <= 2.0 * M_PI);
      sincos(a, s, c);
    }
  };

  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline __host__ __device__ void sincos(const T& a, T *s, T *c) { target::dispatch<sincos_impl>(a, s, c); }

  template <bool is_device> struct sincosf_impl {
    inline void operator()(const float& a, float *s, float *c) { ::sincosf(a, s, c); }
  };

  template <> struct sincosf_impl<true> {
    __device__ inline void operator()(const float& a, float *s, float *c) { __sincosf(a, s, c); }
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
  inline __host__ __device__ void sincos(const float& a, float *s, float *c) { target::dispatch<sincosf_impl>(a, s, c); }


  template <bool is_device> struct sincospi_impl {
    template <typename T> inline void operator()(const T& a, T *s, T *c) { ::sincos(a * static_cast<T>(M_PI), s, c); }
  };

  template <> struct sincospi_impl<true> {
    template <typename T> __device__ inline void operator()(const T& a, T *s, T *c) { sincospi(a, s, c); }
  };


  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline __host__ __device__ void sincospi(const T& a, T *s, T *c) { target::dispatch<sincospi_impl>(a, s, c); }

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments.  Use sincos so that Device function calls CUDA intrinsic.
   */
  template<>
  inline __host__ __device__ void sincospi(const float& a, float *s, float *c) { quda::sincos(a * static_cast<float>(M_PI), s, c); }


  /**
   * @brief Sine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the sin(a * pi)
   */
  template<typename T> inline __host__ __device__ T sinpi(T a) { return ::sinpi(a); }

#ifndef _NVHPC_CUDA
  template <bool is_device> struct sinpif_impl { inline float operator()(float a) { return ::sinpif(a); } };
#else
  template <bool is_device> struct sinpif_impl { inline float operator()(float a) { return ::sinf(a * static_cast<float>(M_PI)); } };
#endif
  template <> struct sinpif_impl<true> { __device__ inline float operator()(float a) { return __sinf(a * static_cast<float>(M_PI)); } };

  /**
   * @brief Sine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the sin(a * pi)
   *
   * Specialization to float.  Device function will call CUDA intrinsic.
   */
  template<> inline __host__ __device__ float sinpi(float a) { return target::dispatch<sinpif_impl>(a); }


  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the cos(a * pi)
   */
  template<typename T> inline __host__ __device__ T cospi(T a) { return ::cospi(a); }

#ifndef _NVHPC_CUDA
  template <bool is_device> struct cospif_impl { inline float operator()(float a) { return ::cospif(a); } };
#else
  template <bool is_device> struct cospif_impl { inline float operator()(float a) { return ::cosf(a * static_cast<float>(M_PI)); } };
#endif
  template <> struct cospif_impl<true> { __device__ inline float operator()(float a) { return __cosf(a * static_cast<float>(M_PI)); } };

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the cos(a * pi)
   *
   * Specialization to float.  Device function will call CUDA intrinsic.
   */
  template<> inline __host__ __device__ float cospi(float a) { return target::dispatch<cospif_impl>(a); }


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
