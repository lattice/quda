#pragma once

#include <math.h>

namespace quda {

  /**
   * @brief Round
   * @param a the argument
   *
   * Round to nearest integer
   *
   */
  template<typename T>
  inline __host__ __device__ T round(T a)
  {
    return sycl::round(a);
  }

  /**
   * @brief Max
   * @param a the argument
   * @param b the argument
   *
   * Max
   *
   */
  template<typename T>
  inline __host__ __device__ T max(T a, T b)
  {
    return sycl::max(a, b);
  }

  /**
   * @brief Combined sin and cos colculation in QUDA NAMESPACE  
   * @param a the angle 
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   */
  template<typename T>
  inline __host__ __device__ void sincos(const T& a, T* s, T* c)
  {
    *s = sycl::sincos(a, c);
  }

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   *
   * some math libraries provide a fast inverse sqrt() function.
   * this implementation uses the CUDA builtins
   */
  template<typename T>
  inline __host__ __device__ T rsqrt(T a)
  {
    return ::rsqrt(a);
  }

  /**
     Generic wrapper for Trig functions -- used in gauge field order 
  */
  template <bool isFixed, typename T>
  struct Trig {
    static T Atan2( const T &a, const T &b) { return ::atan2(a,b); }
    static T Sin( const T &a ) { return ::sin(a); }
    static T Cos( const T &a ) { return ::cos(a); }
    static void SinCos(const T &a, T *s, T *c) { quda::sincos(a, s, c); }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return ::atan2f(a,b)/M_PI; }
    __device__ __host__ static float Sin(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __sinf(a * static_cast<float>(M_PI));
#else
      return ::sinf(a * static_cast<float>(M_PI));
#endif
    }

    __device__ __host__ static float Cos(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __cosf(a * static_cast<float>(M_PI));
#else
      return ::cosf(a * static_cast<float>(M_PI));
#endif
    }

    static void SinCos(const float &a, float *s, float *c)
    {
      *s = sycl::sincos(a * static_cast<float>(M_PI), c);
    }
  };

/*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real __fast_pow(real a, int b)
  {
#ifdef __CUDA_ARCH__
    if (sizeof(real) == sizeof(double)) {
      return ::pow(a, b);
    } else {
      float sign = signbit(a) ? -1.0f : 1.0f;
      float power = __powf(fabsf(a), b);
      return b & 1 ? sign * power : power;
    }
#else
    return std::pow(a, b);
#endif
  }

  template <typename real> inline real fdivide(real a, real b)
  {
    return a / b;
  }

  template <typename real> inline real fdividef(real a, real b)
  {
    return a / b;
  }

}
