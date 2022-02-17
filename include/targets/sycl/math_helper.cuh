#pragma once

#include <math.h>

namespace quda {

#if 0
  /**
   * @brief Round
   * @param a the argument
   *
   * Round to nearest integer
   *
   */
  template<typename T>
  inline T round(T a)
  {
    return sycl::round(a);
  }
#endif

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

  /**
   * @brief Combined sin and cos colculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   */
  template<typename T>
  inline void sincos(const T a, T* s, T* c)
  {
    //*s = sycl::sincos(a, c);
    *s = sycl::sin(a);
    *c = sycl::cos(a);
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
    return sycl::rsqrt(a);
  }

  /**
     Generic wrapper for Trig functions -- used in gauge field order
  */
  template <bool isFixed, typename T>
  struct Trig {
    static T Atan2( const T &a, const T &b) { return sycl::atan2(a,b); }
    static T Sin( const T &a ) { return sycl::sin(a); }
    static T Cos( const T &a ) { return sycl::cos(a); }
    static void SinCos(const T &a, T *s, T *c) { quda::sincos(a, s, c); }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) {
      return sycl::atan2(a,b)/M_PI;
    }
    __device__ __host__ static float Sin(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __sinf(a * static_cast<float>(M_PI));
#else
      return sycl::sin(a * static_cast<float>(M_PI));
#endif
    }

    __device__ __host__ static float Cos(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __cosf(a * static_cast<float>(M_PI));
#else
      return sycl::cos(a * static_cast<float>(M_PI));
#endif
    }

    static void SinCos(const float &a, float *s, float *c)
    {
      //*s = sycl::sincos(a * static_cast<float>(M_PI), c);
      *s = sycl::sinpi(a);
      *c = sycl::cospi(a);
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
    return sycl::pow(a, (real)b);
  }

  /**
     @brief Optimized division routine on the device
  */
  inline float fdividef(float a, float b) { return a/b; }

}


#if 0
template <typename real> inline real fdivide(real a, real b)
{
  return a / b;
}

inline float fdividef(float a, float b)
{
  return a / b;
}

inline float fmaxf(float a, float b)
{
  return sycl::max(a,b);
}
#endif
