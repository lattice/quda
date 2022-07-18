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
   * @brief Sine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the sin(a * pi)
   */
  template<typename T> inline T sinpi(T a) { return sycl::sinpi(a); }

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the cos(a * pi)
   */
  template<typename T> inline T cospi(T a) { return sycl::cospi(a); }

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline void sincospi(const T& a, T *s, T *c)
  {
    *s = sycl::sinpi(a);
    *c = sycl::cospi(a);
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
