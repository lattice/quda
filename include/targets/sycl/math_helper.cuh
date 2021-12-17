#pragma once

#include <math.h>
#include <array.h>

namespace quda {

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

  /**
   * @brief Max
   * @param a the argument
   * @param b the argument
   *
   * Max
   *
   */
  template<typename T>
  inline T max(T a, T b)
  {
    return sycl::max(a, b);
  }

  template<typename T, int N>
  inline array<T,N> max(const array<T,N> &a, const array<T,N> &b)
  {
    array<T,N> result;
    for(int i=0; i<N; i++) {
      result[i] = sycl::max(a[i], b[i]);
    }
    return result;
  }

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
    return sycl::pow(a, b);
#endif
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
