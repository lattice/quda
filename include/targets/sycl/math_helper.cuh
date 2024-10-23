#pragma once

//#include <math.h>

namespace quda {

  inline int abs(const int a) { return sycl::abs(a); }
  inline float abs(const float a) { return sycl::fabs(a); }
  inline double abs(const double a) { return sycl::fabs(a); }

  template<typename T> inline int rint(const T a) { return (int)sycl::round(a); }
  template<typename T> inline T fmod(const T a, const T b) { return sycl::fmod(a, b); }

  /**
   * @brief Maximum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T max(const T a, const T b) { return a > b ? a : b; }

  /**
   * @brief Minimum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T min(const T &a, const T &b) { return a < b ? a : b; }

  /**
   * @brief Sine calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the sin(a)
   */
  template<typename T> inline T sin(T a) { return sycl::sin(a); }

  /**
   * @brief Cosine calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the cos(a)
   */
  template<typename T> inline T cos(T a) { return sycl::cos(a); }

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
    //*s = sycl::sincos(static_cast<T>(M_PI)*a, c);
    *s = sycl::sinpi(a);
    *c = sycl::cospi(a);
  }

  /**
   * @brief Arc cosine calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the acos(a)
   */
  template<typename T> inline T acos(const T a) { return sycl::acos(a); }

  template<typename T> inline T atan2(const T a, const T b) { return sycl::atan2(a, b); }

  template<typename T> inline T sinh(const T a) { return sycl::sinh(a); }
  template<typename T> inline T cosh(const T a) { return sycl::cosh(a); }

  /**
   * @brief Square root function (sqrt)
   * @param a the argument
   */
  template<typename T>
  inline __host__ __device__ T sqrt(T a)
  {
    return sycl::sqrt(a);
  }

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument
   */
  template<typename T>
  inline __host__ __device__ T rsqrt(T a)
  {
    return sycl::rsqrt(a);
  }

  template<typename T> inline T hypot(const T a, const T b) { return sycl::hypot(a, b); }

  /**
   * @brief Exponential function
   * @param a the argument
   */
  template<typename T>
  inline __host__ __device__ T exp(T a)
  {
    return sycl::exp(a);
  }

  /**
   * @brief Natural log function
   * @param a the argument
   */
  template<typename T>
  inline __host__ __device__ T log(T a)
  {
    return sycl::log(a);
  }

  /*
    @brief Power function
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real pow(real a, real b)
  {
    return sycl::pow(a, b);
  }
  template <typename real> __device__ __host__ inline real pow(real a, int b)
  {
    return sycl::pown(a, b);
  }

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real fpow(real a, int b)
  {
    return sycl::pown(a, b);
  }

  /**
     @brief Optimized division routine on the device
  */
  //inline float fdividef(float a, float b) { return a/b; }
  inline float fdividef(float a, float b) { return sycl::native::divide(a,b); }

}
