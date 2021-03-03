#pragma once

#include <cmath>

namespace quda {

  
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
      *s = std::sin(a);
      *c = std::cos(a);
    }

 
  /**
   * @brief Combined sin and cos colculation in QUDA NAMESPACE
   *
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments. Device function calls CUDA intrinsic
   *
  template<>
  inline  __host__ __device__ void sincos(const float& a, float * s, float *c)
  {
#ifdef __CUDA_ARCH__
    __sincosf(a,s,c);
#else
    ::sincosf(a,s,c);
#endif
  }
   */ 

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
      return 1.0/std::sqrt(a);
    }


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
     Specialization of Trig functions using floats
   */
  template <>
    struct Trig<false,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return std::atan2(a,b); }
    __device__ __host__ static float Sin(const float &a)
    {
      return std::sin(a);
    }
    __device__ __host__ static float Cos(const float &a)
    {
      return std::cos(a); 
    }

    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
       sincos(a, s, c);
    }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return std::atan2(a,b)/M_PI; }
    __device__ __host__ static float Sin(const float &a)
    {
      return std::sin(a * static_cast<float>(M_PI));
    }

    __device__ __host__ static float Cos(const float &a)
    {
      return std::cos(a * static_cast<float>(M_PI));
    }

    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
      sincos(a * static_cast<float>(M_PI), s, c);
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
    return std::pow(a, b);
  }

  template <typename real> __device__ __host__ inline real fdivide(real a, real b)
  {
    return a / b;
  }

}
