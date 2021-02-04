#pragma once

#include <math.h>

namespace quda {



  
  /**
   * Combined sinc and cos colculation in QUDA NAMESPACE  
   */
  template<typename T>
  inline __host__ __device__ void sincos(const T& a, T* s, T* c)
    {
      ::sincos(a,s,c);
    }

  
  template<>
  inline  __host__ __device__ void sincos(const float& a, float * s, float *c)
  {
#ifdef __CUDA_ARCH__
    __sincosf(a,s,c);
#else
    ::sincosf(a,s,c);
#endif
  }

    /**
   * Combined sinc and cos colculation in QUDA NAMESPACE
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
    __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b); }
    __device__ __host__ static T Sin( const T &a ) { return sin(a); }
    __device__ __host__ static T Cos( const T &a ) { return cos(a); }
    __device__ __host__ static void SinCos(const T &a, T *s, T *c) { sincos(a, s, c); }
  };
  
  /**
     Specialization of Trig functions using floats
   */
  template <>
    struct Trig<false,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return atan2f(a,b); }
    __device__ __host__ static float Sin(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __sinf(a); 
#else
      return sinf(a);
#endif
    }
    __device__ __host__ static float Cos(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __cosf(a); 
#else
      return cosf(a); 
#endif
    }

    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
#ifdef __CUDA_ARCH__
       __sincosf(a, s, c);
#else
       sincosf(a, s, c);
#endif
    }
  };

  /**
     Specialization of Trig functions using fixed b/c gauge reconstructs are -1 -> 1 instead of -Pi -> Pi
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return atan2f(a,b)/M_PI; }
    __device__ __host__ static float Sin(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __sinf(a * static_cast<float>(M_PI));
#else
      return sinf(a * static_cast<float>(M_PI));
#endif
    }

    __device__ __host__ static float Cos(const float &a)
    {
#ifdef __CUDA_ARCH__
      return __cosf(a * static_cast<float>(M_PI));
#else
      return cosf(a * static_cast<float>(M_PI));
#endif
    }

    __device__ __host__ static void SinCos(const float &a, float *s, float *c)
    {
#ifdef __CUDA_ARCH__
      __sincosf(a * static_cast<float>(M_PI), s, c);
#else
      ::sincosf(a * static_cast<float>(M_PI), s, c);
#endif
    }
  };

  

}
