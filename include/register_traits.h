#ifndef REGISTER_TRAITS_H
#define REGISTER_TRAITS_H

#include <quda_internal.h>

namespace quda {

  /*
    Here we use traits to define the mapping between storage type and
    register type:
    double -> double
    float -> float
    short -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
  template<> struct mapper<double> { typedef double type; };
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };

  template<> struct mapper<double2> { typedef double2 type; };
  template<> struct mapper<float2> { typedef float2 type; };
  template<> struct mapper<short2> { typedef float2 type; };

  template<> struct mapper<double4> { typedef double4 type; };
  template<> struct mapper<float4> { typedef float4 type; };
  template<> struct mapper<short4> { typedef float4 type; };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };

  template<typename T1, typename T2> __host__ __device__ inline void copy (T1 &a, const T2 &b) { a = b; }

  // specializations for short-float conversion
  template<> __host__ __device__ inline void copy(float &a, const short &b) {
    a = (float)b/MAX_SHORT;
  }
  template<> __host__ __device__ inline void copy(short &a, const float &b) {
    a = (short)(b*MAX_SHORT);
  }

  template<> __host__ __device__ inline void copy(float2 &a, const short2 &b) {
    a.x = (float)b.x/MAX_SHORT;
    a.y = (float)b.y/MAX_SHORT;
  }
  template<> __host__ __device__ inline void copy(short2 &a, const float2 &b) {
    a.x = (short)(b.x*MAX_SHORT);
    a.y = (short)(b.y*MAX_SHORT);
  }

  template<> __host__ __device__ inline void copy(float4 &a, const short4 &b) {
    a.x = (float)b.x/MAX_SHORT;
    a.y = (float)b.y/MAX_SHORT;
    a.z = (float)b.z/MAX_SHORT;
    a.w = (float)b.w/MAX_SHORT;
  }
  template<> __host__ __device__ inline void copy(short4 &a, const float4 &b) {
    a.x = (short)(b.x*MAX_SHORT);
    a.y = (short)(b.y*MAX_SHORT);
    a.z = (short)(b.z*MAX_SHORT);
    a.w = (short)(b.w*MAX_SHORT);
  }

  
  /**
     Generic wrapper for Trig functions
  */
  template <bool isHalf, typename T>
    struct Trig {
      __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b); }
      __device__ __host__ static T Sin( const T &a ) { return sin(a); }
      __device__ __host__ static T Cos( const T &a ) { return cos(a); }
      __device__ __host__ static void SinCos(const T& a, T *s, T *c) { *s = sin(a); *c = cos(a); }
    };
  
  /**
     Specialization of Trig functions using floats
   */
  template <>
    struct Trig<false,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return atan2f(a,b); }
    __device__ __host__ static float Sin( const float &a ) { 
#ifdef __CUDA_ARCH__
      return __sinf(a); 
#else
      return sinf(a);
#endif
    }
    __device__ __host__ static float Cos( const float &a ) { 
#ifdef __CUDA_ARCH__
      return __cosf(a); 
#else
      return cosf(a); 
#endif
    }

    __device__ __host__ static void SinCos(const float& a, float *s, float *c) { 
#ifdef __CUDA_ARCH__
       __sincosf(a, s, c);
#else
       sincosf(a, s, c);
#endif
    }

  };

  /**
     Specialization of Trig functions using shorts
   */
  template <>
    struct Trig<true,float> {
    __device__ __host__ static float Atan2( const float &a, const float &b) { return atan2f(a,b)/M_PI; }
    __device__ __host__ static float Sin( const float &a ) { 
#ifdef __CUDA_ARCH__
      return __sinf(a*M_PI); 
#else
      return sinf(a*M_PI); 
#endif
    }
    __device__ __host__ static float Cos( const float &a ) { 
#ifdef __CUDA_ARCH__
      return __cosf(a*M_PI); 
#else
      return cosf(a*M_PI); 
#endif
    }
  };

  


} // namespace quda

#endif
