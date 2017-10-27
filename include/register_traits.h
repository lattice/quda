#ifndef _REGISTER_TRAITS_H
#define _REGISTER_TRAITS_H

/**
 * @file  register_traits.h
 * @brief Provides precision abstractions and defines the register
 * precision given the storage precision using C++ traits.
 *
 */

#include <quda_internal.h>
#include <generics/ldg.h>
#include <complex_quda.h>
#include <inline_ptx.h>

namespace quda {

  /*
    Here we use traits to define the greater type used for mixing types of computation involving these types
  */
  template<class T, class U> struct PromoteTypeId { typedef T Type; };
  template<> struct PromoteTypeId<complex<float>, float> { typedef complex<float> Type; };
  template<> struct PromoteTypeId<float, complex<float> > { typedef complex<float> Type; };
  template<> struct PromoteTypeId<complex<double>, double> { typedef complex<double> Type; };
  template<> struct PromoteTypeId<double, complex<double> > { typedef complex<double> Type; };
  template<> struct PromoteTypeId<double,int> { typedef double Type; };
  template<> struct PromoteTypeId<int,double> { typedef double Type; };
  template<> struct PromoteTypeId<float,int> { typedef float Type; };
  template<> struct PromoteTypeId<int,float> { typedef float Type; };
  template<> struct PromoteTypeId<double,float> { typedef double Type; };
  template<> struct PromoteTypeId<float,double> { typedef double Type; };

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

  template<typename,typename> struct bridge_mapper { };
  template<> struct bridge_mapper<double2,double2> { typedef double2 type; };
  template<> struct bridge_mapper<double2,float2> { typedef double2 type; };
  template<> struct bridge_mapper<double2,short2> { typedef float2 type; };
  template<> struct bridge_mapper<double2,float4> { typedef double4 type; };
  template<> struct bridge_mapper<double2,short4> { typedef float4 type; };
  template<> struct bridge_mapper<float4,double2> { typedef float2 type; };
  template<> struct bridge_mapper<float4,float4> { typedef float4 type; };
  template<> struct bridge_mapper<float4,short4> { typedef float4 type; };
  template<> struct bridge_mapper<float2,double2> { typedef float2 type; };
  template<> struct bridge_mapper<float2,float2> { typedef float2 type; };
  template<> struct bridge_mapper<float2,short2> { typedef float2 type; };

  template<typename> struct vec_length { static const int value = 0; };
  template<> struct vec_length<double4> { static const int value = 4; };
  template<> struct vec_length<double2> { static const int value = 2; };
  template<> struct vec_length<double> { static const int value = 1; };
  template<> struct vec_length<float4> { static const int value = 4; };
  template<> struct vec_length<float2> { static const int value = 2; };
  template<> struct vec_length<float> { static const int value = 1; };
  template<> struct vec_length<short4> { static const int value = 4; };
  template<> struct vec_length<short2> { static const int value = 2; };
  template<> struct vec_length<short> { static const int value = 1; };

  template<typename, int N> struct vector { };

  template<> struct vector<double, 2> {
    typedef double2 type;
    type a;
    vector(const type &a) { this->a.x = a.x; this->a.y = a.y; }
    operator type() const { return a; }
  };

  template<> struct vector<float, 2> {
    typedef float2 type;
    float2 a;
    vector(const double2 &a) { this->a.x = a.x; this->a.y = a.y; }
    operator float2() const { return a; }
  };

  template<typename> struct scalar { };
  template<> struct scalar<double4> { typedef double type; };
  template<> struct scalar<double3> { typedef double type; };
  template<> struct scalar<double2> { typedef double type; };
  template<> struct scalar<double> { typedef double type; };
  template<> struct scalar<float4> { typedef float type; };
  template<> struct scalar<float3> { typedef float type; };
  template<> struct scalar<float2> { typedef float type; };
  template<> struct scalar<float> { typedef float type; };
  template<> struct scalar<short4> { typedef short type; };
  template<> struct scalar<short3> { typedef short type; };
  template<> struct scalar<short2> { typedef short type; };
  template<> struct scalar<short> { typedef short type; };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };
  template<> struct isHalf<short2>{ static const bool value = true; };
  template<> struct isHalf<short4>{ static const bool value = true; };

  template<typename T1, typename T2> __host__ __device__ inline void copy (T1 &a, const T2 &b) { a = b; }

  template<> __host__ __device__ inline void copy(double &a, const int2 &b) {
#ifdef __CUDA_ARCH__
    a = __hiloint2double(b.y, b.x);
#else
    errorQuda("Undefined");
#endif
  }

  template<> __host__ __device__ inline void copy(double2 &a, const int4 &b) {
#ifdef __CUDA_ARCH__
    a.x = __hiloint2double(b.y, b.x); a.y = __hiloint2double(b.w, b.z);
#else
    errorQuda("Undefined");
#endif
  }

  // specializations for short-float conversion
#define MAX_SHORT_INV 3.051850948e-5
  static inline __host__ __device__ float s2f(const short &a) { return static_cast<float>(a) * MAX_SHORT_INV; }
  static inline __host__ __device__ double s2d(const short &a) { return static_cast<double>(a) * MAX_SHORT_INV; }

  // Fast float to integer round
  __device__ __host__ inline int f2i(float f) {
#ifdef __CUDA_ARCH__
    f += 12582912.0f; return reinterpret_cast<int&>(f);
#else
    return static_cast<int>(f);
#endif
  }

  // Fast double to integer round
  __device__ __host__ inline int d2i(double d) {
#ifdef __CUDA_ARCH__
    d += 6755399441055744.0; return reinterpret_cast<int&>(d);
#else
    return static_cast<int>(d);
#endif
  }

  template<> __host__ __device__ inline void copy(float &a, const short &b) { a = s2f(b); }
  template<> __host__ __device__ inline void copy(short &a, const float &b) { a = f2i(b*MAX_SHORT); }

  template<> __host__ __device__ inline void copy(float2 &a, const short2 &b) {
    a.x = s2f(b.x); a.y = s2f(b.y);
  }

  template<> __host__ __device__ inline void copy(short2 &a, const float2 &b) {
    a.x = f2i(b.x*MAX_SHORT); a.y = f2i(b.y*MAX_SHORT);
  }

  template<> __host__ __device__ inline void copy(float4 &a, const short4 &b) {
    a.x = s2f(b.x); a.y = s2f(b.y); a.z = s2f(b.z); a.w = s2f(b.w);
  }

  template<> __host__ __device__ inline void copy(short4 &a, const float4 &b) {
    a.x = f2i(b.x*MAX_SHORT); a.y = f2i(b.y*MAX_SHORT); a.z = f2i(b.z*MAX_SHORT); a.w = f2i(b.w*MAX_SHORT);
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

  
  template <typename Float, int number> struct VectorType;

  // double precision
  template <> struct VectorType<double, 1>{typedef double type; };
  template <> struct VectorType<double, 2>{typedef double2 type; };
  template <> struct VectorType<double, 4>{typedef double4 type; };

  // single precision
  template <> struct VectorType<float, 1>{typedef float type; };
  template <> struct VectorType<float, 2>{typedef float2 type; };
  template <> struct VectorType<float, 4>{typedef float4 type; };

  // half precision
  template <> struct VectorType<short, 1>{typedef short type; };
  template <> struct VectorType<short, 2>{typedef short2 type; };
  template <> struct VectorType<short, 4>{typedef short4 type; };

  // This trait returns the matching texture type (needed for double precision)
  template <typename Float, int number> struct TexVectorType;

  // double precision
  template <> struct TexVectorType<double, 1>{typedef int2 type; };
  template <> struct TexVectorType<double, 2>{typedef int4 type; };

  // single precision
  template <> struct TexVectorType<float, 1>{typedef float type; };
  template <> struct TexVectorType<float, 2>{typedef float2 type; };
  template <> struct TexVectorType<float, 4>{typedef float4 type; };

  // half precision
  template <> struct TexVectorType<short, 1>{typedef short type; };
  template <> struct TexVectorType<short, 2>{typedef short2 type; };
  template <> struct TexVectorType<short, 4>{typedef short4 type; };

  template <typename VectorType>
    __device__ __host__ inline VectorType vector_load(void *ptr, int idx) {
#define USE_LDG
#if defined(__CUDA_ARCH__) && defined(USE_LDG)
    return __ldg(reinterpret_cast< VectorType* >(ptr) + idx);
#else
    return reinterpret_cast< VectorType* >(ptr)[idx];
#endif
  }

  template <typename VectorType>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const VectorType &value) {
    reinterpret_cast< __restrict__ VectorType* >(ptr)[idx] = value;
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const double2 &value) {
#if defined(__CUDA_ARCH__)
    store_streaming_double2(reinterpret_cast<double2*>(ptr)+idx, value.x, value.y);
#else
    reinterpret_cast<double2*>(ptr)[idx] = value;
#endif
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const float4 &value) {
#if defined(__CUDA_ARCH__)
    store_streaming_float4(reinterpret_cast<float4*>(ptr)+idx, value.x, value.y, value.z, value.w);
#else
    reinterpret_cast<float4*>(ptr)[idx] = value;
#endif
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const float2 &value) {
#if defined(__CUDA_ARCH__)
    store_streaming_float2(reinterpret_cast<float2*>(ptr)+idx, value.x, value.y);
#else
    reinterpret_cast<float2*>(ptr)[idx] = value;
#endif
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const short4 &value) {
#if defined(__CUDA_ARCH__)
    store_streaming_short4(reinterpret_cast<short4*>(ptr)+idx, value.x, value.y, value.z, value.w);
#else
    reinterpret_cast<short4*>(ptr)[idx] = value;
#endif
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const short2 &value) {
#if defined(__CUDA_ARCH__)
    store_streaming_short2(reinterpret_cast<short2*>(ptr)+idx, value.x, value.y);
#else
    reinterpret_cast<short2*>(ptr)[idx] = value;
#endif
  }

  template<bool large_alloc> struct AllocType { };
  template<> struct AllocType<true> { typedef size_t type; };
  template<> struct AllocType<false> { typedef int type; };

} // namespace quda

#endif // _REGISTER_TRAITS_H
