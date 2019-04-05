#ifndef _REGISTER_TRAITS_H
#define _REGISTER_TRAITS_H

/**
 * @file  register_traits.h
 * @brief Provides precision abstractions and defines the register
 * precision given the storage precision using C++ traits.
 *
 */

#include <quda_internal.h>
#include <convert.h>
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
    quarter -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
  template<> struct mapper<double> { typedef double type; };
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };
  template<> struct mapper<char> { typedef float type; };

  template<> struct mapper<double2> { typedef double2 type; };
  template<> struct mapper<float2> { typedef float2 type; };
  template<> struct mapper<short2> { typedef float2 type; };
  template<> struct mapper<char2> { typedef float2 type; };

  template<> struct mapper<double4> { typedef double4 type; };
  template<> struct mapper<float4> { typedef float4 type; };
  template<> struct mapper<short4> { typedef float4 type; };
  template<> struct mapper<char4> { typedef float4 type; };

  template<typename,typename> struct bridge_mapper { };
  template<> struct bridge_mapper<double2,double2> { typedef double2 type; };
  template<> struct bridge_mapper<double2,float2> { typedef double2 type; };
  template<> struct bridge_mapper<double2,short2> { typedef float2 type; };
  template<> struct bridge_mapper<double2,char2> { typedef float2 type; };
  template<> struct bridge_mapper<double2,float4> { typedef double4 type; };
  template<> struct bridge_mapper<double2,short4> { typedef float4 type; };
  template<> struct bridge_mapper<double2,char4> { typedef float4 type; };
  template<> struct bridge_mapper<float4,double2> { typedef float2 type; };
  template<> struct bridge_mapper<float4,float4> { typedef float4 type; };
  template<> struct bridge_mapper<float4,short4> { typedef float4 type; };
  template<> struct bridge_mapper<float4,char4> { typedef float4 type; };
  template<> struct bridge_mapper<float2,double2> { typedef float2 type; };
  template<> struct bridge_mapper<float2,float2> { typedef float2 type; };
  template<> struct bridge_mapper<float2,short2> { typedef float2 type; };
  template<> struct bridge_mapper<float2,char2> { typedef float2 type; };

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
  template<> struct vec_length<char4> { static const int value = 4; };
  template<> struct vec_length<char2> { static const int value = 2; };
  template<> struct vec_length<char> { static const int value = 1; };

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
    operator type() const { return a; }
  };

  template<> struct vector<int, 2> {
    typedef int2 type;
    int2 a;
    vector(const int2 &a) { this->a.x = a.x; this->a.y = a.y; }
    operator type() const { return a; }
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
  template<> struct scalar<char4> { typedef char type; };
  template<> struct scalar<char3> { typedef char type; };
  template<> struct scalar<char2> { typedef char type; };
  template<> struct scalar<char> { typedef char type; };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };
  template<> struct isHalf<short2>{ static const bool value = true; };
  template<> struct isHalf<short4>{ static const bool value = true; };

  /* Traits used to determine if a variable is quarter precision or not */
  template< typename T > struct isQuarter{ static const bool value = false; };
  template<> struct isQuarter<char>{ static const bool value = true; };
  template<> struct isQuarter<char2>{ static const bool value = true; };
  template<> struct isQuarter<char4>{ static const bool value = true; };

  /* Traits used to determine if a variable is fixed precision or not */
  template< typename T > struct isFixed{ static const bool value = false; };
  template<> struct isFixed<short>{ static const bool value = true; };
  template<> struct isFixed<short2>{ static const bool value = true; };
  template<> struct isFixed<short4>{ static const bool value = true; };
  template<> struct isFixed<char>{ static const bool value = true; };
  template<> struct isFixed<char2>{ static const bool value = true; };
  template<> struct isFixed<char4>{ static const bool value = true; };

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

  template<> __host__ __device__ inline void copy(float &a, const short &b) { a = s2f(b); }
  template<> __host__ __device__ inline void copy(short &a, const float &b) { a = f2i(b*fixedMaxValue<short>::value); }

  template<> __host__ __device__ inline void copy(float2 &a, const short2 &b) {
    a.x = s2f(b.x); a.y = s2f(b.y);
  }

  template<> __host__ __device__ inline void copy(short2 &a, const float2 &b) {
    a.x = f2i(b.x*fixedMaxValue<short>::value); a.y = f2i(b.y*fixedMaxValue<short>::value);
  }

  template<> __host__ __device__ inline void copy(float4 &a, const short4 &b) {
    a.x = s2f(b.x); a.y = s2f(b.y); a.z = s2f(b.z); a.w = s2f(b.w);
  }

  template<> __host__ __device__ inline void copy(short4 &a, const float4 &b) {
    a.x = f2i(b.x*fixedMaxValue<short>::value); a.y = f2i(b.y*fixedMaxValue<short>::value); a.z = f2i(b.z*fixedMaxValue<short>::value); a.w = f2i(b.w*fixedMaxValue<short>::value);
  }

  template<> __host__ __device__ inline void copy(float &a, const char &b) { a = c2f(b); }
  template<> __host__ __device__ inline void copy(char &a, const float &b) { a = f2i(b*fixedMaxValue<char>::value); }

  template<> __host__ __device__ inline void copy(float2 &a, const char2 &b) {
    a.x = c2f(b.x); a.y = c2f(b.y);
  }

  template<> __host__ __device__ inline void copy(char2 &a, const float2 &b) {
    a.x = f2i(b.x*fixedMaxValue<char>::value); a.y = f2i(b.y*fixedMaxValue<char>::value);
  }

  template<> __host__ __device__ inline void copy(float4 &a, const char4 &b) {
    a.x = c2f(b.x); a.y = c2f(b.y); a.z = c2f(b.z); a.w = c2f(b.w);
  }

  template<> __host__ __device__ inline void copy(char4 &a, const float4 &b) {
    a.x = f2i(b.x*fixedMaxValue<char>::value); a.y = f2i(b.y*fixedMaxValue<char>::value); a.z = f2i(b.z*fixedMaxValue<char>::value); a.w = f2i(b.w*fixedMaxValue<char>::value);
  }

  // specialized variants of the copy function that assumes fixed-point scaling already done
  template <typename T1, typename T2> __host__ __device__ inline void copy_scaled(T1 &a, const T2 &b) { copy(a, b); }

  template <> __host__ __device__ inline void copy_scaled(short4 &a, const float4 &b)
  {
    a.x = f2i(b.x);
    a.y = f2i(b.y);
    a.z = f2i(b.z);
    a.w = f2i(b.w);
  }

  template <> __host__ __device__ inline void copy_scaled(char4 &a, const float4 &b)
  {
    a.x = f2i(b.x);
    a.y = f2i(b.y);
    a.z = f2i(b.z);
    a.w = f2i(b.w);
  }

  template <> __host__ __device__ inline void copy_scaled(short2 &a, const float2 &b)
  {
    a.x = f2i(b.x);
    a.y = f2i(b.y);
  }

  template <> __host__ __device__ inline void copy_scaled(char2 &a, const float2 &b)
  {
    a.x = f2i(b.x);
    a.y = f2i(b.y);
  }

  template <> __host__ __device__ inline void copy_scaled(short &a, const float &b) { a = f2i(b); }

  template <> __host__ __device__ inline void copy_scaled(char &a, const float &b) { a = f2i(b); }

  /**
     @brief Specialized variants of the copy function that include an
     additional scale factor.  Note the scale factor is ignored unless
     the input type (b) is either a short or char vector.
  */
  template <typename T1, typename T2, typename T3>
  __host__ __device__ inline void copy_and_scale(T1 &a, const T2 &b, const T3 &c)
  {
    copy(a, b);
  }

  template <> __host__ __device__ inline void copy_and_scale(float4 &a, const short4 &b, const float &c)
  {
    a.x = s2f(b.x, c);
    a.y = s2f(b.y, c);
    a.z = s2f(b.z, c);
    a.w = s2f(b.w, c);
  }

  template <> __host__ __device__ inline void copy_and_scale(float4 &a, const char4 &b, const float &c)
  {
    a.x = c2f(b.x, c);
    a.y = c2f(b.y, c);
    a.z = c2f(b.z, c);
    a.w = c2f(b.w, c);
  }

  template <> __host__ __device__ inline void copy_and_scale(float2 &a, const short2 &b, const float &c)
  {
    a.x = s2f(b.x, c);
    a.y = s2f(b.y, c);
  }

  template <> __host__ __device__ inline void copy_and_scale(float2 &a, const char2 &b, const float &c)
  {
    a.x = c2f(b.x, c);
    a.y = c2f(b.y, c);
  }

  template <> __host__ __device__ inline void copy_and_scale(float &a, const short &b, const float &c)
  {
    a = s2f(b, c);
  }

  template <> __host__ __device__ inline void copy_and_scale(float &a, const char &b, const float &c) { a = c2f(b, c); }

  /**
     Generic wrapper for Trig functions
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
      sincosf(a * static_cast<float>(M_PI), s, c);
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

  // quarter precision
  template <> struct VectorType<char, 1>{typedef char type; };
  template <> struct VectorType<char, 2>{typedef char2 type; };
  template <> struct VectorType<char, 4>{typedef char4 type; };

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

  // quarter precision
  template <> struct TexVectorType<char, 1>{typedef char type; };
  template <> struct TexVectorType<char, 2>{typedef char2 type; };
  template <> struct TexVectorType<char, 4>{typedef char4 type; };

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
    reinterpret_cast< VectorType* >(ptr)[idx] = value;
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

  // A char4 is the same size as a short2
  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const char4 &value) {
#if defined(__CUDA_ARCH__)

    store_streaming_short2(reinterpret_cast<short2*>(ptr)+idx, reinterpret_cast<const short2*>(&value)->x, reinterpret_cast<const short2*>(&value)->y);
#else
    reinterpret_cast<char4*>(ptr)[idx] = value;
    //reinterpret_cast<short2*>(ptr)[idx] = *reinterpret_cast<const short2*>(&value);
#endif
  }

  template <>
    __device__ __host__ inline void vector_store(void *ptr, int idx, const char2 &value) {
#if defined(__CUDA_ARCH__)
    vector_store(ptr, idx, *reinterpret_cast<const short*>(&value));
    //store_streaming_char2(reinterpret_cast<char2*>(ptr)+idx, reinterpret_cast<const char2*>(&value)->x, reinterpret_cast<const char2*>(&value)->y);
#else
    reinterpret_cast<char2*>(ptr)[idx] = value;
#endif
  }

  template<bool large_alloc> struct AllocType { };
  template<> struct AllocType<true> { typedef size_t type; };
  template<> struct AllocType<false> { typedef int type; };

} // namespace quda

#endif // _REGISTER_TRAITS_H
