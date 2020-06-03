#pragma once

/**
 * @file convert.h
 *
 * @section DESCRIPTION 
 * Conversion functions that are used as building blocks for
 * arbitrary field and register ordering.
 */

#include <quda_internal.h> // for maximum short, char traits.
#include <register_traits.h>

namespace quda
{

  inline __host__ __device__ float s2f_(short a_16)
  {
#if 0
    return static_cast<float>(a_16);
#else
    union {
      uint a_32;
      float f_32;
    };
    a_32 = 0x4b7f8000 ^ (a_16 & 0xFFFF);
    return f_32 - 16744448.f;
#endif
  }

  inline __host__ __device__ float c2f_(char a_8)
  {
#if 1
    return static_cast<float>(a_8);
#else
    uint a_32 = 0x4b7fff80;
    a_32 ^= (a_8) & 0xFFFF;
    return reinterpret_cast<float&>(a_32) - 16777088.f;
#endif
  }

  // specializations for short-float conversion
  inline __host__ __device__ float s2f(short a) { return s2f_(a) * fixedInvMaxValue<short>::value; }
  inline __host__ __device__ double s2d(short a) { return s2f_(a) * fixedInvMaxValue<short>::value; }

  // specializations for char-float conversion
  inline __host__ __device__ float c2f(char a) { return c2f_(a) * fixedInvMaxValue<char>::value; }
  inline __host__ __device__ double c2d(char a) { return c2f_(a) * fixedInvMaxValue<char>::value; }

  // specializations for short-float conversion with additional scale factor
  inline __host__ __device__ float s2f(short a, float c)
  {
    return s2f_(a) * (fixedInvMaxValue<short>::value * c);
  }
  inline __host__ __device__ double s2d(short a, double c)
  {
    return s2f_(a) * (fixedInvMaxValue<short>::value * c);
  }

  // specializations for char-float conversion with additional scale factor
  inline __host__ __device__ float c2f(char a, float c)
  {
    return c2f_(a) * (fixedInvMaxValue<char>::value * c);
  }
  inline __host__ __device__ double c2d(char a, double c)
  {
    return c2f_(a) * (fixedInvMaxValue<char>::value * c);
  }

  template <typename FloatN> __device__ inline void copyFloatN(FloatN &a, const FloatN &b) { a = b; }

  // This is emulating the texture normalized return: char
  __device__ inline void copyFloatN(float2 &a, const char2 &b) { a = make_float2(c2f(b.x), c2f(b.y)); }
  __device__ inline void copyFloatN(float4 &a, const char4 &b)
  {
    a = make_float4(c2f(b.x), c2f(b.y), c2f(b.z), c2f(b.w));
  }
  __device__ inline void copyFloatN(double2 &a, const char2 &b) { a = make_double2(c2d(b.x), c2d(b.y)); }
  __device__ inline void copyFloatN(double4 &a, const char4 &b)
  {
    a = make_double4(c2d(b.x), c2d(b.y), c2d(b.z), c2d(b.w));
  }

  // This is emulating the texture normalized return: short
  __device__ inline void copyFloatN(float2 &a, const short2 &b) { a = make_float2(s2f(b.x), s2f(b.y)); }
  __device__ inline void copyFloatN(float4 &a, const short4 &b)
  {
    a = make_float4(s2f(b.x), s2f(b.y), s2f(b.z), s2f(b.w));
  }
  __device__ inline void copyFloatN(double2 &a, const short2 &b) { a = make_double2(s2d(b.x), s2d(b.y)); }
  __device__ inline void copyFloatN(double4 &a, const short4 &b)
  {
    a = make_double4(s2d(b.x), s2d(b.y), s2d(b.z), s2d(b.w));
  }

  __device__ inline void copyFloatN(float2 &a, const double2 &b) { a = make_float2(b.x, b.y); }
  __device__ inline void copyFloatN(double2 &a, const float2 &b) { a = make_double2(b.x, b.y); }
  __device__ inline void copyFloatN(float4 &a, const double4 &b) { a = make_float4(b.x, b.y, b.z, b.w); }
  __device__ inline void copyFloatN(double4 &a, const float4 &b) { a = make_double4(b.x, b.y, b.z, b.w); }

  // Fast float to integer round
  __device__ __host__ inline int f2i(float f)
  {
#ifdef __CUDA_ARCH__
    f += 12582912.0f;
    return reinterpret_cast<int &>(f);
#else
    return static_cast<int>(f);
#endif
  }

  // Fast double to integer round
  __device__ __host__ inline int d2i(double d)
  {
#ifdef __CUDA_ARCH__
    d += 6755399441055744.0;
    return reinterpret_cast<int &>(d);
#else
    return static_cast<int>(d);
#endif
  }

  /* Here we assume that the input data has already been normalized and shifted. */
  __device__ inline void copyFloatN(short2 &a, const float2 &b) { a = make_short2(f2i(b.x), f2i(b.y)); }
  __device__ inline void copyFloatN(short4 &a, const float4 &b)
  {
    a = make_short4(f2i(b.x), f2i(b.y), f2i(b.z), f2i(b.w));
  }
  __device__ inline void copyFloatN(short2 &a, const double2 &b) { a = make_short2(d2i(b.x), d2i(b.y)); }
  __device__ inline void copyFloatN(short4 &a, const double4 &b)
  {
    a = make_short4(d2i(b.x), d2i(b.y), d2i(b.z), d2i(b.w));
  }

  __device__ inline void copyFloatN(char2 &a, const float2 &b) { a = make_char2(f2i(b.x), f2i(b.y)); }
  __device__ inline void copyFloatN(char4 &a, const float4 &b)
  {
    a = make_char4(f2i(b.x), f2i(b.y), f2i(b.z), f2i(b.w));
  }
  __device__ inline void copyFloatN(char2 &a, const double2 &b) { a = make_char2(d2i(b.x), d2i(b.y)); }
  __device__ inline void copyFloatN(char4 &a, const double4 &b)
  {
    a = make_char4(d2i(b.x), d2i(b.y), d2i(b.z), d2i(b.w));
  }

  /* float-8 overloads - these just call the float-4 components */
  __device__ inline void copyFloatN(char8 &a, const float8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(short8 &a, const float8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(char8 &a, const double8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(short8 &a, const double8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(float8 &a, const char8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(double8 &a, const char8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(float8 &a, const short8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(double8 &a, const short8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(float8 &a, const double8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }
  __device__ inline void copyFloatN(double8 &a, const float8 &b)
  {
    copyFloatN(a.x, b.x);
    copyFloatN(a.y, b.y);
  }

  /**
     Convert a vector of type InputType to type OutputType.

     The main current limitation is that there is an implicit assumption
     that N * sizeof(OutputType) / sizeof(InputType) is an integer.  E.g.,
     you cannot convert a vector 9 float2s into a vector of 5 float4s.

     @param x Output vector.
     @param y Input vector.
     @param N Length of output vector.
  */
  template <typename OutputType, typename InputType>
  __device__ inline void convert(OutputType x[], InputType y[], const int N)
  {
    static_assert(vec_length<decltype(x[0])>::value == vec_length<decltype(y[0])>::value, "mismatched vector lengths");
    // default is one-2-one conversion, e.g., matching vector lengths and precisions
#pragma unroll
    for (int j = 0; j < N; j++) copyFloatN(x[j], y[j]);
  }

// 4 <-> 2 vector conversion

template <> __device__ inline void convert<double4, double2>(double4 x[], double2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_double4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<double2, double4>(double2 x[], double4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_double2(y[j].x, y[j].y);
    x[2 * j + 1] = make_double2(y[j].z, y[j].w);
  }
}

template <> __device__ inline void convert<float4, float2>(float4 x[], float2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_float4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<float2, float4>(float2 x[], float4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_float2(y[j].x, y[j].y);
    x[2 * j + 1] = make_float2(y[j].z, y[j].w);
  }
}

template <> __device__ inline void convert<float4, double2>(float4 x[], double2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_float4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<double2, float4>(double2 x[], float4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_double2(y[j].x, y[j].y);
    x[2 * j + 1] = make_double2(y[j].z, y[j].w);
  }
}

template <> __device__ inline void convert<double4, float2>(double4 x[], float2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_double4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<float2, double4>(float2 x[], double4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_float2(y[j].x, y[j].y);
    x[2 * j + 1] = make_float2(y[j].z, y[j].w);
  }
}

template <typename T1, typename T2> __host__ __device__ inline void copy(T1 &a, const T2 &b) { a = b; }

template <> __host__ __device__ inline void copy(double2 &a, const float2 &b)
{
  a.x = b.x;
  a.y = b.y;
}

template <> __host__ __device__ inline void copy(double4 &a, const float4 &b)
{
  a.x = b.x;
  a.y = b.y;
  a.z = b.z;
  a.w = b.w;
}

template <> __host__ __device__ inline void copy(double8 &a, const float8 &b)
{
  copy(a.x, b.x);
  copy(a.y, b.y);
}

template <> __host__ __device__ inline void copy(double &a, const int2 &b)
{
#ifdef __CUDA_ARCH__
  a = __hiloint2double(b.y, b.x);
#else
  errorQuda("Undefined");
#endif
}

template <> __host__ __device__ inline void copy(double2 &a, const int4 &b)
{
#ifdef __CUDA_ARCH__
  a.x = __hiloint2double(b.y, b.x);
  a.y = __hiloint2double(b.w, b.z);
#else
  errorQuda("Undefined");
#endif
}

template <> __host__ __device__ inline void copy(float &a, const short &b) { a = s2f(b); }
template <> __host__ __device__ inline void copy(short &a, const float &b) { a = f2i(b * fixedMaxValue<short>::value); }

template <> __host__ __device__ inline void copy(float2 &a, const short2 &b)
{
  a.x = s2f(b.x);
  a.y = s2f(b.y);
}

template <> __host__ __device__ inline void copy(short2 &a, const float2 &b)
{
  a.x = f2i(b.x * fixedMaxValue<short>::value);
  a.y = f2i(b.y * fixedMaxValue<short>::value);
}

template <> __host__ __device__ inline void copy(float4 &a, const short4 &b)
{
  a.x = s2f(b.x);
  a.y = s2f(b.y);
  a.z = s2f(b.z);
  a.w = s2f(b.w);
}

template <> __host__ __device__ inline void copy(float8 &a, const short8 &b)
{
  copy(a.x, b.x);
  copy(a.y, b.y);
}

template <> __host__ __device__ inline void copy(short4 &a, const float4 &b)
{
  a.x = f2i(b.x * fixedMaxValue<short>::value);
  a.y = f2i(b.y * fixedMaxValue<short>::value);
  a.z = f2i(b.z * fixedMaxValue<short>::value);
  a.w = f2i(b.w * fixedMaxValue<short>::value);
}

template <> __host__ __device__ inline void copy(float &a, const char &b) { a = c2f(b); }
template <> __host__ __device__ inline void copy(char &a, const float &b) { a = f2i(b * fixedMaxValue<char>::value); }

template <> __host__ __device__ inline void copy(float2 &a, const char2 &b)
{
  a.x = c2f(b.x);
  a.y = c2f(b.y);
}

template <> __host__ __device__ inline void copy(char2 &a, const float2 &b)
{
  a.x = f2i(b.x * fixedMaxValue<char>::value);
  a.y = f2i(b.y * fixedMaxValue<char>::value);
}

template <> __host__ __device__ inline void copy(float4 &a, const char4 &b)
{
  a.x = c2f(b.x);
  a.y = c2f(b.y);
  a.z = c2f(b.z);
  a.w = c2f(b.w);
}

template <> __host__ __device__ inline void copy(float8 &a, const char8 &b)
{
  copy(a.x, b.x);
  copy(a.y, b.y);
}

template <> __host__ __device__ inline void copy(char4 &a, const float4 &b)
{
  a.x = f2i(b.x * fixedMaxValue<char>::value);
  a.y = f2i(b.y * fixedMaxValue<char>::value);
  a.z = f2i(b.z * fixedMaxValue<char>::value);
  a.w = f2i(b.w * fixedMaxValue<char>::value);
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

template <> __host__ __device__ inline void copy_and_scale(float &a, const short &b, const float &c) { a = s2f(b, c); }

template <> __host__ __device__ inline void copy_and_scale(float &a, const char &b, const float &c) { a = c2f(b, c); }

} // namespace quda
