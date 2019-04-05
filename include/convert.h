#pragma once

/**
 * @file convert.h
 *
 * @section DESCRIPTION 
 * Conversion functions that are used as building blocks for
 * arbitrary field and register ordering.
 */

#include <quda_internal.h> // for maximum short, char traits.

namespace quda
{

  template <typename type> inline int vecLength() { return 0; }

  template <> inline int vecLength<char>() { return 1; }
  template <> inline int vecLength<short>() { return 1; }
  template <> inline int vecLength<float>() { return 1; }
  template <> inline int vecLength<double>() { return 1; }

  template <> inline int vecLength<char2>() { return 2; }
  template <> inline int vecLength<short2>() { return 2; }
  template <> inline int vecLength<float2>() { return 2; }
  template <> inline int vecLength<double2>() { return 2; }

  template <> inline int vecLength<char4>() { return 4; }
  template <> inline int vecLength<short4>() { return 4; }
  template <> inline int vecLength<float4>() { return 4; }
  template <> inline int vecLength<double4>() { return 4; }

  // specializations for short-float conversion
  inline __host__ __device__ float s2f(short a) { return static_cast<float>(a) * fixedInvMaxValue<short>::value; }
  inline __host__ __device__ double s2d(short a) { return static_cast<double>(a) * fixedInvMaxValue<short>::value; }

  // specializations for char-float conversion
  inline __host__ __device__ float c2f(char a) { return static_cast<float>(a) * fixedInvMaxValue<char>::value; }
  inline __host__ __device__ double c2d(char a) { return static_cast<double>(a) * fixedInvMaxValue<char>::value; }

  // specializations for short-float conversion with additional scale factor
  inline __host__ __device__ float s2f(short a, float c)
  {
    return static_cast<float>(a) * (fixedInvMaxValue<short>::value * c);
  }
  inline __host__ __device__ double s2d(short a, double c)
  {
    return static_cast<double>(a) * (fixedInvMaxValue<short>::value * c);
  }

  // specializations for char-float conversion with additional scale factor
  inline __host__ __device__ float c2f(char a, float c)
  {
    return static_cast<float>(a) * (fixedInvMaxValue<char>::value * c);
  }
  inline __host__ __device__ double c2d(char a, double c)
  {
    return static_cast<double>(a) * (fixedInvMaxValue<char>::value * c);
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
    // default is one-2-one conversion, e.g., matching vector lengths and precisions
#pragma unroll
    for (int j = 0; j < N; j++) copyFloatN(x[j], y[j]);
  }

  template <> __device__ inline void convert<float2, short2>(float2 x[], short2 y[], const int N)
  {
#pragma unroll
    for (int j = 0; j < N; j++) x[j] = make_float2(y[j].x, y[j].y);
  }

  template <> __device__ inline void convert<float4, short4>(float4 x[], short4 y[], const int N)
  {
#pragma unroll
    for (int j = 0; j < N; j++) x[j] = make_float4(y[j].x, y[j].y, y[j].z, y[j].w);
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

template <> __device__ inline void convert<short4, float2>(short4 x[], float2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++)
    x[j] = make_short4(f2i(y[2 * j].x), f2i(y[2 * j].y), f2i(y[2 * j + 1].x), f2i(y[2 * j + 1].y));
}

template <> __device__ inline void convert<float2, short4>(float2 x[], short4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_float2(y[j].x, y[j].y);
    x[2 * j + 1] = make_float2(y[j].z, y[j].w);
  }
}

template <> __device__ inline void convert<float4, short2>(float4 x[], short2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_float4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<short2, float4>(short2 x[], float4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_short2(f2i(y[j].x), f2i(y[j].y));
    x[2 * j + 1] = make_short2(f2i(y[j].z), f2i(y[j].w));
  }
}

template <> __device__ inline void convert<short4, double2>(short4 x[], double2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++)
    x[j] = make_short4(d2i(y[2 * j].x), d2i(y[2 * j].y), d2i(y[2 * j + 1].x), d2i(y[2 * j + 1].y));
}

template <> __device__ inline void convert<double2, short4>(double2 x[], short4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_double2(y[j].x, y[j].y);
    x[2 * j + 1] = make_double2(y[j].z, y[j].w);
  }
}

template <> __device__ inline void convert<double4, short2>(double4 x[], short2 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N; j++) x[j] = make_double4(y[2 * j].x, y[2 * j].y, y[2 * j + 1].x, y[2 * j + 1].y);
}

template <> __device__ inline void convert<short2, double4>(short2 x[], double4 y[], const int N)
{
#pragma unroll
  for (int j = 0; j < N / 2; j++) {
    x[2 * j] = make_short2(d2i(y[j].x), d2i(y[j].y));
    x[2 * j + 1] = make_short2(d2i(y[j].z), d2i(y[j].w));
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

} // namespace quda
