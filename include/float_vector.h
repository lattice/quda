#include <complex_quda.h>

/**
   @file float_vector.h

   @section DESCRIPTION
   Inline device functions for elementary operations on short vectors, e.g., float4, etc.
*/

#pragma once

namespace quda {

  __host__ __device__ inline double2 operator+(const double2 &x, const double2 &y)
  {
    return make_double2(x.x + y.x, x.y + y.y);
  }

  __host__ __device__ inline double2 operator-(const double2 &x, const double2 &y)
  {
    return make_double2(x.x - y.x, x.y - y.y);
  }

  __host__ __device__ inline float2 operator-(const float2 &x, const float2 &y)
  {
    return make_float2(x.x - y.x, x.y - y.y);
  }

  __host__ __device__ inline float4 operator-(const float4 &x, const float4 &y)
  {
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
  }

  __host__ __device__ inline float8 operator-(const float8 &x, const float8 &y)
  {
    float8 z;
    z.x = x.x - y.x;
    z.y = x.y - y.y;
    return z;
  }

  __host__ __device__ inline double3 operator+(const double3 &x, const double3 &y)
  {
    return make_double3(x.x + y.x, x.y + y.y, x.z + y.z);
  }

  __host__ __device__ inline double4 operator+(const double4 &x, const double4 &y)
  {
    return make_double4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }

  __host__ __device__ inline float4 operator*(const float &a, const float4 &x)
  {
    float4 y;
    y.x = a*x.x;
    y.y = a*x.y;
    y.z = a*x.z;
    y.w = a*x.w;
    return y;
  }

  __host__ __device__ inline float2 operator*(const float &a, const float2 &x)
  {
    float2 y;
    y.x = a*x.x;
    y.y = a*x.y;
    return y;
  }

  __host__ __device__ inline double2 operator*(const double &a, const double2 &x)
  {
    double2 y;
    y.x = a*x.x;
    y.y = a*x.y;
    return y;
  }

  __host__ __device__ inline double4 operator*(const double &a, const double4 &x)
  {
    double4 y;
    y.x = a*x.x;
    y.y = a*x.y;
    y.z = a*x.z;
    y.w = a*x.w;
    return y;
  }

  __host__ __device__ inline float8 operator*(const float &a, const float8 &x)
  {
    float8 y;
    y.x = a * x.x;
    y.y = a * x.y;
    return y;
  }

  __host__ __device__ inline float2 operator+(const float2 &x, const float2 &y)
  {
    float2 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    return z;
  }

  __host__ __device__ inline float4 operator+(const float4 &x, const float4 &y)
  {
    float4 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;
    z.w = x.w + y.w;
    return z;
  }

  __host__ __device__ inline float8 operator+(const float8 &x, const float8 &y)
  {
    float8 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    return z;
  }

  __host__ __device__ inline float4 operator+=(float4 &x, const float4 &y)
  {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    return x;
  }

  __host__ __device__ inline float2 operator+=(float2 &x, const float2 &y)
  {
    x.x += y.x;
    x.y += y.y;
    return x;
  }

  __host__ __device__ inline float8 operator+=(float8 &x, const float8 &y)
  {
    x.x += y.x;
    x.y += y.y;
    return x;
  }

  __host__ __device__ inline double2 operator+=(double2 &x, const double2 &y)
  {
    x.x += y.x;
    x.y += y.y;
    return x;
  }

  __host__ __device__ inline double3 operator+=(double3 &x, const double3 &y)
  {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    return x;
  }

  __host__ __device__ inline double4 operator+=(double4 &x, const double4 &y)
  {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    return x;
  }

  __host__ __device__ inline float4 operator-=(float4 &x, const float4 &y)
  {
    x.x -= y.x;
    x.y -= y.y;
    x.z -= y.z;
    x.w -= y.w;
    return x;
  }

  __host__ __device__ inline float2 operator-=(float2 &x, const float2 &y)
  {
    x.x -= y.x;
    x.y -= y.y;
    return x;
  }

  __host__ __device__ inline float8 operator-=(float8 &x, const float8 &y)
  {
    x.x -= y.x;
    x.y -= y.y;
    return x;
  }

  __host__ __device__ inline double2 operator-=(double2 &x, const double2 &y)
  {
    x.x -= y.x;
    x.y -= y.y;
    return x;
  }

  __host__ __device__ inline float2 operator*=(float2 &x, const float &a)
  {
    x.x *= a;
    x.y *= a;
    return x;
  }

  __host__ __device__ inline double2 operator*=(double2 &x, const float &a)
  {
    x.x *= a;
    x.y *= a;
    return x;
  }

  __host__ __device__ inline float4 operator*=(float4 &a, const float &b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
  }

  __host__ __device__ inline float8 operator*=(float8 &a, const float &b)
  {
    a.x *= b;
    a.y *= b;
    return a;
  }

  __host__ __device__ inline double2 operator*=(double2 &a, const double &b) {
    a.x *= b;
    a.y *= b;
    return a;
  }

  __host__ __device__ inline double4 operator*=(double4 &a, const double &b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
  }

  __host__ __device__ inline float2 operator-(const float2 &x) {
    return make_float2(-x.x, -x.y);
  }

  __host__ __device__ inline double2 operator-(const double2 &x) {
    return make_double2(-x.x, -x.y);
  }

  template<typename T> struct RealType {};
  template<> struct RealType<double> { typedef double type; };
  template<> struct RealType<double2> { typedef double type; };
  template<> struct RealType<complex<double> > { typedef double type; };
  template<> struct RealType<float> { typedef float type; };
  template<> struct RealType<float2> { typedef float type; };
  template<> struct RealType<complex<float> > { typedef float type; };
  template<> struct RealType<float4> { typedef float type; };
  template<> struct RealType<short> { typedef short type; };
  template<> struct RealType<short2> { typedef short type; };
  template<> struct RealType<complex<short> > { typedef short type; };
  template<> struct RealType<short4> { typedef short type; };
  template<> struct RealType<char> { typedef char type; };
  template<> struct RealType<char2> { typedef char type; };
  template<> struct RealType<complex<char> > { typedef char type; };
  template<> struct RealType<char4> { typedef char type; };

}
