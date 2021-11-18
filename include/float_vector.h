#pragma once

/**
   @file float_vector.h

   @section DESCRIPTION
   Inline device functions for elementary operations on short vectors, e.g., float4, etc.
*/

#include <complex_quda.h>
#include <register_traits.h>
#include <array.h>

namespace quda {

  __host__ __device__ inline double2 operator+(const double2 &x, const double2 &y)
  {
    return make_double2(x.x + y.x, x.y + y.y);
  }

  __host__ __device__ inline double3 operator+(const double3 &x, const double3 &y)
  {
    return make_double3(x.x + y.x, x.y + y.y, x.z + y.z);
  }

  __host__ __device__ inline double4 operator+(const double4 &x, const double4 &y)
  {
    return make_double4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }

  __host__ __device__ inline float2 operator+(const float2 &x, const float2 &y)
  {
    return make_float2(x.x + y.x, x.y + y.y);
  }

  template <typename T, int n>
  __device__ __host__ inline array<T, n> operator+(const array<T, n> &a, const array<T, n> &b)
  {
    array<T, n> c;
#pragma unroll
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    return c;
  }

  template <typename T> constexpr T zero() { return static_cast<T>(0); }
  template <> constexpr double2 zero() { return double2 {0.0, 0.0}; }
  template <> constexpr double3 zero() { return double3 {0.0, 0.0, 0.0}; }
  template <> constexpr double4 zero() { return double4 {0.0, 0.0, 0.0, 0.0}; }

  template <> constexpr float2 zero() { return float2 {0.0f, 0.0f}; }
  template <> constexpr float3 zero() { return float3 {0.0f, 0.0f, 0.0f}; }
  template <> constexpr float4 zero() { return float4 {0.0f, 0.0f, 0.0f, 0.0f}; }

#ifdef QUAD_SUM
  template <> __device__ __host__ inline doubledouble zero() { return doubledouble(); }
  template <> __device__ __host__ inline doubledouble2 zero() { return doubledouble2(); }
  template <> __device__ __host__ inline doubledouble3 zero() { return doubledouble3(); }
#endif

  template <typename T, int n> __device__ __host__ inline array<T, n> zero()
  {
    array<T, n> v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = zero<T>();
    return v;
  }

  template <typename T> struct RealType {
  };
  template <> struct RealType<double> {
    typedef double type;
  };
  template <> struct RealType<double2> {
    typedef double type;
  };
  template <> struct RealType<complex<double>> {
    typedef double type;
  };
  template <> struct RealType<float> {
    typedef float type;
  };
  template <> struct RealType<float2> {
    typedef float type;
  };
  template <> struct RealType<complex<float>> {
    typedef float type;
  };
  template <> struct RealType<float4> {
    typedef float type;
  };
  template <> struct RealType<short> {
    typedef short type;
  };
  template <> struct RealType<short2> {
    typedef short type;
  };
  template <> struct RealType<complex<short>> {
    typedef short type;
  };
  template <> struct RealType<short4> {
    typedef short type;
  };
  template <> struct RealType<int8_t> {
    typedef int8_t type;
  };
  template <> struct RealType<char2> {
    typedef int8_t type;
  };
  template <> struct RealType<complex<int8_t>> {
    typedef int8_t type;
  };
  template <> struct RealType<char4> {
    typedef int8_t type;
  };

#ifndef __CUDACC_RTC__
  inline std::ostream &operator<<(std::ostream &output, const double2 &a)
  {
    output << "(" << a.x << ", " << a.y << ")";
    return output;
  }

  inline std::ostream &operator<<(std::ostream &output, const double3 &a)
  {
    output << "(" << a.x << ", " << a.y << "," << a.z << ")";
    return output;
  }

  inline std::ostream &operator<<(std::ostream &output, const double4 &a)
  {
    output << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
    return output;
  }
#endif
}
