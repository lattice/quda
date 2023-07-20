#pragma once

/**
   @file float_vector.h

   @section DESCRIPTION
   Inline device functions for elementary operations on short vectors, e.g., float4, etc.
*/

#include <complex_quda.h>
#include <register_traits.h>
#include <array.h>
#include <limits>
#include <type_traits>
#include "dbldbl.h"

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

  /**
     Container used when we want to track the reference value when
     computing an infinity norm
   */
  template <typename T> struct deviation_t {
    T diff;
    T ref;

    template <typename U> constexpr deviation_t<T> &operator=(const deviation_t<U> &other)
    {
      diff = other.diff;
      ref = other.ref;
      return *this;
    }
  };

  template <class T> struct get_scalar<deviation_t<T>> { using type = typename get_scalar<T>::type; };

  template <typename T> __host__ __device__ inline bool operator>(const deviation_t<T> &a, const deviation_t<T> &b)
  {
    return a.diff > b.diff;
  }

  template <typename T> struct low {
    static constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> value() { return std::numeric_limits<T>::lowest(); }
  };

  template <> struct low<doubledouble> {
    static constexpr doubledouble value() { return std::numeric_limits<double>::lowest(); }
  };

  template <typename T, int N> struct low<array<T, N>> {
    static inline __host__ __device__ array<T, N> value()
    {
      array<T, N> v;
#pragma unroll
      for (int i = 0; i < N; i++) v[i] = low<T>::value();
      return v;
    }
  };

  template <typename T> struct low<deviation_t<T>> {
    static inline __host__ __device__ deviation_t<T> value() { return {low<T>::value(), low<T>::value()}; }
  };

  template <typename T> struct high {
    static constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> value() { return std::numeric_limits<T>::max(); }
  };

  template <> struct high<doubledouble> {
    static constexpr doubledouble value() { return std::numeric_limits<double>::max(); }
  };

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
