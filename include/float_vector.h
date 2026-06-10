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
#include "reproducible_floating_accumulator.hpp"

namespace quda {

#if defined(QUDA_REDUCTION_ALGORITHM_NAIVE)
  using device_reduce_t = reduction_t;
#elif defined(QUDA_REDUCTION_ALGORITHM_KAHAN)
  using device_reduce_t = kahan_t<reduction_t>;
#elif defined(QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE)
  using device_reduce_t = rfa_t<reduction_t>;
  // nvcc does not match get_scalar through the rfa_t alias template
  template <> struct get_scalar<device_reduce_t> { using type = device_reduce_t; };
#endif

  /** Convert a device reduction value to a scalar (host or device). */
  template <typename T = real_t>
  __host__ __device__ inline T reduction_to_scalar(const device_reduce_t &x)
  {
#if defined(QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE)
    return static_cast<T>(x.conv());
#else
    return static_cast<T>(x);
#endif
  }

  /** Convert a device reduction value to host scalar real_t. */
  __host__ inline real_t reduction_to_real(const device_reduce_t &x) { return reduction_to_scalar<real_t>(x); }

  __host__ __device__ inline double2 operator+(const double2 &x, const double2 &y) { return {x.x + y.x, x.y + y.y}; }

  __host__ __device__ inline double3 operator+(const double3 &x, const double3 &y)
  {
    return {x.x + y.x, x.y + y.y, x.z + y.z};
  }

  __host__ __device__ inline double4 operator+(const double4 &x, const double4 &y)
  {
    return {x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w};
  }

  __host__ __device__ inline float2 operator+(const float2 &x, const float2 &y) { return {x.x + y.x, x.y + y.y}; }

#ifdef QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE
  __host__ __device__ inline device_reduce_t operator+(const device_reduce_t &x, const device_reduce_t &y)
  {
    device_reduce_t z = x;
    z.operator+=(y);
    return z;
  }
#endif

  template <typename T, int n>
  __device__ __host__ inline array<T, n> operator+(const array<T, n> &a, const array<T, n> &b)
  {
    array<T, n> c;
#pragma unroll
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    return c;
  }

  template <typename T> constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> zero() { return static_cast<T>(0); }
  template <typename T> constexpr std::enable_if_t<std::is_same_v<T, complex<typename T::value_type>>, T> zero()
  {
    return static_cast<T>(0);
  }

  template <typename T, typename U> using specialize = std::enable_if_t<std::is_same_v<T, U>, U>;

  template <typename T> constexpr specialize<T, double2> zero() { return double2 {0.0, 0.0}; }
  template <typename T> constexpr specialize<T, double3> zero() { return double3 {0.0, 0.0, 0.0}; }
  template <typename T> constexpr specialize<T, double4> zero() { return double4 {0.0, 0.0, 0.0, 0.0}; }

  template <typename T> constexpr specialize<T, float2> zero() { return float2 {0.0f, 0.0f}; }
  template <typename T> constexpr specialize<T, float3> zero() { return float3 {0.0f, 0.0f, 0.0f}; }
  template <typename T> constexpr specialize<T, float4> zero() { return float4 {0.0f, 0.0f, 0.0f, 0.0f}; }

#ifdef QUAD_SUM
  template <typename T> __device__ __host__ inline specialize<T, doubledouble> zero() { return doubledouble(); }
  template <typename T> __device__ __host__ inline specialize<T, doubledouble2> zero() { return doubledouble2(); }
  template <typename T> __device__ __host__ inline specialize<T, doubledouble3> zero() { return doubledouble3(); }
#endif

  template <typename T, int n> __device__ __host__ inline array<T, n> zero()
  {
    array<T, n> v;
#pragma unroll
    for (int i = 0; i < n; i++) v[i] = zero<T>();
    return v;
  }

  // array of arithmetic types specialization
  template <typename T>
  __device__ __host__ inline std::enable_if_t<
    std::is_same_v<T, array<typename T::value_type, T::N>> && std::is_arithmetic_v<typename T::value_type>, T>
  zero()
  {
    return zero<typename T::value_type, T::N>();
  }

  // array of array specialization
  template <typename T>
  __device__ __host__ inline std::enable_if_t<
    std::is_same_v<T, array<array<typename T::value_type::value_type, T::value_type::N>, T::N>>, T>
  zero()
  {
    T v;
#pragma unroll
    for (int i = 0; i < v.size(); i++) v[i] = zero<typename T::value_type>();
    return v;
  }

  // array of complex specialization
  template <typename T>
  __device__
    __host__ inline std::enable_if_t<std::is_same_v<T, array<complex<typename T::value_type::value_type>, T::N>>, T>
    zero()
  {
    T v;
#pragma unroll
    for (int i = 0; i < v.size(); i++) v[i] = zero<typename T::value_type>();
    return v;
  }

  /**
     Container used when we want to track the reference value when
     computing an infinity norm
   */
  template <typename T> struct deviation_t {
    using value_type = T;

    T diff;
    T ref;
  };

  template <class T> struct get_scalar<deviation_t<T>> { using type = typename get_scalar<T>::type; };

  template <typename T> constexpr specialize<T, deviation_t<double>> zero() { return {0.0, 0.0}; }
  template <typename T> constexpr specialize<T, deviation_t<float>> zero() { return {0.0f, 0.0f}; }

  template <typename T> __host__ __device__ inline bool operator>(const deviation_t<T> &a, const deviation_t<T> &b)
  {
    return a.diff > b.diff;
  }

  template <typename T> struct low {
    static constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> value() { return std::numeric_limits<T>::lowest(); }
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

  template <> struct low<doubledouble> {
    static inline __host__ __device__ doubledouble value() { return doubledouble(std::numeric_limits<double>::lowest()); }
  };

  template <typename T> struct high {
    static constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> value() { return std::numeric_limits<T>::max(); }
  };

  template <> struct high<doubledouble> {
    static inline __host__ __device__ doubledouble value() { return doubledouble(std::numeric_limits<double>::max()); }
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
