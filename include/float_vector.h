#pragma once

/**
   @file float_vector.h

   @section DESCRIPTION
   Inline device functions for elementary operations on short vectors, e.g., float4, etc.
*/

#include <complex_quda.h>
#include <register_traits.h>

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

  template <typename T> __device__ __host__ inline T zero() { return static_cast<T>(0); }
  template <> __device__ __host__ inline double2 zero() { return make_double2(0.0, 0.0); }
  template <> __device__ __host__ inline double3 zero() { return make_double3(0.0, 0.0, 0.0); }
  template <> __device__ __host__ inline double4 zero() { return make_double4(0.0, 0.0, 0.0, 0.0); }

  template <> __device__ __host__ inline float2 zero() { return make_float2(0.0, 0.0); }
  template <> __device__ __host__ inline float3 zero() { return make_float3(0.0, 0.0, 0.0); }
  template <> __device__ __host__ inline float4 zero() { return make_float4(0.0, 0.0, 0.0, 0.0); }

#ifdef QUAD_SUM
  template <> __device__ __host__ inline doubledouble zero() { return doubledouble(); }
  template <> __device__ __host__ inline doubledouble2 zero() { return doubledouble2(); }
  template <> __device__ __host__ inline doubledouble3 zero() { return doubledouble3(); }
#endif

  /**
     struct which acts as a wrapper to a vector of data.
   */
  template <typename scalar_, int n> struct vector_type {
    using scalar = scalar_;
    scalar data[n];
    __device__ __host__ inline scalar &operator[](int i) { return data[i]; }
    __device__ __host__ inline const scalar &operator[](int i) const { return data[i]; }
    constexpr int size() const { return n; }
    __device__ __host__ inline void operator+=(const vector_type &a)
    {
#pragma unroll
      for (int i = 0; i < n; i++) data[i] += a[i];
    }
    __device__ __host__ vector_type()
    {
#pragma unroll
      for (int i = 0; i < n; i++) data[i] = zero<scalar>();
    }

    vector_type(const vector_type<scalar, n> &) = default;
    vector_type(vector_type<scalar, n> &&) = default;

    template <typename... T> constexpr vector_type(scalar first, const T... data) : data {first, data...} { }

    template <typename... T> constexpr vector_type(const scalar &a)
    {
      for (auto &e : data) e = a;
    }

    vector_type<scalar, n> &operator=(const vector_type<scalar, n> &) = default;
    vector_type<scalar, n> &operator=(vector_type<scalar, n> &&) = default;
  };

  template <typename T, int n> std::ostream &operator<<(std::ostream &output, const vector_type<T, n> &a)
  {
    output << "{ ";
    for (int i = 0; i < n - 1; i++) output << a[i] << ", ";
    output << a[n - 1] << " }";
    return output;
  }

  template <typename scalar, int n> __device__ __host__ inline vector_type<scalar, n> zero()
  {
    vector_type<scalar, n> v;
#pragma unroll
    for (int i = 0; i < n; i++) v.data[i] = zero<scalar>();
    return v;
  }

  template <typename scalar, int n>
  __device__ __host__ inline vector_type<scalar, n> operator+(const vector_type<scalar, n> &a,
                                                              const vector_type<scalar, n> &b)
  {
    vector_type<scalar, n> c;
#pragma unroll
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    return c;
  }
}
