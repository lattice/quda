#include <complex_quda.h>

/**
   @file float_vector.h

   @section DESCRIPTION
   Inline device functions for elementary operations on short vectors, e.g., float4, etc.
*/

#pragma once

namespace quda {

  __device__ __host__ inline void zero(double &a) { a = 0.0; }
  __device__ __host__ inline void zero(double2 &a) { a.x = 0.0; a.y = 0.0; }
  __device__ __host__ inline void zero(double3 &a) { a.x = 0.0; a.y = 0.0; a.z = 0.0; }
  __device__ __host__ inline void zero(double4 &a) { a.x = 0.0; a.y = 0.0; a.z = 0.0; a.w = 0.0; }

  __device__ __host__ inline void zero(float &a) { a = 0.0; }
  __device__ __host__ inline void zero(float2 &a) { a.x = 0.0; a.y = 0.0; }
  __device__ __host__ inline void zero(float3 &a) { a.x = 0.0; a.y = 0.0; a.z = 0.0; }
  __device__ __host__ inline void zero(float4 &a) { a.x = 0.0; a.y = 0.0; a.z = 0.0; a.w = 0.0; }

  __host__ __device__ inline double2 operator+(const double2& x, const double2 &y) {
    return make_double2(x.x + y.x, x.y + y.y);
  }

  __host__ __device__ inline double2 operator-(const double2& x, const double2 &y) {
    return make_double2(x.x - y.x, x.y - y.y);
  }

  __host__ __device__ inline float2 operator-(const float2& x, const float2 &y) {
    return make_float2(x.x - y.x, x.y - y.y);
  }

  __host__ __device__ inline float4 operator-(const float4& x, const float4 &y) {
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
  }

  __host__ __device__ inline double3 operator+(const double3& x, const double3 &y) {
    return make_double3(x.x + y.x, x.y + y.y, x.z + y.z);
  }

  __host__ __device__ inline double4 operator+(const double4& x, const double4 &y) {
    return make_double4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }

  __host__ __device__ inline float4 operator*(const float a, const float4 x) {
    float4 y;
    y.x = a*x.x;
    y.y = a*x.y;
    y.z = a*x.z;
    y.w = a*x.w;
    return y;
  }

  __host__ __device__ inline float2 operator*(const float a, const float2 x) {
    float2 y;
    y.x = a*x.x;
    y.y = a*x.y;
    return y;
  }

  __host__ __device__ inline double2 operator*(const double a, const double2 x) {
    double2 y;
    y.x = a*x.x;
    y.y = a*x.y;
    return y;
  }

  __host__ __device__ inline double4 operator*(const double a, const double4 x) {
    double4 y;
    y.x = a*x.x;
    y.y = a*x.y;
    y.z = a*x.z;
    y.w = a*x.w;
    return y;
  }

  __host__ __device__ inline float2 operator+(const float2 x, const float2 y) {
    float2 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    return z;
  }

  __host__ __device__ inline float4 operator+(const float4 x, const float4 y) {
    float4 z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;
    z.w = x.w + y.w;
    return z;
  }

  __host__ __device__ inline float4 operator+=(float4 &x, const float4 y) {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    return x;
  }

  __host__ __device__ inline float2 operator+=(float2 &x, const float2 y) {
    x.x += y.x;
    x.y += y.y;
    return x;
  }

  __host__ __device__  inline double2 operator+=(double2 &x, const double2 y) {
    x.x += y.x;
    x.y += y.y;
    return x;
  }

  __host__ __device__ inline double3 operator+=(double3 &x, const double3 y) {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    return x;
  }

  __host__ __device__ inline double4 operator+=(double4 &x, const double4 y) {
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    return x;
  }

  __host__ __device__ inline float4 operator-=(float4 &x, const float4 y) {
    x.x -= y.x;
    x.y -= y.y;
    x.z -= y.z;
    x.w -= y.w;
    return x;
  }

  __host__ __device__ inline float2 operator-=(float2 &x, const float2 y) {
    x.x -= y.x;
    x.y -= y.y;
    return x;
  }

  __host__ __device__ inline double2 operator-=(double2 &x, const double2 y) {
    x.x -= y.x;
    x.y -= y.y;
    return x;
  }

  __host__ __device__ inline float2 operator*=(float2 &x, const float a) {
    x.x *= a;
    x.y *= a;
    return x;
  }

  __host__ __device__ inline double2 operator*=(double2 &x, const float a) {
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


  /*
    Operations to return the maximium absolute value of a FloatN vector
  */

  __forceinline__ __host__ __device__ float max_fabs(const float4 &c) {
    float a = fmaxf(fabsf(c.x), fabsf(c.y));
    float b = fmaxf(fabsf(c.z), fabsf(c.w));
    return fmaxf(a, b);
  };

  __forceinline__ __host__ __device__ float max_fabs(const float2 &b) {
    return fmaxf(fabsf(b.x), fabsf(b.y));
  };

  __forceinline__ __host__ __device__ double max_fabs(const double4 &c) {
    double a = fmax(fabs(c.x), fabs(c.y));
    double b = fmax(fabs(c.z), fabs(c.w));
    return fmax(a, b);
  };

  __forceinline__ __host__ __device__ double max_fabs(const double2 &b) {
    return fmax(fabs(b.x), fabs(b.y));
  };

  /*
    Precision conversion routines for vector types
  */

  __forceinline__ __host__ __device__ float2 make_FloatN(const double2 &a) {
    return make_float2(a.x, a.y);
}

  __forceinline__ __host__ __device__ float4 make_FloatN(const double4 &a) {
    return make_float4(a.x, a.y, a.z, a.w);
  }

  __forceinline__ __host__ __device__ double2 make_FloatN(const float2 &a) {
    return make_double2(a.x, a.y);
  }

  __forceinline__ __host__ __device__ double4 make_FloatN(const float4 &a) {
    return make_double4(a.x, a.y, a.z, a.w);
  }

  __forceinline__ __host__ __device__ short4 make_shortN(const float4 &a) {
    return make_short4(a.x, a.y, a.z, a.w);
  }

  __forceinline__ __host__ __device__ short2 make_shortN(const float2 &a) {
    return make_short2(a.x, a.y);
  }

  __forceinline__ __host__ __device__ short4 make_shortN(const double4 &a) {
    return make_short4(a.x, a.y, a.z, a.w);
  }

  __forceinline__ __host__ __device__ short2 make_shortN(const double2 &a) {
    return make_short2(a.x, a.y);
  }


  /* Helper functions for converting between float2/double2 and complex */
  template<typename Float2, typename Complex>
    inline Float2 make_Float2(const Complex &a) { return (Float2)0; }

  template<>
    inline double2 make_Float2(const complex<double> &a) { return make_double2( a.real(), a.imag() ); }
  template<>
    inline double2 make_Float2(const complex<float> &a) { return make_double2( a.real(), a.imag() ); }
  template<>
    inline float2 make_Float2(const complex<double> &a) { return make_float2( a.real(), a.imag() ); }
  template<>
    inline float2 make_Float2(const complex<float> &a) { return make_float2( a.real(), a.imag() ); }

    template<>
      inline double2 make_Float2(const std::complex<double> &a) { return make_double2( a.real(), a.imag() ); }
    template<>
      inline double2 make_Float2(const std::complex<float> &a) { return make_double2( a.real(), a.imag() ); }
    template<>
      inline float2 make_Float2(const std::complex<double> &a) { return make_float2( a.real(), a.imag() ); }
    template<>
      inline float2 make_Float2(const std::complex<float> &a) { return make_float2( a.real(), a.imag() ); }


  inline complex<double> make_Complex(const double2 &a) { return complex<double>(a.x, a.y); }
  inline complex<float> make_Complex(const float2 &a) { return complex<float>(a.x, a.y); }

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

}
