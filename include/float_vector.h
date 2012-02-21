/**
   @file float_vector.h
   @author M Clark

   @section DESCRIPTION 
   Inline device functions for elementary operations on short vectors, e.g., float4, etc. 
 */

#pragma once

__host__ __device__ double2 operator+(const double2& x, const double2 &y) {
  return make_double2(x.x + y.x, x.y + y.y);
}

__host__ __device__ double2 operator-(const double2& x, const double2 &y) {
  return make_double2(x.x - y.x, x.y - y.y);
}

__host__ __device__ float2 operator-(const float2& x, const float2 &y) {
  return make_float2(x.x - y.x, x.y - y.y);
}

__host__ __device__ float4 operator-(const float4& x, const float4 &y) {
  return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
}

__host__ double3 operator+(const double3& x, const double3 &y) {
  double3 z;
  z.x = x.x + y.x; z.y = x.y + y.y; z.z = x.z + y.z;
  return z;
}

__device__ float4 operator*(const float a, const float4 x) {
  float4 y;
  y.x = a*x.x;
  y.y = a*x.y;
  y.z = a*x.z;
  y.w = a*x.w;
  return y;
}

__device__ float2 operator*(const float a, const float2 x) {
  float2 y;
  y.x = a*x.x;
  y.y = a*x.y;
  return y;
}

__device__ double2 operator*(const double a, const double2 x) {
  double2 y;
  y.x = a*x.x;
  y.y = a*x.y;
  return y;
}

__device__ double4 operator*(const double a, const double4 x) {
  double4 y;
  y.x = a*x.x;
  y.y = a*x.y;
  y.z = a*x.z;
  y.w = a*x.w;
  return y;
}

__device__ float2 operator+(const float2 x, const float2 y) {
  float2 z;
  z.x = x.x + y.x;
  z.y = x.y + y.y;
  return z;
}

__device__ float4 operator+(const float4 x, const float4 y) {
  float4 z;
  z.x = x.x + y.x;
  z.y = x.y + y.y;
  z.z = x.z + y.z;
  z.w = x.w + y.w;
  return z;
}

__device__ float4 operator+=(float4 &x, const float4 y) {
  x.x += y.x;
  x.y += y.y;
  x.z += y.z;
  x.w += y.w;
  return x;
}

__device__ float2 operator+=(float2 &x, const float2 y) {
  x.x += y.x;
  x.y += y.y;
  return x;
}

__host__ __device__ double2 operator+=(double2 &x, const double2 y) {
  x.x += y.x;
  x.y += y.y;
  return x;
}

__host__ __device__ double3 operator+=(double3 &x, const double3 y) {
  x.x += y.x;
  x.y += y.y;
  x.z += y.z;
  return x;
}

__device__ float4 operator-=(float4 &x, const float4 y) {
  x.x -= y.x;
  x.y -= y.y;
  x.z -= y.z;
  x.w -= y.w;
  return x;
}

__device__ float2 operator-=(float2 &x, const float2 y) {
  x.x -= y.x;
  x.y -= y.y;
  return x;
}

__device__ double2 operator-=(double2 &x, const double2 y) {
  x.x -= y.x;
  x.y -= y.y;
  return x;
}

__device__ float2 operator*=(float2 &x, const float a) {
  x.x *= a;
  x.y *= a;
  return x;
}

__device__ float4 operator*=(float4 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}

__device__ double2 operator*=(double2 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  return a;
}

__device__ double4 operator*=(double4 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}

__device__ float2 operator-(const float2 &x) {
  return make_float2(-x.x, -x.y);
}

__device__ double2 operator-(const double2 &x) {
  return make_double2(-x.x, -x.y);
}


/*
  Operations to return the maximium absolute value of a FloatN vector
 */

__forceinline__ __device__ float max_fabs(const float4 &c) {
  float a = fmaxf(fabsf(c.x), fabsf(c.y));
  float b = fmaxf(fabsf(c.z), fabsf(c.w));
  return fmaxf(a, b);
};

__forceinline__ __device__ float max_fabs(const float2 &b) {
  return fmaxf(fabsf(b.x), fabsf(b.y));
};

__forceinline__ __device__ double max_fabs(const double4 &c) {
  double a = fmaxf(fabsf(c.x), fabsf(c.y));
  double b = fmaxf(fabsf(c.z), fabsf(c.w));
  return fmaxf(a, b);
};

__forceinline__ __device__ double max_fabs(const double2 &b) {
  return fmaxf(fabsf(b.x), fabsf(b.y));
};

/*
  Precision conversion routines for vector types
 */

__forceinline__ __device__ float2 make_FloatN(const double2 &a) {
  return make_float2(a.x, a.y);
}

__forceinline__ __device__ float4 make_FloatN(const double4 &a) {
  return make_float4(a.x, a.y, a.z, a.w);
}

__forceinline__ __device__ double2 make_FloatN(const float2 &a) {
  return make_double2(a.x, a.y);
}

__forceinline__ __device__ double4 make_FloatN(const float4 &a) {
  return make_double4(a.x, a.y, a.z, a.w);
}

__forceinline__ __device__ short4 make_shortN(const float4 &a) {
  return make_short4(a.x, a.y, a.z, a.w);
}

__forceinline__ __device__ short2 make_shortN(const float2 &a) {
  return make_short2(a.x, a.y);
}

__forceinline__ __device__ short4 make_shortN(const double4 &a) {
  return make_short4(a.x, a.y, a.z, a.w);
}

__forceinline__ __device__ short2 make_shortN(const double2 &a) {
  return make_short2(a.x, a.y);
}

