#pragma once

template <typename T>
struct vec1 {
  T x;
} __attribute__((aligned(1*sizeof(T))));
template <typename T>
inline vec1<T> make_vec1(T a) { return vec1<T>{a}; }

template <typename T>
struct vec2 {
  T x,y;
} __attribute__((aligned(2*sizeof(T))));
template <typename T>
inline vec2<T> make_vec2(T a, T b) { return vec2<T>{a,b}; }

template <typename T>
struct vec3 {
  T x,y,z;
  inline vec3() {}
  //inline vec3(T a, T b, T c): x(a),y(b),z(c) {}
  constexpr inline vec3(T a): x(a),y(0),z(0) {}
  constexpr inline vec3(T a, T b): x(a),y(b),z(0) {}
  constexpr inline vec3(T a, T b, T c): x(a),y(b),z(c) {}
};
template <typename T, typename A1, typename A2, typename A3>
inline vec3<T> make_vec3(A1 a, A2 b, A3 c) { return vec3<T>{a,b,c}; }
//template <typename T>
//inline vec3<T> vec3<T>::vec3(T a, T b, T c) { return vec3<T>{a,b,c}; }
//inline vec3<unsigned int> vec3<unsigned int>::vec3(unsigned int a, unsigned int b, unsigned int c) { return vec3<unsigned int>{a,b,c}; }

template <typename T>
struct vec4 {
  T x,y,z,w;
} __attribute__((aligned(4*sizeof(T))));
template <typename T>
inline vec4<T> make_vec4(T a, T b, T c, T d) { return vec4<T>{a,b,c,d}; }

typedef vec1<char> char1;
typedef vec2<char> char2;
typedef vec3<char> char3;
typedef vec4<char> char4;
#define make_char1(a) make_vec1<char>(a)
#define make_char2(a,b) make_vec2<char>(a,b)
#define make_char3(a,b,c) make_vec3<char>(a,b,c)
#define make_char4(a,b,c,d) make_vec4<char>(a,b,c,d)

typedef vec1<short> short1;
typedef vec2<short> short2;
typedef vec3<short> short3;
typedef vec4<short> short4;
#define make_short1(a) make_vec1<short>(a)
#define make_short2(a,b) make_vec2<short>(a,b)
#define make_short3(a,b,c) make_vec3<short>(a,b,c)
#define make_short4(a,b,c,d) make_vec4<short>(a,b,c,d)

typedef vec1<int> int1;
typedef vec2<int> int2;
typedef vec3<int> int3;
typedef vec4<int> int4;
#define make_int1(a) make_vec1<int>(a)
#define make_int2(a,b) make_vec2<int>(a,b)
#define make_int3(a,b,c) make_vec3<int>(a,b,c)
#define make_int4(a,b,c,d) make_vec4<int>(a,b,c,d)

typedef vec1<unsigned int> uint1;
typedef vec2<unsigned int> uint2;
typedef vec3<unsigned int> uint3;
typedef vec4<unsigned int> uint4;
#define make_uint1(a) make_vec1<uint>(a)
#define make_uint2(a,b) make_vec2<uint>(a,b)
#define make_uint3(a,b,c) make_vec3<uint>(a,b,c)
//#define make_uint3(a,b,c) uint3{a,b,c}
#define make_uint4(a,b,c,d) make_vec4<uint>(a,b,c,d)

typedef vec1<float> float1;
typedef vec2<float> float2;
typedef vec3<float> float3;
typedef vec4<float> float4;
#define make_float1(a) make_vec1<float>(a)
#define make_float2(a,b) make_vec2<float>(a,b)
//#define make_float3(a,b,c) make_vec3<float>(a,b,c)
#define make_float3(a,b,c) float3{a,b,c}
#define make_float4(a,b,c,d) make_vec4<float>(a,b,c,d)

typedef vec1<double> double1;
typedef vec2<double> double2;
typedef vec3<double> double3;
typedef vec4<double> double4;
#define make_double1(a) make_vec1<double>(a)
#define make_double2(a,b) make_vec2<double>(a,b)
#define make_double3(a,b,c) make_vec3<double>(a,b,c)
#define make_double4(a,b,c,d) make_vec4<double>(a,b,c,d)

typedef vec3<unsigned int> dim3;
//constexpr dim3(unsigned int a) { return make_uint3(a,0,0); }
