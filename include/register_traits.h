#pragma once

/**
 * @file  register_traits.h
 * @brief Provides precision abstractions and defines the register
 * precision given the storage precision using C++ traits.
 *
 */

#include <quda_internal.h>
#include <complex_quda.h>
#include <target_device.h>

namespace quda {

  struct alignas(8) char8 {
    char4 x;
    char4 y;
  };

  struct alignas(16) short8 {
    short4 x;
    short4 y;
  };

  struct alignas(32) float8 {
    float4 x;
    float4 y;
  };

  struct alignas(64) double8 {
    double4 x;
    double4 y;
  };

  /*
    Here we use traits to define the greater type used for mixing types of computation involving these types
  */
  template <class T, class U> struct PromoteTypeId {
    typedef T type;
  };
  template <> struct PromoteTypeId<complex<float>, float> {
    typedef complex<float> type;
  };
  template <> struct PromoteTypeId<float, complex<float>> {
    typedef complex<float> type;
  };
  template <> struct PromoteTypeId<complex<double>, double> {
    typedef complex<double> type;
  };
  template <> struct PromoteTypeId<double, complex<double>> {
    typedef complex<double> type;
  };
  template <> struct PromoteTypeId<double, int> {
    typedef double type;
  };
  template <> struct PromoteTypeId<int, double> {
    typedef double type;
  };
  template <> struct PromoteTypeId<float, int> {
    typedef float type;
  };
  template <> struct PromoteTypeId<int, float> {
    typedef float type;
  };
  template <> struct PromoteTypeId<double, float> {
    typedef double type;
  };
  template <> struct PromoteTypeId<float, double> {
    typedef double type;
  };
  template <> struct PromoteTypeId<double, short> {
    typedef double type;
  };
  template <> struct PromoteTypeId<short, double> {
    typedef double type;
  };
  template <> struct PromoteTypeId<double, int8_t> {
    typedef double type;
  };
  template <> struct PromoteTypeId<int8_t, double> {
    typedef double type;
  };
  template <> struct PromoteTypeId<float, short> {
    typedef float type;
  };
  template <> struct PromoteTypeId<short, float> {
    typedef float type;
  };
  template <> struct PromoteTypeId<float, int8_t> {
    typedef float type;
  };
  template <> struct PromoteTypeId<int8_t, float> {
    typedef float type;
  };
  template <> struct PromoteTypeId<short, int8_t> {
    typedef short type;
  };
  template <> struct PromoteTypeId<int8_t, short> {
    typedef short type;
  };

  /*
    Here we use traits to define the mapping between storage type and
    register type:
    double -> double
    float -> float
    short -> float
    quarter -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
  template<> struct mapper<double> { typedef double type; };
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };
  template <> struct mapper<int8_t> {
    typedef float type;
  };

  template<> struct mapper<double2> { typedef double2 type; };
  template<> struct mapper<float2> { typedef float2 type; };
  template<> struct mapper<short2> { typedef float2 type; };
  template<> struct mapper<char2> { typedef float2 type; };

  template<> struct mapper<double4> { typedef double4 type; };
  template<> struct mapper<float4> { typedef float4 type; };
  template<> struct mapper<short4> { typedef float4 type; };
  template<> struct mapper<char4> { typedef float4 type; };

  template <> struct mapper<double8> {
    typedef double8 type;
  };
  template <> struct mapper<float8> {
    typedef float8 type;
  };
  template <> struct mapper<short8> {
    typedef float8 type;
  };
  template <> struct mapper<char8> {
    typedef float8 type;
  };

  template<typename> struct vec_length { static const int value = 0; };
  template <> struct vec_length<double8> {
    static const int value = 8;
  };
  template<> struct vec_length<double4> { static const int value = 4; };
  template <> struct vec_length<double3> {
    static const int value = 3;
  };
  template<> struct vec_length<double2> { static const int value = 2; };
  template<> struct vec_length<double> { static const int value = 1; };
  template <> struct vec_length<float8> {
    static const int value = 8;
  };
  template<> struct vec_length<float4> { static const int value = 4; };
  template <> struct vec_length<float3> {
    static const int value = 3;
  };
  template<> struct vec_length<float2> { static const int value = 2; };
  template<> struct vec_length<float> { static const int value = 1; };
  template <> struct vec_length<short8> {
    static const int value = 8;
  };
  template<> struct vec_length<short4> { static const int value = 4; };
  template <> struct vec_length<short3> {
    static const int value = 3;
  };
  template<> struct vec_length<short2> { static const int value = 2; };
  template<> struct vec_length<short> { static const int value = 1; };
  template <> struct vec_length<char8> {
    static const int value = 8;
  };
  template<> struct vec_length<char4> { static const int value = 4; };
  template <> struct vec_length<char3> {
    static const int value = 3;
  };
  template<> struct vec_length<char2> { static const int value = 2; };
  template <> struct vec_length<int8_t> {
    static const int value = 1;
  };

  template <> struct vec_length<Complex> {
    static const int value = 2;
  };
  template <> struct vec_length<complex<double>> {
    static const int value = 2;
  };
  template <> struct vec_length<complex<float>> {
    static const int value = 2;
  };
  template <> struct vec_length<complex<short>> {
    static const int value = 2;
  };
  template <> struct vec_length<complex<int8_t>> {
    static const int value = 2;
  };

  template<typename, int N> struct vector { };

  template<> struct vector<double, 2> {
    typedef double2 type;
    type a;
    vector(const type &a) { this->a.x = a.x; this->a.y = a.y; }
    operator type() const { return a; }
  };

  template<> struct vector<float, 2> {
    typedef float2 type;
    float2 a;
    vector(const double2 &a) { this->a.x = a.x; this->a.y = a.y; }
    operator type() const { return a; }
  };

  template<> struct vector<int, 2> {
    typedef int2 type;
    int2 a;
    vector(const int2 &a) { this->a.x = a.x; this->a.y = a.y; }
    operator type() const { return a; }
  };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };
  template<> struct isHalf<short2>{ static const bool value = true; };
  template<> struct isHalf<short4>{ static const bool value = true; };
  template <> struct isHalf<short8> {
    static const bool value = true;
  };

  /* Traits used to determine if a variable is quarter precision or not */
  template< typename T > struct isQuarter{ static const bool value = false; };
  template <> struct isQuarter<int8_t> {
    static const bool value = true;
  };
  template<> struct isQuarter<char2>{ static const bool value = true; };
  template<> struct isQuarter<char4>{ static const bool value = true; };
  template <> struct isQuarter<char8> {
    static const bool value = true;
  };

  /* Traits used to determine if a variable is fixed precision or not */
  template< typename T > struct isFixed{ static const bool value = false; };
  template<> struct isFixed<short>{ static const bool value = true; };
  template<> struct isFixed<short2>{ static const bool value = true; };
  template<> struct isFixed<short4>{ static const bool value = true; };
  template <> struct isFixed<short8> {
    static const bool value = true;
  };
  template <> struct isFixed<int8_t> {
    static const bool value = true;
  };
  template<> struct isFixed<char2>{ static const bool value = true; };
  template<> struct isFixed<char4>{ static const bool value = true; };
  template <> struct isFixed<char8> {
    static const bool value = true;
  };

  template <typename Float, int number> struct VectorType;

  // double precision
  template <> struct VectorType<double, 1>{typedef double type; };
  template <> struct VectorType<double, 2>{typedef double2 type; };
  template <> struct VectorType<double, 3> {
    typedef double3 type;
  };
  template <> struct VectorType<double, 4>{typedef double4 type; };
  template <> struct VectorType<double, 8> {
    typedef double8 type;
  };

  // single precision
  template <> struct VectorType<float, 1>{typedef float type; };
  template <> struct VectorType<float, 2>{typedef float2 type; };
  template <> struct VectorType<float, 3> {
    typedef float3 type;
  };
  template <> struct VectorType<float, 4>{typedef float4 type; };
  template <> struct VectorType<float, 8> {
    typedef float8 type;
  };

  // single precision
  template <> struct VectorType<int, 1> {
    typedef int type;
  };
  template <> struct VectorType<int, 2> {
    typedef int2 type;
  };
  template <> struct VectorType<int, 4> {
    typedef int4 type;
  };

  // half precision
  template <> struct VectorType<short, 1>{typedef short type; };
  template <> struct VectorType<short, 2>{typedef short2 type; };
  template <> struct VectorType<short, 3> {
    typedef short3 type;
  };
  template <> struct VectorType<short, 4>{typedef short4 type; };
  template <> struct VectorType<short, 8> {
    typedef short8 type;
  };

  // quarter precision
  template <> struct VectorType<int8_t, 1> {
    typedef int8_t type;
  };
  template <> struct VectorType<int8_t, 2> {
    typedef char2 type;
  };
  template <> struct VectorType<int8_t, 3> {
    typedef char3 type;
  };
  template <> struct VectorType<int8_t, 4> {
    typedef char4 type;
  };
  template <> struct VectorType<int8_t, 8> {
    typedef char8 type;
  };

  template<bool large_alloc> struct AllocType { };
  template<> struct AllocType<true> { typedef size_t type; };
  template<> struct AllocType<false> { typedef int type; };

} // namespace quda
