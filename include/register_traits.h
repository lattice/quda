#pragma once

/**
 * @file  register_traits.h
 * @brief Provides precision abstractions and defines the register
 * precision given the storage precision using C++ traits.
 *
 */

#include <cstdint>

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
#ifdef QUDA_USE_QUAD_SCALAR
  template <> struct PromoteTypeId<float128_t, double> {
    typedef float128_t type;
  };
  template <> struct PromoteTypeId<double, float128_t> {
    typedef float128_t type;
  };
  template <> struct PromoteTypeId<float128_t, float> {
    typedef float128_t type;
  };
  template <> struct PromoteTypeId<float, float128_t> {
    typedef float128_t type;
  };
  template <> struct PromoteTypeId<float128_t, int> {
    typedef float128_t type;
  };
  template <> struct PromoteTypeId<int, float128_t> {
    typedef float128_t type;
  };
#endif
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
#ifdef QUDA_FPMP_FLOATFLOAT
  // An fp32mp2 accuracy always wins against a narrower builtin type.  Declared
  // per concrete accuracy so that a reduction accuracy differing from the bulk
  // accuracy still promotes correctly.
#define QUDA_FPMP_DECLARE_PROMOTE(FF)                                                                                  \
  template <> struct PromoteTypeId<FF, double> {                                                                       \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<double, FF> {                                                                       \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<FF, float> {                                                                        \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<float, FF> {                                                                        \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<FF, short> {                                                                        \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<short, FF> {                                                                        \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<FF, int8_t> {                                                                       \
    using type = FF;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<int8_t, FF> {                                                                       \
    using type = FF;                                                                                                   \
  };

  QUDA_FPMP_DECLARE_PROMOTE(floatfloat_low)
  QUDA_FPMP_DECLARE_PROMOTE(floatfloat_mid)
  QUDA_FPMP_DECLARE_PROMOTE(floatfloat_high)

#undef QUDA_FPMP_DECLARE_PROMOTE

  // Between two fp32mp2 accuracies the more accurate one wins, so that mixing a
  // low-accuracy field value with a higher-accuracy accumulator promotes up.
#define QUDA_FPMP_DECLARE_PROMOTE_PAIR(LO, HI)                                                                         \
  template <> struct PromoteTypeId<LO, HI> {                                                                           \
    using type = HI;                                                                                                   \
  };                                                                                                                   \
  template <> struct PromoteTypeId<HI, LO> {                                                                           \
    using type = HI;                                                                                                   \
  };

  QUDA_FPMP_DECLARE_PROMOTE_PAIR(floatfloat_low, floatfloat_mid)
  QUDA_FPMP_DECLARE_PROMOTE_PAIR(floatfloat_low, floatfloat_high)
  QUDA_FPMP_DECLARE_PROMOTE_PAIR(floatfloat_mid, floatfloat_high)

#undef QUDA_FPMP_DECLARE_PROMOTE_PAIR
#endif

  /*
    Here we use traits to define the mapping between storage type and
    register type:
    double -> double (or floatfloat when QUDA_FPMP_FLOATFLOAT)
    float -> float
    short -> float
    quarter -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
#ifdef QUDA_FPMP_FLOATFLOAT
  template <> struct mapper<double> {
    using type = floatfloat;
  };
  // mapper is also applied to already-mapped register types, so each fp32mp2
  // accuracy maps to itself
  template <> struct mapper<floatfloat_low> {
    using type = floatfloat_low;
  };
  template <> struct mapper<floatfloat_mid> {
    using type = floatfloat_mid;
  };
  template <> struct mapper<floatfloat_high> {
    using type = floatfloat_high;
  };
#else
  template<> struct mapper<double> { typedef double type; };
#endif
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };
  template <> struct mapper<int8_t> {
    typedef float type;
  };

  /**
     @brief Map a storage or precision type to the register type used
     by kernels that must compute above native storage precision.
     On LOW-accuracy fp32mp2 builds, native double fields store
     fp32mp2_low but use renormalized fp32mp2_mid in registers so
     iterated SU(N) updates stay in the group.  Otherwise this is
     identical to mapper.
     @tparam T Storage or precision type
   */
  template <typename T> struct promote_mapper : mapper<T> {
  };
#if defined(QUDA_FPMP_FLOATFLOAT) && defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_LOW)
  template <> struct promote_mapper<double> {
    using type = floatfloat_mid;
  };
  template <> struct promote_mapper<floatfloat_low> {
    using type = floatfloat_mid;
  };
#endif

  /**
     Map a precision tag to the scalar representation used by native
     field-order storage.  Legacy field orders continue to use the
     precision tag itself as their storage type.
   */
  template <typename T> struct native_store {
    using type = T;
  };
#ifdef QUDA_FPMP_FLOATFLOAT
  template <> struct native_store<double> {
    using type = floatfloat;
  };
#endif
  template <typename T> using native_store_t = typename native_store<T>::type;

  /**
     Apply native_store only for native field orders.  Legacy orders
     keep the caller-provided storage type (IEEE double for
     QUDA_DOUBLE_PRECISION).
   */
  template <typename T, bool is_native> struct order_store {
    using type = T;
  };
#ifdef QUDA_FPMP_FLOATFLOAT
  template <> struct order_store<double, true> {
    using type = floatfloat;
  };
#endif
  template <typename T, bool is_native> using order_store_t = typename order_store<T, is_native>::type;

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };

  /* Traits used to determine if a variable is quarter precision or not */
  template< typename T > struct isQuarter{ static const bool value = false; };
  template <> struct isQuarter<int8_t> {
    static const bool value = true;
  };

  /* Traits used to determine if a variable is fixed precision or not */
  template< typename T > struct isFixed{ static const bool value = false; };
  template<> struct isFixed<short>{ static const bool value = true; };
  template <> struct isFixed<int8_t> {
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

#ifdef QUDA_FPMP_FLOATFLOAT
  // floatfloat is an eight-byte pair.  Prefer f32 vector types over the
  // matching double-width types so the compiler emits better loads and stores.
#define QUDA_FPMP_DECLARE_VECTOR_TYPE(FF)                                                                              \
  template <> struct VectorType<FF, 1> {                                                                               \
    using type = float2;                                                                                               \
  };                                                                                                                   \
  template <> struct VectorType<FF, 2> {                                                                               \
    using type = float4;                                                                                               \
  };                                                                                                                   \
  template <> struct VectorType<FF, 3> {                                                                               \
    using type = double3;                                                                                              \
  };                                                                                                                   \
  template <> struct VectorType<FF, 4> {                                                                               \
    using type = float8;                                                                                               \
  };                                                                                                                   \
  template <> struct VectorType<FF, 8> {                                                                               \
    using type = double8;                                                                                              \
  };

  QUDA_FPMP_DECLARE_VECTOR_TYPE(floatfloat_low)
  QUDA_FPMP_DECLARE_VECTOR_TYPE(floatfloat_mid)
  QUDA_FPMP_DECLARE_VECTOR_TYPE(floatfloat_high)

#undef QUDA_FPMP_DECLARE_VECTOR_TYPE
#endif

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
  template <> struct VectorType<short, 16> {
    typedef float8 type;
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
  template <> struct VectorType<int8_t, 16> {
    typedef double2 type;
  };

  // demote vector type to underlying scalar type
  template <class T, class Enable = void> struct get_scalar;
  template <> struct get_scalar<float> {
    using type = float;
  };
  template <> struct get_scalar<double> {
    using type = double;
  };
  template <> struct get_scalar<double2> {
    using type = double;
  };
#ifdef QUDA_FPMP_DOUBLEDOUBLE
  template <typename T> struct get_scalar<T, std::enable_if_t<is_doubledouble_v<T>>> {
    using type = T;
  };
  template <typename T> struct get_scalar<doubledouble_pair<T>> {
    using type = T;
  };
#else
  template <> struct get_scalar<doubledouble> {
    using type = doubledouble;
  };
#endif
  template <> struct get_scalar<doubledouble2> {
    using type = doubledouble;
  };
#ifdef QUDA_FPMP_FLOATFLOAT
  template <typename T> struct get_scalar<T, std::enable_if_t<is_floatfloat_v<T>>> {
    using type = T;
  };
  template <typename T> struct get_scalar<fpmp2_pair<T>> {
    using type = T;
  };
#endif
#ifdef QUDA_USE_QUAD_SCALAR
  template <> struct get_scalar<float128_t> {
    using type = float128_t;
  };
#endif
  template <> struct get_scalar<complex<float>> {
    using type = float;
  };
  template <> struct get_scalar<complex<double>> {
    using type = double;
  };
#ifdef QUDA_FPMP_FLOATFLOAT
  template <typename T> struct get_scalar<complex<T>, std::enable_if_t<is_floatfloat_v<T>>> {
    using type = T;
  };
#endif
#ifdef QUDA_FPMP_DOUBLEDOUBLE
  template <typename T> struct get_scalar<complex<T>, std::enable_if_t<is_doubledouble_v<T>>> {
    using type = T;
  };
#endif
  template <> struct get_scalar<complex_t> {
    using type = real_t;
  };
  template <class T, int n> struct get_scalar<array<T, n>> {
    using type = typename get_scalar<T>::type;
  };

  template <class T> using get_scalar_t = typename get_scalar<T>::type;

#ifdef QUDA_64BIT_INDEXING
  using index_t = uint64_t;
#else
  using index_t = uint32_t;
#endif

} // namespace quda
