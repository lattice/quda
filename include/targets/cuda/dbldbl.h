#pragma once

#include <quda_define.h>

#if defined(QUDA_FPMP_DOUBLEDOUBLE)

#include <cuda/fpmp_math>
#include <math_helper.h>

#include <ostream>
#include <type_traits>

// The three fp64mp2 accuracies are distinct C++ types, so reductions
// (QUDA_REDUCTION_TYPE) can pin low/mid/high independently of the
// `doubledouble` alias (QUDA_FPMP_DOUBLEDOUBLE_ACCURACY).
using doubledouble_low = cuda::experimental::fp64mp2_low;
using doubledouble_mid = cuda::experimental::fp64mp2_mid;
using doubledouble_high = cuda::experimental::fp64mp2_high;

#if defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_LOW)
using doubledouble = doubledouble_low;
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_MID)
using doubledouble = doubledouble_mid;
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_HIGH)
using doubledouble = doubledouble_high;
#else
#error "No QUDA double-double accuracy selected"
#endif

static_assert(sizeof(doubledouble) == 2 * sizeof(double));
static_assert(alignof(doubledouble) == 2 * alignof(double));
static_assert(std::is_trivially_copyable_v<doubledouble>);
static_assert(std::is_trivially_copyable_v<doubledouble_low>);
static_assert(std::is_trivially_copyable_v<doubledouble_mid>);
static_assert(std::is_trivially_copyable_v<doubledouble_high>);

namespace quda
{
  template <typename T> struct is_doubledouble : std::false_type {
  };
  template <cuda::experimental::fpmp2_accuracy Acc>
  struct is_doubledouble<cuda::experimental::fpmp2<double, Acc>> : std::true_type {
  };
  template <typename T> constexpr bool is_doubledouble_v = is_doubledouble<T>::value;
  template <typename T> using enable_if_doubledouble = std::enable_if_t<is_doubledouble_v<T>, int>;
} // namespace quda

// CCCL spells this operation fabs; keep QUDA's existing unqualified abs interface.
template <typename T, quda::enable_if_doubledouble<T> = 0> __device__ __host__ inline T abs(const T &a)
{
  return cuda::experimental::fabs(a);
}
namespace cuda::experimental
{
  template <typename T, quda::enable_if_doubledouble<T> = 0>
  __device__ __host__ inline T abs(const T &a)
  {
    return fabs(a);
  }
} // namespace cuda::experimental

template <typename T> struct doubledouble_pair {
  T x;
  T y;

  doubledouble_pair() = default;
  constexpr doubledouble_pair(const doubledouble_pair &) = default;
  constexpr doubledouble_pair &operator=(const doubledouble_pair &) = default;
  constexpr doubledouble_pair(const double2 &a) : x(a.x), y(a.y) { }
  constexpr doubledouble_pair(const T &x, const T &y) : x(x), y(y) { }

  __device__ __host__ doubledouble_pair &operator+=(const doubledouble_pair &a)
  {
    x += a.x;
    y += a.y;
    return *this;
  }

  __device__ __host__ void print() const
  {
    printf("vec2: (%16.14e + %16.14e) (%16.14e + %16.14e)\n", x.hi(), x.lo(), y.hi(), y.lo());
  }
};

using doubledouble_low2 = doubledouble_pair<doubledouble_low>;
using doubledouble_mid2 = doubledouble_pair<doubledouble_mid>;
using doubledouble_high2 = doubledouble_pair<doubledouble_high>;
using doubledouble2 = doubledouble_pair<doubledouble>;

template <typename T>
__device__ __host__ inline doubledouble_pair<T> operator+(const doubledouble_pair<T> &a, const doubledouble_pair<T> &b)
{
  return {a.x + b.x, a.y + b.y};
}

template <typename T> __device__ __host__ inline doubledouble_pair<T> operator-(const doubledouble_pair<T> &a)
{
  return {-a.x, -a.y};
}

inline std::ostream &operator<<(std::ostream &output, const doubledouble &a)
{
  output << "{" << a.hi() << ", " << a.lo() << "}";
  return output;
}

namespace quda
{
  inline std::ostream &operator<<(std::ostream &output, const doubledouble &a)
  {
    output << "{" << a.hi() << ", " << a.lo() << "}";
    return output;
  }

  template <typename T> __device__ __host__ inline doubledouble_pair<T> add2(doubledouble_pair<T> a, doubledouble_pair<T> b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  template <typename T> __device__ __host__ inline doubledouble_pair<T> mul2(doubledouble_pair<T> a, doubledouble_pair<T> b)
  {
    return {a.x * b.x, a.y * b.y};
  }

  template <typename T>
  __device__ __host__ inline doubledouble_pair<T> fma2(doubledouble_pair<T> a, doubledouble_pair<T> b, doubledouble_pair<T> c)
  {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
  }
} // namespace quda

#else

#include <targets/generic/dbldbl.h>

#endif // QUDA_FPMP_DOUBLEDOUBLE
