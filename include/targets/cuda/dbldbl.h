#pragma once

#include <quda_define.h>

#if defined(QUDA_FPMP_DOUBLEDOUBLE)

#include <cuda/fpmp_math>
#include <math_helper.h>

#include <ostream>
#include <type_traits>

#if defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_LOW)
using doubledouble = cuda::experimental::fp64mp2_low;
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_MID)
using doubledouble = cuda::experimental::fp64mp2_mid;
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_HIGH)
using doubledouble = cuda::experimental::fp64mp2_high;
#else
#error "No QUDA double-double accuracy selected"
#endif

static_assert(sizeof(doubledouble) == 2 * sizeof(double));
static_assert(alignof(doubledouble) == 2 * alignof(double));
static_assert(std::is_trivially_copyable_v<doubledouble>);

// CCCL spells this operation fabs; keep QUDA's existing unqualified abs interface.
__device__ __host__ inline doubledouble abs(const doubledouble &a) { return cuda::experimental::fabs(a); }
namespace cuda::experimental
{
  // Supply the standard spelling in the associated namespace so generic code can find it through ADL.
  __device__ __host__ inline ::doubledouble abs(const ::doubledouble &a) { return fabs(a); }
} // namespace cuda::experimental

struct doubledouble2 {
  doubledouble x;
  doubledouble y;

  doubledouble2() = default;
  constexpr doubledouble2(const doubledouble2 &) = default;
  constexpr doubledouble2 &operator=(const doubledouble2 &) = default;
  constexpr doubledouble2(const double2 &a) : x(a.x), y(a.y) { }
  constexpr doubledouble2(const doubledouble &x, const doubledouble &y) : x(x), y(y) { }

  __device__ __host__ doubledouble2 &operator+=(const doubledouble2 &a)
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

__device__ __host__ inline doubledouble2 operator+(const doubledouble2 &a, const doubledouble2 &b)
{
  return doubledouble2(a.x + b.x, a.y + b.y);
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
} // namespace quda

#else

#include <targets/generic/dbldbl.h>

#endif // QUDA_FPMP_DOUBLEDOUBLE
