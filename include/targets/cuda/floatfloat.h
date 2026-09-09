#pragma once

#include <quda_define.h>

#if defined(QUDA_FPMP_FLOATFLOAT)

#include <cuda/fpmp>       // cuda::std::numeric_limits specialization
#include <cuda/fpmp_math>

#include <ostream>
#include <type_traits>

#if defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_LOW)
using floatfloat = cuda::experimental::fp32mp2_low;
#elif defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_MID)
using floatfloat = cuda::experimental::fp32mp2_mid;
#elif defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_HIGH)
using floatfloat = cuda::experimental::fp32mp2_high;
#else
#error "No QUDA float-float accuracy selected"
#endif

static_assert(sizeof(floatfloat) == sizeof(double));
static_assert(std::is_trivially_copyable_v<floatfloat>);

__device__ __host__ inline floatfloat abs(const floatfloat &a) { return cuda::experimental::fabs(a); }
namespace cuda::experimental
{
  __device__ __host__ inline ::floatfloat abs(const ::floatfloat &a) { return fabs(a); }
} // namespace cuda::experimental

struct floatfloat2 {
  floatfloat x;
  floatfloat y;

  floatfloat2() = default;
  constexpr floatfloat2(const floatfloat2 &) = default;
  constexpr floatfloat2 &operator=(const floatfloat2 &) = default;
  constexpr floatfloat2(const double2 &a) : x(a.x), y(a.y) { }
  constexpr floatfloat2(const floatfloat &x, const floatfloat &y) : x(x), y(y) { }

  __device__ __host__ floatfloat2 &operator+=(const floatfloat2 &a)
  {
    x += a.x;
    y += a.y;
    return *this;
  }
};

__device__ __host__ inline floatfloat2 operator+(const floatfloat2 &a, const floatfloat2 &b)
{
  return floatfloat2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline floatfloat2 operator-(const floatfloat2 &a) { return floatfloat2(-a.x, -a.y); }

namespace quda
{
  __device__ __host__ inline floatfloat2 add2(floatfloat2 a, floatfloat2 b) { return {a.x + b.x, a.y + b.y}; }
  __device__ __host__ inline floatfloat2 mul2(floatfloat2 a, floatfloat2 b) { return {a.x * b.x, a.y * b.y}; }
  __device__ __host__ inline floatfloat2 fma2(floatfloat2 a, floatfloat2 b, floatfloat2 c)
  {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
  }

  inline std::ostream &operator<<(std::ostream &output, const floatfloat &a)
  {
    output << "{" << a.hi() << ", " << a.lo() << "}";
    return output;
  }
} // namespace quda

#endif // QUDA_FPMP_FLOATFLOAT
