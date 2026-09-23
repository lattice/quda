#pragma once

#include <quda_define.h>

#if defined(QUDA_FPMP_FLOATFLOAT)

#include <cuda/fpmp>       // cuda::std::numeric_limits specialization
#include <cuda/fpmp_math>

#include <ostream>
#include <type_traits>

// The three fp32mp2 accuracies are always distinct C++ types, so every trait,
// wrapper and operator below is keyed on the concrete type rather than on the
// accuracy the bulk build happens to select.  `floatfloat` is then only ever an
// alias for one of them, which lets the reduction accuracy
// (QUDA_REDUCTION_TYPE) be chosen independently of the bulk accuracy
// (QUDA_FPMP_FLOATFLOAT_ACCURACY).
using floatfloat_low = cuda::experimental::fp32mp2_low;

/**
   MID/HIGH arithmetic assumes a renormalized (hi, lo) pair.  LOW arithmetic
   does not, so a raw limb copy into MID/HIGH is not a valid conversion.
   These wrappers keep CCCL's layout and operators but renormalize on
   construction from fp32mp2_low.
 */
template <typename CccT> struct fp32mp2_normalized : CccT {
  using base = CccT;
  using CccT::CccT;

  fp32mp2_normalized() = default;
  fp32mp2_normalized(const fp32mp2_normalized &) = default;
  fp32mp2_normalized(fp32mp2_normalized &&) = default;
  fp32mp2_normalized &operator=(const fp32mp2_normalized &) = default;
  fp32mp2_normalized &operator=(fp32mp2_normalized &&) = default;
  using CccT::operator=;

  __host__ __device__ constexpr fp32mp2_normalized(const CccT &x) : CccT(x) { }
  __host__ __device__ constexpr fp32mp2_normalized(float x) : CccT(x) { }
  __host__ __device__ explicit constexpr fp32mp2_normalized(double x) : CccT(x) { }

  // One arithmetic overload so `x = 1 - 2*(t%2)` (int) is not ambiguous
  // between operator=(float) and operator=(double).
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  __host__ __device__ fp32mp2_normalized &operator=(U x)
  {
    if constexpr (std::is_same_v<U, double>) {
      static_cast<CccT &>(*this) = CccT(x);
    } else {
      static_cast<CccT &>(*this) = CccT(static_cast<float>(x));
    }
    return *this;
  }

  __host__ __device__ explicit fp32mp2_normalized(const cuda::experimental::fp32mp2_low &x) :
    CccT(cuda::experimental::renormalize(CccT(x.hi(), x.lo())))
  {
  }

  __host__ __device__ fp32mp2_normalized &operator=(const cuda::experimental::fp32mp2_low &x)
  {
    static_cast<CccT &>(*this) = cuda::experimental::renormalize(CccT(x.hi(), x.lo()));
    return *this;
  }

  // CCCL's operators return the unwrapped type.  Keep arithmetic in the
  // wrapper so traits and complex<> specializations stay on this type.
  friend __host__ __device__ fp32mp2_normalized operator+(fp32mp2_normalized a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) + static_cast<const CccT &>(b));
  }
  friend __host__ __device__ fp32mp2_normalized operator-(fp32mp2_normalized a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) - static_cast<const CccT &>(b));
  }
  friend __host__ __device__ fp32mp2_normalized operator*(fp32mp2_normalized a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) * static_cast<const CccT &>(b));
  }
  friend __host__ __device__ fp32mp2_normalized operator/(fp32mp2_normalized a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) / static_cast<const CccT &>(b));
  }
  friend __host__ __device__ fp32mp2_normalized operator-(fp32mp2_normalized a)
  {
    return fp32mp2_normalized(-static_cast<const CccT &>(a));
  }

  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator+(fp32mp2_normalized a, U b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) + b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator+(U a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(a + static_cast<const CccT &>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator-(fp32mp2_normalized a, U b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) - b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator-(U a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(a - static_cast<const CccT &>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator*(fp32mp2_normalized a, U b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) * b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator*(U a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(a * static_cast<const CccT &>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator/(fp32mp2_normalized a, U b)
  {
    return fp32mp2_normalized(static_cast<const CccT &>(a) / b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ fp32mp2_normalized operator/(U a, fp32mp2_normalized b)
  {
    return fp32mp2_normalized(a / static_cast<const CccT &>(b));
  }

  // Prefer wrapper vs arithmetic over converting the wrapper to double
  // (CCCL) and matching the built-in operators.
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator==(fp32mp2_normalized a, U b)
  {
    return static_cast<const CccT &>(a) == CccT(static_cast<std::conditional_t<std::is_same_v<U, double>, double, float>>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator==(U a, fp32mp2_normalized b)
  {
    return b == a;
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator!=(fp32mp2_normalized a, U b)
  {
    return !(a == b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator!=(U a, fp32mp2_normalized b)
  {
    return !(b == a);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator<(fp32mp2_normalized a, U b)
  {
    return static_cast<const CccT &>(a)
      < CccT(static_cast<std::conditional_t<std::is_same_v<U, double>, double, float>>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator<(U a, fp32mp2_normalized b)
  {
    return b > a;
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator>(fp32mp2_normalized a, U b)
  {
    return static_cast<const CccT &>(a)
      > CccT(static_cast<std::conditional_t<std::is_same_v<U, double>, double, float>>(b));
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator>(U a, fp32mp2_normalized b)
  {
    return b < a;
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator<=(fp32mp2_normalized a, U b)
  {
    return !(a > b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator<=(U a, fp32mp2_normalized b)
  {
    return !(b < a);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator>=(fp32mp2_normalized a, U b)
  {
    return !(a < b);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  friend __host__ __device__ bool operator>=(U a, fp32mp2_normalized b)
  {
    return !(b > a);
  }
};

using floatfloat_mid = fp32mp2_normalized<cuda::experimental::fp32mp2_mid>;
using floatfloat_high = fp32mp2_normalized<cuda::experimental::fp32mp2_high>;

#if defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_LOW)
using floatfloat = floatfloat_low;
#elif defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_MID)
using floatfloat = floatfloat_mid;
#elif defined(QUDA_FPMP_FLOATFLOAT_ACCURACY_HIGH)
using floatfloat = floatfloat_high;
#else
#error "No QUDA float-float accuracy selected"
#endif

static_assert(sizeof(floatfloat) == sizeof(double));
static_assert(std::is_trivially_copyable_v<floatfloat>);
static_assert(std::is_trivially_copyable_v<floatfloat_mid>);
static_assert(std::is_trivially_copyable_v<floatfloat_high>);

namespace quda
{
  /** True for any fp32mp2 accuracy, not just the one aliased by floatfloat. */
  template <typename T> struct is_floatfloat : std::false_type {
  };
  template <cuda::experimental::fpmp2_accuracy Acc>
  struct is_floatfloat<cuda::experimental::fpmp2<float, Acc>> : std::true_type {
  };
  template <typename CccT> struct is_floatfloat<fp32mp2_normalized<CccT>> : is_floatfloat<CccT> {
  };
  template <typename T> constexpr bool is_floatfloat_v = is_floatfloat<T>::value;

  /** CCCL type that carries numeric_limits / math overloads for T. */
  template <typename T> struct floatfloat_cccl {
    using type = T;
  };
  template <typename CccT> struct floatfloat_cccl<fp32mp2_normalized<CccT>> {
    using type = CccT;
  };
  template <typename T> using floatfloat_cccl_t = typename floatfloat_cccl<T>::type;

  template <typename T> using enable_if_floatfloat = std::enable_if_t<is_floatfloat_v<T>, int>;
} // namespace quda

template <typename T, quda::enable_if_floatfloat<T> = 0> __device__ __host__ inline T abs(const T &a)
{
  using ccc_t = quda::floatfloat_cccl_t<T>;
  return T(cuda::experimental::fabs(static_cast<const ccc_t &>(a)));
}

/** Two-component container mirroring double2 for an fp32mp2 accuracy. */
template <typename T> struct fpmp2_pair {
  T x;
  T y;

  fpmp2_pair() = default;
  constexpr fpmp2_pair(const fpmp2_pair &) = default;
  constexpr fpmp2_pair &operator=(const fpmp2_pair &) = default;
  constexpr fpmp2_pair(const double2 &a) : x(T(a.x)), y(T(a.y)) { }
  constexpr fpmp2_pair(const T &x, const T &y) : x(x), y(y) { }

  __device__ __host__ fpmp2_pair &operator+=(const fpmp2_pair &a)
  {
    x += a.x;
    y += a.y;
    return *this;
  }
};

using floatfloat_low2 = fpmp2_pair<floatfloat_low>;
using floatfloat_mid2 = fpmp2_pair<floatfloat_mid>;
using floatfloat_high2 = fpmp2_pair<floatfloat_high>;
using floatfloat2 = fpmp2_pair<floatfloat>;

template <typename T> __device__ __host__ inline fpmp2_pair<T> operator+(const fpmp2_pair<T> &a, const fpmp2_pair<T> &b)
{
  return {a.x + b.x, a.y + b.y};
}

template <typename T> __device__ __host__ inline fpmp2_pair<T> operator-(const fpmp2_pair<T> &a)
{
  return {-a.x, -a.y};
}

namespace quda
{
  template <typename T> __device__ __host__ inline fpmp2_pair<T> add2(fpmp2_pair<T> a, fpmp2_pair<T> b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  template <typename T> __device__ __host__ inline fpmp2_pair<T> mul2(fpmp2_pair<T> a, fpmp2_pair<T> b)
  {
    return {a.x * b.x, a.y * b.y};
  }

  template <typename T>
  __device__ __host__ inline fpmp2_pair<T> fma2(fpmp2_pair<T> a, fpmp2_pair<T> b, fpmp2_pair<T> c)
  {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
  }

  template <typename T, enable_if_floatfloat<T> = 0> inline std::ostream &operator<<(std::ostream &output, const T &a)
  {
    output << "{" << a.hi() << ", " << a.lo() << "}";
    return output;
  }
} // namespace quda

#endif // QUDA_FPMP_FLOATFLOAT
