#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include <cmath>
#include <iostream>
// std overloads for host .cpp (nvcc host pass uses math_helper / nvvm-latest)
#if !defined(__CUDACC__)

#include <quadmath.h>

namespace std
{

  inline __float128 sqrt(__float128 x) { return sqrtq(x); }
  inline __float128 abs(__float128 x) { return fabsq(x); }
  inline __float128 cbrt(__float128 x) { return cbrtq(x); }
  inline __float128 sin(__float128 x) { return sinq(x); }
  inline __float128 cos(__float128 x) { return cosq(x); }
  inline __float128 sinh(__float128 x) { return sinhq(x); }
  inline __float128 cosh(__float128 x) { return coshq(x); }
  inline __float128 acos(__float128 x) { return acosq(x); }
  inline __float128 asinh(__float128 x) { return asinhq(x); }
  inline __float128 acosh(__float128 x) { return acoshq(x); }
  inline __float128 fabs(__float128 x) { return fabsq(x); }
  inline __float128 exp(__float128 x) { return expq(x); }
  inline __float128 log(__float128 x) { return logq(x); }
  inline __float128 pow(__float128 x, __float128 y) { return powq(x, y); }
  inline __float128 fmod(__float128 x, __float128 y) { return fmodq(x, y); }
  inline bool isinf(__float128 x) { return isinfq(x); }
  inline bool isnan(__float128 x) { return isnanq(x); }
  inline bool isfinite(__float128 x) { return !isnanq(x) && !isinfq(x); }

  inline std::ostream &operator<<(std::ostream &os, __float128 x);

} // namespace std

#endif

namespace quda
{

  // Host .cpp only; .cu host pass uses math_helper fabs(__float128) dispatch
#if !defined(__CUDACC__)
  inline __float128 fabs(__float128 x) { return fabsq(x); }
#endif

  inline double to_double(__float128 x) { return static_cast<double>(x); }

  inline __float128 real_cast(double x) { return static_cast<__float128>(x); }

  template <typename T> inline __float128 max(const __float128 &a, const T &b)
  {
    __float128 y = static_cast<__float128>(b);
    return a > y ? a : y;
  }

  template <typename T> inline __float128 max(const T &a, const __float128 &b) { return max(b, a); }

  template <typename T> inline __float128 min(const __float128 &a, const T &b)
  {
    __float128 y = static_cast<__float128>(b);
    return a < y ? a : y;
  }

  template <typename T> inline __float128 min(const T &a, const __float128 &b) { return min(b, a); }

} // namespace quda

#if !defined(__CUDACC__)
inline std::ostream &std::operator<<(std::ostream &os, __float128 x)
{
  return os << quda::to_double(x);
}
#endif

#endif // QUDA_USE_QUAD_SCALAR
