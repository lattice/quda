#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include <complex>
#include <Eigen/Core>

#if defined(__CUDACC__)
#include <crt/device_fp128_functions.h>
#endif
#if !defined(__CUDA_ARCH__)
#include <quadmath.h>
#endif

namespace quda
{

  __host__ __device__ inline __float128 eigen_fp128_pow(__float128 a, __float128 b)
  {
#if defined(__CUDA_ARCH__)
    return __nv_fp128_pow(a, b);
#else
    return powq(a, b);
#endif
  }

} // namespace quda

namespace Eigen
{

  template <> struct NumTraits<__float128> : GenericNumTraits<__float128>
  {
    typedef __float128 Real;
    typedef __float128 NonInteger;
    typedef __float128 Nested;
    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 1, // do not inherit is_signed from numeric_limits<__float128>;
                    // many standard libraries leave it unspecialized (is_signed=0),
                    // which makes Eigen's numext::abs a no-op and breaks eigensolves.
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 3,
      MulCost = 3,
    };

    __host__ __device__ static inline __float128 epsilon()
    {
      return quda::eigen_fp128_pow(static_cast<__float128>(2), static_cast<__float128>(-112));
    }
    __host__ __device__ static inline __float128 dummy_precision()
    {
      return quda::eigen_fp128_pow(static_cast<__float128>(10), static_cast<__float128>(-32));
    }
    __host__ __device__ static inline __float128 highest()
    {
      // Finite binary128 max: (2 - 2^-112) * 2^16383
      const __float128 two = static_cast<__float128>(2);
      return (two - quda::eigen_fp128_pow(two, static_cast<__float128>(-112)))
        * quda::eigen_fp128_pow(two, static_cast<__float128>(16383));
    }
    __host__ __device__ static inline __float128 lowest() { return -highest(); }
    static inline int digits10() { return 33; }
  };

  template <> struct NumTraits<std::complex<__float128>> : NumTraits<__float128>
  {
    enum { IsComplex = 1 };
    typedef __float128 Real;
  };

  namespace internal
  {

    template <> struct cast_impl<__float128, __float128>
    {
      __host__ __device__ static inline __float128 run(const __float128 &x) { return x; }
    };

    template <> struct cast_impl<double, __float128>
    {
      __host__ __device__ static inline __float128 run(const double &x) { return static_cast<__float128>(x); }
    };

    template <> struct cast_impl<__float128, double>
    {
      __host__ __device__ static inline double run(const __float128 &x) { return static_cast<double>(x); }
    };

  } // namespace internal

} // namespace Eigen

#endif // QUDA_USE_QUAD_SCALAR
