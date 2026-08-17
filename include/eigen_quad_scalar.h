#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include "float128_t.h"
#include "float128_math.h"
#include <complex>
#include <Eigen/Core>
#include <target_device.h>

#if defined(QUDA_CUDA_CC)
#include <crt/device_fp128_functions.h>
#endif

namespace quda
{

#if defined(QUDA_CUDA_CC)
  template <bool is_device> struct eigen_fp128_pow_impl;
  template <> struct eigen_fp128_pow_impl<false> {
    __host__ float128_t operator()(float128_t a, float128_t b) { return fp128::pow(a, b); }
  };
  template <> struct eigen_fp128_pow_impl<true> {
    __device__ float128_t operator()(float128_t a, float128_t b) { return __nv_fp128_pow(a, b); }
  };
  inline __host__ __device__ float128_t eigen_fp128_pow(float128_t a, float128_t b)
  {
    return target::dispatch<eigen_fp128_pow_impl>(a, b);
  }
#else
  inline float128_t eigen_fp128_pow(float128_t a, float128_t b) { return fp128::pow(a, b); }
#endif

} // namespace quda

namespace Eigen
{

  using quda::float128_t;

  template <> struct NumTraits<float128_t> : GenericNumTraits<float128_t>
  {
    typedef float128_t Real;
    typedef float128_t NonInteger;
    typedef float128_t Nested;
    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 1, // do not inherit is_signed from numeric_limits<float128_t>;
                    // many standard libraries leave it unspecialized (is_signed=0),
                    // which makes Eigen's numext::abs a no-op and breaks eigensolves.
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 3,
      MulCost = 3,
    };

    __host__ __device__ static inline float128_t epsilon()
    {
      return quda::eigen_fp128_pow(static_cast<float128_t>(2), static_cast<float128_t>(-112));
    }
    __host__ __device__ static inline float128_t dummy_precision()
    {
      return quda::eigen_fp128_pow(static_cast<float128_t>(10), static_cast<float128_t>(-32));
    }
    __host__ __device__ static inline float128_t highest()
    {
      // Finite binary128 max: (2 - 2^-112) * 2^16383
      const float128_t two = static_cast<float128_t>(2);
      return (two - quda::eigen_fp128_pow(two, static_cast<float128_t>(-112)))
        * quda::eigen_fp128_pow(two, static_cast<float128_t>(16383));
    }
    __host__ __device__ static inline float128_t lowest() { return -highest(); }
    static inline int digits10() { return 33; }
  };

  template <> struct NumTraits<std::complex<float128_t>> : NumTraits<float128_t>
  {
    enum { IsComplex = 1 };
    typedef float128_t Real;
  };

  namespace internal
  {

    template <> struct cast_impl<float128_t, float128_t>
    {
      __host__ __device__ static inline float128_t run(const float128_t &x) { return x; }
    };

    template <> struct cast_impl<double, float128_t>
    {
      __host__ __device__ static inline float128_t run(const double &x) { return static_cast<float128_t>(x); }
    };

    template <> struct cast_impl<float128_t, double>
    {
      __host__ __device__ static inline double run(const float128_t &x) { return static_cast<double>(x); }
    };

  } // namespace internal

} // namespace Eigen

#endif // QUDA_USE_QUAD_SCALAR
