#pragma once

#if defined(__CUDACC__)

#include <cmath>
#include <type_traits>
#include <target_device.h>

#if defined(__CUDACC__) || defined(_NVHPC_CUDA) || (defined(__clang__) && defined(__CUDA__))
#define QUDA_CUDA_CC
#endif

#if (CUDA_VERSION >= 11070) && defined(QUDA_CUDA_CC) && !defined(_NVHPC_CUDA)
#define BUILTIN_ASSUME(x) \
  bool p = x;             \
  __builtin_assume(p);
#else
#define BUILTIN_ASSUME(x)
#endif

#define FTZ

namespace quda {

  inline __host__ __device__ float abs(const float a) { return fabsf(a); }
  inline __host__ __device__ double abs(const double a) { return ::fabs(a); }

  template <typename T>
  __host__ __device__ inline std::enable_if_t<std::is_same_v<T, float>, float> fabs(const T a)
  {
    return fabsf(a);
  }
  template <typename T>
  __host__ __device__ inline std::enable_if_t<std::is_same_v<T, double>, double> fabs(const T a)
  {
    return ::fabs(a);
  }

  template <bool is_device> struct sqrt_impl {
    template <typename T> inline __host__ __device__ T operator()(const T a) { return ::sqrt(a); }
  };

  /** Device float sqrt via PTX sqrt.approx.ftz.f32 (matches fast math / avoids denormal range sequence). */
  template <> struct sqrt_impl<true> {
    template <typename T> __device__ inline std::enable_if_t<std::is_same_v<T, float>, float> operator()(const T a)
    {
      float s;
#ifdef FTZ
      asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(a));
#else
      asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(s) : "f"(a));
#endif
      return s;
    }
    template <typename T> __device__ inline std::enable_if_t<!std::is_same_v<T, float>, T> operator()(const T a)
    {
      return ::sqrt(a);
    }
  };

  template <typename T> inline __host__ __device__ T sqrt(const T a) { return target::dispatch<sqrt_impl>(a); }

  template <typename T> inline __host__ __device__ T exp(const T a) { return ::exp(a); }
  template <typename T> inline __host__ __device__ T log(const T a) { return ::log(a); }
  template <typename T> inline __host__ __device__ T sin(const T a) { return ::sin(a); }
  template <typename T> inline __host__ __device__ T cos(const T a) { return ::cos(a); }
  template <typename T> inline __host__ __device__ T sinh(const T a) { return ::sinh(a); }
  template <typename T> inline __host__ __device__ T cosh(const T a) { return ::cosh(a); }
  template <typename T> inline __host__ __device__ T acos(const T a) { return ::acos(a); }
  template <typename T> inline __host__ __device__ T acosh(const T a) { return ::acosh(a); }
  template <typename T> inline __host__ __device__ T asinh(const T a) { return ::asinh(a); }
  template <typename T> inline __host__ __device__ T cbrt(const T a) { return ::cbrt(a); }
  inline __host__ __device__ bool isnan(float a) { return ::isnan(a); }
  inline __host__ __device__ bool isnan(double a) { return ::isnan(a); }
  template <typename T> inline __host__ __device__ T pow(const T a, const int b) { return ::pow(a, b); }
  template <typename T> inline __host__ __device__ T pow(const T a, const T b) { return ::pow(a, b); }
  template <typename T> inline __host__ __device__ T fmod(const T a, const T b) { return ::fmod(a, b); }

  /**
   * @brief Maximum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T max(const T &a, const T &b) { return a > b ? a : b; }

  /**
   * @brief Maximum of two numbers (float specialization)
   * @param a first number
   * @param b second number
   */
  template <> inline __host__ __device__ float max(const float &a, const float &b) { return fmaxf(a, b); }

  /**
   * @brief Minimum of two numbers
   * @param a first number
   * @param b second number
   */
  template<typename T>
  inline __host__ __device__ T min(const T &a, const T &b) { return a < b ? a : b; }


  template <bool is_device> struct sincos_impl {
    template <typename T> inline void operator()(const T& a, T *s, T *c) { ::sincos(a, s, c); }
  };

#ifdef QUDA_CUDA_CC
  template <> struct sincos_impl<true> {
    template <typename T> __device__ inline void operator()(const T& a, T *s, T *c)
    {
      BUILTIN_ASSUME(fabs(a) <= 2.0 * M_PI);
      sincos(a, s, c);
    }
  };
#endif

  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline __host__ __device__ void sincos(const T& a, T *s, T *c) { target::dispatch<sincos_impl>(a, s, c); }

  template <bool is_device> struct sincosf_impl {
    inline void operator()(const float& a, float *s, float *c) { ::sincosf(a, s, c); }
  };

#ifdef QUDA_CUDA_CC
  template <> struct sincosf_impl<true> {
    __device__ inline void operator()(const float& a, float *s, float *c) { __sincosf(a, s, c); }
  };
#endif

  /**
   * @brief Combined sin and cos calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments. Device function calls CUDA intrinsic
   */
  template<>
  inline __host__ __device__ void sincos(const float& a, float *s, float *c) { target::dispatch<sincosf_impl>(a, s, c); }


  template <bool is_device> struct sincospi_impl {
    template <typename T> inline void operator()(const T& a, T *s, T *c) { ::sincos(a * static_cast<T>(M_PI), s, c); }
  };

#ifdef QUDA_CUDA_CC
  template <> struct sincospi_impl<true> {
    template <typename T> __device__ inline void operator()(const T& a, T *s, T *c) { sincospi(a, s, c); }
  };
#endif

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   */
  template<typename T>
  inline __host__ __device__ void sincospi(const T& a, T *s, T *c) { target::dispatch<sincospi_impl>(a, s, c); }

  /**
   * @brief Combined sinpi and cospi calculation in QUDA NAMESPACE
   * @param a the angle
   * @param s pointer to the storage for the result of the sin
   * @param c pointer to the storage for the result of the cos
   *
   * Specialization to float arguments.  Use sincos so that Device function calls CUDA intrinsic.
   */
  template<>
  inline __host__ __device__ void sincospi(const float& a, float *s, float *c) { quda::sincos(a * static_cast<float>(M_PI), s, c); }

  template <bool is_device> struct sinpi_impl {
    template <typename T> inline T operator()(T a) { return ::sin(a * static_cast<T>(M_PI)); }
  };
#ifdef QUDA_CUDA_CC
  template <> struct sinpi_impl<true> { template <typename T> __device__ inline T operator()(T a) { return ::sinpi(a); } };
#endif

  /**
   * @brief Sine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the sin(a * pi)
   */
  template<typename T> inline __host__ __device__ T sinpi(T a) { return target::dispatch<sinpi_impl>(a); }


  template <bool is_device> struct sinpif_impl { inline float operator()(float a) { return ::sinf(a * static_cast<float>(M_PI)); } };
#ifdef QUDA_CUDA_CC
  template <> struct sinpif_impl<true> { __device__ inline float operator()(float a) { return __sinf(a * static_cast<float>(M_PI)); } };
#endif

  /**
   * @brief Sine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the sin(a * pi)
   *
   * Specialization to float.  Device function will call CUDA intrinsic.
   */
  template<> inline __host__ __device__ float sinpi(float a) { return target::dispatch<sinpif_impl>(a); }

  template <bool is_device> struct cospi_impl {
    template <typename T> inline T operator()(T a) { return ::cos(a * static_cast<T>(M_PI)); }
  };
#ifdef QUDA_CUDA_CC
  template <> struct cospi_impl<true> {
    template <typename T> __device__ inline T operator()(T a) { return ::cospi(a); }
  };
#endif

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE
   * @param a the angle
   * @return result of the cos(a * pi)
   */
  template<typename T> inline __host__ __device__ T cospi(T a) { return target::dispatch<cospi_impl>(a); }


  template <bool is_device> struct cospif_impl { inline float operator()(float a) { return ::cosf(a * static_cast<float>(M_PI)); } };
#ifdef QUDA_CUDA_CC
  template <> struct cospif_impl<true> { __device__ inline float operator()(float a) { return __cosf(a * static_cast<float>(M_PI)); } };
#endif

  /**
   * @brief Cosine pi calculation in QUDA NAMESPACE.
   * @param a the angle
   * @return result of the cos(a * pi)
   *
   * Specialization to float.  Device function will call CUDA intrinsic.
   */
  template<> inline __host__ __device__ float cospi(float a) { return target::dispatch<cospif_impl>(a); }


  template <bool is_device> struct rsqrt_impl {
    template <typename T> inline T operator()(T a) { return static_cast<T>(1.0) / sqrt(a); }
  };

#ifdef QUDA_CUDA_CC
  template <> struct rsqrt_impl<true> {
    template <typename T> __device__ inline T operator()(T a) { return ::rsqrt(a); }
  };
#endif

  /**
   * @brief Reciprocal square root function (rsqrt)
   * @param a the argument  (In|out)
   *
   * some math libraries provide a fast inverse sqrt() function.
   * this implementation uses the CUDA builtins
   */
  template<typename T> inline __host__ __device__ T rsqrt(T a) { return target::dispatch<rsqrt_impl>(a); }


  template <bool is_device> struct fpow_impl { template <typename real> inline real operator()(real a, int b) { return std::pow(a, b); } };

#ifdef QUDA_CUDA_CC
  template <> struct fpow_impl<true> {
    __device__ inline double operator()(double a, int b) { return ::pow(a, b); }

    __device__ inline float operator()(float a, int b)
    {
      float sign = signbit(a) ? -1.0f : 1.0f;
      float power = __powf(fabsf(a), b);
      return b & 1 ? sign * power : power;
    }
  };
#endif

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real fpow(real a, int b) { return target::dispatch<fpow_impl>(a, b); }

  template <bool is_device> struct fdivide_impl {
    inline float operator()(float a, float b) { return a / b; }
    inline double operator()(double a, double b) { return a / b; }
  };

  /** Device float divide via PTX div.approx{.ftz}.f32 (same FTZ toggle as quda::sqrt; avoids long rcp range prologue). */
  template <> struct fdivide_impl<true> {
    __device__ inline float operator()(float a, float b)
    {
      float d;
#ifdef FTZ
      asm volatile("div.approx.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
#else
      asm volatile("div.approx.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
#endif
      return d;
    }
    __device__ inline double operator()(double a, double b) { return a / b; }
  };

  /**
   * @brief Fast division on host/device (float uses device PTX on CUDA GPU).
   */
  __device__ __host__ inline float fdivide(float a, float b) { return target::dispatch<fdivide_impl>(a, b); }
  __device__ __host__ inline double fdivide(double a, double b) { return target::dispatch<fdivide_impl>(a, b); }

  template <bool is_device> struct ffma2_impl {
    inline float2 operator()(float2 a, float2 b, float2 c)
    {
      return {std::fmaf(a.x, b.x, c.x), std::fmaf(a.y, b.y, c.y)};
    }
  };

  template <> struct ffma2_impl<true> {
    __device__ inline float2 operator()(float2 a, float2 b, float2 c)
    {
#ifdef QUDA_VECTORIZE_SINGLE
      if constexpr (target::vectorize<float>())
        return __ffma2_rn(a, b, c);
      else
#endif
        return {__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y)};
    }
  };

  __device__ __host__ inline float2 fma2(float2 a, float2 b, float2 c) { return target::dispatch<ffma2_impl>(a, b, c); }

  template <bool is_device> struct dfma2_impl {
    inline double2 operator()(double2 a, double2 b, double2 c)
    {
      return {std::fma(a.x, b.x, c.x), std::fma(a.y, b.y, c.y)};
    }
  };

  template <> struct dfma2_impl<true> {
    __device__ inline double2 operator()(double2 a, double2 b, double2 c)
    {
      return {__fma_rn(a.x, b.x, c.x), __fma_rn(a.y, b.y, c.y)};
    }
  };

  __device__ __host__ inline double2 fma2(double2 a, double2 b, double2 c)
  {
    return target::dispatch<dfma2_impl>(a, b, c);
  }

  template <bool is_device> struct fmul2_impl {
    inline float2 operator()(float2 a, float2 b) { return {a.x * b.x, a.y * b.y}; }
  };

  template <> struct fmul2_impl<true> {
    __device__ inline float2 operator()(float2 a, float2 b)
    {
#ifdef QUDA_VECTORIZE_SINGLE
      if constexpr (target::vectorize<float>())
        return __fmul2_rn(a, b);
      else
#endif
        return {a.x * b.x, a.y * b.y};
    }
  };

  __device__ __host__ inline float2 mul2(float2 a, float2 b) { return target::dispatch<fmul2_impl>(a, b); }
  __device__ __host__ inline double2 mul2(double2 a, double2 b) { return {a.x * b.x, a.y * b.y}; }

  template <bool is_device> struct fadd2_impl {
    inline float2 operator()(float2 a, float2 b) { return {a.x * b.x, a.y * b.y}; }
  };

  template <> struct fadd2_impl<true> {
    __device__ inline float2 operator()(float2 a, float2 b)
    {
#ifdef QUDA_VECTORIZE_SINGLE
      if constexpr (target::vectorize<float>())
        return __fadd2_rn(a, b);
      else
#endif
        return {a.x + b.x, a.y + b.y};
    }
  };

  __device__ __host__ inline float2 add2(float2 a, float2 b) { return target::dispatch<fadd2_impl>(a, b); }
  __device__ __host__ inline double2 add2(double2 a, double2 b) { return {a.x + b.x, a.y + b.y}; }

  template <bool is_device> struct dmul_rn_impl {
    inline double operator()(double a, double b) { return a * b; }
  };
#ifdef QUDA_CUDA_CC
  template <> struct dmul_rn_impl<true> {
    __device__ inline double operator()(double a, double b) { return ::__dmul_rn(a, b); }
  };
#endif

  /**
     @brief IEEE double precision multiplication
  */
  __device__ __host__ inline double dmul_rn(double a, double b) { return target::dispatch<dmul_rn_impl>(a, b); }


  template <bool is_device> struct dadd_rn_impl {
    inline double operator()(double a, double b) { return a + b; }
  };
#ifdef QUDA_CUDA_CC
  template <> struct dadd_rn_impl<true> {
    __device__ inline double operator()(double a, double b) { return ::__dadd_rn(a, b); }
  };
#endif

  /**
     @brief IEEE double precision addition
  */
  __device__ __host__ inline double dadd_rn(double a, double b) { return target::dispatch<dadd_rn_impl>(a, b); }


  template <bool is_device> struct fma_rn_impl {
    inline double operator()(double a, double b, double c) { return std::fma(a, b, c); }
  };
#ifdef QUDA_CUDA_CC
  template <> struct fma_rn_impl<true> {
    __device__ inline double operator()(double a, double b, double c) { return ::__fma_rn(a, b, c); }
  };
#endif

  /**
     @brief IEEE double precision fused multiply add
  */
  __device__ __host__ inline double fma_rn(double a, double b, double c) { return target::dispatch<fma_rn_impl>(a, b, c); }

#ifdef QUDA_USE_QUAD_SCALAR
#include <quadmath.h>
#include <crt/device_fp128_functions.h>

  template <bool is_device> struct fabs_fp128_impl;
  template <> struct fabs_fp128_impl<false> {
    __host__ __float128 operator()(__float128 a) { return fabsq(a); }
  };
  template <> struct fabs_fp128_impl<true> {
    __device__ __float128 operator()(__float128 a) { return __nv_fp128_fabs(a); }
  };
  inline __host__ __device__ __float128 fabs(__float128 a) { return target::dispatch<fabs_fp128_impl>(a); }
  inline __host__ __device__ __float128 abs(__float128 a) { return target::dispatch<fabs_fp128_impl>(a); }

  template <bool is_device> struct sqrt_fp128_impl;
  template <> struct sqrt_fp128_impl<false> {
    __host__ __float128 operator()(__float128 a) { return sqrtq(a); }
  };
  template <> struct sqrt_fp128_impl<true> {
    __device__ __float128 operator()(__float128 a) { return __nv_fp128_sqrt(a); }
  };
  inline __host__ __device__ __float128 sqrt(__float128 a) { return target::dispatch<sqrt_fp128_impl>(a); }

#define QUDA_FP128_UNARY_CUDA(FUNC, HOSTFN, DEVICEFN)                                                                \
  template <bool is_device> struct fp128_##FUNC##_impl;                                                              \
  template <> struct fp128_##FUNC##_impl<false> {                                                                    \
    __host__ __float128 operator()(__float128 a) { return HOSTFN(a); }                                               \
  };                                                                                                                 \
  template <> struct fp128_##FUNC##_impl<true> {                                                                     \
    __device__ __float128 operator()(__float128 a) { return DEVICEFN(a); }                                            \
  };                                                                                                                 \
  inline __host__ __device__ __float128 FUNC(__float128 a) { return target::dispatch<fp128_##FUNC##_impl>(a); }

  template <bool is_device> struct fp128_cbrt_impl;
  template <> struct fp128_cbrt_impl<false> {
    __host__ __float128 operator()(__float128 a) { return cbrtq(a); }
  };
  template <> struct fp128_cbrt_impl<true> {
    __device__ __float128 operator()(__float128 a)
    {
      return __nv_fp128_pow(a, static_cast<__float128>(1.0) / static_cast<__float128>(3.0));
    }
  };
  inline __host__ __device__ __float128 cbrt(__float128 a) { return target::dispatch<fp128_cbrt_impl>(a); }
  QUDA_FP128_UNARY_CUDA(cos, cosq, __nv_fp128_cos)
  QUDA_FP128_UNARY_CUDA(acos, acosq, __nv_fp128_acos)
  QUDA_FP128_UNARY_CUDA(cosh, coshq, __nv_fp128_cosh)
  QUDA_FP128_UNARY_CUDA(acosh, acoshq, __nv_fp128_acosh)
  QUDA_FP128_UNARY_CUDA(sinh, sinhq, __nv_fp128_sinh)
  QUDA_FP128_UNARY_CUDA(asinh, asinhq, __nv_fp128_asinh)

#undef QUDA_FP128_UNARY_CUDA

#define QUDA_FP128_PRED_CUDA(FUNC, HOSTFN, DEVICEFN)                                                                   \
  template <bool is_device> struct fp128_##FUNC##_impl;                                                                \
  template <> struct fp128_##FUNC##_impl<false> {                                                                      \
    __host__ bool operator()(__float128 a) { return HOSTFN(a); }                                                        \
  };                                                                                                                   \
  template <> struct fp128_##FUNC##_impl<true> {                                                                      \
    __device__ bool operator()(__float128 a) { return DEVICEFN(a); }                                                    \
  };                                                                                                                   \
  inline __host__ __device__ bool FUNC(__float128 a) { return target::dispatch<fp128_##FUNC##_impl>(a); }

  QUDA_FP128_PRED_CUDA(isnan, isnanq, __nv_fp128_isnan)

#undef QUDA_FP128_PRED_CUDA

#endif

}

#undef QUDA_CUDA_CC

#else

#include "../generic/math_helper.h"

#endif
