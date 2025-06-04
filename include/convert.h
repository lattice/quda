#pragma once

/**
 * @file convert.h
 *
 * @section DESCRIPTION
 * Conversion functions that are used as building blocks for
 * arbitrary field and register ordering.
 */

#include <type_traits>
#include <target_device.h>
#include <register_traits.h>
#include <math_helper.cuh>

namespace quda
{

  /**
   * Traits for determining the maximum and inverse maximum
   * value of a (signed) char and short. Relevant for
   * fixed-precision types.
   */
  template <typename T1> struct fixedMaxValue {
    static constexpr float value = 0.0f;
  };
  template <> struct fixedMaxValue<short> {
    static constexpr float value = 32767.0f;
  };

  template <> struct fixedMaxValue<int8_t> {
    static constexpr float value = 127.0f;
  };

  template <typename T1> struct fixedInvMaxValue {
    static constexpr float value = 3.402823e+38f;
  };

  template <> struct fixedInvMaxValue<short> {
    static constexpr float value = 3.0518509476e-5f;
  };

  template <> struct fixedInvMaxValue<int8_t> {
    static constexpr float value = 7.874015748031e-3f;
  };

  /**
     @brief Regular integer to float used on the host
  */
  template <bool is_device> struct i2f {
    template <typename T> constexpr float operator()(int a, T) { return static_cast<float>(a); }
    template <typename T> constexpr float2 operator()(int a, int b, T)
    {
      return {static_cast<float>(a), static_cast<float>(b)};
    }
  };

#if QUDA_ALTERNATIVE_I_TO_F == 100
  constexpr bool i2f_i[4] = {true, true, true, true};
#elif QUDA_ALTERNATIVE_I_TO_F == 75
  constexpr bool i2f_i[4] = {true, false, true, true};
#elif QUDA_ALTERNATIVE_I_TO_F == 50
  constexpr bool i2f_i[4] = {true, false, true, false};
#elif QUDA_ALTERNATIVE_I_TO_F == 25
  constexpr bool i2f_i[4] = {false, true, false, false};
#elif QUDA_ALTERNATIVE_I_TO_F == 0
  constexpr bool i2f_i[4] = {false, false, false, false};
#endif

  /**
     @brief Fast float to integer round used on the device
  */
  template <> struct i2f<true> {
    template <typename T, typename alternative_t>
    __device__ std::enable_if_t<std::is_same_v<alternative_t, std::integral_constant<bool, alternative_t::value>>, float>
    operator()(T a, alternative_t)
    {
      if constexpr (!alternative_t::value) {
        return static_cast<float>(a);
      } else {
        // will work for up to 23-bit int
        int32_t i = a + 0x4B400000;
        float f;
        memcpy(&f, &i, sizeof(int32_t));
        return f - 12582912.0f;
      }
    }

    template <typename T, typename alternative_t>
    __device__ std::enable_if_t<std::is_same_v<alternative_t, std::integral_constant<bool, alternative_t::value>>, float2>
    operator()(const T &a, const T &b, alternative_t)
    {
      if constexpr (!alternative_t::value) {
        return {static_cast<float>(a), static_cast<float>(b)};
      } else {
        // will work for up to 23-bit int
        int2 i = {a + 0x4B400000, b + 0x4B400000};
        float2 f;
        memcpy(&f, &i, sizeof(int2));
        return add2(f, {-12582912.0f, -12582912.0f});
      }
    }
  };

  /**
     @brief Regular float to integer round used on the host
  */
  template <bool is_device> struct f2i {
    constexpr int operator()(float f) { return static_cast<int>(rintf(f)); }
    constexpr int2 operator()(float2 f) { return {static_cast<int>(rintf(f.x)), static_cast<int>(rintf(f.y))}; }
    constexpr int2 operator()(float2 f, float c)
    {
      return {static_cast<int>(rintf(f.x * c)), static_cast<int>(rintf(f.y * c))};
    }
  };

  /**
     @brief Fast float to integer round used on the device
  */
  template <> struct f2i<true> {
    __device__ inline int operator()(float f)
    {
      f += 12582912.0f;
      int i;
      memcpy(&i, &f, sizeof(int));
      return i;
    }

    __device__ inline int2 operator()(float2 f, float c)
    {
      f = fma2(f, {c, c}, {12582912.0f, 12582912.0f});
      int2 i;
      memcpy(&i, &f, sizeof(int2));
      return i;
    }
  };

  /**
     @brief Regular double to integer round used on the host
  */
  template <bool is_device> struct d2i {
    constexpr int operator()(double d) { return static_cast<int>(rint(d)); }
  };

  /**
     @brief Fast double to integer round used on the device
  */
  template <> struct d2i<true> {
    __device__ inline int operator()(double d)
    {
      d += 6755399441055744.0;
      return reinterpret_cast<int &>(d);
    }
  };

  template <auto Start, auto End, auto Inc, class F> constexpr void constexpr_for(F &&f)
  {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      constexpr_for<Start + Inc, End, Inc>(f);
    }
  }

  /**
     @brief Copy function which is trival between floating point
     types.  When converting to an integer type, the input float is
     assumed to be in the range [-1,1] and we rescale to saturate the
     integer range.  When converting from an integer type, we scale
     the output to be on the same range.
  */
  template <typename T1, typename T2>
  constexpr std::enable_if_t<!isFixed<T1>::value && !isFixed<T2>::value, void> copy(T1 &a, const T2 &b)
  {
    a = b;
  }

  template <typename T1, typename T2>
  constexpr std::enable_if_t<!isFixed<T1>::value && isFixed<T2>::value, void> copy(T1 &a, const T2 &b)
  {
    a = target::dispatch<i2f>(b, std::integral_constant<bool, i2f_i[0]>()) * fixedInvMaxValue<T2>::value;
  }

  template <typename T1, typename T2>
  constexpr std::enable_if_t<isFixed<T1>::value && !isFixed<T2>::value, void> copy(T1 &a, const T2 &b)
  {
    a = target::dispatch<f2i>(b * fixedMaxValue<T1>::value);
  }

  template <typename T1, typename T2, int n>
  constexpr std::enable_if_t<!isFixed<T1>::value && !isFixed<T2>::value, void> copy(T1 *a, const array<T2, n> &b)
  {
    for (int i = 0; i < n; i++) a[i] = b[i];
  }

  template <typename T1, typename T2, int n>
  constexpr std::enable_if_t<!isFixed<T1>::value && isFixed<T2>::value, void> copy(T1 *a, const array<T2, n> &b)
  {
    static_assert(n % 2 == 0);
    constexpr_for<0, n, 2>([&](auto i) {
      auto bi = target::dispatch<i2f>(b[i + 0], b[i + 1], std::integral_constant<bool, i2f_i[(i / 2) % 4]>());
      auto ai = mul2(bi, {fixedInvMaxValue<T2>::value, fixedInvMaxValue<T2>::value});
      a[i + 0] = ai.x;
      a[i + 1] = ai.y;
    });
  }

  template <typename T1, typename T2, int n>
  constexpr std::enable_if_t<isFixed<T1>::value && !isFixed<T2>::value, void> copy(T1 *a, const array<T2, n> &b)
  {
    static_assert(n % 2 == 0);
    constexpr_for<0, n, 2>([&](auto i) {
      auto bi = mul2({b[i], b[i + 1]}, {fixedMaxValue<T1>::value, fixedMaxValue<T1>::value});
      auto ai = target::dispatch<f2i>(bi);
      a[i + 0] = ai.x;
      a[i + 1] = ai.y;
    });
  }

  /**
     @brief Specialized variants of the copy function that include an
     additional scale factor.  Note the scale factor is ignored unless
     the input type (b) is either a short or char vector.
  */
  template <typename T1, typename T2, typename T3>
  constexpr std::enable_if_t<!isFixed<T1>::value && !isFixed<T2>::value, void> copy_and_scale(T1 &a, const T2 &b,
                                                                                              const T3 &)
  {
    copy(a, b);
  }

  template <typename T1, typename T2, typename T3>
  constexpr std::enable_if_t<!isFixed<T1>::value && isFixed<T2>::value, void> copy_and_scale(T1 &a, const T2 &b,
                                                                                             const T3 &c)
  {
    a = target::dispatch<i2f>(b, std::integral_constant<bool, i2f_i[0]>()) * c;
  }

  template <typename T1, typename T2, typename T3>
  constexpr std::enable_if_t<isFixed<T1>::value && !isFixed<T2>::value, void> copy_and_scale(T1 &a, const T2 &b,
                                                                                             const T3 &c)
  {
    a = target::dispatch<f2i>(b * c);
  }

  template <typename T1, typename T2, int n, typename T3>
  constexpr std::enable_if_t<!isFixed<T1>::value && !isFixed<T2>::value, void>
  copy_and_scale(T1 *a, const array<T2, n> &b, const T3 &)
  {
    for (int i = 0; i < n; i++) copy(a[i], b[i]);
  }

  template <typename T1, typename T2, int n, typename T3>
  constexpr std::enable_if_t<!isFixed<T1>::value && !isFixed<T2>::value, void> copy_and_scale(array<T1, n> &a,
                                                                                              const T2 *b, const T3 &)
  {
    for (int i = 0; i < n; i++) copy(a[i], b[i]);
  }

  template <typename T1, typename T2, int n, typename T3>
  constexpr std::enable_if_t<!isFixed<T1>::value && isFixed<T2>::value, void>
  copy_and_scale(T1 *a, const array<T2, n> &b, const T3 &c)
  {
    static_assert(n % 2 == 0);
    constexpr_for<0, n, 2>([&](auto i) {
      auto bi = target::dispatch<i2f>(b[i + 0], b[i + 1], std::integral_constant<bool, i2f_i[(i / 2) % 4]>());
      auto ai = mul2(bi, {c, c});
      a[i + 0] = ai.x;
      a[i + 1] = ai.y;
    });
  }

  template <typename T1, typename T2, int n, typename T3>
  constexpr std::enable_if_t<isFixed<T1>::value && !isFixed<T2>::value, void> copy_and_scale(array<T1, n> &a,
                                                                                             const T2 *b, const T3 &c)
  {
    static_assert(n % 2 == 0);
    constexpr_for<0, n, 2>([&](auto i) {
      auto ai = target::dispatch<f2i>(float2 {(float)b[i + 0], (float)b[i + 1]}, c);
      a[i + 0] = ai.x;
      a[i + 1] = ai.y;
    });
  }

  template <class fixed_t, class float_t> __device__ __host__ fixed_t f2i_round(float_t f)
  {
#if 1
    fixed_t i = {};
    if constexpr (sizeof(fixed_t) < 4) {
      i = static_cast<fixed_t>(target::dispatch<f2i>(f));
    } else {
      i = static_cast<fixed_t>(rint(f));
    }
    return i;
#else
    return static_cast<fixed_t>(rint(f));
#endif
  }
} // namespace quda
