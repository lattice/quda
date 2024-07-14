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

namespace quda
{

  /**
   * Traits for determining the maximum and inverse maximum
   * value of a (signed) char and short. Relevant for
   * fixed-precision types.
   */
  template <typename T> struct fixedMaxValue {
    static constexpr float value = 0.0f;
  };
  template <> struct fixedMaxValue<short> {
    static constexpr float value = 32767.0f;
  };
  template <> struct fixedMaxValue<short2> {
    static constexpr float value = 32767.0f;
  };
  template <> struct fixedMaxValue<short4> {
    static constexpr float value = 32767.0f;
  };
  template <> struct fixedMaxValue<short8> {
    static constexpr float value = 32767.0f;
  };
  template <> struct fixedMaxValue<int8_t> {
    static constexpr float value = 127.0f;
  };
  template <> struct fixedMaxValue<char2> {
    static constexpr float value = 127.0f;
  };
  template <> struct fixedMaxValue<char4> {
    static constexpr float value = 127.0f;
  };
  template <> struct fixedMaxValue<char8> {
    static constexpr float value = 127.0f;
  };

  template <typename T> struct fixedInvMaxValue {
    static constexpr float value = 3.402823e+38f;
  };
  template <> struct fixedInvMaxValue<short> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short2> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short4> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short8> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<int8_t> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char2> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char4> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char8> {
    static constexpr float value = 7.874015748031e-3f;
  };

  template <typename T> constexpr float i2f(T a)
  {
#ifndef QUDA_ALTERNATIVE_I_TO_F
    return static_cast<float>(a);
#else
    // will work for up to 23-bit int
    int32_t i = a + 0x4B400000;
    float &f = reinterpret_cast<float &>(i);
    return f - 12582912.0f;
#endif
  }

  /**
     @brief Regular float to integer round used on the host
  */
  template <bool is_device> struct f2i {
    constexpr int operator()(float f) { return static_cast<int>(rintf(f)); }
  };

  /**
     @brief Fast float to integer round used on the device
  */
  template <> struct f2i<true> {
    __device__ inline int operator()(float f)
    {
      f += 12582912.0f;
      return reinterpret_cast<int &>(f);
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
    a = i2f(b) * fixedInvMaxValue<T2>::value;
  }

  template <typename T1, typename T2>
  constexpr std::enable_if_t<isFixed<T1>::value && !isFixed<T2>::value, void> copy(T1 &a, const T2 &b)
  {
    a = target::dispatch<f2i>(b * fixedMaxValue<T1>::value);
  }

  /**
     @brief Specialized variants of the copy function that assumes the
     scaling factor has already been done.
  */
  template <typename T1, typename T2>
  constexpr std::enable_if_t<!isFixed<T1>::value, void> copy_scaled(T1 &a, const T2 &b)
  {
    copy(a, b);
  }

  template <typename T1, typename T2>
  constexpr std::enable_if_t<isFixed<T1>::value, void> copy_scaled(T1 &a, const T2 &b)
  {
    a = target::dispatch<f2i>(b);
  }

  /**
     @brief Specialized variants of the copy function that include an
     additional scale factor.  Note the scale factor is ignored unless
     the input type (b) is either a short or char vector.
  */
  template <typename T1, typename T2, typename T3>
  constexpr std::enable_if_t<!isFixed<T2>::value, void> copy_and_scale(T1 &a, const T2 &b, const T3 &)
  {
    copy(a, b);
  }

  template <typename T1, typename T2, typename T3>
  constexpr std::enable_if_t<isFixed<T2>::value, void> copy_and_scale(T1 &a, const T2 &b, const T3 &c)
  {
    a = i2f(b) * c;
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
