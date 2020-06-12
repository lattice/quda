#pragma once

/**
 * @file convert.h
 *
 * @section DESCRIPTION
 * Conversion functions that are used as building blocks for
 * arbitrary field and register ordering.
 */

#include <quda_internal.h> // for maximum short, char traits.
#include <register_traits.h>

namespace quda
{

  template <typename T> __host__ __device__ inline float i2f(T a)
  {
#if 1
    return static_cast<float>(a);
#else
    // will work for up to 23-bit int
    union {
      int32_t i;
      float f;
    };
    i  = a + 0x4B400000;
    return f - 12582912.0f;
#endif
  }

  // Fast float to integer round
  __device__ __host__ inline int f2i(float f)
  {
#ifdef __CUDA_ARCH__
    f += 12582912.0f;
    return reinterpret_cast<int &>(f);
#else
    return static_cast<int>(f);
#endif
  }

  // Fast double to integer round
  __device__ __host__ inline int d2i(double d)
  {
#ifdef __CUDA_ARCH__
    d += 6755399441055744.0;
    return reinterpret_cast<int &>(d);
#else
    return static_cast<int>(d);
#endif
  }

  /**
     @brief Copy function which is trival between floating point
     types.  When converting to an integer type, the input float is
     assumed to be in the range [-1,1] and we rescale to saturate the
     integer range.  When converting from an integer type, we scale
     the output to be on the same range.
  */
  template <typename T1, typename T2> __host__ __device__ inline void copy(T1 &a, const T2 &b) { a = b; }
  template <> __host__ __device__ inline void copy(float &a, const short &b) { a = i2f(b) * fixedInvMaxValue<short>::value; }
  template <> __host__ __device__ inline void copy(short &a, const float &b) { a = f2i(b * fixedMaxValue<short>::value); }
  template <> __host__ __device__ inline void copy(float &a, const char &b) { a = i2f(b) * fixedInvMaxValue<char>::value; }
  template <> __host__ __device__ inline void copy(char &a, const float &b) { a = f2i(b * fixedMaxValue<char>::value); }

  /**
     @brief Specialized variants of the copy function that assumes the
     scaling factor has already been done.
  */
  template <typename T1, typename T2> __host__ __device__ inline void copy_scaled(T1 &a, const T2 &b) { copy(a, b); }
  template <> __host__ __device__ inline void copy_scaled(short &a, const float &b) { a = f2i(b); }
  template <> __host__ __device__ inline void copy_scaled(char &a, const float &b) { a = f2i(b); }

  /**
     @brief Specialized variants of the copy function that include an
     additional scale factor.  Note the scale factor is ignored unless
     the input type (b) is either a short or char vector.
  */
  template <typename T1, typename T2, typename T3>
    __host__ __device__ inline void copy_and_scale(T1 &a, const T2 &b, const T3 &c) { copy(a, b); }
  template <> __host__ __device__ inline void copy_and_scale(float &a, const short &b, const float &c) { a = i2f(b) * (fixedInvMaxValue<short>::value * c); }
  template <> __host__ __device__ inline void copy_and_scale(float &a, const char &b, const float &c) { a = i2f(b) * (fixedInvMaxValue<char>::value * c); }

} // namespace quda
