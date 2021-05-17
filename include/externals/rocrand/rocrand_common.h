// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCRAND_COMMON_H_
#define ROCRAND_COMMON_H_

#define ROCRAND_2POW16_INV (1.5258789e-05f)
#define ROCRAND_2POW16_INV_2PI (1.5258789e-05f * 6.2831855f)
#define ROCRAND_2POW32_INV (2.3283064e-10f)
#define ROCRAND_2POW32_INV_DOUBLE (2.3283064365386963e-10)
#define ROCRAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define ROCRAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define ROCRAND_PI  (3.1415926f)
#define ROCRAND_PI_DOUBLE  (3.1415926535897932)
#define ROCRAND_2PI (6.2831855f)
#define ROCRAND_SQRT2 (1.4142135f)
#define ROCRAND_SQRT2_DOUBLE (1.4142135623730951)

#include <math.h>

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#if __HIP_DEVICE_COMPILE__ && (defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_NVCC__) && (__CUDA_ARCH__ >= 530)))
#define ROCRAND_HALF_MATH_SUPPORTED
#endif

namespace rocrand_device {
namespace detail {

  #if ( defined(__gfx801__) || \
        defined(__gfx802__) || \
        defined(__gfx803__) || \
        defined(__gfx810__) || \
        defined(__gfx900__) || \
        defined(__gfx902__) || \
        defined(__gfx904__) || \
        defined(__gfx906__) || \
        defined(__gfx908__) || \
        defined(__gfx909__) )
  #ifndef ROCRAND_ENABLE_INLINE_ASM
    #define ROCRAND_ENABLE_INLINE_ASM
  #endif
#else
  #ifdef ROCRAND_ENABLE_INLINE_ASM
    #undef ROCRAND_ENABLE_INLINE_ASM
    #pragma warning "Disabled inline asm, because the build target does not support it."
  #endif
#endif

FQUALIFIERS
unsigned long long mad_u64_u32(const unsigned int x, const unsigned int y, const unsigned long long z)
{
  #if defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__) \
    && defined(ROCRAND_ENABLE_INLINE_ASM)

    unsigned long long r;
    unsigned long long c; // carry bits, SGPR, unused
    // x has "r" constraint. This allows to use both VGPR and SGPR
    // (to save VGPR) as input.
    // y and z have "v" constraints, because only one SGPR or literal
    // can be read by the instruction.
    asm volatile("v_mad_u64_u32 %0, %1, %2, %3, %4"
      : "=v"(r), "=s"(c) : "r"(x), "v"(y), "v"(z)
    );
    return r;
  #elif defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_DEVICE_COMPILE__) \
        && defined(ROCRAND_ENABLE_INLINE_ASM)

    unsigned long long r;
    asm("mad.wide.u32 %0, %1, %2, %3;"
        : "=l"(r) : "r"(x), "r"(y), "l"(z)
    );
    return r;

  #else // host code

    return static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y) + z;

  #endif
}

// This helps access fields of engine's internal state which
// saves floats and doubles generated using the Box–Muller transform
template<typename Engine>
struct engine_boxmuller_helper
{
    static FQUALIFIERS
    bool has_float(const Engine * engine)
    {
        return engine->m_state.boxmuller_float_state != 0;
    }

    static FQUALIFIERS
    float get_float(Engine * engine)
    {
        engine->m_state.boxmuller_float_state = 0;
        return engine->m_state.boxmuller_float;
    }

    static FQUALIFIERS
    void save_float(Engine * engine, float f)
    {
        engine->m_state.boxmuller_float_state = 1;
        engine->m_state.boxmuller_float = f;
    }

    static FQUALIFIERS
    bool has_double(const Engine * engine)
    {
        return engine->m_state.boxmuller_double_state != 0;
    }

    static FQUALIFIERS
    float get_double(Engine * engine)
    {
        engine->m_state.boxmuller_double_state = 0;
        return engine->m_state.boxmuller_double;
    }

    static FQUALIFIERS
    void save_double(Engine * engine, double d)
    {
        engine->m_state.boxmuller_double_state = 1;
        engine->m_state.boxmuller_double = d;
    }
};

} // end namespace detail
} // end namespace rocrand_device

#endif // ROCRAND_COMMON_H_
