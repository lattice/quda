#pragma once

/*
  Inline ptx instructions for low-level control of code generation.
  Primarily these are for doing stores avoiding L1 cache and minimal
  impact on L2 (streaming through L2).
*/

// Define a different pointer storage size for 64 and 32 bit
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

namespace quda {

  // If you're bored...
  // http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st

// Helper macro for prefetch size validation
#define VALIDATE_PREFETCH_SIZE(prefetch_size)                                                                          \
  static_assert(prefetch_size == 0 || prefetch_size == 64 || prefetch_size == 128 || prefetch_size == 256,             \
                "prefetch_size must be 0, 64, 128, or 256")

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_streaming_double4(double4 &a, const double4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    double x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain streaming load, no prefetch hint
      asm volatile("ld.global.cs.v4.f64 {%0, %1, %2, %3}, [%4];\n" : "=d"(x), "=d"(y), "=d"(z), "=d"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cs.L2::64B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cs.L2::128B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cs.L2::256B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    }

    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_streaming_double2(double2 &a, const double2 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    double x, y;

    if constexpr (prefetch_ == 0) {
      // Plain streaming load, no prefetch hint
      asm volatile("ld.global.cs.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cs.L2::64B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cs.L2::128B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cs.L2::256B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    }

    a.x = x; a.y = y;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_streaming_float8(float8 &v, const float8 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y, z, w, a, b, c, d;

    if constexpr (prefetch_ == 0) {
      // Plain streaming load, no prefetch hint
      asm volatile("ld.global.cs.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cs.L2::64B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cs.L2::128B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cs.L2::256B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    }

    v = {{x, y, z, w}, {a, b, c, d}};
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_streaming_float4(float4 &a, const float4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain streaming load, no prefetch hint
      asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cs.L2::64B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cs.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cs.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    }

    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_short4(short4 &a, const short4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    short x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v4.s16 {%0, %1, %2, %3}, [%4];\n" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    }

    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_short2(short2 &a, const short2 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    short x, y;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    }

    a.x = x;
    a.y = y;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_global_short4(short4 &a, const short4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    short x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain global load, no prefetch hint
      asm volatile("ld.global.cg.v4.s16 {%0, %1, %2, %3}, [%4];\n" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cg.L2::64B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cg.L2::128B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cg.L2::256B.v4.s16 {%0, %1, %2, %3}, [%4];\n"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(addr));
    }

    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_global_short2(short2 &a, const short2 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    short x, y;

    if constexpr (prefetch_ == 0) {
      // Plain global load, no prefetch hint
      asm volatile("ld.global.cg.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cg.L2::64B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cg.L2::128B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cg.L2::256B.v2.s16 {%0, %1}, [%2];\n" : "=h"(x), "=h"(y) : "l"(addr));
    }

    a.x = x;
    a.y = y;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_global_float4(float4 &a, const float4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain global load, no prefetch hint
      asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.cg.L2::64B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.cg.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.cg.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    }

    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_float4(float4 &a, const float4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(addr));
    }

    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_float8(float8 &v, const float8 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y, z, w, a, b, c, d;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w), "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                   : "l"(addr));
    }

    v = {{x, y, z, w}, {a, b, c, d}};
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_float2(float2 &a, const float2 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x, y;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v2.f32 {%0, %1}, [%2];\n" : "=f"(x), "=f"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v2.f32 {%0, %1}, [%2];\n" : "=f"(x), "=f"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v2.f32 {%0, %1}, [%2];\n" : "=f"(x), "=f"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v2.f32 {%0, %1}, [%2];\n" : "=f"(x), "=f"(y) : "l"(addr));
    }

    a.x = x; a.y = y;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_float(float &a, const float *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    float x;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.f32 {%0}, [%1];\n" : "=f"(x) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.f32 {%0}, [%1];\n" : "=f"(x) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.f32 {%0}, [%1];\n" : "=f"(x) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.f32 {%0}, [%1];\n" : "=f"(x) : "l"(addr));
    }

    a = x;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_double4(double4 &a, const double4 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    double x, y, z, w;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v4.f64 {%0, %1, %2, %3}, [%4];\n" : "=d"(x), "=d"(y), "=d"(z), "=d"(w) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v4.f64 {%0, %1, %2, %3}, [%4];\n"
                   : "=d"(x), "=d"(y), "=d"(z), "=d"(w)
                   : "l"(addr));
    }

    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
  }

  // Valid values for prefetch_size: 0 (no prefetch), 64, 128, 256
  // Note: 256B prefetch requires SM 80+. For older architectures, 256B -> 128B
  template <size_t prefetch_size = 0> __device__ inline void load_cached_double2(double2 &a, const double2 *addr)
  {
    VALIDATE_PREFETCH_SIZE(prefetch_size);
    constexpr size_t prefetch_ = __COMPUTE_CAPABILITY__ < 800 ? 0 : prefetch_size;

    double x, y;

    if constexpr (prefetch_ == 0) {
      // Plain cached load, no prefetch hint
      asm volatile("ld.global.ca.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 64) {
      asm volatile("ld.global.ca.L2::64B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 128) {
      asm volatile("ld.global.ca.L2::128B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    } else if constexpr (prefetch_ == 256) {
      asm volatile("ld.global.ca.L2::256B.v2.f64 {%0, %1}, [%2];\n" : "=d"(x), "=d"(y) : "l"(addr));
    }

    a.x = x; a.y = y;
  }

  __device__ inline void store_streaming_float8(float8 *addr, const float8 &v)
  {
    asm("st.cs.global.v8.f32 [%0+0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: __PTR(addr), "f"(v.x.x), "f"(v.x.y),
        "f"(v.x.z), "f"(v.x.w), "f"(v.y.x), "f"(v.y.y), "f"(v.y.z), "f"(v.y.w));
  }

  __device__ inline void store_streaming_float4(float4* addr, float x, float y, float z, float w)
  {
    asm("st.cs.global.v4.f32 [%0+0], {%1, %2, %3, %4};" :: __PTR(addr), "f"(x), "f"(y), "f"(z), "f"(w));
  }

  __device__ inline void store_streaming_short4(short4* addr, short x, short y, short z, short w)
  {
    asm("st.cs.global.v4.s16 [%0+0], {%1, %2, %3, %4};" :: __PTR(addr), "h"(x), "h"(y), "h"(z), "h"(w));
  }

  __device__ inline void store_streaming_double4(double4 *addr, double x, double y, double z, double w)
  {
    asm("st.cs.global.v4.f64 [%0+0], {%1, %2, %3, %4};" :: __PTR(addr), "d"(x), "d"(y), "d"(z), "d"(w));
  }

  __device__ inline void store_streaming_double2(double2* addr, double x, double y)
  {
    asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" :: __PTR(addr), "d"(x), "d"(y));
  }

  __device__ inline void store_streaming_float2(float2* addr, float x, float y)
  {
    asm("st.cs.global.v2.f32 [%0+0], {%1, %2};" :: __PTR(addr), "f"(x), "f"(y));
  }

  __device__ inline void store_streaming_short2(short2* addr, short x, short y)
  {
    asm("st.cs.global.v2.s16 [%0+0], {%1, %2};" :: __PTR(addr), "h"(x), "h"(y));
  }

  __device__ inline void prefetch_L1(void *smem_ptr_, const void *gmem_ptr)
  {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_ptr_);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_ptr), "l"(gmem_ptr));
  }

  __device__ __forceinline__ void prefetch_L1(const void *p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

  __device__ __forceinline__ void prefetch_L2(const void *p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

  __device__ __forceinline__ void prefetch_tma(const void *p, size_t bytes)
  {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(static_cast<uint32_t>(bytes)));
  }

  using tensor_desc_t = CUtensorMap;

  __device__ __forceinline__ void prefetch_tma_3d(const CUtensorMap &tensor_map, int x, int y, int z)
  {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(&tensor_map), "r"(x),
                 "r"(y), "r"(z)
                 : "memory");
  }

  __device__ __forceinline__ void prefetch_tma_4d(const CUtensorMap &tensor_map, int x, int y, int z, int w)
  {
    asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" ::"l"(&tensor_map), "r"(x),
                 "r"(y), "r"(z), "r"(w)
                 : "memory");
  }

  __device__ __forceinline__ void prefetch_tma_5d(const CUtensorMap &tensor_map, int x, int y, int z, int w, int u)
  {
    asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" ::"l"(&tensor_map),
                 "r"(x), "r"(y), "r"(z), "r"(w), "r"(u)
                 : "memory");
  }

} // namespace quda
