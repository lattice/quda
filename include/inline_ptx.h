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

  __device__ inline void load_streaming_double2(double2 &a, const double2* addr)
  {
    a.x = ((double *)addr)[0];a.y = ((double *)addr)[1];
/*
    double x, y;
    asm("ld.cs.global.v2.f64 {%0, %1}, [%2+0];" : "=d"(x), "=d"(y) : __PTR(addr));
    a.x = x; a.y = y;
*/
  }

  __device__ inline void load_streaming_float4(float4 &a, const float4* addr)
  {
    a.x = ((float *)addr)[0];a.y = ((float *)addr)[1];a.z = ((float *)addr)[2];a.w = ((float *)addr)[3];
/*
    float x, y, z, w;
    asm("ld.cs.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : __PTR(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
*/
  }

  __device__ inline void load_cached_short4(short4 &a, const short4 *addr)
  {
    a.x = ((short *)addr)[0];a.y = ((short *)addr)[1];a.z = ((short *)addr)[2];a.w = ((short *)addr)[3];
/*
    short x, y, z, w;
    asm("ld.ca.global.v4.s16 {%0, %1, %2, %3}, [%4+0];" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : __PTR(addr));
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
*/
  }

  __device__ inline void load_cached_short2(short2 &a, const short2 *addr)
  {
    a.x = ((short*)addr)[0];a.y = ((short*)addr)[1];
/*
    short x, y;
    asm("ld.ca.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : __PTR(addr));
    a.x = x;
    a.y = y;
*/
  }

  __device__ inline void load_global_short4(short4 &a, const short4 *addr)
  {
    a.x = ((short *)addr)[0];a.y = ((short *)addr)[1];a.z = ((short *)addr)[2];a.w = ((short *)addr)[3];
/*
    short x, y, z, w;
    asm("ld.cg.global.v4.s16 {%0, %1, %2, %3}, [%4+0];" : "=h"(x), "=h"(y), "=h"(z), "=h"(w) : __PTR(addr));
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
*/
  }

  __device__ inline void load_global_short2(short2 &a, const short2 *addr)
  {
    a.x = ((short*)addr)[0];a.y = ((short*)addr)[1];
/*
    short x, y;
    asm("ld.cg.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : __PTR(addr));
    a.x = x;
    a.y = y;
*/
  }

  __device__ inline void load_global_float4(float4 &a, const float4* addr)
  {
    a.x = ((float *)addr)[0];a.y = ((float *)addr)[1];a.z = ((float *)addr)[2];a.w = ((float *)addr)[3];
/*
    float x, y, z, w;
    asm("ld.cg.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : __PTR(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
*/
  }

  __device__ inline void store_streaming_float4(float4* addr, float x, float y, float z, float w)
  {
    ((float *)addr)[0] = x;((float *)addr)[1] = y;((float *)addr)[2] = z;((float *)addr)[3] = w;
//    asm("st.cs.global.v4.f32 [%0+0], {%1, %2, %3, %4};" :: __PTR(addr), "f"(x), "f"(y), "f"(z), "f"(w));
  }

  __device__ inline void store_streaming_short4(short4* addr, short x, short y, short z, short w)
  {
    ((short *)addr)[0] = x;((short *)addr)[1] = y;((short *)addr)[2] = z;((short *)addr)[3] = w;
//    asm("st.cs.global.v4.s16 [%0+0], {%1, %2, %3, %4};" :: __PTR(addr), "h"(x), "h"(y), "h"(z), "h"(w));
  }

  __device__ inline void store_streaming_double2(double2* addr, double x, double y)
  {
    ((double *)addr)[0] = x;((double *)addr)[1] = y;
//    asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" :: __PTR(addr), "d"(x), "d"(y));
  }

  __device__ inline void store_streaming_float2(float2* addr, float x, float y)
  {
    ((float *)addr)[0] = x;((float *)addr)[1] = y;
//    asm("st.cs.global.v2.f32 [%0+0], {%1, %2};" :: __PTR(addr), "f"(x), "f"(y));
  }

  __device__ inline void store_streaming_short2(short2* addr, short x, short y)
  {
    ((short *)addr)[0] = x;((short *)addr)[1] = y;
//    asm("st.cs.global.v2.s16 [%0+0], {%1, %2};" :: __PTR(addr), "h"(x), "h"(y));
  }

} // namespace quda
