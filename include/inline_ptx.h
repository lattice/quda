/*
  Inline ptx instructions for low-level control of code generation.
  Primarily these are for doing stores avoiding L1 cache and minimal
  impact on L2 (streaming through L2).
*/

#if (__COMPUTE_CAPABILITY__ >= 200)

namespace quda {

#if (POINTER_SIZE==8) // 64-bit pointers

  __device__ inline void load_streaming_double2(double2 &a, const double2* addr)
  {
    double x, y;
    asm("ld.cs.global.v2.f64 {%0, %1}, [%2+0];" : "=d"(x), "=d"(y) : "l"(addr));
    a.x = x; a.y = y;
  }

  __device__ inline void load_streaming_float4(float4 &a, const float4* addr)
  {
    float x, y, z, w;
    asm("ld.cs.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  __device__ inline void load_global_float4(float4 &a, const float4* addr)
  {
    float x, y, z, w;
    asm("ld.cg.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  __device__ inline void store_streaming_float4(float4* addr, float x, float y, float z, float w)
  {
    asm("st.cs.global.v4.f32 [%0+0], {%1, %2, %3, %4};" :: "l"(addr), "f"(x), "f"(y), "f"(z), "f"(w));
  }

  __device__ inline void store_streaming_short4(short4* addr, short x, short y, short z, short w)
  {
    asm("st.cs.global.v4.s16 [%0+0], {%1, %2, %3, %4};" :: "l"(addr), "h"(x), "h"(y), "h"(z), "h"(w));
  }

  __device__ inline void store_streaming_double2(double2* addr, double x, double y)
  {
    asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" :: "l"(addr), "d"(x), "d"(y));
  }

  __device__ inline void store_streaming_float2(float2* addr, float x, float y)
  {
    asm("st.cs.global.v2.f32 [%0+0], {%1, %2};" :: "l"(addr), "f"(x), "f"(y));
  }

  __device__ inline void store_streaming_short2(short2* addr, short x, short y)
  {
    asm("st.cs.global.v2.s16 [%0+0], {%1, %2};" :: "l"(addr), "h"(x), "h"(y));
  }

#else // 32-bit pointers

  __device__ inline void load_streaming_double2(double2 &a, const double2* addr)
  {
    double x, y;
    asm("ld.cs.global.v2.f64 {%0, %1}, [%2+0];" : "=d"(x), "=d"(y) : "r"(addr));
    a.x = x; a.y = y;
  }

  __device__ inline void load_streaming_float4(float4 &a, const float4* addr)
  {
    float x, y, z, w;
    asm("ld.cs.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "r"(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  __device__ inline void load_global_float4(float4 &a, const float4* addr)
  {
    float x, y, z, w;
    asm("ld.cg.global.v4.f32 {%0, %1, %2, %3}, [%4+0];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "r"(addr));
    a.x = x; a.y = y; a.z = z; a.w = w;
  }

  __device__ inline void store_streaming_float4(float4* addr, float x, float y, float z, float w)
  {
    asm("st.cs.global.v4.f32 [%0+0], {%1, %2, %3, %4};" :: "r"(addr), "f"(x), "f"(y), "f"(z), "f"(w));
  }

  __device__ inline void store_streaming_short4(short4* addr, short x, short y, short z, short w)
  {
    asm("st.cs.global.v4.s16 [%0+0], {%1, %2, %3, %4};" :: "r"(addr), "h"(x), "h"(y), "h"(z), "h"(w));
  }

  __device__ inline void store_streaming_double2(double2* addr, double x, double y)
  {
    asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" :: "r"(addr), "d"(x), "d"(y));
  }

  __device__ inline void store_streaming_float2(float2* addr, float x, float y)
  {
    asm("st.cs.global.v2.f32 [%0+0], {%1, %2};" :: "r"(addr), "f"(x), "f"(y));
  }

  __device__ inline void store_streaming_short2(short2* addr, short x, short y)
  {
    asm("st.cs.global.v2.s16 [%0+0], {%1, %2};" :: "r"(addr), "h"(x), "h"(y));
  }

#endif // POINTER_SIZE

} // namespace quda

#endif // COMPUTE_CAPABILITY


