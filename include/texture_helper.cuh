#pragma once

template <typename T>
__device__ __forceinline__ T tex1Dfetch_(cudaTextureObject_t tex, int i)
{
  return tex1Dfetch<T>(tex, i);
}

// clang-cuda seem incompatable with the CUDA texture headers, so we must resort to ptx
#if defined(__clang__) && defined(__CUDA__)

template <>
__device__ __forceinline__ float tex1Dfetch_(cudaTextureObject_t tex, int i)
{
  float4 temp;
  asm("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];" :
      "=f"(temp.x), "=f"(temp.y), "=f"(temp.z), "=f"(temp.w) : "l"(tex), "r"(i));
  return temp.x;
}

template <>
__device__ __forceinline__ float2 tex1Dfetch_(cudaTextureObject_t tex, int i)
{
  float4 temp;
  asm("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];" :
      "=f"(temp.x), "=f"(temp.y), "=f"(temp.z), "=f"(temp.w) : "l"(tex), "r"(i));
  return make_float2(temp.x, temp.y);
}

template <>
__device__ __forceinline__ float4 tex1Dfetch_(cudaTextureObject_t tex, int i)
{
  float4 temp;
  asm("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];" :
      "=f"(temp.x), "=f"(temp.y), "=f"(temp.z), "=f"(temp.w) : "l"(tex), "r"(i));
  return temp;
}

template <>
__device__ __forceinline__ int4 tex1Dfetch_(cudaTextureObject_t tex, int i)
{
  int4 temp;
  asm("tex.1d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5}];" :
      "=r"(temp.x), "=r"(temp.y), "=r"(temp.z), "=r"(temp.w) : "l"(tex), "r"(i));
  return temp;
}

#endif
