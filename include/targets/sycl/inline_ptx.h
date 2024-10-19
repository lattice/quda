#pragma once

/*
  Inline ptx instructions for low-level control of code generation.
  Primarily these are for doing stores avoiding L1 cache and minimal
  impact on L2 (streaming through L2).
*/

namespace quda {

  inline void load_streaming_double2(double2 &a, const double2* addr)
  {
    a.x = addr->x; a.y = addr->y;
  }

  inline void load_streaming_float4(float4 &a, const float4* addr)
  {
    a.x = addr->x; a.y = addr->y; a.z = addr->z; a.w = addr->w;
  }

  inline void load_cached_short4(short4 &a, const short4 *addr)
  {
    a.x = addr->x;
    a.y = addr->y;
    a.z = addr->z;
    a.w = addr->w;
  }

  inline void load_cached_short2(short2 &a, const short2 *addr)
  {
    a.x = addr->x;
    a.y = addr->y;
  }

  inline void load_global_short4(short4 &a, const short4 *addr)
  {
    a.x = addr->x;
    a.y = addr->y;
    a.z = addr->z;
    a.w = addr->w;
  }

  inline void load_global_short2(short2 &a, const short2 *addr)
  {
    a.x = addr->x;
    a.y = addr->y;
  }

  inline void load_global_float4(float4 &a, const float4* addr)
  {
    a.x = addr->x; a.y = addr->y; a.z = addr->z; a.w = addr->w;
  }

  inline void store_streaming_float4(float4* addr, float x, float y, float z, float w)
  {
    addr->x = x;
    addr->y = y;
    addr->z = z;
    addr->w = w;
  }

  inline void store_streaming_short4(short4* addr, short x, short y, short z, short w)
  {
    addr->x = x;
    addr->y = y;
    addr->z = z;
    addr->w = w;
  }

  inline void store_streaming_double2(double2* addr, double x, double y)
  {
    addr->x = x;
    addr->y = y;
  }

  inline void store_streaming_float2(float2* addr, float x, float y)
  {
    addr->x = x;
    addr->y = y;
  }

  inline void store_streaming_short2(short2* addr, short x, short y)
  {
    addr->x = x;
    addr->y = y;
  }

} // namespace quda
