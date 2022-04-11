#pragma once

#ifndef QUDA_UNROLL
#define QUDA_UNROLL
#endif

// OMPTARGET SPECIFIC workarounds
#define __host__
#define __device__
#define __global__
#define __constant__ static
#define __launch_bounds__(...)
#define __syncthreads() _Pragma("omp barrier")
#define __threadfence() _Pragma("omp barrier")
#define __forceinline__ inline __attribute__((always_inline))

#define CUDA_SUCCESS QUDA_SUCCESS

#define QUDA_RT_CONSTS \
  const dim3\
    blockDim=target::omptarget::launch_param.block,\
    gridDim=target::omptarget::launch_param.grid,\
    threadIdx(omp_get_thread_num()%target::omptarget::launch_param.block.x, (omp_get_thread_num()/target::omptarget::launch_param.block.x)%target::omptarget::launch_param.block.y, omp_get_thread_num()/(target::omptarget::launch_param.block.x*target::omptarget::launch_param.block.y)),\
    blockIdx(omp_get_team_num()%target::omptarget::launch_param.grid.x, (omp_get_team_num()/target::omptarget::launch_param.grid.x)%target::omptarget::launch_param.grid.y, omp_get_team_num()/(target::omptarget::launch_param.grid.x*target::omptarget::launch_param.grid.y))

#include <functional>
#include <iostream>
#include <omp.h>

struct dim3 {unsigned int x,y,z;
  constexpr dim3():x(1u),y(1u),z(1u){}
  constexpr dim3(unsigned int x):x(x),y(1u),z(1u){}
  constexpr dim3(unsigned int x,unsigned int y):x(x),y(y),z(1u){}
  constexpr dim3(unsigned int x,unsigned int y,unsigned int z):x(x),y(y),z(z){}
};
struct int2 {int x,y;};
struct int4 {int x,y,z,w;};
static inline int4 make_int4(int x,int y,int z,int w){return int4{x,y,z,w};}
struct uint2 {unsigned int x,y;};
static inline uint2 make_uint2(unsigned int x,unsigned int y){return uint2{x,y};}
struct uint3 {unsigned int x,y,z;};
static inline uint3 make_uint3(unsigned int x,unsigned int y,unsigned int z){return uint3{x,y,z};}
struct char2 {char x,y;};
static inline char2 make_char2(char x,char y){return char2{x,y};}
struct char3 {char x,y,z;};
struct char4 {char x,y,z,w;};
struct short2 {short x,y;};
static inline short2 make_short2(short x,short y){return short2{x,y};}
struct short3 {short x,y,z;};
struct short4 {short x,y,z,w;};
struct float2 {float x,y;};
static inline float2 make_float2(float x,float y){return float2{x,y};}
struct float3 {float x,y,z;};
static inline float3 make_float3(float x,float y,float z){return float3{x,y,z};}
struct float4 {float x,y,z,w;};
static inline float4 make_float4(float x,float y,float z,float w){return float4{x,y,z,w};}
struct double2 {double x,y;};
static inline double2 make_double2(double x,double y){return double2{x,y};}
struct double3 {double x,y,z;};
static inline double3 make_double3(double x,double y,double z){return double3{x,y,z};}
struct double4 {double x,y,z,w;};
static inline double4 make_double4(double x,double y,double z,double w){return double4{x,y,z,w};}

template <typename... Fmt>
static inline int
ompwip_(const char * const file, const size_t line, const char * const func, std::function<void(void)>f, Fmt... fmt)
{
  if(0==omp_get_team_num()&&0==omp_get_thread_num()){
    printf("OMP WIP: ");
    printf(fmt...);
    printf(" %s:%ld %s\n", file, line, func);
  }
  f();
  return 0;
}
template <typename... Fmt>
static inline int
ompwip_(const char * const file, const size_t line, const char * const func, const char * const msg, Fmt... fmt)
{
  return ompwip_(file,line,func,[](){},msg,fmt...);
}
static inline int
ompwip_(const char * const file, const size_t line, const char * const func, std::function<void(void)>f=[](){})
{
  return ompwip_(file,line,func,f,"");
}
#define ompwip(...) ompwip_(__FILE__,__LINE__,__PRETTY_FUNCTION__,##__VA_ARGS__)

using cudaStream_t = int;  // device.h:/cudaStream_t
using CUresult = int;  // ../lib/coarse_op.cuh:/CUresult

namespace quda {
  namespace target {
    namespace omptarget {
      struct SharedCache{
        int *addr;
      };
      extern SharedCache shared_cache;

      struct LaunchParam{
        dim3 block;
        dim3 grid;
      };
      extern LaunchParam launch_param;
    }
  }
}

#include "../../quda_api.h"
