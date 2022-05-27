#pragma once

#include <CL/sycl.hpp>
#include <cstddef>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif


using cudaStream_t = int;

//#include <error.h>
#include "shortvec.h"
//#include <stream.h>
//#include <event.h>

#define __host__
#define __device__
//#define __global__
#define __forceinline__ __attribute__((always_inline)) inline
//#define __launch_bounds__(x)

// FIXME
//#define __constant__ static
#define __shared__
//#define __shfl_down_sync(m, x, o) x

inline std::string str(dim3 x)
{
  std::ostringstream ss;
  ss << "(" << x.x << "," << x.y << "," << x.z << ")";
  return ss.str();
}

inline std::string str(sycl::id<3> x)
{
  std::ostringstream ss;
  ss << "(" << x[0] << "," << x[1] << "," << x[2] << ")";
  return ss.str();
}

inline std::string str(sycl::range<3> x)
{
  std::ostringstream ss;
  ss << "(" << x[0] << "," << x[1] << "," << x[2] << ")";
  return ss.str();
}

template <typename T>
inline std::string str(std::vector<T> v)
{
  std::ostringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(ss, " "));
  return ss.str();
}

static inline auto getGroup()
{
  //return sycl::this_group<3>();
  return sycl::ext::oneapi::experimental::this_group<3>();
}
static inline auto getNdItem()
{
  //return sycl::this_nd_item<3>();
  return sycl::ext::oneapi::experimental::this_nd_item<3>();
}

inline dim3 getGridDim()
{
  dim3 r;
  //#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = getNdItem();
  r.x = ndi.get_group_range(0);
  r.y = ndi.get_group_range(1);
  r.z = ndi.get_group_range(2);
  //#endif
  return r;
}

inline dim3 getBlockIdx()
{
  dim3 r;
  //#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = getNdItem();
  r.x = ndi.get_group(0);
  r.y = ndi.get_group(1);
  r.z = ndi.get_group(2);
  //#endif
  return r;
}

inline dim3 getBlockDim()
{
  dim3 r;
  //#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = getNdItem();
  r.x = ndi.get_local_range(0);
  r.y = ndi.get_local_range(1);
  r.z = ndi.get_local_range(2);
  //#endif
  return r;
}

inline dim3 getThreadIdx()
{
  dim3 r;
#ifdef __SYCL_DEVICE_ONLY__
  //auto ndi = cl::sycl::detail::Builder::getNDItem<3>();
  auto ndi = getNdItem();
  r.x = ndi.get_local_id(0);
  r.y = ndi.get_local_id(1);
  r.z = ndi.get_local_id(2);
#endif
  return r;
}

inline uint getLocalLinearId()
{
  int id = 0;
#ifdef __SYCL_DEVICE_ONLY__
  //auto ndi = getNdItem();
  //auto ndi = sycl::this_nd_item<3>();
  auto ndi = getNdItem();
  id = ndi.get_local_linear_id();
#endif
  return id;
}

inline void __syncthreads(void)
{
  //auto ndi = getNdItem();
  //ndi.barrier();
  group_barrier(getGroup());
}

#define gridDim getGridDim()
#define blockIdx getBlockIdx()
#define blockDim getBlockDim()
#define threadIdx getThreadIdx()