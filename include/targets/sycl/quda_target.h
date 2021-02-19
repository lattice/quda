#pragma once

#include <CL/sycl.hpp>
#include <cstddef>

//#define cudaMemcpyHostToHost qudaMemcpyHostToHost
//#define cudaMemcpyHostToDevice qudaMemcpyHostToDevice
//#define cudaMemcpyDeviceToHost qudaMemcpyDeviceToHost
//#define cudaMemcpyDeviceToDevice qudaMemcpyDeviceToDevice
//#define cudaMemcpyDefault qudaMemcpyDefault

//using qudaEvent_t = double;
//using cudaEvent_t = qudaEvent_t;
using cudaStream_t = int;

//#include <error.h>
#include <shortvec.h>
//#include <stream.h>
//#include <event.h>

#define __host__
#define __device__
#define __global__
#define __forceinline__ __attribute__((always_inline)) inline
#define __launch_bounds__(x)

// FIXME
#define __constant__ static
#define __shfl_down_sync(m, x, o) x

//#define rsqrt(x) (1/sqrt(x))
//inline float rsqrt(float x) { return 1.0f/sqrt(x); }
//inline void sincos(float x, float *s, float *c)
//{
  //*s = sin(x);
  //*c = cos(x);
  //*s = sycl::sincos(x, c);
//}

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

//extern "C" {
//size_t __builtin_get_global_id(size_t);
//size_t __builtin_get_global_range(size_t);
//size_t __builtin_get_local_id(size_t);
//size_t __builtin_get_local_range(size_t);
//}
//#define GR_(i) cl::sycl::detail::Builder::getNDItem<3>().get_global_range(i)
//#define GI_(i) (cl::sycl::detail::Builder::getNDItem<3>().get_global_id(i))
//#define LR_(i) (cl::sycl::detail::Builder::getNDItem<3>().get_local_range(i))
//#define LI_(i) (cl::sycl::detail::Builder::getNDItem<3>().get_local_id(i))
//#define gridDim dim3{(uint)GR_(0),(uint)GR_(1),(uint)GR_(2)}
//#define blockIdx dim3{(uint)GI_(0),(uint)GI_(1),(uint)GI_(2)}
//#define blockDim dim3{(uint)LR_(0),(uint)LR_(1),(uint)LR_(2)}
//#define threadIdx dim3{(uint)LI_(0),(uint)LI_(1),(uint)LI_(2)}
//auto b = __spirv::initGlobalInvocationId<3, sycl::id<3>>();
//auto c = __spirv::initGlobalSize<3, sycl::range<3>>();
//auto gr = cl::sycl::detail::Builder::groupRange<3>();

#ifdef __SYCL_DEVICE_ONLY__
static inline auto getGroup()
{
  return sycl::detail::Builder::getElement(static_cast<sycl::group<3>*>(nullptr));
}
static inline auto getNdItem()
{
  return sycl::detail::Builder::getNDItem<3>();
}
#else
#include <util_quda.h>
static inline sycl::group<3> getGroup()
{
  sycl::range<3> r{0,0,0};
  sycl::id<3> i{0,0,0};
  errorQuda("Can't use getGroup() in host code");
  return sycl::detail::Builder::createGroup(r, r, i);
}
static inline sycl::nd_item<3> getNdItem()
{
  sycl::range<3> r{0,0,0};
  sycl::id<3> i{0,0,0};
  auto g = sycl::detail::Builder::createItem<3,true>(r, i, i);
  auto l = sycl::detail::Builder::createItem<3,false>(r, i);
  errorQuda("Can't use getNdItem() in host code");
  return sycl::detail::Builder::createNDItem(g, l, getGroup());
}
#endif


inline dim3 getGridDim()
{
  dim3 r;
#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = cl::sycl::detail::Builder::getNDItem<3>();
  r.x = ndi.get_group_range(0);
  r.y = ndi.get_group_range(1);
  r.z = ndi.get_group_range(2);
#endif
  return r;
}

inline dim3 getBlockIdx()
{
  dim3 r;
#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = cl::sycl::detail::Builder::getNDItem<3>();
  r.x = ndi.get_group(0);
  r.y = ndi.get_group(1);
  r.z = ndi.get_group(2);
#endif
  return r;
}

inline dim3 getBlockDim()
{
  dim3 r;
#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = cl::sycl::detail::Builder::getNDItem<3>();
  r.x = ndi.get_local_range(0);
  r.y = ndi.get_local_range(1);
  r.z = ndi.get_local_range(2);
#endif
  return r;
}

inline dim3 getThreadIdx()
{
  dim3 r;
#ifdef __SYCL_DEVICE_ONLY__
  auto ndi = cl::sycl::detail::Builder::getNDItem<3>();
  r.x = ndi.get_local_id(0);
  r.y = ndi.get_local_id(1);
  r.z = ndi.get_local_id(2);
#endif
  return r;
}

#define gridDim getGridDim()
#define blockIdx getBlockIdx()
#define blockDim getBlockDim()
#define threadIdx getThreadIdx()

namespace quda
{
  namespace device
  {
    sycl::queue get_target_stream(const qudaStream_t &stream);
    sycl::queue defaultQueue(void);
  }
}

#define qudaLaunchKernel(a,b,c,...) \
  qudaLaunchKernel_(__FILE__,__LINE__,__func__,#a)
