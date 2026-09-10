#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <sstream>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include "shortvec.h"

#define __host__
#define __device__
#define __forceinline__ __attribute__((always_inline)) inline

#define RANGE_X 2
#define RANGE_Y 1
#define RANGE_Z 0

inline std::string str(dim3 x)
{
  std::ostringstream ss;
  ss << "(" << x.x << "," << x.y << "," << x.z << ")";
  return ss.str();
}

inline std::string str(sycl::id<3> x)
{
  std::ostringstream ss;
  ss << "(" << x[RANGE_X] << "," << x[RANGE_Y] << "," << x[RANGE_Z] << ")";
  return ss.str();
}

inline std::string str(sycl::range<3> x)
{
  std::ostringstream ss;
  ss << "(" << x[RANGE_X] << "," << x[RANGE_Y] << "," << x[RANGE_Z] << ")";
  return ss.str();
}

template <typename T> inline std::string str(std::vector<T> v)
{
  std::ostringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(ss, " "));
  return ss.str();
}

static inline auto getGroup()
{
  return sycl::ext::oneapi::this_work_item::get_work_group<3>();
}
static inline auto getNdItem()
{
  return sycl::ext::oneapi::this_work_item::get_nd_item<3>();
}

static inline unsigned int globalRange(int d) { return getNdItem().get_global_range(d); }
static inline unsigned int globalId(int d) { return getNdItem().get_global_id(d); }
static inline unsigned int groupRange(int d) { return getNdItem().get_group_range(d); }
static inline unsigned int groupId(int d) { return getNdItem().get_group(d); }
static inline unsigned int localRange(int d) { return getNdItem().get_local_range(d); }
static inline unsigned int localId(int d) { return getNdItem().get_local_id(d); }

#define globalRangeX ::globalRange(RANGE_X)
#define globalRangeY ::globalRange(RANGE_Y)
#define globalRangeZ ::globalRange(RANGE_Z)
#define globalIdX ::globalId(RANGE_X)
#define globalIdY ::globalId(RANGE_Y)
#define globalIdZ ::globalId(RANGE_Z)

#define localRangeX ::localRange(RANGE_X)
#define localRangeY ::localRange(RANGE_Y)
#define localRangeZ ::localRange(RANGE_Z)
#define localIdX ::localId(RANGE_X)
#define localIdY ::localId(RANGE_Y)
#define localIdZ ::localId(RANGE_Z)

#define groupRangeX ::groupRange(RANGE_X)
#define groupRangeY ::groupRange(RANGE_Y)
#define groupRangeZ ::groupRange(RANGE_Z)
#define groupIdX ::groupId(RANGE_X)
#define groupIdY ::groupId(RANGE_Y)
#define groupIdZ ::groupId(RANGE_Z)

inline dim3 makeDim3(sycl::range<3> &&r)
{
  dim3 d;
  d.x = r[RANGE_X];
  d.y = r[RANGE_Y];
  d.z = r[RANGE_Z];
  return d;
}

inline dim3 getGridDim()
{
  dim3 r;
  auto ndi = getNdItem();
  r.x = ndi.get_group_range(RANGE_X);
  r.y = ndi.get_group_range(RANGE_Y);
  r.z = ndi.get_group_range(RANGE_Z);
  return r;
}

inline dim3 getBlockIdx()
{
  dim3 r;
  auto ndi = getNdItem();
  r.x = ndi.get_group(RANGE_X);
  r.y = ndi.get_group(RANGE_Y);
  r.z = ndi.get_group(RANGE_Z);
  return r;
}

inline dim3 getBlockDim()
{
  dim3 r;
  auto ndi = getNdItem();
  r.x = ndi.get_local_range(RANGE_X);
  r.y = ndi.get_local_range(RANGE_Y);
  r.z = ndi.get_local_range(RANGE_Z);
  return r;
}

inline dim3 getBlockDim(sycl::nd_item<3> &ndi)
{
  dim3 r;
  r.x = ndi.get_local_range(RANGE_X);
  r.y = ndi.get_local_range(RANGE_Y);
  r.z = ndi.get_local_range(RANGE_Z);
  return r;
}

inline dim3 getThreadIdx()
{
  dim3 r;
  auto ndi = getNdItem();
  r.x = ndi.get_local_id(RANGE_X);
  r.y = ndi.get_local_id(RANGE_Y);
  r.z = ndi.get_local_id(RANGE_Z);
  return r;
}

#define gridDim getGridDim()
#define blockIdx getBlockIdx()
#define blockDim getBlockDim()
#define threadIdx getThreadIdx()

inline void syncthreads(void)
{
  group_barrier(getGroup());
}
#define __syncthreads syncthreads

namespace quda
{
  namespace device
  {
    unsigned int max_parameter_size();
  }
} // namespace quda
