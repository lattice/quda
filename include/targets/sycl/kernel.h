#pragma once

#include <device.h>
#include <tunable_kernel.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <utility>
#include <quda_sycl_api.h>

namespace quda {

  // Kernel1D

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1DImpl(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += globalRangeX; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1DImplB(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);
    auto tid = globalIdX;
    auto nid = globalRangeX;
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  struct Kernel1DS {
    using SpecialOpsT = getSpecialOps<Functor<Arg>>;
    Kernel1DS(const Arg &arg, const sycl::nd_item<3> &ndi)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel1DImplB<Functor,Arg,grid_stride>(arg, ndi);
#else
      Kernel1DImpl<Functor,Arg,grid_stride>(arg, ndi);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel1D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    static_assert(!hasSpecialOps<Functor<Arg>>);
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //  return QUDA_ERROR;
    //}
#if 0
    if (localSize[RANGE_X] > arg.threads.x) {
      localSize[RANGE_X] = arg.threads.x;
      globalSize[RANGE_X] = arg.threads.x;
    } else if (grid_stride) {
      if (globalSize[RANGE_X] > arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    } else {
      if (globalSize[RANGE_X] != arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    }
#endif
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel1D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  SpecialOps: %s\n", typeid(getSpecialOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>, needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //if(needsFullBlock<Functor<Arg>>) {
    //std::ostringstream what;
    //what << "localSizeX (" << localSize[RANGE_X] << ") % warp_size (" << device::warp_size() << ") != 0";
    //target::sycl::set_error(what.str(), "pre-launch", __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
    //return QUDA_ERROR;
    //}
    //}
    //if (arg.threads.x%localSize[RANGE_X] != 0) {
      //warningQuda("arg.threads.x (%i) %% localSize X (%lu) != 0", arg.threads.x, localSize[RANGE_X]);
    //  return QUDA_ERROR;
    //}
    //sycl::nd_range<3> ndRange{globalSize, localSize};
    //err = launch<Kernel1DS<Functor, Arg, grid_stride>>(stream, ndRange, arg);
    err = launch<Kernel1DS<Functor, Arg, grid_stride>>(tp, stream, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel1D\n");
    }
    return err;
  }

  // Kernel2D

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename ...S>
  std::enable_if_t<!needsFullBlock<Functor<Arg>>, void>
  Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    Functor<Arg> f(arg);
    if constexpr (hasSpecialOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride) i += globalRangeX; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride, typename ...S>
  std::enable_if_t<needsFullBlock<Functor<Arg>>, void>
  Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    Functor<Arg> f(arg);
    if constexpr (hasSpecialOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }

    bool active = true;
    auto j = globalIdY;
    if (j >= arg.threads.y) active = false;
    auto i = globalIdX;
    while (i-localIdX < arg.threads.x) {
      if (i >= arg.threads.x) active = false;
      f.template operator()<true>(i, j, active);
      if (grid_stride) i += globalRangeX; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2DImplB(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);
    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto tid = globalIdX;
    auto nid = globalRangeX;
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i, j);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  struct Kernel2DS {
    using SpecialOpsT = getSpecialOps<Functor<Arg>>;
    template <typename ...S>
    Kernel2DS(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel2DImplB<Functor,Arg,grid_stride>(arg, ndi);
#else
      Kernel2DImpl<Functor,Arg,grid_stride>(arg, ndi, smem...);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel2D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //  return QUDA_ERROR;
    //}
#if 0
    if (localSize[RANGE_X] > arg.threads.x) {
      localSize[RANGE_X] = arg.threads.x;
      globalSize[RANGE_X] = arg.threads.x;
    } else if (grid_stride) {
      if (globalSize[RANGE_X] > arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    } else {
      if (globalSize[RANGE_X] != arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    }
#endif
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  SpecialOps: %s\n", typeid(getSpecialOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>, needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //if(needsFullBlock<Functor<Arg>>) {
    //std::ostringstream what;
    //what << "localSizeX (" << localSize[RANGE_X] << ") % warp_size (" << device::warp_size() << ") != 0";
    //target::sycl::set_error(what.str(), "pre-launch", __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
    //return QUDA_ERROR;
    //}
    //}
    //if (arg.threads.x%localSize[RANGE_X] != 0) {
      //warningQuda("arg.threads.x (%i) %% localSize X (%lu) != 0", arg.threads.x, localSize[RANGE_X]);
    //  return QUDA_ERROR;
    //}
    //if (globalSize[RANGE_Y] != arg.threads.y) {
      //warningQuda("globalSize Y (%lu) != arg.threads.y (%i)", globalSize[RANGE_Y], arg.threads.y);
    //  return QUDA_ERROR;
    //}
    //auto t0 = __rdtsc();
    sycl::nd_range<3> ndRange{globalSize, localSize};
    err = launch<Kernel2DS<Functor, Arg, grid_stride>>(stream, ndRange, arg);
    //auto t1 = __rdtsc();
    //printf("%llu\n", t1-t0);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel2D\n");
    }
    return err;
  }

  // Kernel3D

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename ...S>
  std::enable_if_t<!needsFullBlock<Functor<Arg>>, void>
  Kernel3DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    Functor<Arg> f(arg);
    if constexpr (hasSpecialOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto k = globalIdZ;
    if (k >= arg.threads.z) return;
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride) i += globalRangeX; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride, typename ...S>
  std::enable_if_t<needsFullBlock<Functor<Arg>>, void>
  Kernel3DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
  {
    Functor<Arg> f(arg);
    if constexpr (hasSpecialOps<Functor<Arg>>) {
      f.setNdItem(ndi);
    }
    if constexpr (needsSharedMem<Functor<Arg>>) {
      f.setSharedMem(smem...);
    }

    bool active = true;
    auto j = globalIdY;
    if (j >= arg.threads.y) active = false;
    auto k = globalIdZ;
    if (k >= arg.threads.z) active = false;
    auto i = globalIdX;
    while (i-localIdX < arg.threads.x) {
      if (i >= arg.threads.x) active = false;
      f.template operator()<true>(i, j, k, active);
      if (grid_stride) i += globalRangeX; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride>
  void Kernel3DImplB(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto k = globalIdZ;
    if (k >= arg.threads.z) return;
    auto tid = globalIdX;
    auto nid = globalRangeX;
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i, j, k);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  struct Kernel3DS {
    using SpecialOpsT = getSpecialOps<Functor<Arg>>;
    template <typename ...S>
    Kernel3DS(const Arg &arg, const sycl::nd_item<3> &ndi, S ...smem)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel3DImplB<Functor,Arg,grid_stride>(arg, ndi);
#else
      Kernel3DImpl<Functor,Arg,grid_stride>(arg, ndi, smem...);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel3D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
#if 0
    if (localSize[RANGE_X] > arg.threads.x) {
      localSize[RANGE_X] = arg.threads.x;
      globalSize[RANGE_X] = arg.threads.x;
    } else if (grid_stride) {
      if (globalSize[RANGE_X] > arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    } else {
      if (globalSize[RANGE_X] != arg.threads.x) {
	globalSize[RANGE_X] = ((arg.threads.x+localSize[RANGE_X]-1)/localSize[RANGE_X])*localSize[RANGE_X];
      }
    }
#endif
    //printfQuda("Kernel3D %s\n", typeid(Functor<Arg>).name());
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel3D param grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  SpecialOps: %s\n", typeid(getSpecialOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>, needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
      //fflush(stdout);
    }
    //if (localSize[RANGE_X] % device::warp_size() != 0) {
    //if(needsFullBlock<Functor<Arg>>) {
    //std::ostringstream what;
    //what << "localSizeX (" << localSize[RANGE_X] << ") % warp_size (" << device::warp_size() << ") != 0";
    //target::sycl::set_error(what.str(), "pre-launch", __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
    //return QUDA_ERROR;
    //}
    //}
    //if (arg.threads.x%localSize[RANGE_X] != 0) {
    //warningQuda("arg.threads.x (%i) %% localSize X (%lu) != 0", arg.threads.x, localSize[RANGE_X]);
    //return QUDA_ERROR;
    //}
    //if (globalSize[RANGE_Y] != arg.threads.y) {
      //warningQuda("globalSize Y (%lu) != arg.threads.y (%i)", globalSize[RANGE_Y], arg.threads.y);
    //  return QUDA_ERROR;
    //}
    //if (globalSize[RANGE_Z] != arg.threads.z) {
      //warningQuda("globalSize Z (%lu) != arg.threads.z (%i)", globalSize[RANGE_Z], arg.threads.z);
    //  return QUDA_ERROR;
    //}
    sycl::nd_range<3> ndRange{globalSize, localSize};
    err = launch<Kernel3DS<Functor, Arg, grid_stride>>(stream, ndRange, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel3D\n");
      //fflush(stdout);
    }
    return err;
  }

}
