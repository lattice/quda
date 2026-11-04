#pragma once

#include <device.h>
#include <tunable_kernel.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <utility>
#include <quda_sycl_api.h>

namespace quda
{

  // Kernel1D

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1DImpl(const Arg &arg, const sycl::nd_item<3> &)
  {
    Functor<Arg> f(arg);
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i);
      if (grid_stride)
        i += globalRangeX;
      else
        break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1DImplB(const Arg &arg, const sycl::nd_item<3> &)
  {
    Functor<Arg> f(arg);
#if 0
    auto tid = globalIdX;
    auto nid = globalRangeX;
    auto n = arg.threads.x;
    auto i0 = (tid * n) / nid;
    auto i1 = ((tid + 1) * n) / nid;
    for (auto i = i0; i < i1; i++) { f(i); }
#endif
    // keep warp together
    auto n = arg.threads.x;
    auto tid = globalIdX / QUDA_WARP_SIZE;
    auto nid = globalRangeX / QUDA_WARP_SIZE;
    auto i0 = (tid * n) / nid + (localIdX % QUDA_WARP_SIZE);
    auto i1 = ((tid + 1) * n) / nid;
    for (auto i = i0; i < i1; i+=QUDA_WARP_SIZE) { f(i); }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> struct Kernel1DS {
    using KernelOpsT = getKernelOps<Functor<Arg>>;
    Kernel1DS(const Arg &arg, const sycl::nd_item<3> &ndi)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel1DImplB<Functor, Arg, grid_stride>(arg, ndi);
#else
      Kernel1DImpl<Functor, Arg, grid_stride>(arg, ndi);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t Kernel1D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    static_assert(!hasKernelOps<Functor<Arg>>);
    auto err = QUDA_SUCCESS;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      auto globalSize = globalRange(tp);
      auto localSize = localRange(tp);
      printfQuda("Kernel1D grid_stride: %s  sizeof(arg): %lu\n", grid_stride ? "true" : "false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(), str(localSize).c_str(),
                 str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>,
                 needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    err = launch<Kernel1DS<Functor, Arg, grid_stride>>(tp, stream, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("end Kernel1D\n"); }
    return err;
  }

  // Kernel2D

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  std::enable_if_t<!needsFullBlock<Functor<Arg>>, void> Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi,
                                                                     S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride)
        i += globalRangeX;
      else
        break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  std::enable_if_t<needsFullBlock<Functor<Arg>>, void> Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);

    bool active = true;
    auto j = globalIdY;
    if (j >= arg.threads.y) active = false;
    auto i = globalIdX;
    while (i - localIdX < arg.threads.x) {
      if (i >= arg.threads.x) active = false;
      f.template operator()<true>(i, j, active);
      if (grid_stride)
        i += globalRangeX;
      else
        break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  void Kernel2DImplB(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);
    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto n = arg.threads.x;
    auto tid = globalIdX / QUDA_WARP_SIZE;
    auto nid = globalRangeX / QUDA_WARP_SIZE;
    auto i0 = (tid * n) / nid + (localIdX % QUDA_WARP_SIZE);
    auto i1 = ((tid + 1) * n) / nid;
    for (auto i = i0; i < i1; i+=QUDA_WARP_SIZE) { f(i, j); }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> struct Kernel2DS {
    using KernelOpsT = getKernelOps<Functor<Arg>>;
    template <typename... S> Kernel2DS(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel2DImplB<Functor, Arg, grid_stride>(arg, ndi, smem...);
#else
      Kernel2DImpl<Functor, Arg, grid_stride>(arg, ndi, smem...);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t Kernel2D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      auto globalSize = globalRange(tp);
      auto localSize = localRange(tp);
      printfQuda("Kernel2D grid_stride: %s  sizeof(arg): %lu\n", grid_stride ? "true" : "false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(), str(localSize).c_str(),
                 str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>,
                 needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    err = launch<Kernel2DS<Functor, Arg, grid_stride>>(tp, stream, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("end Kernel2D\n"); }
    return err;
  }

  // Kernel3D

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  std::enable_if_t<!needsFullBlock<Functor<Arg>>, void> Kernel3DImpl(const Arg &arg, const sycl::nd_item<3> &ndi,
                                                                     S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto k = globalIdZ;
    if (k >= arg.threads.z) return;
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride)
        i += globalRangeX;
      else
        break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  std::enable_if_t<needsFullBlock<Functor<Arg>>, void> Kernel3DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);

    bool active = true;
    auto j = globalIdY;
    if (j >= arg.threads.y) active = false;
    auto k = globalIdZ;
    if (k >= arg.threads.z) active = false;
    auto i = globalIdX;
    while (i - localIdX < arg.threads.x) {
      if (i >= arg.threads.x) active = false;
      f.template operator()<true>(i, j, k, active);
      if (grid_stride)
        i += globalRangeX;
      else
        break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride, typename... S>
  void Kernel3DImplB(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
  {
    Ftor<Functor<Arg>> f(arg, ndi, smem...);

    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    auto k = globalIdZ;
    if (k >= arg.threads.z) return;
    auto n = arg.threads.x;
    auto tid = globalIdX / QUDA_WARP_SIZE;
    auto nid = globalRangeX / QUDA_WARP_SIZE;
    auto i0 = (tid * n) / nid + (localIdX % QUDA_WARP_SIZE);
    auto i1 = ((tid + 1) * n) / nid;
    for (auto i = i0; i < i1; i+=QUDA_WARP_SIZE) { f(i, j, k); }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> struct Kernel3DS {
    using KernelOpsT = getKernelOps<Functor<Arg>>;
    template <typename... S> Kernel3DS(const Arg &arg, const sycl::nd_item<3> &ndi, S... smem)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel3DImplB<Functor, Arg, grid_stride>(arg, ndi, smem...);
#else
      Kernel3DImpl<Functor, Arg, grid_stride>(arg, ndi, smem...);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t Kernel3D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      auto globalSize = globalRange(tp);
      auto localSize = localRange(tp);
      printfQuda("Kernel3D param grid_stride: %s  sizeof(arg): %lu\n", grid_stride ? "true" : "false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(), str(localSize).c_str(),
                 str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>,
                 needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    err = launch<Kernel3DS<Functor, Arg, grid_stride>>(tp, stream, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel3D\n");
    }
    return err;
  }

} // namespace quda
