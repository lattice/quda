#pragma once

#include <device.h>
#include <tunable_kernel.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <utility>
#include <quda_sycl_api.h>

namespace quda {

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
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    //if (globalSize[0] > arg.threads.x) {
    //  globalSize[0] = arg.threads.x;
    //  if (localSize[0] > globalSize[0]) localSize[0] = globalSize[0];
    //} else {
    //  if (arg.threads.x%tp.block.x) {
    //warningQuda("K1D %s nondiv %s %s %s", grid_stride?"true":"false",
    //	    str(arg.threads).c_str(), str(tp.block).c_str(), typeid(Arg).name());
    //}
    //}
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel1D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
    }
    sycl::nd_range<3> ndRange{globalSize, localSize};
    err = launch<Kernel1DS<Functor, Arg, grid_stride>>(stream, ndRange, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel1D\n");
    }
    return err;
  }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);
    auto i = globalIdX;
    auto j = globalIdY;
    if (j >= arg.threads.y) return;
    while (i < arg.threads.x) {
      f(i, j);
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
    Kernel2DS(const Arg &arg, const sycl::nd_item<3> &ndi)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel2DImplB<Functor,Arg,grid_stride>(arg, ndi);
#else
      Kernel2DImpl<Functor,Arg,grid_stride>(arg, ndi);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel2D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    //if(grid_stride==false) {
    //  globalSize[RANGE_X] = arg.threads.x;
    //  globalSize[RANGE_Y] = arg.threads.y;
    //}
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
    }
    if (globalSize[RANGE_Y]!=arg.threads.y) {
      errorQuda("globalSize Y (%lu) != arg.threads.y (%i)", globalSize[RANGE_Y], arg.threads.y);
    }
    //if (arg.threads.x%tp.block.x+arg.threads.y%tp.block.y) {
    //  if (Arg::hasBlockOps()) {
    //warningQuda("BlockOps");
    //}
    //warningQuda("K2D %s nondiv %s %s %s", grid_stride?"true":"false",
    //	  str(arg.threads).c_str(), str(tp.block).c_str(), typeid(Arg).name());
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


  template <template <typename> class Functor, typename Arg, bool grid_stride>
  void Kernel3DImpl(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);

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
    Kernel3DS(const Arg &arg, const sycl::nd_item<3> &ndi)
    {
#ifdef QUDA_THREADS_BLOCKED
      Kernel3DImplB<Functor,Arg,grid_stride>(arg, ndi);
#else
      Kernel3DImpl<Functor,Arg,grid_stride>(arg, ndi);
#endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel3D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    if (globalSize[RANGE_Y]!=arg.threads.y) {
      globalSize[RANGE_Y] = arg.threads.y;
    }
    if (globalSize[RANGE_Z]!=arg.threads.z) {
      globalSize[RANGE_Z] = arg.threads.z;
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel3D param grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      //fflush(stdout);
    }
    if (globalSize[RANGE_Y]!=arg.threads.y) {
      errorQuda("globalSize Y (%lu) != arg.threads.y (%i)", globalSize[RANGE_Y], arg.threads.y);
    }
    if (globalSize[RANGE_Z]!=arg.threads.z) {
      errorQuda("globalSize Z (%lu) != arg.threads.z (%i)", globalSize[RANGE_Z], arg.threads.z);
    }
    //if (arg.threads.x%tp.block.x+arg.threads.y%tp.block.y+arg.threads.z%tp.block.z) {
    //warningQuda("K3Da %s nondiv %s %s %s", grid_stride?"true":"false",
    //	  str(arg.threads).c_str(), str(tp.block).c_str(), typeid(Arg).name());
    //}
    //if (localSize[0]>arg.threads.x) {
    //  localSize[0] = arg.threads.x;
    //  printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
    //	 str(localSize).c_str(), str(arg.threads).c_str());
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
