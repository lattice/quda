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
    Functor<Arg> f(const_cast<Arg&>(arg));
    auto i = globalIdX;
    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += globalRangeX; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1DImplB(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(const_cast<Arg&>(arg));
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
  qudaError_t
  Kernel1D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
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
    try {
      q.submit([&](sycl::handler &h) {
	//h.parallel_for<class Kernel1D>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) {
#ifdef QUDA_THREADS_BLOCKED
	     quda::Kernel1DImplB<Functor, Arg, grid_stride>(arg, ndi);
#else
	     quda::Kernel1DImpl<Functor, Arg, grid_stride>(arg, ndi);
#endif
	   });
      });
    } catch (sycl::exception const& e) {
      auto what = e.what();
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n", what);
      }
      target::sycl::set_error(what, "submit", __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());
      err = QUDA_ERROR;
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel1D\n");
    }
    return err;
  }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    //Functor<Arg> f(const_cast<Arg&>(arg));
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
    //Functor<Arg> f(const_cast<Arg&>(arg));
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
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
    }
    //if (arg.threads.x%tp.block.x+arg.threads.y%tp.block.y) {
    //  if (Arg::hasBlockOps()) {
    //warningQuda("BlockOps");
    //}
    //warningQuda("K2D %s nondiv %s %s %s", grid_stride?"true":"false",
    //	  str(arg.threads).c_str(), str(tp.block).c_str(), typeid(Arg).name());
    //}
    //auto t0 = __rdtsc();
    try {
      q.submit([&](sycl::handler &h) {
	//h.parallel_for<class Kernel2D>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) {
#ifdef QUDA_THREADS_BLOCKED
	     quda::Kernel2DImplB<Functor, Arg, grid_stride>(arg, ndi);
#else
	     quda::Kernel2DImpl<Functor, Arg, grid_stride>(arg, ndi);
	     //quda::Kernel2DImpl<Functor, Arg, false>(arg, ndi);
#endif
	   });
      });
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
    //auto t1 = __rdtsc();
    //printf("%llu\n", t1-t0);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel2D\n");
    }
    return err;
  }
#if 0
  template <typename F>
  void Kernel2DImplBx(const F &f, const dim3 &threads, const sycl::nd_item<3> &ndi)
  {
    auto j = globalIdY;
    if (j >= threads.y) return;
    auto tid = globalIdX;
    auto nid = globalRangeX;
    auto n = threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i, j);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  Kernel2Dx(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchKernel2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
    //auto t0 = __rdtsc();
    dim3 thr = arg.threads;
    Arg *arg2 = static_cast<Arg*>(managed_malloc(sizeof(Arg)));
    q.memcpy(arg2, &arg, sizeof(Arg));
    Functor<Arg> f(*arg2);
    Functor<Arg> *f2 = static_cast<Functor<Arg>*>(device_malloc(sizeof(Functor<Arg>)));
    q.memcpy(f2, &f, sizeof(f));
    const Functor<Arg> *const f3 = const_cast<const Functor<Arg>*const>(f2);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class Kernel2D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
	   quda::Kernel2DImplBx<Functor<Arg>>(*f3, thr, ndi);
	 });
    });
    //auto t1 = __rdtsc();
    //printf("%llu\n", t1-t0);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end launchKernel2D\n");
    }
    return QUDA_SUCCESS;
  }
#endif

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

  // kernel args
  template <template <typename> class Functor, typename Arg,
	    bool grid_stride = false>
  std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
  Kernel3D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel3D param grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      //fflush(stdout);
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
    try {
      q.submit([&](sycl::handler& h) {
	//h.parallel_for<struct Kernel3Da>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) [[intel::reqd_sub_group_size(QUDA_WARP_SIZE)]] {
#ifdef QUDA_THREADS_BLOCKED
	     quda::Kernel3DImplB<Functor, Arg, grid_stride>(arg, ndi);
#else
	     quda::Kernel3DImpl<Functor, Arg, grid_stride>(arg, ndi);
#endif
	   });
      });
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel3D\n");
      //fflush(stdout);
    }
    return err;
  }

  // const args
  template <template <typename> class Functor, typename Arg,
	    bool grid_stride = false>
  std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
  Kernel3D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    auto globalSize = globalRange(tp);
    auto localSize = localRange(tp);
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Kernel3D const grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
    }
    //if (arg.threads.x%tp.block.x+arg.threads.y%tp.block.y+arg.threads.z%tp.block.z) {
    //warningQuda("K3Dc %s nondiv %s %s %s", grid_stride?"true":"false",
    //	  str(arg.threads).c_str(), str(tp.block).c_str(), typeid(Arg).name());
    //}
    //if (localSize[0]>arg.threads.x) {
    //  localSize[0] = arg.threads.x;
    //  printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
    //	 str(localSize).c_str(), str(arg.threads).c_str());
    //}
    //while(true) {
    //  auto nl = localSize[0] * localSize[1] * localSize[2];
    //  if(nl<QUDA_WARP_SIZE) {
    //localSize[1]
    //warningQuda("allocating kernel args");
    //auto p = device_malloc(sizeof(arg));
    //q.memcpy(p, &arg, sizeof(arg));
    //sycl::buffer<const Arg,1> buf{&arg, sycl::range(sizeof(arg))};
    //sycl::buffer<const char,1>
    //  buf{reinterpret_cast<const char*>(&arg), sycl::range(sizeof(arg))};
    auto size = sizeof(arg);
    auto p = device::get_arg_buf(stream, size);
    memcpy(p, &arg, size);
    //auto evnt = q.memcpy(p, &arg, size);
    try {
      q.submit([&](sycl::handler& h) {
	//auto a = buf.get_access(h);
	//auto a = buf.get_access<sycl::access_mode::read>(h);
	//auto a = buf.get_access<sycl::access::mode::read,
	//			sycl::access::target::constant_buffer>(h);
	//h.parallel_for<class Kernel3Dc>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) [[intel::reqd_sub_group_size(QUDA_WARP_SIZE)]] {
	     //Arg *arg2 = static_cast<Arg *>(p);
	     //const Arg *arg2 = a.get_pointer();
	     //const char *p = a.get_pointer();
	     const Arg *arg2 = reinterpret_cast<const Arg*>(p);
#ifdef QUDA_THREADS_BLOCKED
	     quda::Kernel3DImplB<Functor, Arg, grid_stride>(*arg2, ndi);
#else
	     quda::Kernel3DImpl<Functor, Arg, grid_stride>(*arg2, ndi);
#endif
	   });
      });
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
    //device::wasSynced(stream);
    //evnt.wait();
    //q.wait();
    //device_free(p);   //  FIXME: host task
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end Kernel3D\n");
    }
    return err;
  }

}
