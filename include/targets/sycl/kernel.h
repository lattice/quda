#pragma once
#include <device.h>
#include <tune_quda.h>
#include <kernel_helper.h>
#include <target_device.h>
#include <utility>

namespace quda {

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1D(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(const_cast<Arg&>(arg));
    auto i = ndi.get_global_id(0);
    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += ndi.get_global_range(0); else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1Db(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(const_cast<Arg&>(arg));
    auto tid = ndi.get_global_id(0);
    auto nid = ndi.get_global_range(0);
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  launchKernel1D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, 1, 1};
    sycl::range<3> localSize{tp.block.x, 1, 1};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchKernel1D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class Kernel1D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
#ifdef QUDA_THREADS_BLOCKED
	   quda::Kernel1Db<Functor, Arg, grid_stride>(arg, ndi);
#else
	   quda::Kernel1D<Functor, Arg, grid_stride>(arg, ndi);
#endif
	 });
    });
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("  end launchKernel1D\n");
    }
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2D(const Arg &arg, sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(const_cast<Arg&>(arg));
    auto i = ndi.get_global_id(0);
    auto j = ndi.get_global_id(1);
    if (j >= arg.threads.y) return;
    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride) i += ndi.get_global_range(0); else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2Db(const Arg &arg, sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(const_cast<Arg&>(arg));
    auto j = ndi.get_global_id(1);
    if (j >= arg.threads.y) return;
    auto tid = ndi.get_global_id(0);
    auto nid = ndi.get_global_range(0);
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i, j);
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  launchKernel2D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, 1};
    sycl::range<3> localSize{tp.block.x, tp.block.y, 1};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchKernel2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class Kernel2D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
#ifdef QUDA_THREADS_BLOCKED
	   quda::Kernel2Db<Functor, Arg, grid_stride>(arg, ndi);
#else
	   quda::Kernel2D<Functor, Arg, grid_stride>(arg, ndi);
#endif
	 });
    });
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("  end launchKernel2D\n");
    }
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride>
  void Kernel3D(const Arg &arg, sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);

    auto j = ndi.get_global_id(1);
    if (j >= arg.threads.y) return;
    auto k = ndi.get_global_id(2);
    if (k >= arg.threads.z) return;
    auto i = ndi.get_global_id(0);
    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride) i += ndi.get_global_range(0); else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride>
  void Kernel3Db(const Arg &arg, sycl::nd_item<3> &ndi)
  {
    Functor<Arg> f(arg);

    auto j = ndi.get_global_id(1);
    if (j >= arg.threads.y) return;
    auto k = ndi.get_global_id(2);
    if (k >= arg.threads.z) return;
    auto tid = ndi.get_global_id(0);
    auto nid = ndi.get_global_range(0);
    auto n = arg.threads.x;
    auto i0 = (tid*n)/nid;
    auto i1 = ((tid+1)*n)/nid;
    for(auto i=i0; i<i1; i++) {
      f(i, j, k);
    }
  }

  template <template <typename> class Functor, typename Arg,
	    bool grid_stride = false>
  std::enable_if_t<device::use_kernel_arg<Arg>(), qudaError_t>
  launchKernel3D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y,
      tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchKernel3D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class Kernel3D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
#ifdef QUDA_THREADS_BLOCKED
	   quda::Kernel3Db<Functor, Arg, grid_stride>(arg, ndi);
#else
	   quda::Kernel3D<Functor, Arg, grid_stride>(arg, ndi);
#endif
	 });
    });
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("  end launchKernel3D\n");
    }
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg,
	    bool grid_stride = false>
  std::enable_if_t<!device::use_kernel_arg<Arg>(), qudaError_t>
  launchKernel3D(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y,
      tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchKernel3D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
    //warningQuda("allocating kernel args");
    //auto p = device_malloc(sizeof(arg));
    //q.memcpy(p, &arg, sizeof(arg));
    //sycl::buffer<const Arg,1> buf{&arg, sycl::range(sizeof(arg))};
    sycl::buffer<const char,1>
      buf{reinterpret_cast<const char*>(&arg), sycl::range(sizeof(arg))};
    q.submit([&](sycl::handler& h) {
      //auto a = buf.get_access(h);
      //auto a = buf.get_access<sycl::access_mode::read>(h);
      auto a = buf.get_access<sycl::access::mode::read,
			      sycl::access::target::constant_buffer>(h);
      h.parallel_for<class Kernel3D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
	   //Arg *arg2 = static_cast<Arg *>(p);
	   //const Arg *arg2 = a.get_pointer();
	   const char *p = a.get_pointer();
	   const Arg *arg2 = reinterpret_cast<const Arg*>(p);
#ifdef QUDA_THREADS_BLOCKED
	   quda::Kernel3Db<Functor, Arg, grid_stride>(*arg2, ndi);
#else
	   quda::Kernel3D<Functor, Arg, grid_stride>(*arg2, ndi);
#endif
	 });
    });
    //q.wait();
    //device_free(p);   //  FIXME: host task
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("  end launchKernel3D\n");
    }
    return QUDA_SUCCESS;
  }

}
