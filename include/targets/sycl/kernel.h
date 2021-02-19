#pragma once
#include <device.h>

namespace quda {

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel1D(Arg arg, sycl::nd_item<3> ndi)
  {
    Functor<Arg> f(arg);

    //auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto i = ndi.get_global_id(0);
    while (i < arg.threads.x) {
      f(i);
      //if (grid_stride) i += gridDim.x * blockDim.x; else break;
      if (grid_stride) i += ndi.get_global_range(0); else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  launchKernel1D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    //auto a = (Arg *)managed_malloc(sizeof(Arg));
    //memcpy(a, &arg, sizeof(Arg));
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    //warningQuda("launchKernel1D %s", grid_stride?"true":"false");
    //warningQuda("%s  %s", str(globalSize).c_str(), str(localSize).c_str());
    //warningQuda("%s", str(arg.threads).c_str());
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Kernel1D>(ndRange,
				 [=](sycl::nd_item<3> ndi)
				 {
				    quda::Kernel1D<Functor, Arg, grid_stride>(arg, ndi);
				  });
	     });
    //managed_free(a);
    //warningQuda("end launchKernel1D");
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel2D(Arg arg, sycl::nd_item<3> ndi)
  {
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= arg.threads.y) return;

    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  launchKernel2D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    auto a = (Arg *)managed_malloc(sizeof(Arg));
    memcpy((void*)a, &arg, sizeof(Arg));
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Kernel2D>(ndRange,
					      [=](sycl::nd_item<3> ndi)
				  {
				    quda::Kernel2D<Functor, Arg, grid_stride>(*a, ndi);
				  });
	     });
    managed_free(a);
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  void Kernel3D(Arg arg, sycl::nd_item<3> ndi)
  {
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z + blockIdx.z * blockDim.z;
    if (j >= arg.threads.y) return;
    if (k >= arg.threads.z) return;

    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  qudaError_t
  launchKernel3D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    auto a = (Arg *)managed_malloc(sizeof(Arg));
    memcpy((void*)a, &arg, sizeof(Arg));
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Kernel3D>(ndRange,
					      [=](sycl::nd_item<3> ndi)
				  {
				    quda::Kernel3D<Functor, Arg, grid_stride>(*a, ndi);
				  });
	     });
    managed_free(a);
    return QUDA_SUCCESS;
  }

  template <template <typename> class Functor, typename Arg>
  void raw_kernel(Arg arg)
  {
    Functor<Arg> f(arg);
    f();
  }

}
