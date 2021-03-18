#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ void Reduction2D(Arg arg, sycl::nd_item<3> ndi)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    //auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx = ndi.get_global_id(0);
    //auto j = threadIdx.y;
    auto j = ndi.get_local_id(1);

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      //if (grid_stride) idx += blockDim.x * gridDim.x; else break;
      if (grid_stride) idx += ndi.get_global_range(0); else break;
    }

    // perform final inter-block reduction and write out result
    quda::reduce<block_size_x, block_size_y>(arg, t, value);
  }
  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  qudaError_t
  launchReduction2D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    //auto a = (Arg *)managed_malloc(sizeof(Arg));
    //memcpy((void*)a, &arg, sizeof(Arg));
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    warningQuda("launchReduction2D %s", grid_stride?"true":"false");
    warningQuda("%s  %s", str(globalSize).c_str(), str(localSize).c_str());
    warningQuda("%s", str(arg.threads).c_str());
    //arg.debug();
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Reduction2D>(ndRange,
					  [=](sycl::nd_item<3> ndi)
					  {
					    quda::Reduction2D<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
					  });
	     });
    //managed_free(a);
    //q.wait();
    //arg.debug();
    warningQuda("end launchReduction2D");
    return QUDA_SUCCESS;
  }


  template <int block_size_x, int block_size_y, template <typename> class Transformer,
	    typename Arg, bool grid_stride = true>
  void MultiReduction(Arg arg, sycl::nd_item<3> ndi)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    //auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx = ndi.get_global_id(0);
    //auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto j = ndi.get_global_id(1);
    //auto k = threadIdx.z;
    auto k = ndi.get_local_id(2);

    if (j >= arg.threads.y) return;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j, k);
      //if (grid_stride) idx += blockDim.x * gridDim.x; else break;
      if (grid_stride) idx += ndi.get_global_range(0); else break;
    }

    // perform final inter-block reduction and write out result
    reduce<block_size_x, block_size_y>(arg, t, value, j);
  }
  template <int block_size_x, int block_size_y, template <typename> class Transformer,
	    typename Arg, bool grid_stride = true>
  qudaError_t
  launchMultiReduction(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    warningQuda("launchMultiReduction %s", grid_stride?"true":"false");
    warningQuda("%s  %s", str(globalSize).c_str(), str(localSize).c_str());
    warningQuda("%s", str(arg.threads).c_str());
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Reduction2D>
		 (ndRange,
		  [=](sycl::nd_item<3> ndi) {
		    MultiReduction<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
		  });
	     });
    warningQuda("end launchMultiReduction");
    return QUDA_SUCCESS;
  }

}
