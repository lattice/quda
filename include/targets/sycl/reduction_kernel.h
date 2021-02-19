#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ void Reduction2D(Arg arg, sycl::nd_item<3> ndi)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      if (grid_stride) idx += blockDim.x * gridDim.x; else break;
    }

    // perform final inter-block reduction and write out result
    quda::reduce<block_size_x, block_size_y>(arg, t, value);
  }
  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  qudaError_t
  launchReduction2D(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    auto a = (Arg *)managed_malloc(sizeof(Arg));
    memcpy((void*)a, &arg, sizeof(Arg));
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Reduction2D>(ndRange,
					  [=](sycl::nd_item<3> ndi)
					  {
					    quda::Reduction2D<block_size_x, block_size_y, Transformer, Arg, grid_stride>(*a, ndi);
					  });
	     });
    managed_free(a);
#if 0
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Reduction2D>(ndRange,
					  [=](sycl::nd_item<3> ndi)
					  {
					    quda::Reduction2D<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
					  });
	     });
#endif
    return QUDA_SUCCESS;
  }


  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ void MultiReduction(Arg arg, sycl::nd_item<3> ndi)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z;

    if (j >= arg.threads.y) return;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, j, k);
      if (grid_stride) idx += blockDim.x * gridDim.x; else break;
    }

    // perform final inter-block reduction and write out result
    reduce<block_size_x, block_size_y>(arg, t, value, j);
  }
  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  qudaError_t
  launchMultiReduction(const TuneParam &tp, const qudaStream_t &stream, Arg arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
#if 0
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class Reduction2D>(ndRange,
					  [=](sycl::nd_item<3> ndi)
					  {
					    MultiReduction<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
					  });
	     });
#endif
    return QUDA_SUCCESS;
  }

}
