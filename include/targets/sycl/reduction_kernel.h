#pragma once
#include <tune_quda.h>
#include <reduce_helper.h>

namespace quda {

#if 0
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
#endif
  template <int block_size_x, int block_size_y,
	    template <typename> class Transformer, typename Arg, typename S, typename RT,
	    bool grid_stride = true>
  void Reduction2Dn(const Arg &arg, sycl::nd_item<3> &ndi, S &sum)
  {
    Transformer<Arg> t(const_cast<Arg&>(arg));
    auto idx = ndi.get_global_id(0);
    auto j = ndi.get_local_id(1);
    auto value = arg.init();
    while (idx < arg.threads.x) {
      value = t(value, idx, j);
      if (grid_stride) idx += ndi.get_global_range(0); else break;
    }
    sum.combine(*(RT*)&value);
  }
  template <int block_size_x, int block_size_y,
	    template <typename> class Transformer,
	    typename Arg, bool grid_stride = true>
  qudaError_t launchReduction2D(const TuneParam &tp,
				const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, 1};
    sycl::range<3> localSize{tp.block.x, tp.block.y, 1};
    //sycl::range<3> globalSize{1,1,1};
    //sycl::range<3> localSize{1,tp.block.y,1};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchReduction2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
#if 0
    //arg.debug();
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class Reduction2D>
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
	   quda::Reduction2D<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
	 });
    });
    //q.wait();
    //arg.debug();
#else
    using reduce_t = typename Transformer<Arg>::reduce_t;
    auto result_h = reinterpret_cast<reduce_t *>(quda::reducer::get_host_buffer());
    *result_h = arg.init();
    reduce_t *result_d = result_h;
    if (commAsyncReduction()) {
      result_d = reinterpret_cast<reduce_t *>(quda::reducer::get_device_buffer());
      q.memcpy(result_d, result_h, sizeof(reduce_t));
    }
    //auto red = sycl::ONEAPI::reduction(result, arg.init(), typename Transformer<Arg>::reducer_t());
    //auto red = sycl::ONEAPI::reduction(result_d, Transformer<Arg>::init(),
    //			       typename Transformer<Arg>::reducer_t());
    //warningQuda("nd: %i\n", nd);
    //using da = double[nd];
    constexpr int nd = sizeof(*result_h)/sizeof(double);
    using da = sycl::vec<double,nd>;
    auto red = sycl::ONEAPI::reduction((da*)result_d, *(da*)result_h,
				       sycl::ONEAPI::plus<da>());
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class Reduction2Dn>
	(ndRange, red,
	 [=](sycl::nd_item<3> ndi, auto &sum) {
	   using Sum = decltype(sum);
	   quda::Reduction2Dn<block_size_x, block_size_y, Transformer,
			      Arg, Sum, da, grid_stride>(arg, ndi, sum);
	 });
    });
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      q.wait();
      printfQuda("  end launchReduction2D result_h: %g\n", *(double *)result_h);
    }
#endif
    //warningQuda("end launchReduction2D");
    return QUDA_SUCCESS;
  }


#if 0
  template <int block_size_x, int block_size_y, template <typename> class Transformer,
	    typename Arg, bool grid_stride = true>
  void MultiReduction(const Arg &arg, sycl::nd_item<3> ndi)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(const_cast<Arg&>(arg));

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
#endif
  template <int block_size_x, int block_size_y,
	    template <typename> class Transformer,
	    typename Arg, typename S, bool grid_stride = true>
  void MultiReduction1(const Arg &arg, sycl::nd_item<3> &ndi, S &sum)
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(const_cast<Arg&>(arg));
    auto idx = ndi.get_global_id(0);
    auto j = ndi.get_global_id(1);
    auto k = ndi.get_local_id(2);
    if (j >= arg.threads.y) return;
    reduce_t value = arg.init();
    while (idx < arg.threads.x) {
      value = t(value, idx, j, k);
      if (grid_stride) idx += ndi.get_global_range(0); else break;
    }
    sum.combine(value);
  }
  template <int block_size_x, int block_size_y,
	    template <typename> class Transformer,
	    typename Arg, bool grid_stride = true>
  qudaError_t launchMultiReduction(const TuneParam &tp,
				   const qudaStream_t &stream, const Arg &arg)
  {
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("launchMultiReduction grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
    }
#if 0
    q.submit([&](sycl::handler& h) {
	       h.parallel_for<class MultiReduction>
		 (ndRange,
		  [=](sycl::nd_item<3> ndi) {
		    MultiReduction<block_size_x, block_size_y, Transformer, Arg, grid_stride>(arg, ndi);
		  });
	     });
#else
    if(arg.threads.y==1) {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      auto result_h = reinterpret_cast<reduce_t *>(quda::reducer::get_host_buffer());
      *result_h = arg.init();
      auto result = reinterpret_cast<reduce_t *>(quda::reducer::get_mapped_buffer());
      auto red = sycl::ONEAPI::reduction(result, arg.init(), typename Transformer<Arg>::reducer_t());
      q.submit([&](sycl::handler& h) {
	h.parallel_for<class MultiReduction1x>
	  (ndRange, red,
	   [=](sycl::nd_item<3> ndi, auto &sum) {
	     using Sum = decltype(sum);
	     MultiReduction1<block_size_x, block_size_y, Transformer, Arg, Sum, grid_stride>(arg, ndi, sum);
	   });
      });
    } else {
      errorQuda("multireduce %i\n", arg.threads.y);
    }
#endif
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("  end launchMultiReduction\n");
    }
    return QUDA_SUCCESS;
  }

}
