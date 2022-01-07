#pragma once
#include <tune_quda.h>
#include <reduce_helper.h>
#include <timer.h>

namespace quda {

  /**
     @brief This class is derived from the arg class that the functor
     creates and curries in the block size.  This allows the block
     size to be set statically at launch time in the actual argument
     class that is passed to the kernel.
  */
  template <int block_size_x_, int block_size_y_, typename Arg_> struct ReduceKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    ReduceKernelArg(const Arg &arg) : Arg(arg) { }
  };

#if 1
  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  void Reduction2DImpl(const Arg &arg, sycl::nd_item<3> &ndi)
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
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value);
  }
#endif
  template <template <typename> class Transformer, typename Arg,
	    typename S, typename RT, bool grid_stride = true>
  void Reduction2DImplN(const Arg &arg, sycl::nd_item<3> &ndi, S &sum)
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
  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  qudaError_t Reduction2D(const TuneParam &tp,
			  const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y, 1};
    sycl::range<3> localSize{tp.block.x, tp.block.y, 1};
    //sycl::range<3> globalSize{1,1,1};
    //sycl::range<3> localSize{1,tp.block.y,1};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    host_timer_t timer;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Reduction2D grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      timer.start();
    }
#if 0
    //arg.debug();
    q.submit([&](sycl::handler& h) {
      //h.parallel_for<class Reduction2D>
      h.parallel_for
	(ndRange,
	 [=](sycl::nd_item<3> ndi) {
	   quda::Reduction2DImpl<Transformer, Arg, grid_stride>(arg, ndi);
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
    auto red = sycl::reduction(result_d, arg.init(),
			       typename Transformer<Arg>::reducer_t());
    //auto red = sycl::ONEAPI::reduction(result_d, Transformer<Arg>::init(),
    //			       typename Transformer<Arg>::reducer_t());
    //warningQuda("nd: %i\n", nd);
    //using da = double[nd];
    //constexpr int nd = sizeof(*result_h)/sizeof(double);
    //using da = sycl::vec<double,nd>;
    //auto red = sycl::reduction((da*)result_d, *(da*)result_h, sycl::plus<da>());
    try {
      q.submit([&](sycl::handler& h) {
	//h.parallel_for<class Reduction2Dn>
	h.parallel_for<>
	  (ndRange, red,
	   [=](sycl::nd_item<3> ndi, auto &sum) {
	     using Sum = decltype(sum);
	     quda::Reduction2DImplN<Transformer, Arg, Sum, reduce_t, grid_stride>(arg, ndi, sum);
	     //quda::Reduction2DImplN<Transformer, Arg, Sum, da, grid_stride>(arg, ndi, sum);
	   });
      });
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      timer.stop();
      printfQuda("  launch time: %g\n", timer.last());
      if (commAsyncReduction()) {
	q.memcpy(result_h, result_d, sizeof(reduce_t));
      }
      q.wait_and_throw();
      printfQuda("end Reduction2D result_h: %g\n", *(double *)result_h);
    }
#endif
    //warningQuda("end launchReduction2D");
    return err;
  }


  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  void MultiReductionImpl(const Arg &arg, const sycl::nd_item<3> &ndi)
  {
    using reduce_t = typename Functor<Arg>::reduce_t;
    Functor<Arg> t(arg);

    auto idx = ndi.get_global_id(0);
    auto k = ndi.get_local_id(1);
    auto j = ndi.get_global_id(2);

    if (j >= arg.threads.z) return;

    reduce_t value = arg.init();

    while (idx < arg.threads.x) {
      value = t(value, idx, k, j);
      if (grid_stride) idx += ndi.get_global_range(0); else break;
    }

    // perform final inter-block reduction and write out result
    reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value, j);
  }
#if 0
  template <template <typename> class Transformer, typename Arg,
	    typename S, bool grid_stride = true>
  void MultiReductionImpl1(const Arg &arg, sycl::nd_item<3> &ndi, S &sum)
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
#endif
  template <template <typename> class Transformer, typename Arg, bool grid_stride = true>
  qudaError_t
  MultiReduction(const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    sycl::range<3> globalSize{tp.grid.x*tp.block.x, tp.grid.y*tp.block.y,
      tp.grid.z*tp.block.z};
    sycl::range<3> localSize{tp.block.x, tp.block.y, tp.block.z};
    sycl::nd_range<3> ndRange{globalSize, localSize};
    auto q = device::get_target_stream(stream);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      printfQuda("MultiReduction grid_stride: %s  sizeof(arg): %lu\n",
		 grid_stride?"true":"false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(),
		 str(localSize).c_str(), str(arg.threads).c_str());
      printfQuda("  reduce_t: %s\n", typeid(reduce_t).name());
    }
#if 1
    sycl::buffer<const char,1>
      buf{reinterpret_cast<const char*>(&arg), sycl::range(sizeof(arg))};
    try {
      q.submit([&](sycl::handler& h) {
	auto a = buf.get_access<sycl::access::mode::read,
				sycl::access::target::constant_buffer>(h);
	//h.parallel_for<class MultiReductionx>
	h.parallel_for<>
	  (ndRange,
	   [=](sycl::nd_item<3> ndi) {
	    //MultiReductionImpl<Transformer,Arg,grid_stride>(arg,ndi);
	    const char *p = a.get_pointer();
	    const Arg *arg2 = reinterpret_cast<const Arg*>(p);
	    MultiReductionImpl<Transformer,Arg,grid_stride>(*arg2,ndi);
	  });
	});
    } catch (sycl::exception const& e) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
	printfQuda("  Caught synchronous SYCL exception:\n  %s\n",e.what());
      }
      err = QUDA_ERROR;
    }
#else
    if(arg.threads.y==1) {
      using reduce_t = typename Transformer<Arg>::reduce_t;
      auto result_h = reinterpret_cast<reduce_t *>(quda::reducer::get_host_buffer());
      *result_h = arg.init();
      auto result = reinterpret_cast<reduce_t *>(quda::reducer::get_mapped_buffer());
      auto red = sycl::reduction(result, arg.init(), typename Transformer<Arg>::reducer_t());
      sycl::buffer<const char,1>
	buf{reinterpret_cast<const char*>(&arg), sycl::range(sizeof(arg))};
      q.submit([&](sycl::handler& h) {
	auto a = buf.get_access<sycl::access::mode::read,
				sycl::access::target::constant_buffer>(h);
	//h.parallel_for<class MultiReduction1x>
	h.parallel_for<>
	  (ndRange, red,
	   [=](sycl::nd_item<3> ndi, auto &sum) {
	     using Sum = decltype(sum);
	     //MultiReductionImpl1<Transformer, Arg, Sum, grid_stride>(arg, ndi, sum);
	     const char *p = a.get_pointer();
	     const Arg *arg2 = reinterpret_cast<const Arg*>(p);
	     MultiReductionImpl1<Transformer, Arg, Sum, grid_stride>(*arg2, ndi, sum);
	   });
      });
    } else {
      errorQuda("multireduce %i\n", arg.threads.y);
    }
#endif
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("end MultiReduction\n");
    }
    return err;
  }

}
