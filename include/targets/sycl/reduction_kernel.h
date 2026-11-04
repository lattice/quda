#pragma once
#include <tunable_kernel.h>
#include <reduce_helper.h>
#include <timer.h>
#include <quda_sycl_api.h>

//#define HIGH_LEVEL_REDUCTIONS

namespace quda
{

#ifndef HIGH_LEVEL_REDUCTIONS
  template <template <typename> class Functor, typename Arg, bool grid_stride = true, typename S>
  void Reduction2DImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S smem)
  {
    Functor<Arg> f(arg);
    typename reduceParams<Arg, Functor<Arg>, typename Functor<Arg>::reduce_t>::Ops rso {smem};
    auto idx = globalIdX;
    auto j = localIdY;
    auto value = f.init();
    while (idx < arg.threads.x) {
      value = f(value, idx, j);
      if (grid_stride)
        idx += globalRangeX;
      else
        break;
    }
    if constexpr (needsSharedMem<Functor<Arg>>) group_barrier(ndi.get_group());
    // perform final inter-block reduction and write out result
    reduce(arg, f, value, 0, rso);
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride = false> struct Reduction2DS {
    using KernelOpsT = typename reduceParams<Arg, Functor<Arg>, typename Functor<Arg>::reduce_t>::Ops;
    template <typename... T> Reduction2DS(const Arg &arg, const sycl::nd_item<3> &ndi, T... smem)
    {
      //#ifdef QUDA_THREADS_BLOCKED
      //Reduction2DImpl<Functor, Arg, grid_stride>(arg, ndi);
      //#else
      Reduction2DImpl<Functor, Arg, grid_stride>(arg, ndi, smem...);
      //#endif
    }
  };
#else
  template <template <typename> class Functor, bool grid_stride, typename Arg, typename R>
  void Reduction2DImplN(const Arg &arg, const sycl::nd_item<3> &ndi, R &reducer)
  {
    Functor<Arg> f(arg);
    auto idx = globalIdX;
    auto j = localIdY;
    auto value = f.init();
    while (idx < arg.threads.x) {
      value = f(value, idx, j);
      if (grid_stride)
        idx += globalRangeX;
      else
        break;
    }
    reducer.combine(value);
  }
  template <template <typename> class Functor, bool grid_stride = false> struct Reduction2DS {
    using KernelOpsT = NoKernelOps;
    template <typename Arg, typename R> static void apply(const Arg &arg, const sycl::nd_item<3> &ndi, R &reducer)
    {
      //#ifdef QUDA_THREADS_BLOCKED
      //Reduction2DImplN<Functor, grid_stride>(arg, ndi, reducer);
      //#else
      Reduction2DImplN<Functor, grid_stride>(arg, ndi, reducer);
      //#endif
    }
  };
#endif
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  qudaError_t Reduction2D(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    static_assert(!hasKernelOps<Functor<Arg>>);
    auto err = QUDA_SUCCESS;
    host_timer_t timer;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      auto globalSize = globalRange(tp);
      auto localSize = localRange(tp);
      printfQuda("Reduction2D grid_stride: %s  sizeof(arg): %lu\n", grid_stride ? "true" : "false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(), str(localSize).c_str(),
                 str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  SLM size: %lu\n", localSize.size() * sizeof(typename Functor<Arg>::reduce_t) / device::warp_size());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>,
                 needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
      timer.start();
    }
#ifndef HIGH_LEVEL_REDUCTIONS
    err = launch<Reduction2DS<Functor, Arg, grid_stride>>(tp, stream, arg);
#else
    err = launchR<Functor, Reduction2DS<Functor, grid_stride>>(tp, stream, arg);
#endif
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      timer.stop();
      printfQuda("end Reduction2D launch time: %g\n", timer.last());
    }
    return err;
  }

  // MultiReduction

  template <template <typename> class Functor, typename Arg, bool grid_stride = true, typename S>
  void MultiReductionImpl(const Arg &arg, const sycl::nd_item<3> &ndi, S smem)
  {
    static_assert(!needsFullBlock<Functor<Arg>>);
    using reduce_t = typename Functor<Arg>::reduce_t;
    Ftor<Functor<Arg>> f(arg, ndi, smem);

    typename reduceParams<Arg, Functor<Arg>, typename Functor<Arg>::reduce_t>::Ops rso {smem};

    auto idx = globalIdX;
    auto k = localIdY;
    auto j = globalIdZ;

    reduce_t value = f.init();

    if (j < arg.threads.z) {
      while (idx < arg.threads.x) {
        value = f(value, idx, k, j);
        if (grid_stride)
          idx += globalRangeX;
        else
          break;
      }
    }
    if constexpr (needsSharedMem<Functor<Arg>>) group_barrier(ndi.get_group());

    // perform final inter-block reduction and write out result
    reduce(arg, f, value, j, rso);
  }
  template <template <typename> class Functor, typename Arg, bool grid_stride> struct MultiReductionS {
    using KernelOpsT
      = combineOps<getKernelOps<Functor<Arg>>, typename reduceParams<Arg, Functor<Arg>, typename Functor<Arg>::reduce_t>::Ops>;
    template <typename... T> MultiReductionS(const Arg &arg, const sycl::nd_item<3> &ndi, T... smem)
    {
      // #ifdef QUDA_THREADS_BLOCKED
      // MultiReductionImpl<Functor,Arg,grid_stride>(arg, ndi);
      // #else
      MultiReductionImpl<Functor, Arg, grid_stride>(arg, ndi, smem...);
      // #endif
    }
  };

  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  qudaError_t MultiReduction(const TuneParam &tp, const qudaStream_t &stream, Arg &arg)
  {
    auto err = QUDA_SUCCESS;
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      auto globalSize = globalRange(tp);
      auto localSize = localRange(tp);
      using reduce_t = typename Functor<Arg>::reduce_t;
      printfQuda("MultiReduction grid_stride: %s  sizeof(arg): %lu\n", grid_stride ? "true" : "false", sizeof(arg));
      printfQuda("  global: %s  local: %s  threads: %s\n", str(globalSize).c_str(), str(localSize).c_str(),
                 str(arg.threads).c_str());
      printfQuda("  Functor: %s\n", typeid(Functor<Arg>).name());
      printfQuda("  Arg: %s\n", typeid(Arg).name());
      printfQuda("  reduce_t: %s\n", typeid(reduce_t).name());
      printfQuda("  Arg::max_n_batch_block: %d\n", Arg::max_n_batch_block);
      printfQuda("  Functor::reduce_block_dim: %d\n", Functor<Arg>::reduce_block_dim);
      printfQuda("  max_block_z: %d\n", device::max_block_size() / (tp.block.x * tp.block.y));
      printfQuda("  SLM size: %lu\n", localSize.size() * sizeof(typename Functor<Arg>::reduce_t) / device::warp_size());
      printfQuda("  KernelOps: %s\n", typeid(getKernelOps<Functor<Arg>>).name());
      printfQuda("  needsFullBlock: %i  needsSharedMem: %i\n", needsFullBlock<Functor<Arg>>,
                 needsSharedMem<Functor<Arg>>);
      printfQuda("  shared_bytes: %i\n", tp.shared_bytes);
    }
    err = launch<MultiReductionS<Functor, Arg, grid_stride>>(tp, stream, arg);
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) { printfQuda("end MultiReduction\n"); }
    return err;
  }

} // namespace quda
