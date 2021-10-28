#pragma once

#include <gauge_field.h>
#include <tune_quda.h>

#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)

#include <kernels/coarse_op_preconditioned_mma.cuh>

#endif

/**
  Here we detail how the MMA kernels for computeYhat should be launched. Specifically:
  - bM, bN, bK: the CTA-local MMA shape sizes.
  - block_y, block_z: number of threads in each direction. (blockDim.x has nothing to do with the MMA shape)
 */
namespace quda
{

  namespace mma
  {

#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)

    template <bool compute_max_only, int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg>
    typename std::enable_if<!Arg::is_mma_compatible, void>::type launch_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                               const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <bool compute_max_only, int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg>
    typename std::enable_if<Arg::is_mma_compatible, void>::type launch_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                              const cudaStream_t &stream)
    {
      tp.block.x = 1;
      tp.block.y = block_y;
      tp.block.z = block_z;
      constexpr int shared_bytes = shared_memory_bytes(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      constexpr bool divide_b_no = bM < Arg::M && bK == Arg::K && bN == Arg::N;

      constexpr int t_m = divide_b_no ? 1 : (Arg::M + bM - 1) / bM;
      constexpr int t_n = divide_b_no ? 1 : (Arg::N + bN - 1) / bN;

      tp.grid = dim3(min_threads * t_m * t_n, 2, 4);

      auto kernel = mma::CalculateYhatGPU<compute_max_only, Arg, bM, bN, bK, block_y, block_z, min_block_cta>;
      tp.set_max_shared_bytes = true;
      qudaLaunchKernel(kernel, tp, stream, arg);
    }

    /**
        The following functions have switch's that list computeYhat MMA kernels instantiations.
        if query_max = true, it will simply return how many instantiations there are; if query_max = false,
        the MMA kernel is launched with the corresponding configuration.
     */
    template <bool compute_max_only, bool query_max = false, class Arg>
    typename std::enable_if<Arg::N == 48, int>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                        const cudaStream_t &stream)
    {
      if (query_max) return 2;
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  48,  48,  48,  24,  12>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  48,  48,  48,   6,  48>(arg, min_threads, tp, stream); break;
      case   2: launch_kernel<compute_max_only,  48,  48,  48,  12,  24>(arg, min_threads, tp, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 48", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool compute_max_only, bool query_max = false, class Arg>
    typename std::enable_if<Arg::N == 12, int>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                        const cudaStream_t &stream)
    {
      if (query_max) return 1;
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  16,  16,  16,   4,   8>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  16,  16,  16,   8,   4>(arg, min_threads, tp, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 12", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool compute_max_only, bool query_max = false, class Arg>
    typename std::enable_if<Arg::N == 64, int>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                        const cudaStream_t &stream)
    {
      if (query_max) return 6;
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  64,  64,  16,  32,   8>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  64,  64,  16,  16,  16>(arg, min_threads, tp, stream); break;
      case   2: launch_kernel<compute_max_only,  64,  64,  16,  32,  16>(arg, min_threads, tp, stream); break;
      case   3: launch_kernel<compute_max_only,  64,  64,  32,  32,  16>(arg, min_threads, tp, stream); break;
      case   4: launch_kernel<compute_max_only,  64,  64,  64,   8,  64>(arg, min_threads, tp, stream); break;
      case   5: launch_kernel<compute_max_only,  64,  64,  64,  16,  32>(arg, min_threads, tp, stream); break;
      case   6: launch_kernel<compute_max_only,  64,  64,  64,  32,  16>(arg, min_threads, tp, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 64", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool compute_max_only, bool query_max = false, class Arg>
    typename std::enable_if<Arg::N == 128, int>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                         const cudaStream_t &stream)
    {
      if (query_max) return 7;
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  64,  64,  16,  32,  16,  2>(arg, min_threads, tp, stream); break;
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      case   1: launch_kernel<compute_max_only,  16, 128, 128,  32,  16,  2>(arg, min_threads, tp, stream); break;
#else
      case   1: launch_kernel<compute_max_only,  32, 128, 128,  32,  16,  2>(arg, min_threads, tp, stream); break;
#endif
      case   2: launch_kernel<compute_max_only, 128, 128,  16,  64,   8    >(arg, min_threads, tp, stream); break;
      case   3: launch_kernel<compute_max_only, 128, 128,  16,  32,  16    >(arg, min_threads, tp, stream); break;
      case   4: launch_kernel<compute_max_only, 128, 128,  32,  16,  32    >(arg, min_threads, tp, stream); break;
      case   5: launch_kernel<compute_max_only, 128, 128,  32,  64,   8    >(arg, min_threads, tp, stream); break;
      case   6: launch_kernel<compute_max_only, 128, 128,  32,  32,  16    >(arg, min_threads, tp, stream); break;
      case   7: launch_kernel<compute_max_only, 128, 128,  32,  32,  32    >(arg, min_threads, tp, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 128", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool compute_max_only, bool query_max = false, class Arg>
    typename std::enable_if<Arg::N == 192, int>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                         const cudaStream_t &stream)
    {
      if (query_max) return 4;
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  64,  64,  16,  16,  16,  2>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  64,  64,  64,  16,  16,  2>(arg, min_threads, tp, stream); break;
      case   2: launch_kernel<compute_max_only,  16, 192, 192,  24,  16    >(arg, min_threads, tp, stream); break;
      case   3: launch_kernel<compute_max_only,  64,  64,  32,  16,  16,  2>(arg, min_threads, tp, stream); break;
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      case   4: launch_kernel<compute_max_only,  16, 192, 192,  96,   8    >(arg, min_threads, tp, stream); break;
#else
      case   4: launch_kernel<compute_max_only,  16, 192, 192,  48,   8    >(arg, min_threads, tp, stream); break;
#endif
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 192", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

#else

    template <bool compute_max_only, bool query_max = false, class Arg>
    int launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp, const cudaStream_t &stream)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

#endif // compute capability >= 700, CUDA >= 10.1

  } // namespace mma

} // namespace quda
