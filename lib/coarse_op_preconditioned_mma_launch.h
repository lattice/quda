#pragma once

#include <gauge_field.h>
#include <tunable_nd.h>

#ifdef QUDA_MMA_AVAILABLE
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

#ifdef QUDA_MMA_AVAILABLE

    template <int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg, typename Tunable>
    typename std::enable_if<!Arg::is_mma_compatible, void>::type launch_kernel(TuneParam &, const qudaStream_t &, Arg &,
                                                                               Tunable &)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg, typename Tunable>
    typename std::enable_if<Arg::is_mma_compatible, void>::type launch_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                              Arg &arg, Tunable &tunable)
    {
      tp.block = dim3(1, block_y, block_z);

      constexpr bool divide_b_no = bM < Arg::M && bK == Arg::K && bN == Arg::N;
      constexpr int t_m = divide_b_no ? 1 : (Arg::M + bM - 1) / bM;
      constexpr int t_n = divide_b_no ? 1 : (Arg::N + bN - 1) / bN;
      tp.grid = dim3(arg.threads.x * t_m * t_n, 2, 4);

      using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;
      tp.shared_bytes = shared_memory_bytes<mma_t>(bM, bN, bK);
      tp.set_max_shared_bytes = true;

      tunable.template launch_cuda<CalculateYhatMMA>(
        tp, stream, CalculateYhatMMAArg<Arg, bM, bN, bK, block_y, block_z, min_block_cta>(arg));
    }

    /**
       The following functions have switch's that list computeYhat MMA kernels instantiations.
       if query_max = true, it will simply return how many instantiations there are; if query_max = false,
       the MMA kernel is launched with the corresponding configuration.
    */
    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if<Arg::N == 48, int>::type launch_yhat_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                        Arg &arg, Tunable &tunable)
    {
      if (query_max) return 2;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_kernel<48, 48, 48, 24,  12>(tp, stream, arg, tunable); break;
      case 1: launch_kernel<48, 48, 48,  6,  48>(tp, stream, arg, tunable); break;
      case 2: launch_kernel<48, 48, 48, 12,  24>(tp, stream, arg, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 48", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if<Arg::N == 12, int>::type launch_yhat_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                        Arg &arg, Tunable &tunable)
    {
      if (query_max) return 1;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_kernel<16, 16, 16, 4, 8>(tp, stream, arg, tunable); break;
      case 1: launch_kernel<16, 16, 16, 8, 4>(tp, stream, arg, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 12", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if<Arg::N == 64, int>::type launch_yhat_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                        Arg &arg, Tunable &tunable)
    {
      if (query_max) return 6;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_kernel<64, 64, 16, 32,  8>(tp, stream, arg, tunable); break;
      case 1: launch_kernel<64, 64, 16, 16, 16>(tp, stream, arg, tunable); break;
      case 2: launch_kernel<64, 64, 16, 32, 16>(tp, stream, arg, tunable); break;
      case 3: launch_kernel<64, 64, 32, 32, 16>(tp, stream, arg, tunable); break;
      case 4: launch_kernel<64, 64, 64,  8, 64>(tp, stream, arg, tunable); break;
      case 5: launch_kernel<64, 64, 64, 16, 32>(tp, stream, arg, tunable); break;
      case 6: launch_kernel<64, 64, 64, 32, 16>(tp, stream, arg, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 64", tp.aux.x);
        }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if<Arg::N == 128, int>::type launch_yhat_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                         Arg &arg, Tunable &tunable)
    {
      if (query_max) return 7;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_kernel< 64,  64,  16,  32,  16,  2>(tp, stream, arg, tunable); break;
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      case 1: launch_kernel< 16, 128, 128,  32,  16,  2>(tp, stream, arg, tunable); break;
#else
      case 1: launch_kernel< 32, 128, 128,  32,  16,  2>(tp, stream, arg, tunable); break;
#endif
      case 2: launch_kernel<128, 128,  16,  64,   8    >(tp, stream, arg, tunable); break;
      case 3: launch_kernel<128, 128,  16,  32,  16    >(tp, stream, arg, tunable); break;
      case 4: launch_kernel<128, 128,  32,  16,  32    >(tp, stream, arg, tunable); break;
      case 5: launch_kernel<128, 128,  32,  64,   8    >(tp, stream, arg, tunable); break;
      case 6: launch_kernel<128, 128,  32,  32,  16    >(tp, stream, arg, tunable); break;
      case 7: launch_kernel<128, 128,  32,  32,  32    >(tp, stream, arg, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 128", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if<Arg::N == 192, int>::type launch_yhat_kernel(TuneParam &tp, const qudaStream_t &stream,
                                                                         Arg &arg, Tunable &tunable)
    {
      if (query_max) return 4;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_kernel<64,  64,  16,  16,  16,  2>(tp, stream, arg, tunable); break;
      case 1: launch_kernel<64,  64,  64,  16,  16,  2>(tp, stream, arg, tunable); break;
      case 2: launch_kernel<16, 192, 192,  24,  16    >(tp, stream, arg, tunable); break;
      case 3: launch_kernel<64,  64,  32,  16,  16,  2>(tp, stream, arg, tunable); break;
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      case 4: launch_kernel<16, 192, 192,  96,   8    >(tp, stream, arg, tunable); break;
#else
      case 4: launch_kernel<16, 192, 192,  48,   8    >(tp, stream, arg, tunable); break;
#endif
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 192", tp.aux.x);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, typename Tunable>
    typename std::enable_if_t<!(Arg::N == 12 || Arg::N == 48 || Arg::N == 64 || Arg::N == 128 || Arg::N == 192), int>
    launch_yhat_kernel(TuneParam &, const qudaStream_t &, Arg &, Tunable &)
    {
      errorQuda("MMA implementation not available for N = %d", Arg::N);
      return -1;
    }
#else

    template <bool query_max = false, class Arg, typename Tunable>
    int launch_yhat_kernel(TuneParam &, const qudaStream_t &, Arg &, Tunable &)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

#endif // QUDA_MMA_AVAILABLE

  } // namespace mma

} // namespace quda
