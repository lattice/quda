#pragma once

#include <tune_quda.h>

#ifdef QUDA_MMA_AVAILABLE
#include <kernels/coarse_op_kernel_mma.cuh>
#endif

/**
  Here we detail how the MMA kernels for computeUV and computeVUV should be launched. Specifically:
  - bM, bN, bK: the CTA-local MMA shape sizes.
  - block_y, block_z: number of threads in each direction. (blockDim.x has nothing to do with the MMA shape)
 */

namespace quda
{

  namespace mma
  {

#ifdef QUDA_MMA_AVAILABLE

    template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg, class Tunable>
    std::enable_if_t<!Arg::is_mma_compatible, void> launch_compute_uv_kernel(TuneParam &, const Arg &, int,
                                                                             const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg, class Tunable>
    std::enable_if_t<Arg::is_mma_compatible, void> launch_compute_uv_kernel(TuneParam &tp, const Arg &arg,
                                                                            int min_threads, const qudaStream_t &stream,
                                                                            Tunable &tunable)
    {
      tp.block.x = 1;
      tp.block.y = block_y;
      tp.block.z = block_z;

      using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;
      constexpr int shared_bytes = shared_memory_bytes<mma_t>(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      constexpr int M = Arg::uvTileType::m * Arg::fineSpin;
      constexpr int N = Arg::uvTileType::n;
      constexpr int K = Arg::uvTileType::k;

      constexpr bool divide_b_no = bM < M && bK >= K && bN == N;
      constexpr int t_m = divide_b_no ? 1 : (Arg::uvTileType::m * Arg::fineSpin + bM - 1) / bM;
      constexpr int t_n = divide_b_no ? 1 : (Arg::uvTileType::n + bN - 1) / bN;

      tp.grid = dim3(min_threads * t_m * t_n, 2, 1);
      tp.set_max_shared_bytes = true;

      tunable.template launch_cuda<ComputeUVMMA>(tp, stream, mmaArg<Arg, dim, dir, bM, bN, bK, block_y, block_z>(arg));
    }

    template <int bM, int bN, int bK, int block_y, int block_z, class Arg, typename Tunable>
    void launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream,
                                  Tunable &tunable)
    {
      if (arg.dir == QUDA_BACKWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_uv_kernel<0, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 1:
          launch_compute_uv_kernel<1, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 2:
          launch_compute_uv_kernel<2, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 3:
          launch_compute_uv_kernel<3, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else if (arg.dir == QUDA_FORWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_uv_kernel<0, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
          break;
        case 1:
          launch_compute_uv_kernel<1, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
          break;
        case 2:
          launch_compute_uv_kernel<2, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
          break;
        case 3:
          launch_compute_uv_kernel<3, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else if (arg.dir == QUDA_IN_PLACE) {
        launch_compute_uv_kernel<0, QUDA_IN_PLACE, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
      } else {
        errorQuda("arg.dir(=%d) is not supported", arg.dir);
      }
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<!Arg::from_coarse, int> launch_compute_uv_kernel(TuneParam &, const Arg &, int,
                                                                      const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation is ONLY built for !from_coarse.");
      return -1;
    }

    /**
        The following functions have switch's that list computeUV and computeVUV MMA kernels instantiations.
        if query_max = true, it will simply return how many instantiations there are; if query_max = false,
        the MMA kernel is launched with the corresponding configuration.
     */

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 1;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 16,  16,   8,   4,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 16,  16,   8,   8,   4>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      if (query_max) return 5;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 48,  24,  24,  24,  12>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 48,  24,  24,  16,   6>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 48,  24,  24,  16,   2>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 48,  24,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 48,  24,  24,   8,   4>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel< 48,  24,  24,   4,   8>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
#else
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 48,  32,  24,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 48,  32,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 48,  32,  24,   8,  24>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 48,  32,  24,  16,   4>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 48,  32,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
#endif
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 3;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 48,  32,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 48,  32,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 48,  32,  24,   8,  24>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 48,  32,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 5;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 48,  64,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 48,  64,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 48,  64,  24,   8,  24>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 48,  64,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 48,  64,  24,  32,  12>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel< 48,  64,  24,  16,  24>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // note --- currently unused, may be revisited in the future
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 48,  96,  24,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 48,  96,  24,   8,  24>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 48,  96,  24,  16,   6>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 48,  96,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 48,  96,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel< 48,  96,  24,  24,  12>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_uv_kernel< 48,  96,  24,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 2;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 64,  32,  32,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 64,  32,  32,   8,  32>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 64,  32,  32,  16,  16>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // note --- currently unused, may be revisited in the future
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<128,  64,  64,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel<128,  64,  64,   8,  32>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel<128,  64,  64,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel<128,  64,  64,  16,  16>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel<128,  64,  64,  16,  32>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel<128,  64,  64,  32,   8>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_uv_kernel<128,  64,  64,  32,  16>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel< 64,  96,  64,  32,  24>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel< 64,  96,  64,  12,  32>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 64,  96,  64,  32,  12>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 64,  96,  64,  16,  24>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 64,  96,  64,  16,  48>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel< 64,  96,  64,  32,   6>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_uv_kernel< 64,  96,  64,  32,   8>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 7;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<192,  96,  48,  24,  12>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_uv_kernel<192,  96,  48,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_uv_kernel< 96,  96,  96,  24,  12>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_uv_kernel< 96,  96,  96,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_uv_kernel< 96,  96,  96,  32,  12>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_uv_kernel< 96,  96,  96,  12,  32>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_uv_kernel< 48,  48,  96,  12,  24>(tp, arg, min_threads, stream, tunable); break;
      case 7: launch_compute_uv_kernel< 48,  48,  96,  24,  12>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // catch any cases that have not been implemented
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<!((Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)),
                     int>
    launch_compute_uv_kernel(TuneParam &, const Arg &, int, const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation not available for fineColor = %d coarseColor = %d", Arg::fineColor, Arg::coarseColor);
      return -1;
    }

    template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg, class Tunable>
    typename std::enable_if_t<!Arg::is_mma_compatible, void> launch_compute_vuv_kernel(TuneParam &, const Arg &, int,
                                                                                       const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg, class Tunable>
    std::enable_if_t<Arg::is_mma_compatible, void> launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg,
                                                                             int min_threads, const qudaStream_t &stream,
                                                                             Tunable &tunable)
    {
      tp.block.x = 1;
      tp.block.y = block_y;
      tp.block.z = block_z;
      using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;
      constexpr int shared_bytes = shared_memory_bytes<mma_t>(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      constexpr int t_m = (Arg::vuvTileType::m + bM - 1) / bM;
      constexpr int t_n = (Arg::vuvTileType::n + bN - 1) / bN;

      tp.grid = dim3(min_threads * t_m * t_n, 2, 1);
      tp.set_max_shared_bytes = true;

      tunable.template launch_cuda<ComputeVUVMMA>(tp, stream, mmaArg<Arg, dim, dir, bM, bN, bK, block_y, block_z>(arg));
    }

    template <int bM, int bN, int bK, int block_y, int block_z, class Arg, class Tunable>
    void launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream,
                                   Tunable &tunable)
    {
      if (arg.dir == QUDA_BACKWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_vuv_kernel<0, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                     tunable);
          break;
        case 1:
          launch_compute_vuv_kernel<1, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                     tunable);
          break;
        case 2:
          launch_compute_vuv_kernel<2, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                     tunable);
          break;
        case 3:
          launch_compute_vuv_kernel<3, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                     tunable);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else if (arg.dir == QUDA_FORWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_vuv_kernel<0, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 1:
          launch_compute_vuv_kernel<1, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 2:
          launch_compute_vuv_kernel<2, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        case 3:
          launch_compute_vuv_kernel<3, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream,
                                                                                    tunable);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else if (arg.dir == QUDA_IN_PLACE) {
        launch_compute_vuv_kernel<0, QUDA_IN_PLACE, bM, bN, bK, block_y, block_z>(tp, arg, min_threads, stream, tunable);
      } else {
        errorQuda("arg.dir(=%d) is not supported", arg.dir);
      }
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<!Arg::from_coarse, int> launch_compute_vuv_kernel(TuneParam &, const Arg &, int,
                                                                       const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation is ONLY built for !from_coarse.");
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 1;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 16,  16,   8,   8,   4>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 16,  16,   8,   4,   8>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
#if (__COMPUTE_CAPABILITY__ >= 750) // Turing or above
      if (query_max) return 1;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 32,  24,  24,  16,   6>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 32,  24,  24,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
#else
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 32,  32,  24,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 32,  32,  24,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 32,  32,  24,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 32,  32,  24,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 32,  32,  24,  32,   4>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
#endif
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 32,  32,  24,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 32,  32,  24,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 32,  32,  24,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 32,  32,  24,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 32,  32,  24,  32,   4>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 7;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 64,  64,  24,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 64,  64,  24,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 64,  64,  24,   8,  32>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 64,  64,  24,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 64,  64,  24,  16,  16>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_vuv_kernel< 64,  64,  24,  16,  32>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_vuv_kernel< 64,  64,  24,  32,   8>(tp, arg, min_threads, stream, tunable); break;
      case 7: launch_compute_vuv_kernel< 64,  64,  24,  32,  16>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // note -- currently unused, may be used in the future
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 6;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel< 96,  96,  24,  12,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 96,  96,  24,  24,   8>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 96,  96,  24,   6,  16>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 96,  96,  24,  12,  16>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 96,  96,  24,  24,  16>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_vuv_kernel< 96,  96,  24,  12,  24>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_vuv_kernel< 96,  96,  24,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 3;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel< 32,  32,  32,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 32,  32,  32,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 32,  32,  32,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 32,  32,  32,  32,   4>(tp, arg, min_threads, stream, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    // note --- currently unused, may be revisited in the future
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 7;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel< 64,  64,  64,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 64,  64,  64,   8,  16>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 64,  64,  64,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 64,  64,  64,  16,  16>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 64,  64,  64,  16,  32>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_vuv_kernel< 64,  64,  64,  32,   4>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_vuv_kernel< 64,  64,  64,  32,   8>(tp, arg, min_threads, stream, tunable); break;
      case 7: launch_compute_vuv_kernel< 64,  64,  64,  32,  16>(tp, arg, min_threads, stream, tunable); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 96,  96,  64,   8,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 96,  96,  64,   8,  12>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 96,  96,  64,  16,   6>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 96,  96,  64,  16,   8>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 96,  96,  64,  16,  12>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_vuv_kernel< 96,  96,  64,  32,   4>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_vuv_kernel< 96,  96,  64,  32,   6>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // note -- currently unused, may be revisited in the future
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2, int>
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const qudaStream_t &stream, Tunable &tunable)
    {
      if (query_max) return 8;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel< 96,  96,  96,  12,   8>(tp, arg, min_threads, stream, tunable); break;
      case 1: launch_compute_vuv_kernel< 96,  96,  96,  24,   8>(tp, arg, min_threads, stream, tunable); break;
      case 2: launch_compute_vuv_kernel< 96,  96,  96,   6,  16>(tp, arg, min_threads, stream, tunable); break;
      case 3: launch_compute_vuv_kernel< 96,  96,  96,  12,  16>(tp, arg, min_threads, stream, tunable); break;
      case 4: launch_compute_vuv_kernel< 96,  96,  96,  24,  16>(tp, arg, min_threads, stream, tunable); break;
      case 5: launch_compute_vuv_kernel< 96,  96,  96,  12,  24>(tp, arg, min_threads, stream, tunable); break;
      case 6: launch_compute_vuv_kernel< 96,  96,  96,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      case 7: launch_compute_vuv_kernel< 48,  48,  96,  12,  24>(tp, arg, min_threads, stream, tunable); break;
      case 8: launch_compute_vuv_kernel< 48,  48,  96,  24,  24>(tp, arg, min_threads, stream, tunable); break;
      // clang-format on
      default:
        errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin,
                  Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    // catch any cases that have not been implemented
    template <bool query_max = false, class Arg, class Tunable>
    std::enable_if_t<!((Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)
                       || (Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2)),
                     int>
    launch_compute_vuv_kernel(TuneParam &, const Arg &, int, const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA implementation not available for fineColor = %d coarseColor = %d", Arg::fineColor, Arg::coarseColor);
      return -1;
    }
#else

    template <bool query_max = false, class Arg, class Tunable>
    int launch_compute_uv_kernel(TuneParam &, const Arg &, int, const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

    template <bool query_max = false, class Arg, class Tunable>
    int launch_compute_vuv_kernel(TuneParam &, const Arg &, int, const qudaStream_t &, Tunable &)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

#endif // QUDA_MMA_AVAILABLE

  } // namespace mma

} // namespace quda
