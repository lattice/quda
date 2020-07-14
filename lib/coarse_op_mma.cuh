#pragma once

#include <tune_quda.h>

#if ((__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1) || (__CUDACC_VER_MAJOR__ > 10))                         \
  && (__COMPUTE_CAPABILITY__ >= 700)

#include <kernels/coarse_op_kernel_mma.cuh>

#endif

namespace quda
{

  namespace mma
  {

#if ((__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1) || (__CUDACC_VER_MAJOR__ > 10))                         \
  && (__COMPUTE_CAPABILITY__ >= 700)

    template <typename F> inline void setMaxDynamicSharedBytesPerBlock(F *func)
    {
      qudaFuncSetAttribute((const void *)func, cudaFuncAttributePreferredSharedMemoryCarveout,
                           (int)cudaSharedmemCarveoutMaxShared);
      qudaFuncSetAttribute((const void *)func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           deviceProp.sharedMemPerBlockOptin);
    }

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    typename std::enable_if<!Arg::is_aos, void>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    typename std::enable_if<Arg::is_aos, void>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      tp.block.x = 1;
      tp.block.y = block_y;
      tp.block.z = block_z;
      constexpr int shared_bytes = shared_memory_bytes(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      // TODO: Fix the split M/N.
      constexpr int t_m = 1;
      constexpr int t_n = 1;

      tp.grid = dim3(min_threads * t_m * t_n, 2, 1);

      auto kernel = ComputeUVMMA<from_coarse, dim, dir, bM, bN, bK, block_y, block_z, Arg>;
      setMaxDynamicSharedBytesPerBlock(kernel);
      kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    template <bool from_coarse, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    void launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (arg.dir == QUDA_BACKWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_uv_kernel<from_coarse, 0, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 1:
          launch_compute_uv_kernel<from_coarse, 1, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 2:
          launch_compute_uv_kernel<from_coarse, 2, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 3:
          launch_compute_uv_kernel<from_coarse, 3, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else {
        switch (arg.dim) {
        case 0:
          launch_compute_uv_kernel<from_coarse, 0, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                stream);
          break;
        case 1:
          launch_compute_uv_kernel<from_coarse, 1, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                stream);
          break;
        case 2:
          launch_compute_uv_kernel<from_coarse, 2, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                stream);
          break;
        case 3:
          launch_compute_uv_kernel<from_coarse, 3, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                stream);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      }
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<!from_coarse, int>::type launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads,
                                                                         const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for !from_coarse.");
      return -1;
    }
    
    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 1;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  16,  16,   8,   4,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  16,  16,   8,   8,   4>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,  24>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse,  48,  32,  24,  16,   4>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse,  48,  32,  24,  16,  12>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 3;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse,  48,  32,  24,   8,  24>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse,  48,  32,  24,  16,  12>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 5;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  48,  64,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  48,  64,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse,  48,  64,  24,   8,  24>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse,  48,  64,  24,  16,  12>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse,  48,  64,  24,  32,  12>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_uv_kernel<from_coarse,  48,  64,  24,  16,  24>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  48,  96,  24,   8,  12>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  48,  96,  24,   8,  24>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse,  48,  96,  24,  16,   6>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse,  48,  96,  24,  16,  12>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse,  48,  96,  24,  16,  12>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_uv_kernel<from_coarse,  48,  96,  24,  24,  12>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_uv_kernel<from_coarse,  48,  96,  24,  24,  24>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 2;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse,  64,  32,  32,   8,  16>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse,  64,  32,  32,   8,  32>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse,  64,  32,  32,  16,  16>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse, 128,  64,  64,   8,  16>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse, 128,  64,  64,   8,  32>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse, 128,  64,  64,  16,   8>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse, 128,  64,  64,  16,  16>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse, 128,  64,  64,  16,  32>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_uv_kernel<from_coarse, 128,  64,  64,  32,   8>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_uv_kernel<from_coarse, 128,  64,  64,  32,  16>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 7;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse, 128,  96,  64,   8,  12>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse, 128,  96,  64,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse, 128,  96,  64,   8,  32>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse, 128,  96,  64,  16,  12>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse, 128,  96,  64,  16,  24>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_uv_kernel<from_coarse, 128,  96,  64,  32,   6>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_uv_kernel<from_coarse, 128,  96,  64,  32,   8>(tp, arg, min_threads, stream); break;
      case 7: launch_compute_uv_kernel<from_coarse, 128,  96,  64,  32,  12>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 5;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_uv_kernel<from_coarse, 192,  96,  96,   8,  12>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_uv_kernel<from_coarse, 192,  96,  96,   8,  24>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_uv_kernel<from_coarse, 192,  96,  96,  16,   6>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_uv_kernel<from_coarse, 192,  96,  96,  16,  12>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_uv_kernel<from_coarse, 192,  96,  96,  24,  12>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_uv_kernel<from_coarse, 192,  96,  96,  24,  24>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    typename std::enable_if<!Arg::is_aos, void>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    typename std::enable_if<Arg::is_aos, void>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      tp.block.x = 1;
      tp.block.y = block_y;
      tp.block.z = block_z;
      constexpr int shared_bytes = shared_memory_bytes(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      // TODO: Fix the split M/N.
      constexpr int t_m = 1;
      constexpr int t_n = 1;

      tp.grid = dim3(min_threads * t_m * t_n, 2, 1);

      auto kernel = ComputeVUVMMA<from_coarse, dim, dir, bM, bN, bK, block_y, block_z, Arg>;
      setMaxDynamicSharedBytesPerBlock(kernel);
      kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    template <bool from_coarse, int bM, int bN, int bK, int block_y, int block_z, class Arg>
    void launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (arg.dir == QUDA_BACKWARDS) {
        switch (arg.dim) {
        case 0:
          launch_compute_vuv_kernel<from_coarse, 0, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                  stream);
          break;
        case 1:
          launch_compute_vuv_kernel<from_coarse, 1, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                  stream);
          break;
        case 2:
          launch_compute_vuv_kernel<from_coarse, 2, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                  stream);
          break;
        case 3:
          launch_compute_vuv_kernel<from_coarse, 3, QUDA_BACKWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                  stream);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      } else {
        switch (arg.dim) {
        case 0:
          launch_compute_vuv_kernel<from_coarse, 0, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 1:
          launch_compute_vuv_kernel<from_coarse, 1, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 2:
          launch_compute_vuv_kernel<from_coarse, 2, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        case 3:
          launch_compute_vuv_kernel<from_coarse, 3, QUDA_FORWARDS, bM, bN, bK, block_y, block_z>(tp, arg, min_threads,
                                                                                                 stream);
          break;
        default: errorQuda("arg.dim(=%d) is NOT supported.", arg.dim);
        }
      }
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<!from_coarse, int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for !from_coarse.");
      return -1;
    }
    
    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 6 && Arg::coarseColor == 6 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 2;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  16,  16,   8,   8,   4>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  16,  16,   8,   4,   8>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }


    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 24 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,  16>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,  16,   8>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,  32,   4>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 4;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,   8,  16>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,  16,   8>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  32,  32,  24,  32,   4>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 7;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,   8,  32>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,  16,   8>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,  16,  16>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,  16,  32>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,  32,   8>(tp, arg, min_threads, stream); break;
      case 7: launch_compute_vuv_kernel<from_coarse,  64,  64,  24,  32,  16>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 24 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 6;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  12,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  24,   8>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,   6,  16>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  12,  16>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  24,  16>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  12,  24>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_vuv_kernel<from_coarse,  96,  96,  24,  24,  24>(tp, arg, min_threads, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 32 && Arg::coarseColor == 32 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 3;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel<from_coarse,  32,  32,  32,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  32,  32,  32,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  32,  32,  32,  16,   8>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  32,  32,  32,  32,   4>(tp, arg, min_threads, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 64 && Arg::coarseColor == 64 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 7;
      // clang-format off
      switch (tp.aux.x) {
      case 0: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,   8,  16>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  16,   8>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  16,  16>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  16,  32>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  32,   4>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  32,   8>(tp, arg, min_threads, stream); break;
      case 7: launch_compute_vuv_kernel<from_coarse,  64,  64,  64,  32,  16>(tp, arg, min_threads, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      // clang-format on
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 64 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,   8,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,   8,  12>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,  16,   6>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,  16,   8>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,  16,  12>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,  32,   4>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_vuv_kernel<from_coarse,  96,  96,  64,  32,   6>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    typename std::enable_if<Arg::fineColor == 96 && Arg::coarseColor == 96 && Arg::fineSpin == 2 && Arg::coarseSpin == 2,
                            int>::type
    launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      if (query_max) return 6;
      switch (tp.aux.x) {
      // clang-format off
      case 0: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  12,   8>(tp, arg, min_threads, stream); break;
      case 1: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  24,   8>(tp, arg, min_threads, stream); break;
      case 2: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,   6,  16>(tp, arg, min_threads, stream); break;
      case 3: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  12,  16>(tp, arg, min_threads, stream); break;
      case 4: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  24,  16>(tp, arg, min_threads, stream); break;
      case 5: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  12,  24>(tp, arg, min_threads, stream); break;
      case 6: launch_compute_vuv_kernel<from_coarse,  96,  96,  96,  24,  24>(tp, arg, min_threads, stream); break;
      // clang-format on
      default: errorQuda("tp.aux.x(=%d) is NOT supported by (%d, %d, %d, %d).", tp.aux.x, Arg::fineSpin, Arg::coarseSpin, Arg::fineColor, Arg::coarseColor);
      }
      return -1;
    }

#else

    template <bool from_coarse, bool query_max = false, class Arg>
    int launch_compute_uv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

    template <bool from_coarse, bool query_max = false, class Arg>
    int launch_compute_vuv_kernel(TuneParam &tp, const Arg &arg, int min_threads, const cudaStream_t &stream)
    {
      errorQuda("MMA multigrid is not available for this setup.");
      return -1;
    }

#endif

  } // namespace mma

} // namespace quda
