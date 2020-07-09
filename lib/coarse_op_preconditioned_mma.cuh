
#include <gauge_field.h>
#include <blas_cublas.h>
#include <blas_quda.h>
#include <tune_quda.h>

#include <jitify_helper.cuh>

#if ((__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1) || (__CUDACC_VER_MAJOR__ > 10))                         \
  && (__COMPUTE_CAPABILITY__ >= 700)

#include <kernels/coarse_op_preconditioned_mma.cuh>

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
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, (const void *)func);
      qudaFuncSetAttribute((const void *)func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           deviceProp.sharedMemPerBlockOptin - attr.sharedSizeBytes);
    }

    template <bool compute_max_only, int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg>
    typename std::enable_if<!Arg::is_aos, void>::type launch_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                    const cudaStream_t &stream)
    {
      errorQuda("MMA implementation is ONLY built for AoS order.");
    }

    template <bool compute_max_only, int bM, int bN, int bK, int block_y, int block_z, int min_block_cta = 1, class Arg>
    typename std::enable_if<Arg::is_aos, void>::type launch_kernel(Arg &arg, int min_threads, TuneParam &tp,
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
      setMaxDynamicSharedBytesPerBlock(kernel);
      kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    template <bool compute_max_only, class Arg>
    typename std::enable_if<Arg::N == 48, void>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                         const cudaStream_t &stream)
    {
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  48,  48,  48,  24,  12>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  48,  48,  48,   6,  48>(arg, min_threads, tp, stream); break;
      case   2: launch_kernel<compute_max_only,  48,  48,  48,  12,  24>(arg, min_threads, tp, stream); break;
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 48", tp.aux.x);
      }
      // clang-format on
    }

    template <bool compute_max_only, class Arg>
    typename std::enable_if<Arg::N == 64, void>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                         const cudaStream_t &stream)
    {
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
    }

    template <bool compute_max_only, class Arg>
    typename std::enable_if<Arg::N == 128, void>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                          const cudaStream_t &stream)
    {
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  64,  64,  16,  32,  16,  2>(arg, min_threads, tp, stream); break;
#if (__COMPUTE_CAPABILITY__ > 700)
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
    }

    template <bool compute_max_only, class Arg>
    typename std::enable_if<Arg::N == 192, void>::type launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp,
                                                                          const cudaStream_t &stream)
    {
      // clang-format off
      switch (tp.aux.x) {
      case   0: launch_kernel<compute_max_only,  64,  64,  16,  16,  16,  2>(arg, min_threads, tp, stream); break;
      case   1: launch_kernel<compute_max_only,  64,  64,  64,  16,  16,  2>(arg, min_threads, tp, stream); break;
      case   2: launch_kernel<compute_max_only,  16, 192, 192,  24,  16    >(arg, min_threads, tp, stream); break;
      case   3: launch_kernel<compute_max_only,  64,  64,  32,  16,  16,  2>(arg, min_threads, tp, stream); break;
#if (__COMPUTE_CAPABILITY__ > 700)
      case   4: launch_kernel<compute_max_only,  16, 192, 192,  96,   8    >(arg, min_threads, tp, stream); break;
#else
      case   4: launch_kernel<compute_max_only,  16, 192, 192,  48,   8    >(arg, min_threads, tp, stream); break;
#endif
      default: errorQuda("tp.aux.x(=%d) is NOT supported by N = 192", tp.aux.x);
      }
      // clang-format on
    }

#else

    template <bool compute_max_only, class Arg>
    void launch_yhat_kernel(Arg &arg, int min_threads, TuneParam &tp, const cudaStream_t &stream)
    {
      errorQuda("MMA multigrid is not available for this setup.");
    }

#endif

  } // namespace mma

} // namespace quda
