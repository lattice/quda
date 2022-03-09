#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#include <mdw_dslash5_tensor_core.cuh>
#endif
#include <kernel.h>

#include <mma_tensor_op/smma.cuh>
#include <mdw_dslash5_tensor_core.cuh>

#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

#if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  /**
    @brief Parameter structure for applying the Dslash
   */
  template <class store_t_, int nColor_, int Ls_, int block_dim_x_, bool dagger, bool xpay>
  struct Dslash5MmaArg : Dslash5Arg<store_t_, nColor_, dagger, xpay, Dslash5Type::M5_INV_MOBIUS> {
    using D5Arg = Dslash5Arg<store_t_, nColor_, dagger, xpay, Dslash5Type::M5_INV_MOBIUS>;
    using store_t = store_t_;
    using real = typename D5Arg::real;

    using D5Arg::nColor;
    using D5Arg::type;
    using D5Arg::m_f;
    using D5Arg::m_5;
    using D5Arg::kernel_param::threads;

    static constexpr int Ls = Ls_;
    static constexpr int block_dim_x = block_dim_x_;
    static constexpr int block_dim = block_dim_x * Ls;
    using Vector = ColorSpinor<real, nColor, 4>;

    static constexpr int m = 4 * Ls;
    static constexpr int n = 6 * block_dim_x;
    static constexpr int k = m;

    using Mma = typename mma_mapper<store_t>::type;
    static constexpr int smem_ld_a = m + Mma::t_pad;
    static constexpr int smem_ld_b = n + Mma::t_pad;
    static constexpr int smem_ld_c = n + Mma::acc_pad;

    static constexpr int tk_dim = k / Mma::mma_k;

    static constexpr int tm_dim = m / Mma::mma_m;
    static constexpr int tn_dim = n / Mma::mma_n;

    static constexpr int total_warp = block_dim_x * Ls / 32;
    static constexpr int total_tile = tm_dim * tn_dim;
    static constexpr int warp_cycle = total_tile / total_warp;

    static constexpr bool use_mma = true;

    Dslash5MmaArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f_,
                  double m_5_, const Complex *b_5, const Complex *c_5, double a) :
      D5Arg(out, in, x, m_f_, m_5_, b_5, c_5, a)
    {
      if (D5Arg::nParity == 2) { errorQuda("nParity = 2 NOT supported, yet.\n"); }
      if (b_5[0] != b_5[1] || b_5[0].imag() != 0) { errorQuda("zMobius is NOT supported yet.\n"); }
    }

    /**
      @brief Round up the number of threads in the x direction such that it is a multiple of the
        block dimension in x. This is primarily to make sure we have complete warps participating
        in the MMA instructions.
      @param block_dim_x block dimension in x
     */
    void round_up_threads_x(unsigned int block_dim_x) {
      threads.x = (threads.x + block_dim_x - 1) / block_dim_x * block_dim_x;

    }

  };

  template <class T, int size> struct RegisterArray {
    T data[size];
  };

  template <bool dagger, class Arg> __device__ void smem_construct_m5inv(Arg &arg)
  {
    SharedMemoryCache<typename Arg::real> shared_memory_data;
    auto *smem_a = shared_memory_data.data();

    smem_construct_m5inv<Arg::smem_ld_a, dagger>(arg, smem_a);
  }

  template <class Arg> __device__ auto construct_m5inv_op(Arg &arg)
  {
    typename Arg::Mma::WarpRegisterMapping wrm(threadIdx.y * blockDim.x + threadIdx.x);
    RegisterArray<typename Arg::Mma::OperandA, Arg::tk_dim> op_a;

    SharedMemoryCache<typename Arg::real> shared_memory_data;
    typename Arg::real *smem_a = shared_memory_data.data();

    const int this_warp = (threadIdx.y * Arg::block_dim_x + threadIdx.x) / 32;
    const int warp_m = this_warp * Arg::warp_cycle / Arg::tn_dim;
#pragma unroll
    for (int tile_k = 0; tile_k < Arg::tk_dim; tile_k++) {
      op_a.data[tile_k].template load<Arg::smem_ld_a>(smem_a, tile_k, warp_m, wrm);
    }

    return op_a;
  }

  template <class OpA, class Vector, class Arg> __device__ Vector m5inv_mma(OpA &op_a, Vector in, Arg &arg)
  {
    typename Arg::Mma::WarpRegisterMapping wrm(threadIdx.y * blockDim.x + threadIdx.x);

    SharedMemoryCache<typename Arg::real> shared_memory_data;
    auto *smem_a = shared_memory_data.data();
    auto *smem_b = smem_a;
    auto *smem_c = smem_b + Arg::smem_ld_b * Arg::k;

    smem_take_vector<Arg::smem_ld_b>(in, smem_b);

    __syncthreads();
    mma_sync_gemm<typename Arg::Mma, Arg::block_dim_x, Arg::Ls, Arg::m, Arg::n, Arg::smem_ld_a, Arg::smem_ld_b, Arg::smem_ld_c, false>(
      op_a.data, smem_a, smem_b, smem_c, wrm);
    __syncthreads();

    Vector out = smem_give_vector<Arg::smem_ld_c, Vector>(smem_c);
    return out;
  }

  /**
    @brief Tensor core kernel for applying Wilson hopping term and then the beta + alpha * M5inv operator
    The integer kernel types corresponds to the enum MdwfFusedDslashType.
   */
  template <typename Arg> struct Dslash5MmaKernel {
    const Arg &arg;
    constexpr Dslash5MmaKernel(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __forceinline__ void operator()(unsigned int, int s, int parity)
    {
      smem_construct_m5inv<Arg::dagger>(arg);
      __syncthreads();

      bool idle = false;
      int x_cb_base = blockIdx.x * blockDim.x; // base.

      auto op_a = construct_m5inv_op(arg);

      while (x_cb_base < arg.volume_4d_cb) {
        int x_cb = x_cb_base + threadIdx.x;
        if (x_cb >= arg.volume_4d_cb) { idle = true; }

        typename Arg::Vector in;
        if (!idle) { in = arg.in(s * arg.volume_4d_cb + x_cb, parity); }
        auto out = m5inv_mma(op_a, in, arg);

        if (!idle) { arg.out(s * arg.volume_4d_cb + x_cb, parity) = out; }

        x_cb_base += gridDim.x * blockDim.x;
      } // while
    }
  };

#endif // #if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

} // namespace quda
