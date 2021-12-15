#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#include <mdw_dslash5_tensor_core.cuh>
#endif
#include <kernel.h>

#include <mma_tensor_op/smma.cuh>

namespace quda {

#if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  /**
    @brief Parameter structure for applying the Dslash
   */
  template <class store_t_, int nColor_, int Ls_, int block_dim_x_, bool dagger_, int min_blocks_per_sm, bool reload_>
    struct Dslash5MmaArg : kernel_param<> {
      using store_t = store_t_;
      using real = typename mapper<store_t>::type; // the compute type for the in kernel computation
      static constexpr int nColor = nColor_;
      static constexpr int Ls = Ls_;
      static constexpr int block_dim_x = block_dim_x_;
      static constexpr int block_dim = block_dim_x * Ls;
      static constexpr int min_blocks = min_blocks_per_sm;
      static constexpr bool dagger = dagger_;
      static constexpr bool reload = reload_; // Whether reload the m5inv matrix
      using F = typename colorspinor_mapper<store_t, 4, nColor>::type; // color spin field order

      F out;      // output vector field
      const F in; // input vector field
      const F x;  // auxiliary input vector field

      const int nParity;      // number of parities we're working on
      const int parity;       // output parity of this dslash operator
      const int volume_cb;    // checkerboarded volume
      const int volume_4d_cb; // 4-d checkerboarded volume

      const real m_f; // fermion mass parameter
      const real m_5; // Wilson mass shift

      real b; // real constant Mobius coefficient
      real c; // real constant Mobius coefficient
      real a; // real xpay coefficient

      real kappa;
      real inv;

      Dslash5MmaArg(ColorSpinorField &out, const ColorSpinorField &in,
          const ColorSpinorField &x, double m_f_, double m_5_, const Complex *b_5, const Complex *c_5,
          double a, int parity) :
        out(out),
        in(in),
        x(x),
        nParity(in.SiteSubset()),
        parity(parity),
        volume_cb(in.VolumeCB()),
        volume_4d_cb(volume_cb / Ls),
        m_f(m_f_),
        m_5(m_5_),
        a(a)
      {
        if (in.Nspin() != 4) { errorQuda("nSpin = %d NOT supported.\n", in.Nspin()); }
        if (nParity == 2) { errorQuda("nParity = 2 NOT supported, yet.\n"); }
        if (b_5[0] != b_5[1] || b_5[0].imag() != 0) { errorQuda("zMobius is NOT supported yet.\n"); }

        b = b_5[0].real();
        c = c_5[0].real();
        kappa = -(c * (4. + m_5) - 1.) / (b * (4. + m_5) + 1.);
        inv = 0.5 / (1. + std::pow(kappa, (int)Ls) * m_f); // 0.5 to normalize the (1 +/- gamma5) in the chiral projector.
      }
    };

  /**
    @brief Tensor core kernel for applying Wilson hopping term and then the beta + alpha * M5inv operator
    The integer kernel types corresponds to the enum MdwfFusedDslashType.
   */
  template <typename Arg> struct Dslash5MmaKernel {
    const Arg &arg;
    constexpr Dslash5MmaKernel(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __forceinline__ void operator()()
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, 4>;
      constexpr int Ls = Arg::Ls;
      const int parity = arg.nParity == 2 ? arg.parity : 0;

      SharedMemoryCache<real> shared_memory_data;

      constexpr int m = 4 * Ls;
      constexpr int n = 6 * Arg::block_dim_x;
      constexpr int k = m;

      using Mma = typename mma_mapper<typename Arg::store_t>::type;

      constexpr int smem_ld_a = m + Mma::t_pad;
      constexpr int smem_ld_b = n + Mma::t_pad;
      constexpr int smem_ld_c = n + Mma::acc_pad;

      real *smem_a = shared_memory_data.data();
      real *smem_b = Arg::reload ? smem_a + smem_ld_a * k : smem_a;
      real *smem_c = smem_b + smem_ld_b * k;

      smem_construct_m5inv<smem_ld_a, Arg::dagger>(arg, smem_a);
      __syncthreads();

      bool idle = false;
      int x_cb_base = blockIdx.x * blockDim.x; // base.
      int x_cb;

      constexpr int tk_dim = k / Mma::mma_k;
      typename Mma::WarpRegisterMapping wrm(threadIdx.y * blockDim.x + threadIdx.x);
      typename Mma::OperandA op_a[Arg::reload ? 1 : tk_dim];
      if (!Arg::reload) { // the data in registers can be resued.
        constexpr int tm_dim = m / Mma::mma_m;
        constexpr int tn_dim = n / Mma::mma_n;

        constexpr int total_warp = Arg::block_dim_x * Ls / 32;
        const int this_warp = (threadIdx.y * Arg::block_dim_x + threadIdx.x) / 32;

        constexpr int total_tile = tm_dim * tn_dim;

        constexpr int warp_cycle = total_tile / total_warp;
        const int warp_m = this_warp * warp_cycle / tn_dim;
#pragma unroll
        for (int tile_k = 0; tile_k < tk_dim; tile_k++) { op_a[tile_k].template load<smem_ld_a>(smem_a, tile_k, warp_m, wrm); }
      }

      while (x_cb_base < arg.volume_4d_cb) {
        x_cb = x_cb_base + threadIdx.x;
        int s = threadIdx.y;
        if (x_cb >= arg.volume_4d_cb) { idle = true; }

        Vector in;
        if (!idle) {
          in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
        }
        smem_take_vector<smem_ld_b>(in, smem_b);

        __syncthreads();
        mma_sync_gemm<Mma, Arg::block_dim_x, Arg::Ls, m, n, smem_ld_a, smem_ld_b, smem_ld_c, Arg::reload>(op_a, smem_a, smem_b, smem_c, wrm);
        __syncthreads();

        if (!idle) {
          Vector out = smem_give_vector<smem_ld_c, Vector>(smem_c);
          arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
        }
        // __syncthreads();

        x_cb_base += gridDim.x * blockDim.x;
      } // while
    }
  };

#endif // #if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

}

