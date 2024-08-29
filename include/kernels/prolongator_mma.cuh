#include <color_spinor_field_order.h>
#include <color_spinor.h>
#include <multigrid_helper.cuh>
#include <mma_tensor_op/gemm.cuh>
#include <kernel.h>

namespace quda
{

  /**
      Kernel argument struct
  */
  template <typename mma_t_, typename Float_, typename vFloat_, int fineSpin_, int fineColor_, int coarseSpin_,
            int coarseColor_, int nVec_, bool to_non_rel_, int bN_, int bM_, int bK_, int block_y_, int block_z_>
  struct ProlongateMmaArg : kernel_param<> {

    static constexpr int block_dim = block_z_ * block_y_;
    static constexpr int min_blocks = 1;

    using Float = Float_;
    using vFloat = vFloat_;
    using real = Float;
    using mma_t = mma_t_;
    static constexpr int nVec = nVec_;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseColor = coarseColor_;
    static constexpr bool to_non_rel = to_non_rel_;

    static constexpr int bN = bN_;
    static constexpr int bM = bM_;
    static constexpr int bK = bK_;
    static constexpr int block_y = block_y_;
    static constexpr int block_z = block_z_;

    static constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    // disable ghost to reduce arg size
    using in_accessor_t =
      typename colorspinor::FieldOrderCB<Float, coarseSpin, coarseColor, nVec, csOrder, Float, Float, true>;
    using out_accessor_t =
      typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, nVec, csOrder, Float, Float, true>;
    using v_accessor_t =
      typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat, vFloat>;

    static constexpr int spin_block_factor = spin_mapper<fineSpin, coarseSpin>::get_spin_block_factor();

    out_accessor_t out;
    const in_accessor_t in;
    const v_accessor_t v;
    const int *geo_map; // need to make a device copy of this
    const spin_mapper<fineSpin, coarseSpin> spin_map;
    const int parity;  // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field

    ProlongateMmaArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v, const int *geo_map,
                     const int parity) :
      kernel_param(dim3(out.VolumeCB() * out.SiteSubset() * out.Nspin(), block_y, block_z)),
      out(out),
      in(in),
      v(v),
      geo_map(geo_map),
      spin_map(),
      parity(parity),
      nParity(out.SiteSubset())
    {
      if (out.Nvec() > static_cast<int>(get_max_multi_rhs()))
        errorQuda("vector set size %d greater than max size %d", out.Nvec(), get_max_multi_rhs());
      if (out.Nvec() != nVec) { errorQuda("out.Nvec() (%d) != nVec (%d)", out.Nvec(), nVec); }
      if (in.Nvec() != nVec) { errorQuda("in.Nvec() (%d) != nVec (%d)", in.Nvec(), nVec); }
    }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <typename Arg>
  __device__ inline void prolongate_mma(int parity, int x_cb, int spin, int m_offset, int n_offset, const Arg &arg)
  {
    int x = parity * arg.out.VolumeCB() + x_cb;
    int x_coarse = arg.geo_map[x];
    int parity_coarse = (x_coarse >= arg.in.VolumeCB()) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse * arg.in.VolumeCB();
    int spinor_parity = (arg.nParity == 2) ? parity : 0;
    int v_parity = (arg.v.Nparity() == 2) ? parity : 0;

    // Everything is dagger'ed since coarseColor >= fineColor

    constexpr int M = Arg::nVec;
    constexpr int N = Arg::fineColor * Arg::spin_block_factor;
    constexpr int K = Arg::coarseColor;

    constexpr int lda = M;
    constexpr int ldb = K;
    constexpr int ldc = M;

    using mma_t = typename Arg::mma_t;
    using Config = mma::MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

    static_assert(M % Arg::bM == 0, "M %% Arg::bM != 0.\n");
    static_assert(K % Arg::bK == 0, "K %% Arg::bK != 0.\n");

    extern __shared__ typename mma_t::compute_t smem_ptr[];

    typename Config::SmemObjA smem_obj_a_real(smem_ptr);
    typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * Arg::bK);

    typename Config::ALoader a_loader;
    typename Config::BLoader b_loader;

    typename Config::Accumulator accumulator((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

    accumulator.zero();

    auto a = arg.in(parity_coarse, x_coarse_cb, arg.spin_map(spin * Arg::spin_block_factor, parity), 0, 0);
    auto b = arg.v(v_parity, x_cb, spin * Arg::spin_block_factor, 0, 0);
    constexpr bool a_dagger = true;
    constexpr bool b_dagger = true;

    for (int k_offset = 0; k_offset < K; k_offset += Arg::bK) {
      __syncthreads();
      a_loader.template g2r<lda, a_dagger>(a, m_offset, k_offset);
      b_loader.template g2r<ldb, b_dagger>(b, n_offset, k_offset);
      a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);
      b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
      __syncthreads();
      accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
    }

    auto c = arg.out(spinor_parity, x_cb, spin * Arg::spin_block_factor, 0, 0);
    constexpr bool c_dagger = true;
    accumulator.template store<M, N, ldc, c_dagger>(c, m_offset, n_offset, assign_t());
  }

  template <typename Arg> struct ProlongatorMma {
    const Arg &arg;
    constexpr ProlongatorMma(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()()
    {
      int n_offset = target::block_idx().z * Arg::bN;
      int m_offset = target::block_idx().y * Arg::bM;

      int parity_x_cb_spin = target::block_idx().x;
      int spin = parity_x_cb_spin % (Arg::fineSpin / Arg::spin_block_factor);
      int parity_x_cb = parity_x_cb_spin / (Arg::fineSpin / Arg::spin_block_factor);
      int parity = (arg.nParity == 2) ? parity_x_cb % 2 : arg.parity;
      int x_cb = (arg.nParity == 2) ? parity_x_cb / 2 : parity_x_cb;

      prolongate_mma(parity, x_cb, spin, m_offset, n_offset, arg);
    }
  };

} // namespace quda
