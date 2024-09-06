#include <color_spinor_field_order.h>
#include <block_reduce_helper.h>
#include <multigrid_helper.cuh>
#include <mma_tensor_op/gemm.cuh>
#include <fast_intdiv.h>

namespace quda
{

  using namespace quda::colorspinor;

  /**
      Kernel argument struct
  */
  template <typename mma_t_, typename out_t_, typename in_t_, typename v_t_, int fineSpin_, int fineColor_, int coarseSpin_,
            int coarseColor_, int nVec_, int aggregate_size_, int bN_, int bM_, int bK_, int block_y_, int block_z_>
  struct RestrictMmaArg : kernel_param<> {

    static constexpr int block_dim = block_z_ * block_y_;
    static constexpr int min_blocks = 1;

    using mma_t = mma_t_;

    using out_t = out_t_;
    using real = out_t;
    using in_t = in_t_;
    using v_t = v_t_;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int coarseColor = coarseColor_;
    static constexpr int nVec = nVec_;
    static constexpr int aggregate_size = aggregate_size_;
    static constexpr int bN = bN_;
    static constexpr int bM = bM_;
    static constexpr int bK = bK_;
    static constexpr int block_y = block_y_;
    static constexpr int block_z = block_z_;

    static constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    // disable ghost to reduce arg size
    using in_accessor_t = FieldOrderCB<real, fineSpin, fineColor, nVec, csOrder, in_t, in_t, false, isFixed<in_t>::value>;
    using out_accessor_t = FieldOrderCB<real, coarseSpin, coarseColor, nVec, csOrder, out_t, out_t, true>;
    using v_accessor_t = FieldOrderCB<real, fineSpin, fineColor, coarseColor, csOrder, v_t, v_t>;

    static constexpr int spin_block_factor = spin_mapper<fineSpin, coarseSpin>::get_spin_block_factor();
    static_assert(bK % (fineColor * spin_block_factor) == 0, "K %% Arg::bK != 0.\n");

    static constexpr int aggregate_per_block = bK / (fineColor * spin_block_factor);
    static_assert(aggregate_size % aggregate_per_block == 0, "aggregate_size %% aggregate_per_block");

    out_accessor_t out;
    in_accessor_t in;
    const v_accessor_t v;
    const int_fastdiv aggregate_size_cb; // number of checkerboard sites that form a single aggregate
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin, coarseSpin> spin_map;
    const int parity;  // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field

    RestrictMmaArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      kernel_param(dim3(out.Volume() * coarseSpin, block_y, block_z)),
      out(out),
      in(in),
      v(v),
      aggregate_size_cb(in.VolumeCB() / out.Volume()),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(in.SiteSubset())
    {
      if (out.Nvec() > static_cast<int>(get_max_multi_rhs()))
        errorQuda("vector set size %d greater than max size %d", out.Nvec(), get_max_multi_rhs());
      if (out.Nvec() != nVec) { errorQuda("out.Nvec() (%d) != nVec (%d)", out.Nvec(), nVec); }
      if (in.Nvec() != nVec) { errorQuda("in.Nvec() (%d) != nVec (%d)", in.Nvec(), nVec); }
    }
  };

  template <int contiguous_dim, bool dagger, int contiguous_limit, class smem_obj_t, class gmem_obj_t, class Arg>
  inline void __device__ load_g2s(smem_obj_t &smem_real, smem_obj_t &smem_imag, const gmem_obj_t &gmem, int x_coarse,
                                  int coarse_spin, int contiguous_dim_offset, int aggregate_k_offset,
                                  int *coarse_to_fine, const Arg &arg)
  { // v as a
    constexpr int elements_per_thread = 16 / (sizeof(typename gmem_obj_t::store_type) * 2);
    static_assert(contiguous_dim % elements_per_thread == 0, "contiguous_dim %% elements_per_thread == 0");
    int thread = target::thread_idx().y + Arg::block_y * target::thread_idx().z;
    while (thread < (contiguous_dim / elements_per_thread) * Arg::spin_block_factor * Arg::fineColor
             * Arg::aggregate_per_block) {
      int thread_idx = thread;
      int contiguous = thread_idx % (contiguous_dim / elements_per_thread) * elements_per_thread;
    constexpr bool check_contiguous_bound = !(contiguous_limit % contiguous_dim == 0);
    if (!check_contiguous_bound || contiguous + contiguous_dim_offset < contiguous_limit) {
      thread_idx /= (contiguous_dim / elements_per_thread);
      int fine_spin_block = thread_idx % Arg::spin_block_factor; // fineSpin / coarseSpin
      thread_idx /= Arg::spin_block_factor;
      int fine_color = thread_idx % Arg::fineColor;
      thread_idx /= Arg::fineColor;
      int x_fine_offset = thread_idx + aggregate_k_offset;

      const int parity_offset = x_fine_offset >= arg.aggregate_size_cb ? 1 : 0;
      const int x_fine_cb_offset = x_fine_offset % arg.aggregate_size_cb;
      const int parity = arg.nParity == 2 ? parity_offset : arg.parity;

      // look-up map is ordered as (coarse-block-id + fine-point-id),
      // with fine-point-id parity ordered
      const int x_fine = coarse_to_fine[parity * arg.aggregate_size_cb + x_fine_cb_offset];
      const int x_fine_cb = x_fine - parity * arg.in.VolumeCB();

      const int v_parity = (gmem.Nparity() == 2) ? parity : 0;

      int fine_spin = fine_spin_block + coarse_spin * Arg::spin_block_factor;
      auto a_gmem = gmem(v_parity, x_fine_cb, fine_spin, fine_color, contiguous + contiguous_dim_offset);
      complex<typename gmem_obj_t::store_type> a[elements_per_thread];
      mma::batch_load_t<complex<typename gmem_obj_t::store_type>, elements_per_thread>::load(a, a_gmem.data());

      int smem_m = contiguous;
      int smem_k = (thread_idx * Arg::spin_block_factor + fine_spin_block) * Arg::fineColor + fine_color;

      typename Arg::real a_real[elements_per_thread];
      typename Arg::real a_imag[elements_per_thread];
      if constexpr (decltype(a_gmem)::fixed) {
        auto scale_inv = a_gmem.get_scale_inv();
#pragma unroll
        for (int e = 0; e < elements_per_thread; e++) {
          a_real[e] = +a[e].real() * scale_inv;
          a_imag[e] = dagger ? -a[e].imag() * scale_inv : +a[e].imag() * scale_inv;
        }
      } else {
#pragma unroll
        for (int e = 0; e < elements_per_thread; e++) {
          a_real[e] = +a[e].real();
          a_imag[e] = dagger ? -a[e].imag() : +a[e].imag();
        }
      }

      static_assert(smem_obj_t::ldm == 1, "smem_obj_t::ldm == 1");
      smem_real.vector_load(smem_m, smem_k, mma::make_vector_t<typename Arg::real, elements_per_thread>::get(a_real));
      smem_imag.vector_load(smem_m, smem_k, mma::make_vector_t<typename Arg::real, elements_per_thread>::get(a_imag));
    }

      thread += Arg::block_y * Arg::block_z;
    }
  }

  template <typename Arg>
  void __device__ inline restrict_mma(int x_coarse, int coarse_spin, int m_offset, int n_offset, const Arg &arg)
  {

    constexpr int M = Arg::nVec;
    constexpr int N = Arg::coarseColor;
    constexpr int K = Arg::fineColor * Arg::spin_block_factor * Arg::aggregate_size;

    constexpr int ldc = M;

    using mma_t = typename Arg::mma_t;
    // The first two ldc's are dummy
    using Config = mma::MmaConfig<mma_t, M, N, K, ldc, ldc, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

    static_assert(M % Arg::bM == 0, "M %% Arg::bM != 0.\n");
    static_assert(K % Arg::bK == 0, "K %% Arg::bK != 0.\n");

    __shared__ int coarse_to_fine[Arg::aggregate_size];
    int index = target::thread_idx().y + Arg::block_y * target::thread_idx().z;
    while (index < Arg::aggregate_size) {
      coarse_to_fine[index] = arg.coarse_to_fine[x_coarse * 2 * arg.aggregate_size_cb + index];
      index += Arg::block_y * Arg::block_z;
    }
    __syncthreads();

    extern __shared__ typename mma_t::compute_t smem_ptr[];

    typename Config::SmemObjA smem_obj_a_real(smem_ptr);
    typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * Arg::bK);

    typename Config::ALoader a_loader;
    typename Config::BLoader b_loader;

    typename Config::Accumulator accumulator((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

    accumulator.zero();

    for (int aggregate_k_offset = 0; aggregate_k_offset < Arg::aggregate_size;
         aggregate_k_offset += Arg::aggregate_per_block) {
      __syncthreads();

      constexpr bool a_dagger = true;
      load_g2s<Arg::bM, a_dagger, M>(smem_obj_a_real, smem_obj_a_imag, arg.in, x_coarse, coarse_spin, m_offset,
                                  aggregate_k_offset, coarse_to_fine, arg);

      constexpr bool b_dagger = false;
      load_g2s<Arg::bN, b_dagger, N>(smem_obj_b_real, smem_obj_b_imag, arg.v, x_coarse, coarse_spin, n_offset,
                                  aggregate_k_offset, coarse_to_fine, arg);

      __syncthreads();
      accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
    }

    const int parity_coarse = x_coarse >= arg.out.VolumeCB() ? 1 : 0;
    const int x_coarse_cb = x_coarse - parity_coarse * arg.out.VolumeCB();

    auto c_gmem = arg.out(parity_coarse, x_coarse_cb, coarse_spin, 0, 0);
    constexpr bool c_dagger = true;
    accumulator.template store<M, N, ldc, c_dagger>(c_gmem, m_offset, n_offset, assign_t());
  }

  template <typename Arg> struct RestrictorMma {
    const Arg &arg;
    constexpr RestrictorMma(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()()
    {
      int coarse_spin = target::block_idx().x % Arg::coarseSpin;
      int x_coarse = target::block_idx().x / Arg::coarseSpin;

      int m_offset = Arg::bM * target::block_idx().y;
      int n_offset = Arg::bN * target::block_idx().z;

      restrict_mma(x_coarse, coarse_spin, m_offset, n_offset, arg);
    }
  };

} // namespace quda
