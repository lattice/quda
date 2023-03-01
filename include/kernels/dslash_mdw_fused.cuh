#include <gauge_field_order.h>
#ifdef QUDA_MMA_AVAILABLE
#include <mdw_dslash5_tensor_core.cuh>
#endif
#include <kernel.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  namespace mobius_tensor_core
  {

#ifdef QUDA_MMA_AVAILABLE

    using mma_t = mma::hmma_t; // simt::simt_t<mma::half, 8, 4, 2, 2>;

    constexpr int sm_m_pad_size(int m) { return mma_t::pad_size(m); }

    constexpr int sm_n_pad_size(int n) { return mma_t::pad_size(n); }

    /**
      @brief Parameter structure for applying the Dslash
    */
    template <class storage_type_, int nColor_, QudaReconstructType recon_, int Ls_, MdwfFusedDslashType type_,
              int block_dim_x_, int min_blocks_per_SM, bool reload_>
    // storage_type is the usual "Float" in other places in QUDA
    struct FusedDslashArg : kernel_param<> {
      using storage_type = storage_type_;
      using real = typename mapper<storage_type>::type; // the compute type for the in kernel computation
      static constexpr int nColor = nColor_;
      static constexpr QudaReconstructType recon = recon_;
      static constexpr int Ls = Ls_;
      static constexpr MdwfFusedDslashType type = type_;
      static constexpr int block_dim_x = block_dim_x_;
      static constexpr int block_dim = block_dim_x * Ls;
      static constexpr int min_blocks = min_blocks_per_SM;
      static constexpr bool reload = reload_;
      static constexpr bool spin_project = true;
      static constexpr bool spinor_direct_load = true; // false means texture load
      using F = typename colorspinor_mapper<storage_type, 4, nColor, spin_project, spinor_direct_load>::type; // color spin field order
      static constexpr bool gauge_direct_load = true;                          // false means texture load
      static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_EXTENDED; // gauge field used is an extended one
      using G = typename gauge_mapper<storage_type, recon, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type; // gauge field order

      F out;      // output vector field
      const F in; // input vector field
      F y;        // auxiliary output vector field
      const F x;  // auxiliary input vector field

      const G U; // The gauge field

      const int nParity;      // number of parities we're working on
      const int parity;       // output parity of this dslash operator
      const int volume_cb;    // checkerboarded volume
      const int volume_4d_cb; // 4-d checkerboarded volume

      const real m_f; // fermion mass parameter
      const real m_5; // Wilson mass shift

      const int dim[4];
      const int shift[4];      // sites where we actually calculate.
      const int halo_shift[4]; // halo means zero. When we are expanding we have halo of cs-field where values are zero.

      const int_fastdiv shrunk_dim[4]; // dimension after shifts are considered.

      // partial kernel and expansion parameters
      const int volume_4d_cb_shift; // number of 4d sites we need calculate
      // const int volume_4d_cb_expansive; //

      //    const bool xpay;        // whether we are doing xpay or not

      real b; // real constant Mobius coefficient
      real c; // real constant Mobius coefficient
      real a; // real xpay coefficient

      real kappa;
      real fac_inv;

      // (beta + alpha*m5inv) * in
      real alpha = 1.;
      real beta = 0.;

      real m_scale = 1.; // scale factor for the matrix

      bool small_kappa = false;

      const bool comm[4];

      FusedDslashArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, ColorSpinorField &y,
                     const ColorSpinorField &x, double m_f_, double m_5_, const Complex *b_5, const Complex *c_5,
                     int parity, int shift_[4], int halo_shift_[4]) :
        out(out),
        in(in),
        y(y),
        x(x),
        U(U),
        nParity(in.SiteSubset()),
        parity(parity),
        volume_cb(in.VolumeCB() > out.VolumeCB() ? in.VolumeCB() : out.VolumeCB()),
        volume_4d_cb(volume_cb / Ls_),
        m_f(m_f_),
        m_5(m_5_),
        dim {(3 - nParity) * (in.VolumeCB() > out.VolumeCB() ? in.X(0) : out.X(0)),
             in.VolumeCB() > out.VolumeCB() ? in.X(1) : out.X(1), in.VolumeCB() > out.VolumeCB() ? in.X(2) : out.X(2),
             in.VolumeCB() > out.VolumeCB() ? in.X(3) : out.X(3)},
        shift {shift_[0], shift_[1], shift_[2], shift_[3]},
        halo_shift {halo_shift_[0], halo_shift_[1], halo_shift_[2], halo_shift_[3]},
        shrunk_dim {dim[0] - 2 * shift[0], dim[1] - 2 * shift[1], dim[2] - 2 * shift[2], dim[3] - 2 * shift[3]},
        volume_4d_cb_shift(shrunk_dim[0] * shrunk_dim[1] * shrunk_dim[2] * shrunk_dim[3] / 2),
        comm {static_cast<bool>(comm_dim_partitioned(0)), static_cast<bool>(comm_dim_partitioned(1)),
              static_cast<bool>(comm_dim_partitioned(2)), static_cast<bool>(comm_dim_partitioned(3))}
      {
        if (in.Nspin() != 4) { errorQuda("nSpin = %d NOT supported.\n", in.Nspin()); }

        if (nParity == 2) { errorQuda("nParity = 2 NOT supported, yet.\n"); }

        if (b_5[0] != b_5[1] || b_5[0].imag() != 0) { errorQuda("zMobius is NOT supported yet.\n"); }

        b = b_5[0].real();
        c = c_5[0].real();
        kappa = -(c * (4. + m_5) - 1.) / (b * (4. + m_5) + 1.); // This is actually -kappa in my(Jiqun Tu) notes.

        if (kappa * kappa < 1e-6) { small_kappa = true; }

        fac_inv
          = 0.5 / (1. + std::pow(kappa, (int)Ls) * m_f); // 0.5 to normalize the (1 +/- gamma5) in the chiral projector.
        switch (type) {
        case MdwfFusedDslashType::D4_D5INV_D5PRE:
        case MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG:
          if (small_kappa) {
            m_scale = b;
            alpha = (c - b * kappa) / (2. * b);
            beta = 1.;
          } else {
            m_scale = b + c / kappa;
            alpha = 1.;
            beta = -1. / (1. + (kappa * b) / c);
          }
          break;
        case MdwfFusedDslashType::D4_D5INV_D5INVDAG:
          m_scale = -0.25 / ((b * (4. + m_5) + 1.) * (b * (4. + m_5) + 1.)); // -kappa_b^2
          break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG:
          m_scale = -0.25 / ((b * (4. + m_5) + 1.) * (b * (4. + m_5) + 1.)) * b; // -kappa_b^2
          alpha = c / (2. * b); // 2 to compensate for the spin projection
          beta = 1.;
          break;
        case MdwfFusedDslashType::D5PRE:
          m_scale = b;
          alpha = c / (2. * b);
          beta = 1.;
          break;
        default: errorQuda("Unknown MdwfFusedDslashType");
        }
      }
    };

    __device__ inline int index_4d_cb_from_coordinate_4d(const int coordinate[4], const int dim[4])
    {
      return (((coordinate[3] * dim[2] + coordinate[2]) * dim[1] + coordinate[1]) * dim[0] + coordinate[0]) / 2;
    }

    __device__ inline bool is_halo_4d(const int coordinate[4], const int dim[4], const int halo_shift[4])
    {
      bool ret = false;
#pragma unroll
      for (int d = 0; d < 4; d++) {
        ret = ret or (coordinate[d] >= dim[d] - halo_shift[d] or coordinate[d] < halo_shift[d]);
      }
      return ret;
    }

    __device__ inline int index_from_extended_coordinate(const int x[4], const int dim[4], const bool comm[4], const int y)
    {
      constexpr int pad = 2;
      int back_x[4];
      int back_dim[4];

#pragma unroll
      for (int d = 0; d < 4; d++) {
        back_x[d] = comm[d] ? x[d] - pad : x[d];
        back_dim[d] = comm[d] ? dim[d] - pad * 2 : dim[d];
      }

      bool is_center = true;
#pragma unroll
      for (int d = 0; d < 4; d++) { is_center = is_center && (back_x[d] >= 0 && back_x[d] < back_dim[d]); }

      if (is_center) {
        int volume_4d_cb_back = back_dim[0] * back_dim[1] * back_dim[2] * back_dim[3] / 2;
        return y * volume_4d_cb_back
          + index_4d_cb_from_coordinate_4d(back_x, back_dim); // the input coordinate is in the center region
      } else {
        return -1;
      }
    }

    /**
    -> Everything should be understood in a 4d checkboarding sense.
    */
    template <class storage_type, bool dagger, bool halo, bool back, class Vector, class Arg>
    __device__ inline void apply_wilson_5d(Vector &out, int coordinate[4], Arg &arg, int s)
    {
      typedef typename mapper<storage_type>::type compute_type;
      typedef Matrix<complex<compute_type>, 3> Link;
      const int their_spinor_parity = arg.nParity == 2 ? 1 - arg.parity : 0;

      const int index_4d_cb = index_4d_cb_from_coordinate_4d(coordinate, arg.dim);

#pragma unroll
      for (int d = 0; d < 4; d++) // loop over dimension
      {
        int x[4] = {coordinate[0], coordinate[1], coordinate[2], coordinate[3]};
        x[d] = (coordinate[d] == arg.dim[d] - 1 && !arg.comm[d]) ? 0 : coordinate[d] + 1;
        if (!halo || !is_halo_4d(x, arg.dim, arg.halo_shift)) {
          // Forward gather - compute fwd offset for vector fetch
          int fwd_idx;
          if (back) {
            fwd_idx = index_from_extended_coordinate(x, arg.dim, arg.comm, s);
          } else {
            fwd_idx = s * arg.volume_4d_cb + index_4d_cb_from_coordinate_4d(x, arg.dim);
          }
          constexpr int proj_dir = dagger ? +1 : -1;

          const Link U = arg.U(d, index_4d_cb, arg.parity);
          const Vector in = arg.in(fwd_idx, their_spinor_parity);
          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
        x[d] = (coordinate[d] == 0 && !arg.comm[d]) ? arg.dim[d] - 1 : coordinate[d] - 1;
        if (!halo || !is_halo_4d(x, arg.dim, arg.halo_shift)) {
          // Backward gather - compute back offset for spinor and gauge fetch
          const int gauge_idx = index_4d_cb_from_coordinate_4d(x, arg.dim);

          int back_idx;
          if (back) {
            back_idx = index_from_extended_coordinate(x, arg.dim, arg.comm, s);
          } else {
            back_idx = s * arg.volume_4d_cb + gauge_idx;
          }
          constexpr int proj_dir = dagger ? -1 : +1;

          const Link U = arg.U(d, gauge_idx, 1 - arg.parity);
          const Vector in = arg.in(back_idx, their_spinor_parity);
          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      } // nDim
    }

    /**
    -> Everything should be understood in a 4d checkboarding sense.
      Given index in the shrunk block, calculate the coordinate in the shrunk block,
      then shift the coordinate to the un-shrunk coordinate, e.g. (0,0,4,1) -> (2,2,6,3) with shift = (2,2,2,2)
    */
    template <class T>
    __device__ inline void coordinate_from_shrunk_index(int coordinate[4], int shrunk_index,
                                                          const T shrunk_dim[4], const int shift[4], int parity)
    {
      int aux[4];
      aux[0] = shrunk_index * 2;

#pragma unroll
      for (int i = 0; i < 3; i++) { aux[i + 1] = aux[i] / shrunk_dim[i]; }

      coordinate[0] = aux[0] - aux[1] * shrunk_dim[0];
      coordinate[1] = aux[1] - aux[2] * shrunk_dim[1];
      coordinate[2] = aux[2] - aux[3] * shrunk_dim[2];
      coordinate[3] = aux[3];

      // Find the full coordinate in the shrunk volume.
      coordinate[0]
        += (shift[0] + shift[1] + shift[2] + shift[3] + parity + coordinate[3] + coordinate[2] + coordinate[1]) & 1;

// Now go back to the extended volume.
#pragma unroll
      for (int d = 0; d < 4; d++) { coordinate[d] += shift[d]; }
    }

    /**
      @brief Tensor core kernel for applying Wilson hopping term and then the beta + alpha * M5inv operator
      The integer kernel types corresponds to the enum MdwfFusedDslashType.
    */
    template <typename Arg> struct FusedMobiusDslash {
      Arg &arg;
      constexpr FusedMobiusDslash(Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __forceinline__ void operator()()
      {
        using storage_type = typename Arg::storage_type;
        using real = typename mapper<storage_type>::type;
        using Vector = ColorSpinor<real, 3, 4>;
        constexpr int Ls = Arg::Ls;
        const int explicit_parity = arg.nParity == 2 ? arg.parity : 0;

        SharedMemoryCache<half2> cache;

        static_assert(Arg::block_dim_x * Ls / 32 < 32, "Number of threads in a threadblock should be less than 1024.");

        constexpr int M = 4 * Ls;
        constexpr int N = 6 * Arg::block_dim_x;

        constexpr int N_sm = N + sm_n_pad_size(N);
        constexpr int M_sm = M + sm_m_pad_size(M);

        half2 *sm_b = cache.data();
        half *sm_c = reinterpret_cast<half *>(sm_b);

        half *sm_a = Arg::reload ? sm_c + M * N_sm : sm_c;
        // This is for type == 1 ONLY.
        half *sm_a_black = sm_a + M * M_sm;

        if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5PRE) {
          if (arg.small_kappa) {
            construct_matrix_a_d5<M_sm, false, Arg>(arg, sm_a); // dagger = false
          } else {
            construct_matrix_a_m5inv<M_sm, false, Arg>(arg, sm_a); // dagger = false
          }
        } else if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG) {
          if (arg.small_kappa) {
            construct_matrix_a_d5<M_sm, true, Arg>(arg, sm_a); // dagger =  true
          } else {
            construct_matrix_a_m5inv<M_sm, true, Arg>(arg, sm_a); // dagger = false
          }
        } else if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
          construct_matrix_a_m5inv<M_sm, false, Arg>(arg, sm_a); // dagger = false
        } else if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG) {
          construct_matrix_a_d5<M_sm, true, Arg>(arg, sm_a); // dagger =  true
        } else if (Arg::type == MdwfFusedDslashType::D5PRE) {
          construct_matrix_a_d5<M_sm, false, Arg>(arg, sm_a); // dagger =  true
        }
        __syncthreads();

        bool idle = false;
        int s4_shift_base = blockIdx.x * blockDim.x; // base.
        int s4_shift, sid;

        constexpr int tm_dim = M / mma_t::MMA_M;
        constexpr int tn_dim = N / mma_t::MMA_N;
        constexpr int tk_dim = M / mma_t::MMA_K;

        constexpr int total_warp = Arg::block_dim_x * Ls >> 5;
        const int this_warp = (threadIdx.y * Arg::block_dim_x + threadIdx.x) >> 5;

        constexpr int total_tile = tm_dim * tn_dim;

        constexpr int warp_cycle = total_tile / total_warp;
        const int warp_m = this_warp * warp_cycle / tn_dim;

        typename mma_t::WarpRegisterMapping wrm(threadIdx.y * blockDim.x + threadIdx.x);
        typename mma_t::OperandA op_a[Arg::reload ? 1 : tk_dim];
        typename mma_t::OperandA op_a_aux[Arg::reload ? 1 : tk_dim];
        if (!Arg::reload) { // the data in registers can be resued.
#pragma unroll
          for (int tile_k = 0; tile_k < tk_dim; tile_k++) { op_a[tile_k].template load<M_sm>(sm_a, tile_k, warp_m, wrm); }
        }

        if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
          arg.alpha = 1.;
          if (!Arg::reload) {                                                           // in the preload case we preload ...
            construct_matrix_a_m5inv<M_sm, true, Arg>(arg, sm_a); // dagger = true
            __syncthreads();

#pragma unroll
            for (int tile_k = 0; tile_k < tk_dim; tile_k++) {
              op_a_aux[tile_k].template load<M_sm>(sm_a, tile_k, warp_m, wrm);
            }

          } else {
            construct_matrix_a_m5inv<M_sm, true, Arg>(arg, sm_a_black); // dagger = true
            __syncthreads();
          }
        }

        while (s4_shift_base < arg.volume_4d_cb_shift) {
          int x[4];
          s4_shift = s4_shift_base + threadIdx.x;
          coordinate_from_shrunk_index(x, s4_shift, arg.shrunk_dim, arg.shift, arg.parity);
          sid = threadIdx.y * arg.volume_4d_cb + index_4d_cb_from_coordinate_4d(x, arg.dim);

          if (s4_shift >= arg.volume_4d_cb_shift) { idle = true; }

          Vector in_vec;
          if (!idle) {
            // the Wilson hopping terms
            if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5PRE) {
              apply_wilson_5d<storage_type, false, true, true>(in_vec, x, arg, threadIdx.y); // dagger = false; halo = true
            } else if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG) {
              apply_wilson_5d<storage_type, true, false, false>(in_vec, x, arg, threadIdx.y); // dagger =  true; halo = false
            } else if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
              apply_wilson_5d<storage_type, false, true, false>(in_vec, x, arg, threadIdx.y); // dagger = false; halo = true
            } else if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG) {
              apply_wilson_5d<storage_type, true, false, false>(in_vec, x, arg, threadIdx.y); // dagger =  true; halo = false
            } else if (Arg::type == MdwfFusedDslashType::D5PRE) {
              int sid_shift = threadIdx.y * arg.volume_4d_cb_shift + s4_shift;
              in_vec = arg.in(sid_shift, explicit_parity);
            }
            // store result to shared memory
          }
          float scale;
          load_matrix_b_vector<N_sm / 2, false>(in_vec, sm_b, scale); // acc(accumulation) = false

          __syncthreads();
          mma_sync_gemm<mma_t, Arg::block_dim_x, Arg::Ls, M, N, M_sm, N_sm, Arg::reload>(op_a, sm_a, sm_c, sm_c, wrm);
          __syncthreads();

          if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
            Vector aux_in_vec;
            int sid_back;
            bool center = false;
            if (!idle) {
              sid_back = index_from_extended_coordinate(x, arg.dim, arg.comm, threadIdx.y);
              if (sid_back >= 0) {
                center = true;
                aux_in_vec = arg.x(sid_back, explicit_parity);
              }
            }
            load_matrix_b_vector<N_sm / 2, true>(aux_in_vec, sm_b, scale, arg.m_scale); // acc = true
            if (!idle && center) { store_matrix_c<storage_type, N_sm>(arg.y, sm_b, sid_back, scale); }
            __syncthreads();
            mma_sync_gemm<mma_t, Arg::block_dim_x, Arg::Ls, M, N, M_sm, N_sm, Arg::reload>(op_a_aux, sm_a_black, sm_c,
                                                                                           sm_c, wrm);
            __syncthreads();

          } else if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG) {
            Vector aux_in_vec;
            int sid_shift = threadIdx.y * arg.volume_4d_cb_shift + s4_shift;
            if (!idle) { aux_in_vec = arg.x(sid_shift, explicit_parity); }
            load_matrix_b_vector<N_sm / 2, true, false>(aux_in_vec, sm_b, scale, arg.m_scale);
            if (!idle) { arg.out(sid_shift, explicit_parity) = aux_in_vec; }
          }

          if (Arg::type == MdwfFusedDslashType::D4DAG_D5PREDAG) {

          } else if (Arg::type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
            if (!idle) { store_matrix_c<storage_type, N_sm>(arg.out, sm_b, sid, scale); }
          } else {
            if (!idle) { store_matrix_c<storage_type, N_sm>(arg.out, sm_b, sid, scale * arg.m_scale); }
          }

          s4_shift_base += gridDim.x * blockDim.x;
        } // while
      }
    };
    
#endif // QUDA_MMA_AVAILABLE
  }

}
