#include <gauge_field.h>
#include <dslash.h>
<<<<<<< HEAD

#include <quda_define.h>
#if defined(QUDA_TARGET_CUDA)
#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#include <mdw_dslash5_tensor_core.cuh>
#endif
#endif // QUDA_TARGET_CUDA

namespace quda
{
  namespace mobius_tensor_core
  {

#if defined(QUDA_TARGET_CUDA)
#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)

    constexpr int sm_m_pad_size(int m)
    {
      return quda::mma::pad_size(m);
    }

    constexpr int sm_n_pad_size(int n)
    {
      return quda::mma::pad_size(n);
    }

    /**
      @brief Parameter structure for applying the Dslash
    */
    template <class storage_type_, QudaReconstructType recon_,
              int Ls_> // storage_type is the usual "Float" in other places in QUDA
    struct FusedDslashArg {
      using storage_type = storage_type_;
      using real = typename mapper<storage_type>::type; // the compute type for the in kernel computation
      static constexpr QudaReconstructType recon = recon_;
      static constexpr int Ls = Ls_;
      static constexpr bool spin_project = true;
      static constexpr bool spinor_direct_load = true; // false means texture load
#ifdef FLOAT8
      using F
        = colorspinor::FloatNOrder<storage_type, 4, 3, 8, spin_project, spinor_direct_load>; // color spin field order
#else
      using F
        = colorspinor::FloatNOrder<storage_type, 4, 3, 4, spin_project, spinor_direct_load>; // color spin field order
#endif
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

      const int dim[4];
      const int shift[4];      // sites where we actually calculate.
      const int halo_shift[4]; // halo means zero. When we are expanding we have halo of cs-field where values are zero.

      const int_fastdiv shrinked_dim[4]; // dimension after shifts are considered.

      // partial kernel and expansion parameters
      const int volume_4d_cb_shift; // number of 4d sites we need calculate
      // const int volume_4d_cb_expansive; //

      const real m_f; // fermion mass parameter
      const real m_5; // Wilson mass shift

      const bool dagger; // dagger
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

      MdwfFusedDslashType type;
      FusedDslashArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, ColorSpinorField &y,
                     const ColorSpinorField &x, double m_f_, double m_5_, const Complex *b_5, const Complex *c_5,
                     bool dagger_, int parity, int shift_[4], int halo_shift_[4], MdwfFusedDslashType type_) :
        out(out),
        in(in),
        U(U),
        y(y),
        x(x),
        nParity(in.SiteSubset()),
        parity(parity),
        volume_cb(in.VolumeCB() > out.VolumeCB() ? in.VolumeCB() : out.VolumeCB()),
        volume_4d_cb(volume_cb / Ls_),
        m_f(m_f_),
        m_5(m_5_),
        dagger(dagger_),
        shift {shift_[0], shift_[1], shift_[2], shift_[3]},
        halo_shift {halo_shift_[0], halo_shift_[1], halo_shift_[2], halo_shift_[3]},
        dim {(3 - nParity) * (in.VolumeCB() > out.VolumeCB() ? in.X(0) : out.X(0)),
             in.VolumeCB() > out.VolumeCB() ? in.X(1) : out.X(1), in.VolumeCB() > out.VolumeCB() ? in.X(2) : out.X(2),
             in.VolumeCB() > out.VolumeCB() ? in.X(3) : out.X(3)},
        shrinked_dim {dim[0] - 2 * shift[0], dim[1] - 2 * shift[1], dim[2] - 2 * shift[2], dim[3] - 2 * shift[3]},
        volume_4d_cb_shift(shrinked_dim[0] * shrinked_dim[1] * shrinked_dim[2] * shrinked_dim[3] / 2),
        type(type_),
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
      Given index in the shrinked block, calculate the coordinate in the shrinked block,
      then shift the coordinate to the un-shrinked coordinate, e.g. (0,0,4,1) -> (2,2,6,3) with shift = (2,2,2,2)
    */
    template <class T>
    __device__ inline void coordinate_from_shrinked_index(int coordinate[4], int shrinked_index,
                                                          const T shrinked_dim[4], const int shift[4], int parity)
    {
      int aux[4];
      aux[0] = shrinked_index * 2;

#pragma unroll
      for (int i = 0; i < 3; i++) { aux[i + 1] = aux[i] / shrinked_dim[i]; }

      coordinate[0] = aux[0] - aux[1] * shrinked_dim[0];
      coordinate[1] = aux[1] - aux[2] * shrinked_dim[1];
      coordinate[2] = aux[2] - aux[3] * shrinked_dim[2];
      coordinate[3] = aux[3];
=======
#include <tunable_nd.h>
#include <kernels/mdw_fused_dslash.cuh>

namespace quda
{
>>>>>>> feature/generic_kernel

  namespace mobius_tensor_core {

#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)

    template <class store_t, int nColor, QudaReconstructType recon> class FusedDslash : public TunableGridStrideKernel2D
    {
      ColorSpinorField &out;
      const ColorSpinorField &in;
      const GaugeField &U;
      ColorSpinorField &y;
      const ColorSpinorField &x;
      double m_f;
      double m_5;
      const Complex *b_5;
      const Complex *c_5;
      int parity;
      int dim[4];
      int *shift;
      int *halo_shift;
      const int Ls;
      const MdwfFusedDslashType type;
      unsigned int volume_4d_cb_active;

      /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
      static constexpr bool var_inverse = true;

      long long flops() const
      {
        auto hop = 7ll * 8ll;
        auto mat = 2ll * 4ll * Ls - 1ll;
        auto volume_4d_cb_halo_shift = (dim[0] - 2 * halo_shift[0]) * (dim[1] - 2 * halo_shift[1])
          * (dim[2] - 2 * halo_shift[2]) * (dim[3] - 2 * halo_shift[3]) / 2;

        long long flops_ = 0;
        switch (type) {
        case MdwfFusedDslashType::D4_D5INV_D5PRE:
          flops_ = volume_4d_cb_halo_shift * 6ll * 4ll * Ls * hop + volume_4d_cb_active * 24ll * Ls * mat;
          break;
        case MdwfFusedDslashType::D4_D5INV_D5INVDAG:
          flops_
            = volume_4d_cb_halo_shift * 6ll * 4ll * Ls * hop + volume_4d_cb_active * 24ll * Ls * 2ll * mat;
          break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG:
        case MdwfFusedDslashType::D4DAG_D5PREDAG:
          flops_ = volume_4d_cb_active * 6ll * 4ll * Ls * (hop + mat); // for 2 and 3 we don't have the halo complication.
          break;
        case MdwfFusedDslashType::D5PRE: flops_ = volume_4d_cb_active * 6ll * 4ll * Ls * (mat); break;
        default: errorQuda("Unknown MdwfFusedDslashType %d", (int)type);
        }

        return flops_;
      }

      long long bytes() const
      {
        auto site_size = Ls * (2ll * in.Nspin() * nColor * in.Precision() + sizeof(float));
        auto b_m0 = ((dim[0] - 0) * (dim[1] - 0) * (dim[2] - 0) * (dim[3] - 0) / 2) * site_size;
        auto b_m1 = ((dim[0] - 1) * (dim[1] - 1) * (dim[2] - 1) * (dim[3] - 1) / 2) * site_size;
        auto b_m2 = ((dim[0] - 2) * (dim[1] - 2) * (dim[2] - 2) * (dim[3] - 2) / 2) * site_size;
        switch (type) {
        case MdwfFusedDslashType::D4_D5INV_D5PRE: return b_m1 + b_m2 + U.Bytes();
        case MdwfFusedDslashType::D4_D5INV_D5INVDAG: return 2 * b_m2 + b_m1 + b_m0 + U.Bytes();
        case MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG: return b_m1 + b_m0 + U.Bytes();
        case MdwfFusedDslashType::D4DAG_D5PREDAG: return 2 * b_m2 + b_m1 + U.Bytes();
        case MdwfFusedDslashType::D5PRE: return 2 * b_m0;
        default: errorQuda("Unknown MdwfFusedDslashType %d", (int)type);
        }
        return 0ll;
      }

      bool tuneAuxDim() const { return true; }

      int blockStep() const { return 16; }
      int blockMin() const { return 16; }
      unsigned int maxBlockSize(const TuneParam &) const { return 32; }

      int gridStep() const { return device::processor_count(); }
      unsigned int maxGridSize() const { return (volume_4d_cb_active + blockMin() - 1) / blockMin(); }
      unsigned int minGridSize() const { return device::processor_count(); }

      unsigned int sharedBytesPerBlock(const TuneParam &param) const
      {
        const int a_size = (param.block.y * 4) * (param.block.y * 4 + sm_m_pad_size(param.block.y * 4));
        const int b_size = (param.block.y * 4) * (param.block.x * 6 + sm_n_pad_size(param.block.x * 6));
        // (Ls*4) by (Ls*4), (Ls*4) by (volume_4d*6 + 16)
        if (param.aux.x == 1) { // aux.x == 1 --> reload == true
          if (type == MdwfFusedDslashType::D4_D5INV_D5INVDAG) {
            return (a_size * 2 + b_size) * sizeof(half) + 128;
          } else {
            return (a_size + b_size) * sizeof(half) + 128;
          }
        } else {
          return (a_size > b_size ? a_size : b_size) * sizeof(half) + 128;
        }
      }

      bool advanceAux(TuneParam &param) const
      {
        bool aux_advanced = false;
        if (param.aux.x == 0) { // first see if aux.x(ONLY 0(false) or 1(true))
          param.aux.x++;
          aux_advanced = true;
        } else {
          if (param.aux.y < 3) { // second see if aux.y
            param.aux.y++;
            aux_advanced = true;
            param.aux.x = 0;
          }
        }
        // shared bytes depends on aux, so update if changed
        if (aux_advanced) param.shared_bytes = sharedBytesPerBlock(param);
        return aux_advanced;
      }

      // overloaded to return max dynamic shared memory if doing shared-memory inverse
      unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }

    public:
      FusedDslash(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, ColorSpinorField &y,
                  const ColorSpinorField &x, double m_f, double m_5, const Complex *b_5, const Complex *c_5,
                  bool dagger, int parity, int shift[4], int halo_shift[4], MdwfFusedDslashType type) :
        TunableGridStrideKernel2D(in, x.X(4)),
        out(out),
        in(in),
        U(U),
        y(y),
        x(x),
        m_f(m_f),
        m_5(m_5),
        b_5(b_5),
        c_5(c_5),
        parity(parity),
        shift(shift),
        halo_shift(halo_shift),
        Ls(in.X(4)),
        type(type)
      {
        resizeStep(in.X(4)); // Ls must be contained in the block
        for (int i = 0; i < 4; i++) dim[i] = in.Volume() > out.Volume() ? in.X(i) : out.X(i);
        dim[0] *= 3 - in.SiteSubset(); // make full lattice dim
        volume_4d_cb_active = (dim[0] - 2 * shift[0]) * (dim[1] - 2 * shift[1]) * (dim[2] - 2 * shift[2]) * (dim[3] - 2 * shift[3]) / 2;

        if (dagger) strcat(aux, ",Dagger");
        char config[512];
        switch (type) {
        case MdwfFusedDslashType::D4_D5INV_D5PRE: sprintf(config, ",f0"); break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG: sprintf(config, ",f2"); break;
        case MdwfFusedDslashType::D4_D5INV_D5INVDAG: sprintf(config, ",f1"); break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG: sprintf(config, ",f3"); break;
        case MdwfFusedDslashType::D5PRE: sprintf(config, ",f4"); break;
        default: errorQuda("Unknown MdwfFusedDslashType %d", (int)type);
        }
        strcat(aux, config);
        sprintf(config, "shift%d%d%d%d,halo%d%d%d%d", shift[0], shift[1], shift[2], shift[3],
                halo_shift[0], halo_shift[1], halo_shift[2], halo_shift[3]);
        strcat(aux, config);
        strcat(aux, comm_dim_partitioned_string());

        apply(device::get_default_stream());
      }

      template <MdwfFusedDslashType type, int Ls, int block_dim_x, int min_blocks, bool reload> using Arg =
        FusedDslashArg<store_t, nColor, recon, Ls, type, block_dim_x, min_blocks, reload>;

      // The following apply<...> functions are used to turn the tune parameters into template arguments.
      // Specifically tp.aux.y dictates the minBlocksPerMultiprocessor in __launch_bounds__(..).
      // tp.aux.x dictates whether or not to reload.
      template <int block_dim_x, int min_blocks, bool reload, MdwfFusedDslashType type>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        switch (Ls) {
        case 4: launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, 4, block_dim_x, min_blocks, reload>
                                               (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift)); break;
        case 8: launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, 8, block_dim_x, min_blocks, reload>
                                               (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift)); break;
        case 12: launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, 12, block_dim_x, min_blocks, reload>
                                                (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift)); break;
        case 16: launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, 16, block_dim_x, min_blocks, reload>
                                                (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift)); break;
        case 20: launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, 20, block_dim_x, min_blocks, reload>
                                                (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift)); break;
        default: errorQuda("Ls = %d not instantiated\n", Ls);
        }          
      }

      template <int block_dim_x, bool reload, MdwfFusedDslashType type>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        switch (tp.aux.y) {
        case 1: apply<block_dim_x, 1, reload, type>(tp, stream); break;
        case 2: apply<block_dim_x, 2, reload, type>(tp, stream); break;
        case 3: apply<block_dim_x, 3, reload, type>(tp, stream); break;
        default: errorQuda("NOT valid tp.aux.y(=%d)\n", tp.aux.y);
        }
      }

      template <MdwfFusedDslashType type> void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        switch (tp.block.x) {
        case 16: tp.aux.x ? apply<16, true, type>(tp, stream) : apply<16, false, type>(tp, stream); break;
        case 32: tp.aux.x ? apply<32, true, type>(tp, stream) : apply<32, false, type>(tp, stream); break;
        default: errorQuda("Invalid tp.block.x(=%d)\n", tp.block.x);
        }
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true;
        switch (type) {
        case MdwfFusedDslashType::D4_D5INV_D5PRE:
          apply<MdwfFusedDslashType::D4_D5INV_D5PRE>(tp, stream); break;
        case MdwfFusedDslashType::D4_D5INV_D5INVDAG:
          apply<MdwfFusedDslashType::D4_D5INV_D5INVDAG>(tp, stream); break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG:
          apply<MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG>(tp, stream); break;
        case MdwfFusedDslashType::D4DAG_D5PREDAG:
          apply<MdwfFusedDslashType::D4DAG_D5PREDAG>(tp, stream); break;
        case MdwfFusedDslashType::D5PRE:
          apply<MdwfFusedDslashType::D5PRE>(tp, stream); break;
        default: errorQuda("Unknown MdwfFusedDslashType %d", (int)type);
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        TunableGridStrideKernel2D::initTuneParam(param);
        param.aux.x = 0;
        param.aux.y = 1;
      }

      void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }
    };

#endif // #if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
#endif // QUDA_TARGET_CUDA

#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
    void apply_fused_dslash(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, ColorSpinorField &y,
                            const ColorSpinorField &x, double m_f, double m_5, const Complex *b_5, const Complex *c_5,
                            bool dagger, int parity, int shift[4], int halo_shift[4], MdwfFusedDslashType type)
    {
<<<<<<< HEAD
#if defined(QUDA_TARGET_CUDA)
#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)
=======
>>>>>>> feature/generic_kernel
      checkLocation(out, in); // check all locations match
      instantiatePreconditioner<FusedDslash>(out, in, U, y, x, m_f, m_5, b_5, c_5, dagger, parity, shift, halo_shift,
                                             type);
    }
#else
    void apply_fused_dslash(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, ColorSpinorField &,
                            const ColorSpinorField &, double, double, const Complex *, const Complex *,
                            bool, int, int [4], int [4], MdwfFusedDslashType)
    {
      errorQuda("Domain wall dslash with tensor cores has not been built");
<<<<<<< HEAD
#endif
#else
	errorQuda("Domain wall dslash with tensor cores can only be built for CUDA TARGET");
#endif // QUDA_TARGET_CUDA

=======
>>>>>>> feature/generic_kernel
    }
#endif
  } // namespace mobius_tensor_core
} // namespace quda
