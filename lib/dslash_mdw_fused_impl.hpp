#include <gauge_field.h>
#include <gauge_field_order.h>

#include <typeinfo>

#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <dslash_quda.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_mdw_fused.cuh>
#include <dslash_mdw_fused.hpp>

namespace quda
{

  namespace mobius_tensor_core {

#ifdef QUDA_MMA_AVAILABLE

    template <class store_t, int nColor, QudaReconstructType recon, int Ls_> class FusedDslash : public TunableGridStrideKernel2D
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
      static constexpr int Ls = Ls_;
      int *halo_shift;
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
            return (a_size * 2 + b_size) * sizeof(half);
          } else {
            return (a_size + b_size) * sizeof(half);
          }
        } else {
          return (a_size > b_size ? a_size : b_size) * sizeof(half);
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
        launch_cuda<FusedMobiusDslash>(tp, stream, Arg<type, Ls, block_dim_x, min_blocks, reload>
                                               (out, in, U, y, x, m_f, m_5, b_5, c_5, parity, shift, halo_shift));
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

#endif // QUDA_MMA_AVAILABLE

#if defined(GPU_DOMAIN_WALL_DIRAC) && defined(QUDA_MMA_AVAILABLE)
    template <int Ls>
    struct FusedDslashLs {
      template <class store_t, int nColor, QudaReconstructType recon>
      using type = FusedDslash<store_t, nColor, recon, Ls>;
    };
#endif
  } // namespace mobius_tensor_core
} // namespace quda
