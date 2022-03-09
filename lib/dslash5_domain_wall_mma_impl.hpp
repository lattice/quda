#include <typeinfo>

#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <dslash_quda.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_domain_wall_m5_mma.cuh>

namespace quda
{

#if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  template <class store_t, int nColor, int Ls> class Dslash5Mma : public TunableKernel3D
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &x;
    double m_f;
    double m_5;
    const Complex *b_5;
    const Complex *c_5;
    double a; // for xpay
    int dagger;
    int xpay;
    int volume_4d_cb;

    using real = typename mapper<store_t>::type; // the compute type for the in kernel computation

    /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
    static constexpr bool var_inverse = true;

    long long flops() const
    {
      long long n = in.Ncolor() * in.Nspin();
      return (2 + 8 * n) * Ls * in.Volume();
    }

    long long bytes() const
    {
      return in.Bytes() + out.Bytes();
    }

    bool tuneAuxDim() const { return true; }
    bool tuneGridDim() const { return true; }

    int blockStep() const { return 16; }
    int blockMin() const { return 16; }
    unsigned int maxBlockSize(const TuneParam &) const { return 32; }

    int gridStep() const { return device::processor_count(); }
    unsigned int maxGridSize() const { return (volume_4d_cb + blockMin() - 1) / blockMin(); }
    unsigned int minGridSize() const { return device::processor_count(); }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const
    {
      using Mma = typename mma_mapper<store_t>::type;
      const int a_size = (param.block.y * 4) * (param.block.y * 4 + Mma::t_pad);
      const int b_size = (param.block.y * 4) * (param.block.x * 6 + Mma::t_pad);
      const int c_size = (param.block.y * 4) * (param.block.x * 6 + Mma::acc_pad);
      if (param.aux.x == 1) { // aux.x == 1 --> reload == true
        return (a_size + b_size + c_size) * sizeof(real);
      } else {
        return (a_size > b_size + c_size ? a_size : b_size + c_size) * sizeof(real);
      }
    }

    bool advanceAux(TuneParam &param) const
    {
      bool aux_advanced = false;
      if (param.aux.x == 0) { // first see if aux.x (reload 0 (false) or 1 (true))
        param.aux.x++;
        aux_advanced = true;
      } else {
#if 0
        if (param.aux.y < 3) { // second see if aux.y
          param.aux.y++;
          aux_advanced = true;
          param.aux.x = 1;
        }
#endif
      }
      // shared bytes depends on aux, so update if changed
      if (aux_advanced) param.shared_bytes = sharedBytesPerBlock(param);
      return aux_advanced;
    }

    unsigned int minThreads() const { return 1; } // We are actually doing grid dim tuning

    // overloaded to return max dynamic shared memory if doing shared-memory inverse
    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }

    public:
    Dslash5Mma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
        double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger) :
      TunableKernel3D(in, x.X(4), in.SiteSubset()),
      out(out),
      in(in),
      x(x),
      m_f(m_f),
      m_5(m_5),
      b_5(b_5),
      c_5(c_5),
      a(a),
      dagger(dagger),
      xpay(a == 0.0 ? false : true),
      volume_4d_cb(in.VolumeCB() / in.X(4))
    {
      if (Ls != out.X(4) || Ls != in.X(4)) {
        errorQuda("Ls (=%d) mismatch: out.X(4) = %d, in.X(4) = %d", Ls, out.X(4), in.X(4));
      }
      resizeBlock(in.X(4), 1); // Ls must be contained in the block
      resizeStep(in.X(4), 1);
      if (dagger) strcat(aux, ",Dagger");
      if (xpay) strcat(aux, ",xpay");
      apply(device::get_default_stream());
    }

    template <int block_dim_x, bool dagger, bool xpay> using Arg =
      Dslash5MmaArg<store_t, nColor, Ls, block_dim_x, dagger, xpay>;

    template <int block_dim_x, bool dagger, bool xpay>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        Arg<block_dim_x, dagger, xpay> arg(out, in, x, m_f, m_5, b_5, c_5, a);
        arg.round_up_threads_x(tp.block.x);
        launch<Dslash5MmaKernel>(tp, stream, arg);
      }

    template <int block_dim_x, bool dagger>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        if (xpay) {
          apply<block_dim_x, dagger, true>(tp, stream);
        } else {
          apply<block_dim_x, dagger, false>(tp, stream);
        }
      }

    template <int block_dim_x>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        if (dagger) {
          apply<block_dim_x, true>(tp, stream);
        } else {
          apply<block_dim_x, false>(tp, stream);
        }
      }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      tp.set_max_shared_bytes = true;
      switch (tp.block.x) {
        case 16: apply<16>(tp, stream); break;
        case 32: apply<32>(tp, stream); break;
        default: errorQuda("Invalid tp.block.x(=%d)\n", tp.block.x);
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableKernel3D::initTuneParam(param);
      param.aux.x = 0;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }
  };

#endif // #if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5*in
#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)
    template <int Ls>
    struct Dslash5MmaLs {
      template <class store_t, int nColor>
      using type = Dslash5Mma<store_t, nColor, Ls>;
    };
#endif

} // namespace quda
