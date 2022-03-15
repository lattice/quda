#include <typeinfo>

#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <dslash_quda.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_domain_wall_m5_mma.cuh>

namespace quda
{

#if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  constexpr int m5_mma_reload() {
    return false;
  }

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

    bool tuneGridDim() const { return true; }

    int blockStep() const { return 16; }
    int blockMin() const { return 16; }
    unsigned int maxBlockSize(const TuneParam &) const { return 32; }

    int gridStep() const { return device::processor_count(); }
    unsigned int maxGridSize() const { return (volume_4d_cb + blockMin() - 1) / blockMin(); }
    unsigned int minGridSize() const { return device::processor_count(); }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const
    {
      return mma_shared_bytes<store_t, m5_mma_reload()>(param.block.x, param.block.y);
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
      Dslash5MmaArg<store_t, nColor, Ls, block_dim_x, dagger, xpay, m5_mma_reload()>;

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
