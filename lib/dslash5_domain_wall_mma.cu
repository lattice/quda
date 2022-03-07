#include <typeinfo>

#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <dslash_quda.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_domain_wall_m5_mma.cuh>

namespace quda
{

#if (CUDA_VERSION >= 10010 && __COMPUTE_CAPABILITY__ >= 700)

  template <class store_t, int nColor, int Ls> class Dslash5Mma : public TunableGridStrideKernel2D
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &x;
    double m_f;
    double m_5;
    const Complex *b_5;
    const Complex *c_5;
    double a; // for xpay
    int parity;
    int dagger;
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
        if (param.aux.y < 3) { // second see if aux.y
          param.aux.y++;
          aux_advanced = true;
          param.aux.x = 1;
        }
      }
      // shared bytes depends on aux, so update if changed
      if (aux_advanced) param.shared_bytes = sharedBytesPerBlock(param);
      return aux_advanced;
    }

    // overloaded to return max dynamic shared memory if doing shared-memory inverse
    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }

    public:
    Dslash5Mma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
        double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger) :
      TunableGridStrideKernel2D(in, x.X(4)),
      out(out),
      in(in),
      x(x),
      m_f(m_f),
      m_5(m_5),
      b_5(b_5),
      c_5(c_5),
      a(a),
      parity(parity),
      dagger(dagger),
      volume_4d_cb(in.VolumeCB() / in.X(4)),
    {
      if (Ls != out.X(4) || Ls != in.X(4)) {
        errorQuda("Ls (=%d) mismatch: out.X(4) = %d, in.X(4) = %d", Ls, out.X(4), in.X(4));
      }
      resizeStep(in.X(4)); // Ls must be contained in the block
      if (dagger) strcat(aux, ",Dagger");
      char config[512];
      apply(device::get_default_stream());
    }

    template <int block_dim_x, bool dagger, int min_blocks, bool reload> using Arg =
      Dslash5MmaArg<store_t, nColor, Ls, block_dim_x, dagger, min_blocks, reload>;

    template <int block_dim_x, int min_blocks, bool reload, bool dagger>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        launch_cuda<Dslash5MmaKernel>(tp, stream, Arg<Ls, block_dim_x, dagger, min_blocks, reload>
        (out, in, x, m_f, m_5, b_5, c_5, a, parity));
      }

    template <int block_dim_x, bool reload, bool dagger>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        switch (tp.aux.y) {
          case 1: apply<block_dim_x, 1, reload, dagger>(tp, stream); break;
          case 2: apply<block_dim_x, 2, reload, dagger>(tp, stream); break;
          case 3: apply<block_dim_x, 3, reload, dagger>(tp, stream); break;
          default: errorQuda("NOT valid tp.aux.y(=%d)\n", tp.aux.y);
        }
      }

    template <int block_dim_x, bool reload>
      void apply(const TuneParam &tp, const qudaStream_t &stream)
      {
        if (dagger) {
          apply<block_dim_x, reload, true>(tp, stream);
        } else {
          apply<block_dim_x, reload, false>(tp, stream);
        }
      }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      tp.set_max_shared_bytes = true;
      switch (tp.block.x) {
        case 16: tp.aux.x ? apply<16, true>(tp, stream) : apply<16, false>(tp, stream); break;
        case 32: tp.aux.x ? apply<32, true>(tp, stream) : apply<32, false>(tp, stream); break;
        default: errorQuda("Invalid tp.block.x(=%d)\n", tp.block.x);
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

#endif // #if (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5*in
#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)
  void ApplyDslash5Mma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
    if (in.PCType() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
    checkLocation(out, in, x); // check all locations match
    instantiate<Dslash5Mma>(out, in, x, m_f, m_5, b_5, c_5, a, dagger);
  }
#else
  void ApplyDslash5Mma(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &, double,
      double, const Complex *, const Complex *, double, bool, Dslash5Type)
  {
    errorQuda("Domain wall dslash has not been built");
  }
#endif

} // namespace quda
