#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_domain_wall_4d.cuh>

/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{

  template <class Base>
  struct TuneY: Tunable
  {
    Base &base;
    const int Ls;
    static constexpr int s_batch_max = 2;

    char aux[TuneKey::aux_n];
    char vol_string[TuneKey::aux_n];

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    TuneY(Base &base, int Ls) : base(base), Ls(Ls) {
      strcpy(aux, "tune_y");
      if (!tuned()) {
        disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

        // before we do policy tuning we must ensure the kernel
        // constituents have been tuned since we can't do nested tuning
        for (int s_batch = 1; s_batch <= s_batch_max; s_batch++) {
          if (Ls % s_batch == 0) {
            base.TunableVectorY::resizeVector(Ls / s_batch);
            base.apply(0);
          }
        }

        enableProfileCount();
        setPolicyTuning(true);
      }
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      base.TunableVectorY::resizeVector(Ls / tp.aux.x);
      base.apply(stream);
    }

    bool advanceAux(TuneParam &param) const
    {
      do {
        param.aux.x++;
      } while(Ls % param.aux.x != 0 && param.aux.x <= s_batch_max);
      if (param.aux.x <= s_batch_max) {
        return true;
      } else {
        param.aux.x = 1;
        return false;
      }
    }

    bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

    void initTuneParam(TuneParam &param) const
    {
      param.aux.x = 1;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const {
      return TuneKey(base.tuneKey().volume, typeid(*this).name(), aux);
    }

    long long flops() const { return base.flops(); }
    long long bytes() const { return base.bytes(); }

    void preTune() {}  // FIXME - use write to determine what needs to be saved
    void postTune() {} // FIXME - use write to determine what needs to be saved
  };

  template <typename Arg> class DomainWall4D : public Dslash<domainWall4D, Arg>
  {
    using Dslash = Dslash<domainWall4D, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall4D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      TunableVectorYZ::resizeVector(in.X(4), arg.nParity);
    }

    void launch_s_batch(TuneParam &tp, const qudaStream_t &stream)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      if (arg.dc.Ls % TunableVectorY::get_vector_y() != 0) { errorQuda("arg.dc.Ls %% get_vector_y() != 0"); }
      int s_batch = arg.dc.Ls / TunableVectorY::get_vector_y();
#ifdef JITIFY
      // we need to break the dslash launch abstraction here to get a handle on the constant memory pointer in the kernel module
      auto instance = Dslash::template kernel_instance<packShmem>();
      cuMemcpyHtoDAsync(instance.get_constant_ptr("quda::mobius_d"), arg.a_5, QUDA_MAX_DWF_LS * sizeof(complex<real>),
                        stream);
      Tunable::jitify_error = instance.configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      cudaMemcpyToSymbolAsync(mobius_d, arg.a_5, QUDA_MAX_DWF_LS * sizeof(complex<real>), 0, cudaMemcpyHostToDevice,
                              streams[Nstream - 1]);
      switch(s_batch) {
      case 1: Dslash::template instantiate<packShmem, 1>(tp, stream); break;
      case 2: Dslash::template instantiate<packShmem, 2>(tp, stream); break;
      default: errorQuda("Unsupported s_batch = %d", s_batch);
      }
#endif
    }

    void apply(const qudaStream_t &stream)
    {
      static bool tuning = false;
      if (!tuning) {
        tuning = true;
        TuneY<decltype(*this)> tune_y(*this, arg.dc.Ls);
        tune_y.apply(stream);
        tuning = false;
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash::setParam(tp);
        launch_s_batch(tp, stream);
      }
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApply {

    inline DomainWall4DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                             double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity,
                             bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      DomainWall4DArg<Float, nColor, nDim, recon> arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, parity, dagger,
                                                      comm_override);
      DomainWall4D<decltype(arg)> dwf(arg, out, in);

      dslash::DslashPolicyTune<decltype(dwf)> policy(
        dwf, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
        in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
      policy.apply(0);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                         const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    instantiate<DomainWall4DApply>(out, in, U, a, m_5, b_5, c_5, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Domain-wall dslash has not been built");
#endif // GPU_DOMAIN_WALL_DIRAC
  }

} // namespace quda
