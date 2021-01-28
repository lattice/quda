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

  template <class First, class Second>
  struct TunePair: Tunable
  {
    First &first;
    Second &second;

    char vol_string[256];
    char aux[TuneKey::aux_n];

    long long flops_;
    long long bytes_;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    TunePair(First &first, Second &second, const char vol_string_[], const char aux_[], long long flops, long long bytes) : first(first), second(second), flops_(flops), bytes_(bytes){
      strcpy(aux, aux_);
      strcpy(vol_string, vol_string_);
      if (!tuned()) {
        disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

        // before we do policy tuning we must ensure the kernel
        // constituents have been tuned since we can't do nested tuning
        first({0});
        second({0});

        enableProfileCount();
        setPolicyTuning(true);
      }
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (tp.aux.x == 0) {
        // first(stream);
        second(stream);
      } else {
        second(stream);
      }
    }

    bool advanceAux(TuneParam &param) const
    {
      if (param.aux.x == 0) {
        param.aux.x = 1;
        return true;
      } else {
        param.aux.x = 0;
        return false;
      }
    }

    bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

    void initTuneParam(TuneParam &param) const
    {
      param.aux.x = 0;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const { return TuneKey(vol_string, typeid(*this).name(), aux); }

    long long flops() const { return flops_; }
    long long bytes() const { return bytes_; }

    void preTune() {}  // FIXME - use write to determine what needs to be saved
    void postTune() {} // FIXME - use write to determine what needs to be saved
  };

  /**
    Having a policy tuning for whether or not using the MMA to perform the stencil
  */
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

    void apply(const qudaStream_t &stream)
    {
      static bool tuning = false;
      if (arg.kernel_type == INTERIOR_KERNEL && !tuning) {

        tuning = true;

        auto strategy = [this] (const qudaStream_t &stream_) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          Dslash::setParam(tp);
          Dslash::template instantiate<packShmem>(tp, stream_);
        };

        auto specialized_strategy = [this] (const qudaStream_t &stream_) {
          TuneParam tp;
          domainWall4D<1, false, false, INTERIOR_KERNEL, Arg>::specialized_launch(tp, stream_, arg); // nPairty = 1, dagger = false, xpay = false
        };

        char aux[TuneKey::aux_n];
        char comm[5];
        comm[0] = (arg.commDim[0] ? '1' : '0');
        comm[1] = (arg.commDim[1] ? '1' : '0');
        comm[2] = (arg.commDim[2] ? '1' : '0');
        comm[3] = (arg.commDim[3] ? '1' : '0');
        comm[4] = '\0';
        strcpy(aux, "policy_kernel=interior,commDim=");
        strcat(aux, comm);
        if (arg.xpay) strcat(aux, ",xpay");
        if (arg.dagger) strcat(aux, ",dagger");

        TunePair<decltype(strategy), decltype(specialized_strategy)> pair(strategy, specialized_strategy, in.VolString(), aux, this->flops(), this->bytes());
        pair.apply(stream);

        tuning = false;

      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash::setParam(tp);
        Dslash::template instantiate<packShmem>(tp, stream);
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
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
#ifdef GPU_DOMAIN_WALL_DIRAC
  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                         const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    instantiate<DomainWall4DApply>(out, in, U, a, m_5, b_5, c_5, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyDomainWall4D(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, double,
                         const Complex *, const Complex *, const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Domain-wall dslash has not been built");
  }
#endif // GPU_DOMAIN_WALL_DIRAC

} // namespace quda
