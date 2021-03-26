#include <gauge_field.h>
#include <color_spinor_field.h>
#include <constant_kernel_arg.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_domain_wall_4d_fused_m5.cuh>

/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{

  template <typename Arg> class DomainWall4DFusedM5 : public Dslash<domainWall4DFusedM5, Arg>
  {
    using Dslash = Dslash<domainWall4DFusedM5, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall4DFusedM5(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      TunableVectorYZ::resizeVector(in.X(4), arg.nParity);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      typedef typename mapper<typename Arg::Float>::type real;
#ifdef JITIFY
      // we need to break the dslash launch abstraction here to get a handle on the constant memory pointer in the kernel module
      auto instance = Dslash::template kernel_instance<packShmem>();
      Tunable::jitify_error = instance.configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      Dslash::template instantiate<packShmem>(tp, stream);
#endif
    }

    unsigned int sharedBytesPerThread() const
    {
      // spin components in shared depend on inversion algorithm
      return 2 * Arg::nSpin * Arg::nColor * sizeof(typename mapper<typename Arg::Float>::type);
    }

    void initTuneParam(TuneParam &param) const
    {
      Dslash::initTuneParam(param);

      param.block.y = arg.Ls; // Ls must be contained in the block
      param.grid.y = 1;
      param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApplyFusedM5 {

    template <Dslash5Type...> struct Dslash5TypeList {
    };

    inline void instantiate_dslash5_type(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double ,
                                         double, const Complex *, const Complex *, const ColorSpinorField &,
                                         ColorSpinorField &, int, bool, const int *,
                                         double, Dslash5Type dslash5_type, Dslash5TypeList<>, TimeProfile &)
    {
      errorQuda("Unsupported dslash5_type of %d", (int)dslash5_type);
    }

    template <Dslash5Type dslash5_type_impl, Dslash5Type... N>
    inline void instantiate_dslash5_type(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                         double a, double m_5, const Complex *b_5, const Complex *c_5,
                                         const ColorSpinorField &x, ColorSpinorField &y, int parity, bool dagger,
                                         const int *comm_override, double m_f, Dslash5Type dslash5_type,
                                         Dslash5TypeList<dslash5_type_impl, N...>, TimeProfile &profile)
    {
      if (dslash5_type == dslash5_type_impl) {
        constexpr int nDim = 4;
        using Arg = DomainWall4DFusedM5Arg<Float, nColor, nDim, recon, dslash5_type_impl>;
        static_assert(sizeof(Arg) <= 4096, "arg size too large");
        Arg arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, y, parity, dagger, comm_override, m_f);
        DomainWall4DFusedM5<Arg> dwf(arg, out, in);

        dslash::DslashPolicyTune<decltype(dwf)> policy(
          dwf, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
      } else {
        instantiate_dslash5_type(out, in, U, a, m_5, b_5, c_5, x, y, parity, dagger, comm_override, m_f, dslash5_type,
                                 Dslash5TypeList<N...>(), profile);
      }
    }

    inline DomainWall4DApplyFusedM5(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                                    double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x,
                                    ColorSpinorField &y, int parity, bool dagger, const int *comm_override, double m_f,
                                    Dslash5Type dslash5_type, TimeProfile &profile)
    {
      auto dummy_list
        = Dslash5TypeList<Dslash5Type::M5_INV_MOBIUS_M5_PRE, Dslash5Type::M5_PRE_MOBIUS_M5_INV, Dslash5Type::M5_INV_MOBIUS,
            Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG, Dslash5Type::DSLASH5_MOBIUS_PRE>();
      instantiate_dslash5_type(out, in, U, a, m_5, b_5, c_5, x, y, parity, dagger, comm_override, m_f, dslash5_type,
                               dummy_list, profile);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall4DFusedM5(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                                double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x,
                                ColorSpinorField &y, int parity, bool dagger, const int *comm_override, double m_f,
                                Dslash5Type dslash5_type, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    instantiate<DomainWall4DApplyFusedM5>(out, in, U, a, m_5, b_5, c_5, x, y, parity, dagger, comm_override, m_f,
                                          dslash5_type, profile);
#else
    errorQuda("Domain-wall dslash has not been built");
#endif // GPU_DOMAIN_WALL_DIRAC
  }

} // namespace quda
