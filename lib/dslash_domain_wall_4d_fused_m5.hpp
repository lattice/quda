#include <gauge_field.h>
#include <color_spinor_field.h>
#include <worker.h>
#include <kernels/dslash_domain_wall_4d_fused_m5.cuh>
#include <dslash_policy.hpp>
#include <dslash.h>

/**
   This is the templated gauged domain-wall 4-d preconditioned operator, but fused with immediately followed fifth
   dimension operators.
*/

namespace quda
{

  template <typename Arg> class DomainWall4DFusedM5 : public Dslash<domainWall4DFusedM5, Arg>
  {
    using Dslash = Dslash<domainWall4DFusedM5, Arg>;
    using Dslash::arg;
    using Dslash::aux_base;
    using Dslash::in;
    cvector_ref<ColorSpinorField> &y;

    inline std::string get_app_base()
    {
      switch (Arg::dslash5_type) {
      case Dslash5Type::DSLASH5_MOBIUS_PRE: return ",fused_type=m5pre";
      case Dslash5Type::DSLASH5_MOBIUS: return ",fused_type=m5mob";
      case Dslash5Type::M5_INV_MOBIUS: return ",fused_type=m5inv";
      case Dslash5Type::M5_INV_MOBIUS_M5_PRE: return ",fused_type=m5inv_m5pre";
      case Dslash5Type::M5_PRE_MOBIUS_M5_INV: return ",fused_type=m5pre_m5inv";
      case Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG: return ",fused_type=m5inv_m5inv";
      case Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB: return ",fused_type=m5pre_m5mob";
      default:
        errorQuda("Unexpected Dslash5Type %d", static_cast<int>(Arg::dslash5_type));
        return ",fused_type=unknown_type";
      }
    }

    int blockStep() const override { return 8; }
    int blockMin() const override { return 8; }

  public:
    DomainWall4DFusedM5(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                        const ColorSpinorField &halo, cvector_ref<ColorSpinorField> &y) :
      Dslash(arg, out, in, halo, get_app_base()), y(y)
    {
      TunableKernel3D::resizeStep(in.X(4), 1); // keep Ls local to the thread block
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }

    unsigned int sharedBytesPerThread() const override
    {
      // spin components in shared depend on inversion algorithm
      if (mobius_m5::use_half_vector()) {
        return 2 * (Arg::nSpin / 2) * Arg::nColor * sizeof(typename mapper<typename Arg::Float>::type);
      } else {
        return 2 * Arg::nSpin * Arg::nColor * sizeof(typename mapper<typename Arg::Float>::type);
      }
    }

    long long m5pre_flops() const
    {
      long long Ls = in.X(4);
      long long bulk = (Ls - 2) * (in.Volume() / Ls);
      long long wall = 2 * in.Volume() / Ls;
      long long n = in.Ncolor() * in.Nspin();
      return n * (8ll * bulk + 10ll * wall + 6ll * in.Volume());
    }

    long long m5mob_flops() const
    {
      long long Ls = in.X(4);
      long long bulk = (Ls - 2) * (in.Volume() / Ls);
      long long wall = 2 * in.Volume() / Ls;
      long long n = in.Ncolor() * in.Nspin();
      return n * (8ll * bulk + 10ll * wall + 4ll * in.Volume());
    }

    long long m5inv_flops() const
    {
      long long Ls = in.X(4);
      long long n = in.Ncolor() * in.Nspin();
      return (12ll * n * Ls) * in.Volume();
    }

    long long flops() const override
    {
      long long flops_ = 0;
      switch (Arg::dslash5_type) {
      case Dslash5Type::DSLASH5_MOBIUS_PRE: flops_ = m5pre_flops(); break;
      case Dslash5Type::DSLASH5_MOBIUS: flops_ = m5mob_flops(); break;
      case Dslash5Type::M5_INV_MOBIUS: flops_ = m5inv_flops(); break;
      case Dslash5Type::M5_INV_MOBIUS_M5_PRE:
      case Dslash5Type::M5_PRE_MOBIUS_M5_INV: flops_ = m5inv_flops() + m5pre_flops(); break;
      case Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG: flops_ = m5inv_flops() + m5inv_flops(); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB: flops_ = m5pre_flops() + m5mob_flops(); break;
      default: errorQuda("Unexpected Dslash5Type %d", static_cast<int>(Arg::dslash5_type));
      }

      return in.size() * flops_ + Dslash::flops();
    }

    long long bytes() const override
    {
      if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG) {
        return y.Bytes() + Dslash::bytes();
      } else {
        return Dslash::bytes();
      }
    }
  };

  template <Dslash5Type...> struct Dslash5TypeList {
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApplyFusedM5 {

    template <Dslash5Type dslash5_type_impl, Dslash5Type... N>
    DomainWall4DApplyFusedM5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             cvector_ref<const ColorSpinorField> &x, const GaugeField &U, cvector_ref<ColorSpinorField> &y,
                             const Complex *b_5, const Complex *c_5, double a, double m_5, int parity,
                             bool dagger, const int *comm_override, double m_f,
                             Dslash5TypeList<dslash5_type_impl, N...>, TimeProfile &profile)
    {
#ifdef NVSHMEM_COMMS
      errorQuda("Fused Mobius/DWF-4D kernels do not currently work with NVSHMEM.");
#else
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      using Arg = DomainWall4DFusedM5Arg<Float, nColor, nDim, recon, dslash5_type_impl>;
      Arg arg(out, in, halo, U, a, m_5, b_5, c_5, a != 0.0, x, y, parity, dagger, comm_override, m_f);
      DomainWall4DFusedM5<Arg> dwf(arg, out, in, halo, y);
      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, halo, profile);
#endif
    }
  };

  // use custom instantiate to deal with field splitting if needed
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  void instantiate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, const GaugeField &U, Args ...args)
  {
    if (in.size() > get_max_multi_rhs()) {
      instantiate<Apply, Recon>({out.begin(), out.begin() + out.size() / 2},
                                {in.begin(), in.begin() + in.size() / 2},
                                {x.begin(), x.begin() + x.size() / 2},
                                {y.begin(), y.begin() + y.size() / 2},
                                U, args...);
      instantiate<Apply, Recon>({out.begin() + out.size() / 2, out.end()},
                                {in.begin() + in.size() / 2, in.end()},
                                {x.begin() + x.size() / 2, x.end()},
                                {y.begin() + y.size() / 2, y.end()},
                                U, args...);
      return;
    }
    instantiate<Apply, Recon>(out, in, x, U, y, args...);
  }

} // namespace quda
