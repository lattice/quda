#pragma once

#include <dslash_domain_wall_4d_fused_m5.hpp>

/**
   This is the templated gauged domain-wall 4-d preconditioned operator, but fused with immediately followed fifth
   dimension operators.
*/

namespace quda
{

  constexpr int domain_wall_4d_fused_m5_mma_block_dim_x(int Ls) {
    return 16;
  }

  constexpr int domain_wall_4d_fused_m5_mma_reload() {
    return true;
  }

  template <template <int, bool, bool, KernelType, typename> class D, class Arg> class DomainWall4DFusedM5Mma : public Dslash<D, Arg>
  {
    using Dslash = Dslash<D, Arg>;
    using Dslash::arg;
    using Dslash::aux_base;
    using Dslash::in;

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

  public:
    DomainWall4DFusedM5Mma(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash(arg, out, in, get_app_base())
    {
      TunableKernel3D::resizeVector(in.X(4), arg.nParity);
      TunableKernel3D::resizeStep(in.X(4), arg.nParity);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }

    unsigned int sharedBytesPerThread() const
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

    long long flops() const
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

      return flops_ + Dslash::flops();
    }

    long long bytes() const
    {
      if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG) {
        return arg.y.Bytes() + Dslash::bytes();
      } else {
        return Dslash::bytes();
      }
    }

    bool tuneGridDim() const { return true; }

    int blockStep() const { return domain_wall_4d_fused_m5_mma_block_dim_x(Arg::Ls); }
    int blockMin() const { return domain_wall_4d_fused_m5_mma_block_dim_x(Arg::Ls); }
    unsigned int maxBlockSize(const TuneParam &) const { return domain_wall_4d_fused_m5_mma_block_dim_x(Arg::Ls); }

    int gridStep() const { return device::processor_count(); }
    unsigned int maxGridSize() const { return (arg.volume_4d_cb + blockMin() - 1) / blockMin(); }
    unsigned int minGridSize() const { return device::processor_count(); }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const
    {
      return mma_shared_bytes<typename Arg::store_t, domain_wall_4d_fused_m5_mma_reload()>(param.block.x, param.block.y);
    }
  };

  template <template <int, bool, bool, KernelType, typename> class D> struct Dslash5KernelList { };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApplyFusedM5Mma {

    template <template <int, bool, bool, KernelType, typename> class D>
    inline DomainWall4DApplyFusedM5Mma(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                                    double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x,
                                    ColorSpinorField &y, int parity, bool dagger, const int *comm_override, double m_f,
                                    Dslash5KernelList<D>, TimeProfile &profile)
    {
#ifdef NVSHMEM_COMMS
      errorQuda("Fused Mobius/DWF-4D kernels do not currently work with NVSHMEM.");
#else
      constexpr int nDim = 4;
      using D4Arg = DomainWall4DArg<Float, nColor, nDim, recon>;
      constexpr int Ls = 12;
      constexpr int block_dim_x = domain_wall_4d_fused_m5_mma_block_dim_x(Ls);
      using D5Arg = Dslash5MmaArg<Float, nColor, Ls, block_dim_x, false, false, domain_wall_4d_fused_m5_mma_reload()>;
      using Arg = DomainWall4DFusedM5Arg<D4Arg, D5Arg>;
      Arg arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, y, parity, dagger, comm_override, m_f);
      DomainWall4DFusedM5Mma<D, Arg> dwf(arg, out, in);

      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, in.getDslashConstant().volume_4d_cb,
                                                     in.getDslashConstant().ghostFaceCB, profile);
#endif
    }
  };

} // namespggace quda
