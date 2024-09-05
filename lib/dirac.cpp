#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda {

  // FIXME: At the moment, it's unsafe for more than one Dirac operator to be active unless
  // they all have the same volume, etc. (used to initialize the various CUDA constants).

  Dirac::Dirac(const DiracParam &param) :
    gauge(param.gauge),
    kappa(param.kappa),
    mass(param.mass),
    laplace3D(param.laplace3D),
    matpcType(param.matpcType),
    this_parity(QUDA_INVALID_PARITY),
    other_parity(QUDA_INVALID_PARITY),
    dagger(param.dagger),
    type(param.type),
    halo_precision(param.halo_precision),
    commDim(param.commDim),
    use_mobius_fused_kernel(param.use_mobius_fused_kernel),
    distance_pc_alpha0(param.distance_pc_alpha0),
    distance_pc_t0(param.distance_pc_t0),
    profile("Dirac", false)
  {
    if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      this_parity = QUDA_EVEN_PARITY;
      other_parity = QUDA_ODD_PARITY;
    } else if (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      this_parity = QUDA_ODD_PARITY;
      other_parity = QUDA_EVEN_PARITY;
    } else {
      errorQuda("Invalid matpcType(%d) in function\n", matpcType);
    }
    symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD);
  }

  Dirac::Dirac(const Dirac &dirac) :
    gauge(dirac.gauge),
    kappa(dirac.kappa),
    laplace3D(dirac.laplace3D),
    matpcType(dirac.matpcType),
    this_parity(dirac.this_parity),
    other_parity(dirac.other_parity),
    symmetric(dirac.symmetric),
    dagger(dirac.dagger),
    type(dirac.type),
    halo_precision(dirac.halo_precision),
    commDim(dirac.commDim),
    distance_pc_alpha0(dirac.distance_pc_alpha0),
    distance_pc_t0(dirac.distance_pc_t0),
    profile("Dirac", false)
  {
  }

  // Destroy
  Dirac::~Dirac() {   
    if (getVerbosity() > QUDA_VERBOSE) profile.Print();
  }

  // Assignment
  Dirac& Dirac::operator=(const Dirac &dirac)
  {
    if (&dirac != this) {
      gauge = dirac.gauge;
      kappa = dirac.kappa;
      laplace3D = dirac.laplace3D;
      matpcType = dirac.matpcType;
      this_parity = dirac.this_parity;
      other_parity = dirac.other_parity;
      symmetric = dirac.symmetric;
      dagger = dirac.dagger;
      commDim = dirac.commDim;
      distance_pc_alpha0 = dirac.distance_pc_alpha0;
      distance_pc_t0 = dirac.distance_pc_t0;
      profile = dirac.profile;

      if (type != dirac.type) errorQuda("Trying to copy between incompatible types %d %d", type, dirac.type);
    }
    return *this;
  }

#define flip(x) (x) = ((x) == QUDA_DAG_YES ? QUDA_DAG_NO : QUDA_DAG_YES)

  void Dirac::Mdag(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    flip(dagger);
    M(out, in);
    flip(dagger);
  }

  void Dirac::MMdag(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    flip(dagger);
    MdagM(out, in);
    flip(dagger);
  }

#undef flip

  void Dirac::checkParitySpinor(cvector_ref<const ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    for (auto i = 0u; i < out.size(); i++) {
      if ((in[i].GammaBasis() != QUDA_UKQCD_GAMMA_BASIS || out[i].GammaBasis() != QUDA_UKQCD_GAMMA_BASIS)
          && in[i].Nspin() == 4) {
        errorQuda("Dirac operator requires UKQCD basis, out = %d, in = %d", out[i].GammaBasis(), in[i].GammaBasis());
      }

      if (in[i].SiteSubset() != QUDA_PARITY_SITE_SUBSET || out[i].SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not single parity: in = %d, out = %d", in[i].SiteSubset(), out[i].SiteSubset());
      }

      if (out[i].Ndim() != 5) {
        if ((out[i].Volume() != gauge->Volume() && out[i].SiteSubset() == QUDA_FULL_SITE_SUBSET)
            || (out[i].Volume() != gauge->VolumeCB() && out[i].SiteSubset() == QUDA_PARITY_SITE_SUBSET)) {
          errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out[i].Volume(), gauge->VolumeCB());
        }
      } else {
        // Domain wall fermions, compare 4d volumes not 5d
        if ((out[i].Volume() / out[i].X(4) != gauge->Volume() && out[i].SiteSubset() == QUDA_FULL_SITE_SUBSET)
            || (out[i].Volume() / out[i].X(4) != gauge->VolumeCB() && out[i].SiteSubset() == QUDA_PARITY_SITE_SUBSET)) {
          errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out[i].Volume(), gauge->VolumeCB());
        }
      }
    }
  }

  void Dirac::checkFullSpinor(cvector_ref<const ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    for (auto i = 0u; i < out.size(); i++) {
      if (in[i].SiteSubset() != QUDA_FULL_SITE_SUBSET || out[i].SiteSubset() != QUDA_FULL_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not full fields: in = %d, out = %d", in[i].SiteSubset(), out[i].SiteSubset());
      }
    }
  }

  void Dirac::checkSpinorAlias(cvector_ref<const ColorSpinorField> &a, cvector_ref<const ColorSpinorField> &b) const
  {
    for (auto i = 0u; i < a.size(); i++)
      if (a[i].data() == b[i].data()) errorQuda("Aliasing pointers");
  }

  // Dirac operator factory
  Dirac* Dirac::create(const DiracParam &param)
  {
    if (param.type == QUDA_WILSON_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracWilson operator\n");
      return new DiracWilson(param);
    } else if (param.type == QUDA_WILSONPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracWilsonPC operator\n");
      return new DiracWilsonPC(param);
    } else if (param.type == QUDA_CLOVER_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracClover operator\n");
      return new DiracClover(param);
    } else if (param.type == QUDA_CLOVERPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverPC operator\n");
      return new DiracCloverPC(param);
    } else if (param.type == QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverHasenbuschTwist operator\n");
      return new DiracCloverHasenbuschTwist(param);
    } else if (param.type == QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverHasenbuschTwistPC operator\n");
      return new DiracCloverHasenbuschTwistPC(param);
    } else if (param.type == QUDA_DOMAIN_WALL_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracDomainWall operator\n");
      return new DiracDomainWall(param);
    } else if (param.type == QUDA_DOMAIN_WALLPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracDomainWallPC operator\n");
      return new DiracDomainWallPC(param);
    } else if (param.type == QUDA_DOMAIN_WALL_4D_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracDomainWall4D operator\n");
      return new DiracDomainWall4D(param);
    } else if (param.type == QUDA_DOMAIN_WALL_4DPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracDomainWall4DPC operator\n");
      return new DiracDomainWall4DPC(param);
    } else if (param.type == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracMobius operator\n");
      return new DiracMobius(param);
    } else if (param.type == QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracMobiusPC operator\n");
      return new DiracMobiusPC(param);
    } else if (param.type == QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracMobiusEofa operator\n");
      return new DiracMobiusEofa(param);
    } else if (param.type == QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracMobiusEofaPC operator\n");
      return new DiracMobiusEofaPC(param);
    } else if (param.type == QUDA_STAGGERED_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracStaggered operator\n");
      return new DiracStaggered(param);
    } else if (param.type == QUDA_STAGGEREDPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracStaggeredPC operator\n");
      return new DiracStaggeredPC(param);
    } else if (param.type == QUDA_STAGGEREDKD_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracStaggeredKD operator\n");
      return new DiracStaggeredKD(param);
    } else if (param.type == QUDA_ASQTAD_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracImprovedStaggered operator\n");
      return new DiracImprovedStaggered(param);
    } else if (param.type == QUDA_ASQTADPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracImprovedStaggeredPC operator\n");
      return new DiracImprovedStaggeredPC(param);
    } else if (param.type == QUDA_ASQTADKD_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracImprovedStaggeredKD operator\n");
      return new DiracImprovedStaggeredKD(param);
    } else if (param.type == QUDA_TWISTED_CLOVER_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedClover operator (%d flavor(s))\n", param.Ls);
      switch (param.Ls) {
      case 1: return new DiracTwistedClover(param, 4);
      case 2: return new DiracTwistedClover(param, 5);
      default: errorQuda("Unexpected Ls = %d", param.Ls);
      }
    } else if (param.type == QUDA_TWISTED_CLOVERPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedCloverPC operator (%d flavor(s))\n", param.Ls);
      switch (param.Ls) {
      case 1: return new DiracTwistedCloverPC(param, 4);
      case 2: return new DiracTwistedCloverPC(param, 5);
      default: errorQuda("Unexpected Ls = %d", param.Ls);
      }
    } else if (param.type == QUDA_TWISTED_MASS_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedMass operator (%d flavor(s))\n", param.Ls);
      switch (param.Ls) {
      case 1: return new DiracTwistedMass(param, 4);
      case 2: return new DiracTwistedMass(param, 5);
      default: errorQuda("Unexpected Ls = %d", param.Ls);
      }
    } else if (param.type == QUDA_TWISTED_MASSPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Creating a DiracTwistedMassPC operator (%d flavor(s))\n", param.Ls);
      switch (param.Ls) {
      case 1: return new DiracTwistedMassPC(param, 4);
      case 2: return new DiracTwistedMassPC(param, 5);
      default: errorQuda("Unexpected Ls = %d", param.Ls);
      }
    } else if (param.type == QUDA_COARSE_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCoarse operator\n");
      return new DiracCoarse(param);
    } else if (param.type == QUDA_COARSEPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCoarsePC operator\n");
      return new DiracCoarsePC(param);
    } else if (param.type == QUDA_GAUGE_COVDEV_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a GaugeCovDev operator\n");
      return new GaugeCovDev(param);
    } else if (param.type == QUDA_GAUGE_LAPLACE_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a GaugeLaplace operator\n");
      return new GaugeLaplace(param);
    } else if (param.type == QUDA_GAUGE_LAPLACEPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a GaugeLaplacePC operator\n");
      return new GaugeLaplacePC(param);
    } else {
      errorQuda("Unsupported Dirac type %d", param.type);
    }

    return nullptr;
  }

  bool Dirac::is_wilson_type(QudaDiracType type) {
    switch (type) {
    case QUDA_WILSON_DIRAC:
    case QUDA_WILSONPC_DIRAC:
    case QUDA_CLOVER_DIRAC:
    case QUDA_CLOVERPC_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
    case QUDA_TWISTED_CLOVER_DIRAC:
    case QUDA_TWISTED_CLOVERPC_DIRAC:
    case QUDA_TWISTED_MASS_DIRAC:
    case QUDA_TWISTED_MASSPC_DIRAC:
      return true; break;
    case QUDA_DOMAIN_WALL_DIRAC:
    case QUDA_DOMAIN_WALLPC_DIRAC:
    case QUDA_DOMAIN_WALL_4D_DIRAC:
    case QUDA_DOMAIN_WALL_4DPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC:
    case QUDA_STAGGERED_DIRAC:
    case QUDA_STAGGEREDPC_DIRAC:
    case QUDA_STAGGEREDKD_DIRAC:
    case QUDA_ASQTAD_DIRAC:
    case QUDA_ASQTADPC_DIRAC:
    case QUDA_ASQTADKD_DIRAC:
    case QUDA_COARSE_DIRAC:
    case QUDA_COARSEPC_DIRAC:
    case QUDA_GAUGE_COVDEV_DIRAC:
    case QUDA_GAUGE_LAPLACE_DIRAC:
    case QUDA_GAUGE_LAPLACEPC_DIRAC:
      return false; break;
    default:
      errorQuda("Invalid QudaDiracType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_wilson_type(QudaDslashType type) {
    switch (type) {
    case QUDA_WILSON_DSLASH:
    case QUDA_CLOVER_WILSON_DSLASH:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH:
    case QUDA_TWISTED_MASS_DSLASH:
    case QUDA_TWISTED_CLOVER_DSLASH:
      return true; break;
    case QUDA_DOMAIN_WALL_DSLASH:
    case QUDA_DOMAIN_WALL_4D_DSLASH:
    case QUDA_MOBIUS_DWF_DSLASH:
    case QUDA_MOBIUS_DWF_EOFA_DSLASH:
    case QUDA_STAGGERED_DSLASH:
    case QUDA_ASQTAD_DSLASH:
    case QUDA_LAPLACE_DSLASH:
    case QUDA_COVDEV_DSLASH:
      return false; break;
    default:
      errorQuda("Invalid QudaDslashType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_staggered_type(QudaDiracType type) {
    switch (type) {
    case QUDA_STAGGERED_DIRAC:
    case QUDA_STAGGEREDPC_DIRAC:
    case QUDA_STAGGEREDKD_DIRAC:
    case QUDA_ASQTAD_DIRAC:
    case QUDA_ASQTADPC_DIRAC:
    case QUDA_ASQTADKD_DIRAC:
      return true; break;
    case QUDA_WILSON_DIRAC:
    case QUDA_WILSONPC_DIRAC:
    case QUDA_CLOVER_DIRAC:
    case QUDA_CLOVERPC_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
    case QUDA_TWISTED_CLOVER_DIRAC:
    case QUDA_TWISTED_CLOVERPC_DIRAC:
    case QUDA_TWISTED_MASS_DIRAC:
    case QUDA_TWISTED_MASSPC_DIRAC:
    case QUDA_DOMAIN_WALL_DIRAC:
    case QUDA_DOMAIN_WALLPC_DIRAC:
    case QUDA_DOMAIN_WALL_4D_DIRAC:
    case QUDA_DOMAIN_WALL_4DPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC:
    case QUDA_COARSE_DIRAC:
    case QUDA_COARSEPC_DIRAC:
    case QUDA_GAUGE_COVDEV_DIRAC:
    case QUDA_GAUGE_LAPLACE_DIRAC:
    case QUDA_GAUGE_LAPLACEPC_DIRAC:
      return false; break;
    default:
      errorQuda("Invalid QudaDiracType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_staggered_type(QudaDslashType type) {
    switch (type) {
    case QUDA_STAGGERED_DSLASH:
    case QUDA_ASQTAD_DSLASH:
      return true; break;
    case QUDA_WILSON_DSLASH:
    case QUDA_CLOVER_WILSON_DSLASH:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH:
    case QUDA_TWISTED_MASS_DSLASH:
    case QUDA_TWISTED_CLOVER_DSLASH:
    case QUDA_DOMAIN_WALL_DSLASH:
    case QUDA_DOMAIN_WALL_4D_DSLASH:
    case QUDA_MOBIUS_DWF_DSLASH:
    case QUDA_MOBIUS_DWF_EOFA_DSLASH:
    case QUDA_LAPLACE_DSLASH:
    case QUDA_COVDEV_DSLASH:
      return false; break;
    default:
      errorQuda("Invalid QudaDslashType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_asqtad(QudaDiracType type) {
    switch (type) {
    case QUDA_ASQTAD_DIRAC:
    case QUDA_ASQTADPC_DIRAC:
    case QUDA_ASQTADKD_DIRAC:
      return true; break;
    case QUDA_WILSON_DIRAC:
    case QUDA_WILSONPC_DIRAC:
    case QUDA_CLOVER_DIRAC:
    case QUDA_CLOVERPC_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
    case QUDA_TWISTED_CLOVER_DIRAC:
    case QUDA_TWISTED_CLOVERPC_DIRAC:
    case QUDA_TWISTED_MASS_DIRAC:
    case QUDA_TWISTED_MASSPC_DIRAC:
    case QUDA_DOMAIN_WALL_DIRAC:
    case QUDA_DOMAIN_WALLPC_DIRAC:
    case QUDA_DOMAIN_WALL_4D_DIRAC:
    case QUDA_DOMAIN_WALL_4DPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC:
    case QUDA_STAGGERED_DIRAC:
    case QUDA_STAGGEREDPC_DIRAC:
    case QUDA_STAGGEREDKD_DIRAC:
    case QUDA_COARSE_DIRAC:
    case QUDA_COARSEPC_DIRAC:
    case QUDA_GAUGE_COVDEV_DIRAC:
    case QUDA_GAUGE_LAPLACE_DIRAC:
    case QUDA_GAUGE_LAPLACEPC_DIRAC:
      return false; break;
    default:
      errorQuda("Invalid QudaDiracType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_asqtad(QudaDslashType type) {
    switch (type) {
    case QUDA_ASQTAD_DSLASH:
      return true; break;
    case QUDA_WILSON_DSLASH:
    case QUDA_CLOVER_WILSON_DSLASH:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH:
    case QUDA_TWISTED_MASS_DSLASH:
    case QUDA_TWISTED_CLOVER_DSLASH:
    case QUDA_DOMAIN_WALL_DSLASH:
    case QUDA_DOMAIN_WALL_4D_DSLASH:
    case QUDA_MOBIUS_DWF_DSLASH:
    case QUDA_MOBIUS_DWF_EOFA_DSLASH:
    case QUDA_STAGGERED_DSLASH:
    case QUDA_LAPLACE_DSLASH:
    case QUDA_COVDEV_DSLASH:
      return false; break;
    default:
      errorQuda("Invalid QudaDslashType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_dwf(QudaDiracType type) {
    switch (type) {
    case QUDA_DOMAIN_WALL_DIRAC:
    case QUDA_DOMAIN_WALLPC_DIRAC:
    case QUDA_DOMAIN_WALL_4D_DIRAC:
    case QUDA_DOMAIN_WALL_4DPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC:
      return true; break;
    case QUDA_WILSON_DIRAC:
    case QUDA_WILSONPC_DIRAC:
    case QUDA_CLOVER_DIRAC:
    case QUDA_CLOVERPC_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
    case QUDA_TWISTED_CLOVER_DIRAC:
    case QUDA_TWISTED_CLOVERPC_DIRAC:
    case QUDA_TWISTED_MASS_DIRAC:
    case QUDA_TWISTED_MASSPC_DIRAC:
    case QUDA_STAGGERED_DIRAC:
    case QUDA_STAGGEREDPC_DIRAC:
    case QUDA_STAGGEREDKD_DIRAC:
    case QUDA_ASQTAD_DIRAC:
    case QUDA_ASQTADPC_DIRAC:
    case QUDA_ASQTADKD_DIRAC:
    case QUDA_COARSE_DIRAC:
    case QUDA_COARSEPC_DIRAC:
    case QUDA_GAUGE_COVDEV_DIRAC:
    case QUDA_GAUGE_LAPLACE_DIRAC:
    case QUDA_GAUGE_LAPLACEPC_DIRAC:
      return false; break;
    default:
      errorQuda("Invalid QudaDiracType %d", type); break;
    }
    return false;
  }

  bool Dirac::is_dwf(QudaDslashType type) {
    switch (type) {
    case QUDA_DOMAIN_WALL_DSLASH:
    case QUDA_DOMAIN_WALL_4D_DSLASH:
    case QUDA_MOBIUS_DWF_DSLASH:
    case QUDA_MOBIUS_DWF_EOFA_DSLASH:
      return true; break;
    case QUDA_WILSON_DSLASH:
    case QUDA_CLOVER_WILSON_DSLASH:
    case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH:
    case QUDA_TWISTED_MASS_DSLASH:
    case QUDA_TWISTED_CLOVER_DSLASH:
    case QUDA_STAGGERED_DSLASH:
    case QUDA_ASQTAD_DSLASH:
    case QUDA_LAPLACE_DSLASH:
    case QUDA_COVDEV_DSLASH:
      return false; break;
    default:
      errorQuda("Invalid QudaDslashType %d", type); break;
    }
    return false;
  }

  QudaDslashType Dirac::dirac_to_dslash_type(QudaDiracType type) {
    switch (type) {
    case QUDA_WILSON_DIRAC:
    case QUDA_WILSONPC_DIRAC:
      return QUDA_WILSON_DSLASH; break;
    case QUDA_CLOVER_DIRAC:
    case QUDA_CLOVERPC_DIRAC:
      return QUDA_CLOVER_WILSON_DSLASH; break;
    case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
    case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
      return QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH; break;
    case QUDA_TWISTED_CLOVER_DIRAC:
    case QUDA_TWISTED_CLOVERPC_DIRAC:
      return QUDA_TWISTED_CLOVER_DSLASH; break;
    case QUDA_TWISTED_MASS_DIRAC:
    case QUDA_TWISTED_MASSPC_DIRAC:
      return QUDA_TWISTED_MASS_DSLASH; break;
    case QUDA_DOMAIN_WALL_DIRAC:
    case QUDA_DOMAIN_WALLPC_DIRAC:
      return QUDA_DOMAIN_WALL_DSLASH; break;
    case QUDA_DOMAIN_WALL_4D_DIRAC:
    case QUDA_DOMAIN_WALL_4DPC_DIRAC:
      return QUDA_DOMAIN_WALL_4D_DSLASH; break;
    case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
      return QUDA_MOBIUS_DWF_DSLASH; break;
    case QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC:
    case QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC:
      return QUDA_MOBIUS_DWF_EOFA_DSLASH; break;
    case QUDA_STAGGERED_DIRAC:
    case QUDA_STAGGEREDPC_DIRAC:
    case QUDA_STAGGEREDKD_DIRAC:
      return QUDA_STAGGERED_DSLASH; break;
    case QUDA_ASQTAD_DIRAC:
    case QUDA_ASQTADPC_DIRAC:
    case QUDA_ASQTADKD_DIRAC:
      return QUDA_ASQTAD_DSLASH; break;
    case QUDA_GAUGE_COVDEV_DIRAC:
      return QUDA_COVDEV_DSLASH; break;
    case QUDA_GAUGE_LAPLACE_DIRAC:
    case QUDA_GAUGE_LAPLACEPC_DIRAC:
      return QUDA_LAPLACE_DSLASH; break;
    case QUDA_COARSE_DIRAC:
    case QUDA_COARSEPC_DIRAC:
      return QUDA_INVALID_DSLASH; break;
    default:
      errorQuda("Invalid QudaDiracType %d", type); break;
    }
    return QUDA_INVALID_DSLASH;
  }

  void Dirac::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (gauge) gauge->prefetch(mem_space, stream);
  }

} // namespace quda
