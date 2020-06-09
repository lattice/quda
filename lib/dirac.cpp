#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

#include <iostream>

namespace quda {

  // FIXME: At the moment, it's unsafe for more than one Dirac operator to be active unless
  // they all have the same volume, etc. (used to initialize the various CUDA constants).

  Dirac::Dirac(const DiracParam &param) :
    gauge(param.gauge),
    kappa(param.kappa),
    mass(param.mass),
    laplace3D(param.laplace3D),
    matpcType(param.matpcType),
    dagger(param.dagger),
    flops(0),
    tmp1(param.tmp1),
    tmp2(param.tmp2),
    type(param.type),
    halo_precision(param.halo_precision),
    profile("Dirac", false)
  {
    for (int i=0; i<4; i++) commDim[i] = param.commDim[i];
  }

  Dirac::Dirac(const Dirac &dirac) :
    gauge(dirac.gauge),
    kappa(dirac.kappa),
    laplace3D(dirac.laplace3D),
    matpcType(dirac.matpcType),
    dagger(dirac.dagger),
    flops(0),
    tmp1(dirac.tmp1),
    tmp2(dirac.tmp2),
    type(dirac.type),
    halo_precision(dirac.halo_precision),
    profile("Dirac", false)
  {
    for (int i=0; i<4; i++) commDim[i] = dirac.commDim[i];
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
      dagger = dirac.dagger;
      flops = 0;
      tmp1 = dirac.tmp1;
      tmp2 = dirac.tmp2;

      for (int i=0; i<4; i++) commDim[i] = dirac.commDim[i];

      profile = dirac.profile;

      if (type != dirac.type) errorQuda("Trying to copy between incompatible types %d %d", type, dirac.type);
    }
    return *this;
  }

  bool Dirac::newTmp(ColorSpinorField **tmp, const ColorSpinorField &a) const {
    if (*tmp) return false;
    ColorSpinorParam param(a);
    param.create = QUDA_ZERO_FIELD_CREATE; // need to zero elements else padded region will be junk

    if (typeid(a) == typeid(cudaColorSpinorField)) *tmp = new cudaColorSpinorField(a, param);
    else *tmp = new cpuColorSpinorField(param);

    return true;
  }

  void Dirac::deleteTmp(ColorSpinorField **a, const bool &reset) const {
    if (reset) {
      delete *a;
      *a = NULL;
    }
  }

#define flip(x) (x) = ((x) == QUDA_DAG_YES ? QUDA_DAG_NO : QUDA_DAG_YES)

  void Dirac::Mdag(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    flip(dagger);
    M(out, in);
    flip(dagger);
  }

  void Dirac::MMdag(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    flip(dagger);
    MdagM(out, in);
    flip(dagger);
  }

#undef flip

  void Dirac::checkParitySpinor(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( (in.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS || out.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) && 
	 in.Nspin() == 4) {
      errorQuda("CUDA Dirac operator requires UKQCD basis, out = %d, in = %d", 
		out.GammaBasis(), in.GammaBasis());
    }

    if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not single parity: in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    }

    if (!static_cast<const cudaColorSpinorField&>(in).isNative()) errorQuda("Input field is not in native order");
    if (!static_cast<const cudaColorSpinorField&>(out).isNative()) errorQuda("Output field is not in native order");

    if (out.Ndim() != 5) {
      if ((out.Volume() != gauge->Volume() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	  (out.Volume() != gauge->VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
        errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), gauge->VolumeCB());
      }
    } else {
      // Domain wall fermions, compare 4d volumes not 5d
      if ((out.Volume()/out.X(4) != gauge->Volume() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	  (out.Volume()/out.X(4) != gauge->VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
        errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), gauge->VolumeCB());
      }
    }
  }

  void Dirac::checkFullSpinor(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() != QUDA_FULL_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not full fields: in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    } 
  }

  void Dirac::checkSpinorAlias(const ColorSpinorField &a, const ColorSpinorField &b) const {
    if (a.V() == b.V()) errorQuda("Aliasing pointers");
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
    } else if (param.type == QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverHasenbuschTwist operator\n");
      return new DiracCloverHasenbuschTwist(param);
    } else if (param.type == QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverHasenbuschTwistPC operator\n");
      return new DiracCloverHasenbuschTwistPC(param);
    } else if (param.type == QUDA_CLOVERPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracCloverPC operator\n");
      return new DiracCloverPC(param);
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
    } else if (param.type == QUDA_ASQTAD_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracImprovedStaggered operator\n");
      return new DiracImprovedStaggered(param);
    } else if (param.type == QUDA_ASQTADPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracImprovedStaggeredPC operator\n");
      return new DiracImprovedStaggeredPC(param);
    } else if (param.type == QUDA_TWISTED_CLOVER_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedClover operator (%d flavor(s))\n", param.Ls);
      if (param.Ls == 1) {
	return new DiracTwistedClover(param, 4);
      } else { 
	errorQuda("Cannot create DiracTwistedClover operator for %d flavors\n", param.Ls);
      }
    } else if (param.type == QUDA_TWISTED_CLOVERPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedCloverPC operator (%d flavor(s))\n", param.Ls);
      if (param.Ls == 1) {
	return new DiracTwistedCloverPC(param, 4);
      } else {
	errorQuda("Cannot create DiracTwistedCloverPC operator for %d flavors\n", param.Ls);
      }
    } else if (param.type == QUDA_TWISTED_MASS_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Creating a DiracTwistedMass operator (%d flavor(s))\n", param.Ls);
        if (param.Ls == 1) return new DiracTwistedMass(param, 4);
        else return new DiracTwistedMass(param, 5);
    } else if (param.type == QUDA_TWISTED_MASSPC_DIRAC) {
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Creating a DiracTwistedMassPC operator (%d flavor(s))\n", param.Ls);
      if (param.Ls == 1)
        return new DiracTwistedMassPC(param, 4);
      else
        return new DiracTwistedMassPC(param, 5);
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
  
  // Count the number of stencil applications per dslash application.
  int Dirac::getStencilSteps() const
  {
    int steps = 0;
    switch (type)
    {
      case QUDA_COARSE_DIRAC: // single fused operator
      case QUDA_GAUGE_LAPLACE_DIRAC:
      case QUDA_GAUGE_COVDEV_DIRAC:
	steps = 1;
	break;
      case QUDA_WILSON_DIRAC:
      case QUDA_CLOVER_DIRAC:
      case QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC:
      case QUDA_DOMAIN_WALL_DIRAC:
      case QUDA_MOBIUS_DOMAIN_WALL_DIRAC:
      case QUDA_STAGGERED_DIRAC:
      case QUDA_ASQTAD_DIRAC:
      case QUDA_TWISTED_CLOVER_DIRAC:
      case QUDA_TWISTED_MASS_DIRAC:
        steps = 2; // For D_{eo} and D_{oe} piece.
        break;
      case QUDA_WILSONPC_DIRAC:
      case QUDA_CLOVERPC_DIRAC:
      case QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC:
      case QUDA_DOMAIN_WALLPC_DIRAC:
      case QUDA_DOMAIN_WALL_4DPC_DIRAC:
      case QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC:
      case QUDA_STAGGEREDPC_DIRAC:
      case QUDA_ASQTADPC_DIRAC:
      case QUDA_TWISTED_CLOVERPC_DIRAC:
      case QUDA_TWISTED_MASSPC_DIRAC:
      case QUDA_COARSEPC_DIRAC:
      case QUDA_GAUGE_LAPLACEPC_DIRAC:
        steps = 2;
        break;
	  default:
	    errorQuda("Unsupported Dslash type %d.\n", type);
        steps = 0;
        break;
    }
    
    return steps; 
  }

  void Dirac::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (gauge) gauge->prefetch(mem_space, stream);
    if (tmp1) tmp1->prefetch(mem_space, stream);
    if (tmp2) tmp2->prefetch(mem_space, stream);
  }

} // namespace quda
