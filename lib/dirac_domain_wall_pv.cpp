#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracDomainWallPV::DiracDomainWallPV(const DiracParam &param) :
      DiracDomainWall(param)
  {
  }

  DiracDomainWallPV::DiracDomainWallPV(const DiracDomainWallPV &dirac) :
      DiracDomainWall(dirac)
  {
  }

  DiracDomainWallPV::~DiracDomainWallPV() { }

  DiracDomainWallPV& DiracDomainWallPV::operator=(const DiracDomainWallPV &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall::operator=(dirac);
    }
    return *this;
  }

  void DiracDomainWallPV::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) { errorQuda("Domain wall dslash requires 5-d fermion fields"); }

    if (in.Precision() != out.Precision()) {
      errorQuda("Input and output spinor precisions don't match in dslash_quda");
    }

    if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not full parity, in = %d, out = %d", in.SiteSubset(), out.SiteSubset());
    }

    if (out.Volume() / out.X(4) != 2 * gauge->VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), gauge->VolumeCB());
    }
  }

  void DiracDomainWallPV::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWallPV::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				   const QudaParity parity, const ColorSpinorField &x,
				   const double &k) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWallPV::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp2, in);

    if (dagger == QUDA_DAG_NO) {
      ApplyDomainWall5D(*tmp2, in, *gauge, -kappa5, mass, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      ApplyDomainWall5D(out, *tmp2, *gauge, -kappa5, 1.0, *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
    } else {
      ApplyDomainWall5D(*tmp2, in, *gauge, -kappa5, 1.0, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      ApplyDomainWall5D(out, *tmp2, *gauge, -kappa5, mass, *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
    }

    deleteTmp(&tmp2, reset);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 2LL * (1320LL + 48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  void DiracDomainWallPV::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracDomainWallPV::ApplyPVDagger(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    if (dagger == QUDA_DAG_NO) {
      ApplyDomainWall5D(out, in, *gauge, -kappa5, 1.0, in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
    } else {
      ApplyDomainWall5D(out, in, *gauge, -kappa5, 1.0, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
    }

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += (1320LL + 48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;

  }

  void DiracDomainWallPV::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				ColorSpinorField &x, ColorSpinorField &b, 
				const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracDomainWallPV::prepareSpecialMG(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                          ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    checkFullSpinor(x, b);
    // need to modify rhs
    bool reset = newTmp(&tmp1, b);

    ApplyPVDagger(*tmp1, b);
    b = *tmp1;

    deleteTmp(&tmp1, reset);
    sol = &x;
    src = &b;

  }

  void DiracDomainWallPV::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				    const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracDomainWallPV::reconstructSpecialMG(ColorSpinorField &x, const ColorSpinorField &b,
            const QudaSolutionType solType) const
  {
    // do nothing
  }


} // namespace quda
