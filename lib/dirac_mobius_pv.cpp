#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracMobiusPV::DiracMobiusPV(const DiracParam &param) :
      DiracMobius(param)
  {

  }

  DiracMobiusPV::DiracMobiusPV(const DiracMobiusPV &dirac) :
      DiracMobius(dirac)
  {

  }

  DiracMobiusPV::~DiracMobiusPV() {
  }

  DiracMobiusPV& DiracMobiusPV::operator=(const DiracMobiusPV &dirac)
  {
    if (&dirac != this) {
      DiracMobius::operator=(dirac);
      if (dirac.tmp3) {
        ColorSpinorParam csParam(*dirac.tmp3);
        csParam.create = QUDA_NULL_FIELD_CREATE;
        tmp3 = ColorSpinorField::Create(csParam);
      }
    }
    return *this;
  }

  void DiracMobiusPV::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
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

  void DiracMobiusPV::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				   const QudaParity parity, const ColorSpinorField &x,
				   const double &k) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }


  void DiracMobiusPV::Dslash4(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4pre(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                   const ColorSpinorField &x, const double &k) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4preXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                      const ColorSpinorField &x, const double &k) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                   const ColorSpinorField &x, const double &k) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset_2 = newTmp(&tmp2, in);
    bool reset_3 = newTmp(&tmp3, in);

    // zMobius breaks the following code. Refer to the zMobius check in DiracMobius::DiracMobius(param)
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

    // Who lets me write stuff like this
    if (dagger == QUDA_DAG_NO) {
      // Apply D_dw
      ApplyDslash5(*tmp2, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(*tmp3, *tmp2, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      ApplyDslash5(*tmp2, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, *tmp3, *tmp2);

      // Apply D_pv^dag
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, *tmp2, *gauge, 0.0, m5, b_5, c_5, *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
      ApplyDslash5(*tmp3, out, *tmp2, 1.0, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, *tmp2, *tmp2, 1.0, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, *tmp3, out);
    } else {
      // Apply D_pv
      ApplyDslash5(*tmp2, in, in, 1.0, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(*tmp3, *tmp2, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      ApplyDslash5(*tmp2, in, in, 1.0, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, *tmp3, *tmp2);

      // Apply D_dw^dag
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, *tmp2, *gauge, 0.0, m5, b_5, c_5, *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
      ApplyDslash5(*tmp3, out, *tmp2, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, *tmp2, *tmp2, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, *tmp3, out);
    }

    deleteTmp(&tmp2, reset_2);
    deleteTmp(&tmp3, reset_3);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 2LL * 72LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // pre
    flops += 2LL * 1320LL * (long long)in.Volume();                            // dslash4
    flops += 2LL * 48LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // dslash5
  }

  void DiracMobiusPV::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracMobiusPV::ApplyPVDagger(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    // zMobius breaks the following code. Refer to the zMobius check in DiracMobius::DiracMobius(param)
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

    ColorSpinorField *tmp = nullptr;
    if (tmp2 && tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = tmp2;
    bool reset = newTmp(&tmp, in);

    if (dagger == QUDA_DAG_YES) {
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(*tmp, out, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, DSLASH5_MOBIUS);
    } else {
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, in, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
      ApplyDslash5(*tmp, out, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, DSLASH5_MOBIUS);
    }
    blas::axpy(-mobius_kappa_b, *tmp, out);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += (1320LL + 48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;

    deleteTmp(&tmp, reset);

  }

  void DiracMobiusPV::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				ColorSpinorField &x, ColorSpinorField &b, 
				const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracMobiusPV::prepareSpecialMG(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
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

  void DiracMobiusPV::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				    const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracMobiusPV::reconstructSpecialMG(ColorSpinorField &x, const ColorSpinorField &b,
            const QudaSolutionType solType) const
  {
    // do nothing
  }


} // namespace quda
