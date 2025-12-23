#include <iostream>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

#include <dslash_mdw_fused.hpp>

namespace quda
{

  DiracMobiusPV::DiracMobiusPV(const DiracParam &param) : DiracMobius(param), mass_pv(1.0) { }

  DiracMobiusPV::DiracMobiusPV(const DiracMobiusPV &dirac) : DiracMobius(dirac), mass_pv(1.0) { }

  DiracMobiusPV::~DiracMobiusPV() { }

  DiracMobiusPV &DiracMobiusPV::operator=(const DiracMobiusPV &dirac)
  {
    if (&dirac != this) {
      DiracMobius::operator=(dirac);
      mass_pv = dirac.mass_pv;
    }

    return *this;
  }

  void DiracMobiusPV::Dslash4(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4pre(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash5(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4Xpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity,
                                  cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash4preXpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                     cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::Dslash5Xpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                  cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The mobius PV operator does not have a single parity form");
  }

  void DiracMobiusPV::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);

    // An intermediate purely for the PV op
    auto pv = getFieldTmp(out);

    // zMobius breaks the following code. Refer to the zMobius check in DiracMobius::DiracMobius(param)
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);
    auto tmp = getFieldTmp(out);

    // cannot use Xpay variants since it will scale incorrectly for this operator
    if (dagger == QUDA_DAG_NO) {
      // Apply D_mob
      ApplyDslash5(pv, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(tmp, pv, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim.data, profile);
      ApplyDslash5(pv, in, in, mass, m5, b_5, c_5, 0.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, tmp, pv);

      // D_pv^dag
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, pv, *gauge, 0.0, m5, b_5, c_5, pv, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data, profile);
      ApplyDslash5(tmp, out, pv, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, pv, pv, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, tmp, out);
    } else {
      // Apply D_pv
      ApplyDslash5(pv, in, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(tmp, pv, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim.data, profile);
      ApplyDslash5(pv, in, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, tmp, pv);

      // D_mob^dag
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, pv, *gauge, 0.0, m5, b_5, c_5, pv, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data, profile);
      ApplyDslash5(tmp, out, pv, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, pv, pv, mass, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS);
      blas::axpy(-mobius_kappa_b, tmp, out);
    }
  }

  void DiracMobiusPV::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracMobiusPV::ApplyMDwf(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);

    DiracMobius::M(out, in);
  }

  void DiracMobiusPV::ApplyPVDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);

    // zMobius breaks the following code. Refer to the zMobius check in DiracMobius::DiracMobius(param)
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);
    auto tmp = getFieldTmp(out);

    ApplyDomainWall4D(out, in, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data, profile);
    ApplyDslash5(tmp, out, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS_PRE);
    ApplyDslash5(out, in, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS);
    blas::axpy(-mobius_kappa_b, tmp, out);
  }

  void DiracMobiusPV::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                              cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                              const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    create_alias(src, b);
    create_alias(sol, x);
  }

  bool DiracMobiusPV::hermitian() const { return (mass_pv == mass); }

  void DiracMobiusPV::prepareSpecialMG(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                       cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }
    checkFullSpinor(x, b);

    create_alias(src, b);
    create_alias(sol, x);
    auto tmp = getFieldTmp(x);

    ApplyPVDagger(tmp, b);
    blas::copy(src, tmp);
  }

  void DiracMobiusPV::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                  const QudaSolutionType) const
  {
    // do nothing
  }

  void DiracMobiusPV::reconstructSpecialMG(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                           const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracMobiusPV::createCoarseOp(GaugeField &, GaugeField &, const Transfer &, double, double, double,
                                     double, bool) const
  {
    errorQuda("DiracMobiusPV::createCoarseOp has not been implemented yet");

    /*if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CloverField *c = nullptr;
    CoarseOp(Y, X, T, *gauge, c, kappa, mass, a, mu_factor, QUDA_WILSON_DIRAC, QUDA_MATPC_INVALID);*/
  }
} // end namespace quda