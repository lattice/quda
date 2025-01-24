#include <iostream>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda
{

  DiracDomainWall4DPV::DiracDomainWall4DPV(const DiracParam &param) : DiracDomainWall4D(param), mass_pv(1.0) { }

  DiracDomainWall4DPV::DiracDomainWall4DPV(const DiracDomainWall4DPV &dirac) :
    DiracDomainWall4D(dirac), mass_pv(1.0) { }

  DiracDomainWall4DPV::~DiracDomainWall4DPV() { }

  DiracDomainWall4DPV &DiracDomainWall4DPV::operator=(const DiracDomainWall4DPV &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall4D::operator=(dirac);
      mass_pv = 1.0;
    }

    return *this;
  }

  void DiracDomainWall4DPV::Dslash4(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                    const QudaParity) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWall4DPV::Dslash5(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWall4DPV::Dslash4Xpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                        const QudaParity, cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWall4DPV::Dslash5Xpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                        cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The domain wall PV operator does not have a single parity form");
  }

  void DiracDomainWall4DPV::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    // printfQuda("Calling DiracDomainWall4DPV::M%s with kappa5 = %e , mass = %e\n", dagger == QUDA_DAG_YES ? "Dag" : "", kappa5, mass);

    if (dagger == QUDA_DAG_NO) {
      // Apply D_dwf
      ApplyDomainWall4D(tmp, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim.data,
                        profile);
      ApplyDslash5(tmp, in, tmp, mass, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_DWF);
      blas::xpay(in, -kappa5, tmp);

      // Apply D_PV^dagger
      ApplyDomainWall4D(out, tmp, *gauge, 0.0, 0.0, nullptr, nullptr, tmp, QUDA_INVALID_PARITY, QUDA_DAG_YES,
                        commDim.data, profile);
      ApplyDslash5(out, tmp, out, mass_pv, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_DWF);
      blas::xpay(tmp, -kappa5, out);
    } else {
      // Apply D_PV
      ApplyDomainWall4D(tmp, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim.data,
                        profile);
      ApplyDslash5(tmp, in, tmp, mass_pv, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_NO, Dslash5Type::DSLASH5_DWF);
      blas::xpay(in, -kappa5, tmp);

      // Apply D_dwf^dagger
      ApplyDomainWall4D(out, tmp, *gauge, 0.0, 0.0, nullptr, nullptr, tmp, QUDA_INVALID_PARITY, QUDA_DAG_YES,
                        commDim.data, profile);
      ApplyDslash5(out, tmp, out, mass, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_DWF);
      blas::xpay(tmp, -kappa5, out);
    }
  }

  void DiracDomainWall4DPV::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall4DPV::ApplyPVDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);

    // printfQuda("Calling DiracDomainWall4DPV::ApplyPVDagger with kappa5 = %e , mass = %e\n", kappa5, mass);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data,
                      profile);
    ApplyDslash5(out, in, out, mass_pv, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_DWF);
    blas::xpay(in, -kappa5, out);
  }

  void DiracDomainWall4DPV::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                    cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    create_alias(src, b);
    create_alias(sol, x);
  }

  bool DiracDomainWall4DPV::hermitian() const { return (mass_pv == mass); }

  void DiracDomainWall4DPV::prepareSpecialMG(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                             cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);

      src[i] = getFieldTmp(x[i]);
      ApplyPVDagger(src[i], b[i]);

      sol[i] = x[i].create_alias();
    }
  }

  void DiracDomainWall4DPV::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                        const QudaSolutionType) const
  {
    // do nothing
  }

  void DiracDomainWall4DPV::reconstructSpecialMG(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                                 const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

} // end namespace quda
