#include <iostream>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracDomainWall4D::DiracDomainWall4D(const DiracParam &param) : DiracDomainWall(param) {}

  DiracDomainWall4D::DiracDomainWall4D(const DiracDomainWall4D &dirac) : DiracDomainWall(dirac) {}

  DiracDomainWall4D::~DiracDomainWall4D() {}

  DiracDomainWall4D &DiracDomainWall4D::operator=(const DiracDomainWall4D &dirac)
  {
    if (&dirac != this) { DiracDomainWall::operator=(dirac); }

    return *this;
  }

// Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4D::Dslash4(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                  const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, parity, dagger, commDim.data, profile);
  }

  void DiracDomainWall4D::Dslash5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, 0.0, nullptr, nullptr, 0.0, dagger, Dslash5Type::DSLASH5_DWF);
  }

  // Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4D::Dslash4Xpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      const QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, k, 0.0, nullptr, nullptr, x, parity, dagger, commDim.data, profile);
  }

  void DiracDomainWall4D::Dslash5Xpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, 0.0, nullptr, nullptr, k, dagger, Dslash5Type::DSLASH5_DWF);
  }

  void DiracDomainWall4D::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, dagger, commDim.data,
                      profile);
    ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, Dslash5Type::DSLASH5_DWF);
    blas::xpay(in, -kappa5, out);
  }

  void DiracDomainWall4D::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(in);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall4D::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                  cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      sol[i] = x[i].create_alias();
    }
  }

  void DiracDomainWall4D::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                      const QudaSolutionType) const
  {
    // do nothing
  }

  // Modification for the 4D preconditioned domain wall operator
  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracParam &param) : DiracDomainWall4D(param) {}

  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracDomainWall4DPC &dirac) : DiracDomainWall4D(dirac) {}

  DiracDomainWall4DPC::~DiracDomainWall4DPC() {}

  DiracDomainWall4DPC &DiracDomainWall4DPC::operator=(const DiracDomainWall4DPC &dirac)
  {
    if (&dirac != this) { DiracDomainWall4D::operator=(dirac); }

    return *this;
  }

  void DiracDomainWall4DPC::M5inv(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, nullptr, nullptr, 0.0, dagger, Dslash5Type::M5_INV_DWF);
  }

  void DiracDomainWall4DPC::M5invXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      cvector_ref<const ColorSpinorField> &x, double b) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, nullptr, nullptr, b, dagger, Dslash5Type::M5_INV_DWF);
  }

  // Apply the 4D even-odd preconditioned domain-wall Dirac operator
  void DiracDomainWall4DPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = kappa5*kappa5;
    auto tmp = getFieldTmp(in);

    if (symmetric && !dagger) {
      // 1 - k^2 M5^-1 D4 M5^-1 D4
      Dslash4(tmp, in, other_parity);
      M5inv(out, tmp);
      Dslash4(tmp, out, this_parity);
      M5invXpay(out, tmp, in, -kappa2);
    } else if (symmetric && dagger) {
      // 1 - k^2 D4 M5^-1 D4 M5^-1
      M5inv(tmp, in);
      Dslash4(out, tmp, other_parity);
      M5inv(tmp, out);
      Dslash4Xpay(out, tmp, this_parity, in, -kappa2);
    } else {
      // 1 - k D5 - k^2 D4 M5^-1 D4_oe
      Dslash4(tmp, in, other_parity);
      M5inv(out, tmp);
      Dslash4Xpay(tmp, out, this_parity, in, -kappa2);
      Dslash5Xpay(out, in, tmp, -kappa5);
    }
  }

  void DiracDomainWall4DPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall4DPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                    cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      for (auto i = 0u; i < b.size(); i++) {
        src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
        sol[i] = x[i].create_alias();
      }
      return;
    }

    // we desire solution to full system
    auto tmp = getFieldTmp(b[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      if (symmetric) {
        // src = M5^-1 (b_e + k D4_eo*M5^-1 b_o)
        src[i] = x[i][other_parity].create_alias();
        M5inv(src[i], b[i][other_parity]);
        Dslash4Xpay(tmp, src[i], this_parity, b[i][this_parity], kappa5);
        M5inv(src[i], tmp);
        sol[i] = x[i][this_parity].create_alias();
      } else {
        // src = b_e + k D4_eo*M5^-1 b_o
        src[i] = x[i][other_parity].create_alias();
        M5inv(tmp, b[i][other_parity]);
        Dslash4Xpay(src[i], tmp, this_parity, b[i][this_parity], kappa5);
        sol[i] = x[i][this_parity].create_alias();
      }
    }
  }

  void DiracDomainWall4DPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                        const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    // create full solution
    auto tmp = getFieldTmp(b[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);
      // x_o = M5^-1 (b_o + k D4_oe x_e)
      Dslash4Xpay(tmp, x[i][this_parity], other_parity, b[i][other_parity], kappa5);
      M5inv(x[i][other_parity], tmp);
    }
  }

} // end namespace quda
