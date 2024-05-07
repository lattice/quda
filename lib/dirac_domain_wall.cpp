#include <iostream>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracDomainWall::DiracDomainWall(const DiracParam &param) :
      DiracWilson(param, 5),
      m5(param.m5),
      kappa5(0.5 / (5.0 + m5)),
      Ls(param.Ls)
  {
  }

  DiracDomainWall::DiracDomainWall(const DiracDomainWall &dirac) :
      DiracWilson(dirac),
      m5(dirac.m5),
      kappa5(0.5 / (5.0 + m5)),
      Ls(dirac.Ls)
  {
  }

  DiracDomainWall::~DiracDomainWall() { }

  DiracDomainWall& DiracDomainWall::operator=(const DiracDomainWall &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      m5 = dirac.m5;
      kappa5 = dirac.kappa5;
    }
    return *this;
  }

  void DiracDomainWall::checkDWF(cvector_ref<const ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions");
    for (auto i = 0u; i < in.size(); i++)
      if (in[i].X(4) != out[i].X(4))
        errorQuda("5th dimension size mismatch: in = %d, out = %d", in[i].X(4), out[i].X(4));
  }

  void DiracDomainWall::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               QudaParity parity) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    ApplyDomainWall5D(out, in, *gauge, 0.0, mass, in, parity, dagger, commDim.data, profile);
  }

  void DiracDomainWall::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                   QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    ApplyDomainWall5D(out, in, *gauge, k, mass, x, parity, dagger, commDim.data, profile);
  }

  void DiracDomainWall::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);

    ApplyDomainWall5D(out, in, *gauge, -kappa5, mass, in, QUDA_INVALID_PARITY, dagger, commDim.data, profile);
  }

  void DiracDomainWall::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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

  void DiracDomainWall::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                    const QudaSolutionType) const
  {
    // do nothing
  }

  DiracDomainWallPC::DiracDomainWallPC(const DiracParam &param)
    : DiracDomainWall(param)
  {

  }

  DiracDomainWallPC::DiracDomainWallPC(const DiracDomainWallPC &dirac) 
    : DiracDomainWall(dirac)
  {

  }

  DiracDomainWallPC::~DiracDomainWallPC()
  {

  }

  DiracDomainWallPC& DiracDomainWallPC::operator=(const DiracDomainWallPC &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall::operator=(dirac);
    }

    return *this;
  }

  // Apply the even-odd preconditioned clover-improved Dirac operator
  void DiracDomainWallPC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkDWF(out, in);
    double kappa2 = -kappa5*kappa5;
    auto tmp = getFieldTmp(out);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(tmp, in, QUDA_ODD_PARITY);
      DslashXpay(out, tmp, QUDA_EVEN_PARITY, in, kappa2);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(tmp, in, QUDA_EVEN_PARITY);
      DslashXpay(out, tmp, QUDA_ODD_PARITY, in, kappa2);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
    }
  }

  void DiracDomainWallPC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWallPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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
    for (auto i = 0u; i < b.size(); i++) {
      // src = b_e + k D_eo b_o
      DslashXpay(x[i][other_parity], b[i][other_parity], this_parity, b[this_parity], kappa5);
      src[i] = x[i][other_parity].create_alias();
      sol[i] = x[i][this_parity].create_alias();
    }
  }

  void DiracDomainWallPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                      const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    // create full solution
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);
      // x_o = b_o + k D_oe x_e
      DslashXpay(x[i][other_parity], x[i][this_parity], other_parity, b[i][other_parity], kappa5);
    }
  }

} // namespace quda
