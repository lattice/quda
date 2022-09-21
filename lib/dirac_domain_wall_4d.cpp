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
  void DiracDomainWall4D::Dslash4(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, parity, dagger, commDim, profile);
    flops += 1320LL*(long long)in.Volume();
  }

  void DiracDomainWall4D::Dslash5(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, 0.0, nullptr, nullptr, 0.0, dagger, Dslash5Type::DSLASH5_DWF);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 96LL*bulk + 120LL*wall;
  }

  // Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4D::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, k, 0.0, nullptr, nullptr, x, parity, dagger, commDim, profile);

    flops += (1320LL+48LL)*(long long)in.Volume();
  }

  void DiracDomainWall4D::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
                                      const double &k) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, 0.0, nullptr, nullptr, k, dagger, Dslash5Type::DSLASH5_DWF);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDomainWall4D::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
    flops += 1320LL * (long long)in.Volume();
    ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, Dslash5Type::DSLASH5_DWF);
    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += (48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;

    blas::xpay(in, -kappa5, out);
  }

  void DiracDomainWall4D::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(in);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall4D::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
      ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracDomainWall4D::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
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

  void DiracDomainWall4DPC::M5inv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, nullptr, nullptr, 0.0, dagger, Dslash5Type::M5_INV_DWF);

    long long Ls = in.X(4);
    flops += 144LL * (long long)in.Volume() * Ls + 3LL * Ls * (Ls - 1LL);
  }

  void DiracDomainWall4DPC::M5invXpay(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
                                      const double &b) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, nullptr, nullptr, b, dagger, Dslash5Type::M5_INV_DWF);

    long long Ls = in.X(4);
    flops += (144LL * Ls + 48LL) * (long long)in.Volume() + 3LL * Ls * (Ls - 1LL);
  }

  // Apply the 4D even-odd preconditioned domain-wall Dirac operator
  void DiracDomainWall4DPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = kappa5*kappa5;
    auto tmp = getFieldTmp(in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric && !dagger) {
      // 1 - k^2 M5^-1 D4 M5^-1 D4
      Dslash4(tmp, in, parity[0]);
      M5inv(out, tmp);
      Dslash4(tmp, out, parity[1]);
      M5invXpay(out, tmp, in, -kappa2);
    } else if (symmetric && dagger) {
      // 1 - k^2 D4 M5^-1 D4 M5^-1
      M5inv(tmp, in);
      Dslash4(out, tmp, parity[0]);
      M5inv(tmp, out);
      Dslash4Xpay(out, tmp, parity[1], in, -kappa2);
    } else {
      // 1 - k D5 - k^2 D4 M5^-1 D4_oe
      Dslash4(tmp, in, parity[0]);
      M5inv(out, tmp);
      Dslash4Xpay(tmp, out, parity[1], in, -kappa2);
      Dslash5Xpay(out, in, tmp, -kappa5);
    }
  }

  void DiracDomainWall4DPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracDomainWall4DPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				    ColorSpinorField &x, ColorSpinorField &b, 
				    const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {  // we desire solution to full system
      auto tmp = getFieldTmp(b.Even());

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = M5^-1 (b_e + k D4_eo*M5^-1 b_o)
        src = &(x.Odd());
        M5inv(*src, b.Odd());
        Dslash4Xpay(tmp, *src, QUDA_EVEN_PARITY, b.Even(), kappa5);
        M5inv(*src, tmp);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = M5^-1 (b_o + k D4_oe*M5^-1 b_e)
        src = &(x.Even());
        M5inv(*src, b.Even());
        Dslash4Xpay(tmp, *src, QUDA_ODD_PARITY, b.Odd(), kappa5);
        M5inv(*src, tmp);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo*M5^-1 b_o
        src = &(x.Odd());
        M5inv(tmp, b.Odd());
        Dslash4Xpay(*src, tmp, QUDA_EVEN_PARITY, b.Even(), kappa5);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe*M5^-1 b_e
        src = &(x.Even());
        M5inv(tmp, b.Even());
        Dslash4Xpay(*src, tmp, QUDA_ODD_PARITY, b.Odd(), kappa5);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }
  }

  void DiracDomainWall4DPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
					const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    checkFullSpinor(x, b);
    auto tmp = getFieldTmp(b.Even());

    // create full solution
    if (matpcType == QUDA_MATPC_EVEN_EVEN ||
	matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // x_o = M5^-1 (b_o + k D4_oe x_e)
      Dslash4Xpay(tmp, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
      M5inv(x.Odd(), tmp);
    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // x_e = M5^-1 (b_e + k D4_eo x_o)
      Dslash4Xpay(tmp, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
      M5inv(x.Even(), tmp);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
    }
  }

} // end namespace quda
