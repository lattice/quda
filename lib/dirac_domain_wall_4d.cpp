#include <iostream>
#include <dirac_quda.h>
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

  void DiracDomainWall4D::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, 0.0, nullptr, nullptr, 0.0, dagger, DSLASH5_DWF);

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

  void DiracDomainWall4D::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, 0.0, nullptr, nullptr, k, dagger, DSLASH5_DWF);

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
    ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, DSLASH5_DWF);
    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += (48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;

    blas::xpay(const_cast<ColorSpinorField &>(in), -kappa5, out);
  }

  void DiracDomainWall4D::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
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

  void DiracDomainWall4D::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
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

  // I think this is misnamed and should be M5inv
  void DiracDomainWall4DPC::Dslash5inv(
      ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, nullptr, nullptr, 0.0, dagger, M5_INV_DWF);

    long long Ls = in.X(4);
    flops += 144LL * (long long)in.Volume() * Ls + 3LL * Ls * (Ls - 1LL);
  }

  void DiracDomainWall4DPC::Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
					   const QudaParity parity, const double &a,
					   const ColorSpinorField &x, const double &b) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, nullptr, nullptr, b, dagger, M5_INV_DWF);

    long long Ls = in.X(4);
    flops +=  (144LL*Ls + 48LL)*(long long)in.Volume() + 3LL*Ls*(Ls-1LL);
  }

  // Apply the 4D even-odd preconditioned domain-wall Dirac operator
  void DiracDomainWall4DPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = kappa5*kappa5;

    bool reset1 = newTmp(&tmp1, in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric && !dagger) {
      // 1 - k^2 M5^-1 D4 M5^-1 D4
      Dslash4(*tmp1, in, parity[0]);
      Dslash5inv(out, *tmp1, parity[0], kappa5);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5invXpay(out, *tmp1, parity[1], kappa5, in, -kappa2);
    } else if (symmetric && dagger) {
      // 1 - k^2 D4 M5^-1 D4 M5^-1
      Dslash5inv(*tmp1, in, parity[1], kappa5);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0], kappa5);
      Dslash4Xpay(out, *tmp1, parity[1], in, -kappa2);
    } else {
      // 1 - k D5 - k^2 D4 M5^-1 D4_oe
      Dslash4(*tmp1, in, parity[0]);
      Dslash5inv(out, *tmp1, parity[0], kappa5);
      Dslash4Xpay(*tmp1, out, parity[1], in, -kappa2);
      Dslash5Xpay(out, in, parity[1], *tmp1, -kappa5);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracDomainWall4DPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
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
      bool reset = newTmp(&tmp1, b.Even());

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = M5^-1 (b_e + k D4_eo*M5^-1 b_o)
        src = &(x.Odd());
        Dslash5inv(*src, b.Odd(), QUDA_ODD_PARITY, kappa5);
        Dslash4Xpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa5);
	Dslash5inv(*src, *tmp1, QUDA_EVEN_PARITY, kappa5);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = M5^-1 (b_o + k D4_oe*M5^-1 b_e)
        src = &(x.Even());
        Dslash5inv(*src, b.Even(), QUDA_EVEN_PARITY, kappa5);
        Dslash4Xpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa5);
	Dslash5inv(*src, *tmp1, QUDA_ODD_PARITY, kappa5);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo*M5^-1 b_o
        src = &(x.Odd());
        Dslash5inv(*tmp1, b.Odd(), QUDA_ODD_PARITY, kappa5);
        Dslash4Xpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa5);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe*M5^-1 b_e
        src = &(x.Even());
        Dslash5inv(*tmp1, b.Even(), QUDA_EVEN_PARITY, kappa5);
        Dslash4Xpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa5);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want

      deleteTmp(&tmp1, reset);
    }
  }

  void DiracDomainWall4DPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
					const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    checkFullSpinor(x, b);

    bool reset1 = newTmp(&tmp1, b.Even());

    // create full solution

    if (matpcType == QUDA_MATPC_EVEN_EVEN ||
	matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // x_o = M5^-1 (b_o + k D4_oe x_e)
      Dslash4Xpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
      Dslash5inv(x.Odd(), *tmp1, QUDA_ODD_PARITY, kappa5);
    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // x_e = M5^-1 (b_e + k D4_eo x_o)
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
      Dslash5inv(x.Even(), *tmp1, QUDA_EVEN_PARITY, kappa5);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

} // end namespace quda
