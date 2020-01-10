#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

namespace quda {

  DiracTwistedMass::DiracTwistedMass(const DiracParam &param, const int nDim) :
      DiracWilson(param, nDim),
      mu(param.mu),
      epsilon(param.epsilon)
  {
  }

  DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) :
      DiracWilson(dirac),
      mu(dirac.mu),
      epsilon(dirac.epsilon)
  {
  }

  DiracTwistedMass::~DiracTwistedMass() { }

  DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
    }
    return *this;
  }

  // Protected method for applying twist
  void DiracTwistedMass::twistedApply(ColorSpinorField &out, const ColorSpinorField &in,
				      const QudaTwistGamma5Type twistType) const
  {
    checkParitySpinor(out, in);
    ApplyTwistGamma(out, in, 4, kappa, mu, epsilon, dagger, twistType);
    flops += 24ll*in.Volume();
  }

  // Public method to apply the twist
  void DiracTwistedMass::Twist(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
  }

  void DiracTwistedMass::Dslash(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // this would really just be a Wilson dslash (not actually instantiated at present)
      ApplyTwistedMass(out, in, *gauge, 0.0, 2 * mu * kappa, in, parity, dagger, commDim, profile);
      flops += 1392ll * in.Volume();
    } else {
      // this would really just be a 2-way vectorized Wilson dslash (not actually instantiated at present)
      ApplyNdegTwistedMass(
          out, in, *gauge, 0.0, 2 * mu * kappa, -2 * kappa * epsilon, in, parity, dagger, commDim, profile);
      flops += (1440ll) * in.Volume();
    }
  }

  void DiracTwistedMass::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // k * D * in + (1 + i*2*mu*kappa*gamma_5) *x
      ApplyTwistedMass(out, in, *gauge, k, 2 * mu * kappa, x, parity, dagger, commDim, profile);
      flops += 1416ll * in.Volume();
    } else {
      // k * D * in + (1 + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1) * x
      ApplyNdegTwistedMass(out, in, *gauge, k, 2 * mu * kappa, -2 * kappa * epsilon, x, parity, dagger, commDim, profile);
      flops += (1464ll) * in.Volume();
    }
  }

  // apply full operator  / (-kappa * D + (1 + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1)) * in
  void DiracTwistedMass::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());
    }

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      ApplyTwistedMass(out, in, *gauge, -kappa, 2 * mu * kappa, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      flops += 1416ll * in.Volume();
    } else {
      ApplyNdegTwistedMass(out, in, *gauge, -kappa, 2 * mu * kappa, -2 * kappa * epsilon, in, QUDA_INVALID_PARITY,
          dagger, commDim, profile);
      flops += (1464ll) * in.Volume();
    }
  }

  void DiracTwistedMass::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMass::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
      ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracTwistedMass::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				     const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracTwistedMass::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
					double kappa, double mass, double mu, double mu_factor) const {
    double a = 2.0 * kappa * mu;
    cudaCloverField *c = NULL;
    CoarseOp(Y, X, T, *gauge, c, kappa, a, mu_factor, QUDA_TWISTED_MASS_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracTwistedMassPC::DiracTwistedMassPC(const DiracTwistedMassPC &dirac) : DiracTwistedMass(dirac) { }

  DiracTwistedMassPC::DiracTwistedMassPC(const DiracParam &param, const int nDim) : DiracTwistedMass(param, nDim){ }

  DiracTwistedMassPC::~DiracTwistedMassPC()
  {

  }

  DiracTwistedMassPC& DiracTwistedMassPC::operator=(const DiracTwistedMassPC &dirac)
  {
    if (&dirac != this) {
      DiracTwistedMass::operator=(dirac);
    }
    return *this;
  }

  // Public method to apply the inverse twist
  void DiracTwistedMassPC::TwistInv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    twistedApply(out, in,  QUDA_TWIST_GAMMA5_INVERSE);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo A_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedMassPC::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * mu; // for inverse twist
      double b = 1.0 / (1.0 + a * a);

      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyTwistedMassPreconditioned(out, in, *gauge, b, a, false, in, parity, dagger, asymmetric, commDim, profile);
      flops += 1392ll * in.Volume(); // flops numbers are approximate since they will vary depending on the dagger or not
    } else {//TWIST doublet :
      double a = 2.0 * kappa * mu;
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a * a - b * b);

      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyNdegTwistedMassPreconditioned(out, in, *gauge, c, -2.0 * mu * kappa, 2.0 * kappa * epsilon, false, in,
          parity, dagger, asymmetric, commDim, profile);
      flops += (1440ll) * in.Volume(); // flops are approx. since they will vary depending on the dagger or not
    }
  }

  // xpay version of the above
  void DiracTwistedMassPC::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
				      const ColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * mu; // for inverse twist
      double b = k / (1.0 + a * a);
      // asymmetric should never be true here since we never need to apply 1 + k * A^{-1} D^\dagger
      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyTwistedMassPreconditioned(out, in, *gauge, b, a, true, x, parity, dagger, asymmetric, commDim, profile);
      flops += 1416ll * in.Volume(); // flops numbers are approximate since they will vary depending on the dagger or not
    } else {//TWIST_DOUBLET:
      double a = 2.0 * kappa * mu;
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a * a - b * b);

      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyNdegTwistedMassPreconditioned(out, in, *gauge, k * c, -2 * mu * kappa, 2 * kappa * epsilon, true, x, parity,
          dagger, asymmetric, commDim, profile);
      flops += (1464ll)
          * in.Volume(); // flops numbers are approximate since they will vary depending on the dagger or not
    }
  }

  void DiracTwistedMassPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    bool reset = newTmp(&tmp1, in);

    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric) {
      Dslash(*tmp1, in, parity[0]);
      DslashXpay(out, *tmp1, parity[1], in, kappa2);
    } else { // asymmetric preconditioning
      Dslash(*tmp1, in, parity[0]);
      DiracTwistedMass::DslashXpay(out, *tmp1, parity[1], in, kappa2);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMassPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracTwistedMassPC::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                   ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
      return;
    }

    bool reset = newTmp(&tmp1, b.Even());

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;

    src = odd_bit ? &(x.Even()) : &(x.Odd());
    sol = odd_bit ? &(x.Odd()) : &(x.Even());

    TwistInv(symmetric ? *src : *tmp1, odd_bit ? b.Even() : b.Odd());

    // we desire solution to full system
    if (b.TwistFlavor() == QUDA_TWIST_SINGLET) {

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }

    } else { // doublet:

      // repurpose the precondiitoned dslash as a vectorized operator: 1+kappa*D
      double mu_ = mu;
      mu = 0.0;
      double epsilon_ = epsilon;
      epsilon = 0.0;

      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = A_ee^-1(b_e + k D_eo A_oo^-1 b_o)
        DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
        DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }

      mu = mu_;
      epsilon = epsilon_;

    } // end of doublet

    if (symmetric) TwistInv(*src, *tmp1);

    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMassPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    checkFullSpinor(x, b);
    bool reset = newTmp(&tmp1, b.Even());
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;

    // create full solution
    if (b.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||   matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)
        DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }
    } else { // twist doublet:
      double mu_ = mu;
      mu = 0.0;
      double epsilon_ = epsilon;
      epsilon = 0.0;

      if (matpcType == QUDA_MATPC_EVEN_EVEN ||  matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||  matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)
        DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }

      mu = mu_;
      epsilon = epsilon_;
    } // end of twist doublet...

    TwistInv(odd_bit ? x.Even() : x.Odd(), *tmp1);
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMassPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
					  double kappa, double mass, double mu, double mu_factor) const {
    double a = -2.0 * kappa * mu;
    cudaCloverField *c = NULL;
    CoarseOp(Y, X, T, *gauge, c, kappa, a, -mu_factor, QUDA_TWISTED_MASSPC_DIRAC, matpcType);
  }
} // namespace quda
