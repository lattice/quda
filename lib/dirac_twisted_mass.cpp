#include <dirac_quda.h>
#include <dslash_quda.h>
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
  void DiracTwistedMass::twistedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
				      const QudaTwistGamma5Type twistType) const
  {
    checkParitySpinor(out, in);
    ApplyTwistGamma(out, in, 4, kappa, mu, epsilon, dagger, twistType);
  }

  // Public method to apply the twist
  void DiracTwistedMass::Twist(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
  }

  void DiracTwistedMass::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                QudaParity parity) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // this would really just be a Wilson dslash (not actually instantiated at present)
      ApplyTwistedMass(out, in, *gauge, 0.0, 2 * mu * kappa, in, parity, dagger, commDim.data, profile);
    } else {
      // this would really just be a 2-way vectorized Wilson dslash (not actually instantiated at present)
      ApplyNdegTwistedMass(out, in, *gauge, 0.0, 2 * mu * kappa, -2 * kappa * epsilon, in, parity, dagger, commDim.data,
                           profile);
    }
  }

  void DiracTwistedMass::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                    QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // k * D * in + (1 + i*2*mu*kappa*gamma_5) *x
      ApplyTwistedMass(out, in, *gauge, k, 2 * mu * kappa, x, parity, dagger, commDim.data, profile);
    } else {
      // k * D * in + (1 + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1) * x
      ApplyNdegTwistedMass(out, in, *gauge, k, 2 * mu * kappa, -2 * kappa * epsilon, x, parity, dagger, commDim.data,
                           profile);
    }
  }

  // apply full operator  / (-kappa * D + (1 + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1)) * in
  void DiracTwistedMass::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());
    }

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      ApplyTwistedMass(out, in, *gauge, -kappa, 2 * mu * kappa, in, QUDA_INVALID_PARITY, dagger, commDim.data, profile);
    } else {
      ApplyNdegTwistedMass(out, in, *gauge, -kappa, 2 * mu * kappa, -2 * kappa * epsilon, in, QUDA_INVALID_PARITY,
                           dagger, commDim.data, profile);
    }
  }

  void DiracTwistedMass::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedMass::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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

  void DiracTwistedMass::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                     const QudaSolutionType) const
  {
    // do nothing
  }

  void DiracTwistedMass::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double,
                                        double mu, double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu;
    CloverField *c = nullptr;
    CoarseOp(Y, X, T, *gauge, c, kappa, mass, a, mu_factor, QUDA_TWISTED_MASS_DIRAC, QUDA_MATPC_INVALID);
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
  void DiracTwistedMassPC::TwistInv(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    twistedApply(out, in,  QUDA_TWIST_GAMMA5_INVERSE);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo A_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedMassPC::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                  QudaParity parity) const
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
      ApplyTwistedMassPreconditioned(out, in, *gauge, b, a, false, in, parity, dagger, asymmetric, commDim.data, profile);
    } else {//TWIST doublet :
      double a = 2.0 * kappa * mu;
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a * a - b * b);

      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyNdegTwistedMassPreconditioned(out, in, *gauge, c, -2.0 * mu * kappa, 2.0 * kappa * epsilon, false, in,
                                         parity, dagger, asymmetric, commDim.data, profile);
    }
  }

  // xpay version of the above
  void DiracTwistedMassPC::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
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
      ApplyTwistedMassPreconditioned(out, in, *gauge, b, a, true, x, parity, dagger, asymmetric, commDim.data, profile);
    } else {//TWIST_DOUBLET:
      double a = 2.0 * kappa * mu;
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a * a - b * b);

      bool asymmetric
          = (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) && dagger;
      ApplyNdegTwistedMassPreconditioned(out, in, *gauge, k * c, -2 * mu * kappa, 2 * kappa * epsilon, true, x, parity,
                                         dagger, asymmetric, commDim.data, profile);
    }
  }

  void DiracTwistedMassPC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(out);

    if (symmetric) {
      Dslash(tmp, in, other_parity);
      DslashXpay(out, tmp, this_parity, in, kappa2);
    } else { // asymmetric preconditioning
      Dslash(tmp, in, other_parity);
      DiracTwistedMass::DslashXpay(out, tmp, this_parity, in, kappa2);
    }
  }

  void DiracTwistedMassPC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedMassPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                   cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                   const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      for (auto i = 0u; i < b.size(); i++) {
        src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
        sol[i] = x[i].create_alias();
      }
      return;
    }

    // we desire solution to full system
    auto tmp = getFieldTmp(x[0].Even());
    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = x[i][other_parity].create_alias();
      sol[i] = x[i][this_parity].create_alias();

      TwistInv(symmetric ? src[i] : static_cast<ColorSpinorField &>(tmp), b[i][other_parity]);

      if (b[0].TwistFlavor() == QUDA_TWIST_SINGLET) {

        if (symmetric) {
          // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
          DiracWilson::DslashXpay(tmp, src[i], this_parity, b[i][this_parity], kappa);
        } else {
          // src = b_e + k D_eo A_oo^-1 b_o
          DiracWilson::DslashXpay(src[i], tmp, this_parity, b[i][this_parity], kappa);
        }

      } else { // doublet:

        // repurpose the preconditioned dslash as a vectorized operator: 1+kappa*D
        double mu_ = mu;
        mu = 0.0;
        double epsilon_ = epsilon;
        epsilon = 0.0;

        if (symmetric) {
          // src = A_ee^-1(b_e + k D_eo A_oo^-1 b_o)
          DslashXpay(tmp, src[i], this_parity, b[i][this_parity], kappa);
        } else {
          // src = b_e + k D_eo A_oo^-1 b_o
          DslashXpay(src[i], tmp, this_parity, b[i][this_parity], kappa);
        }

        mu = mu_;
        epsilon = epsilon_;

      } // end of doublet

      if (symmetric) TwistInv(src[i], tmp);
    }
  }

  void DiracTwistedMassPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    auto tmp = getFieldTmp(x[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);

      // create full solution
      if (b[0].TwistFlavor() == QUDA_TWIST_SINGLET) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DiracWilson::DslashXpay(tmp, x[i][this_parity], other_parity, b[i][other_parity], kappa);
      } else { // twist doublet:
        double mu_ = mu;
        mu = 0.0;
        double epsilon_ = epsilon;
        epsilon = 0.0;

        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DslashXpay(tmp, x[i][this_parity], other_parity, b[i][other_parity], kappa);

        mu = mu_;
        epsilon = epsilon_;
      }

      TwistInv(x[i][other_parity], tmp);
    }
  }

  void DiracTwistedMassPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double,
                                          double mu, double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = -2.0 * kappa * mu;
    CloverField *c = nullptr;
    CoarseOp(Y, X, T, *gauge, c, kappa, mass, a, -mu_factor, QUDA_TWISTED_MASSPC_DIRAC, matpcType);
  }
} // namespace quda
