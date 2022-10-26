#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

namespace quda {

  DiracTwistedClover::DiracTwistedClover(const DiracParam &param, const int nDim) :
    DiracWilson(param, nDim), mu(param.mu), epsilon(param.epsilon), tm_rho(param.tm_rho), clover(param.clover)
  {
  }

  DiracTwistedClover::DiracTwistedClover(const DiracTwistedClover &dirac) :
    DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon), tm_rho(dirac.tm_rho), clover(dirac.clover)
  {
  }

  DiracTwistedClover::~DiracTwistedClover() { }

  DiracTwistedClover& DiracTwistedClover::operator=(const DiracTwistedClover &dirac)
  {
    if (&dirac != this)
      {
	DiracWilson::operator=(dirac);
	clover = dirac.clover;
      }

    return *this;
  }

  void DiracTwistedClover::checkParitySpinor(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    Dirac::checkParitySpinor(out, in);

    if (out.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (out.Volume() != clover->VolumeCB())
        errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out.Volume(),
                  clover->VolumeCB());
    } else {
      //
      if (out.Volume() / 2 != clover->VolumeCB())
        errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out.Volume(),
                  clover->VolumeCB());
    }
  }

  // Protected method for applying twist
  void DiracTwistedClover::twistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const QudaTwistGamma5Type twistType, const int parity) const
  {
    checkParitySpinor(out, in);
    ApplyTwistClover(out, in, *clover, kappa, mu, epsilon, parity, dagger, twistType);

    if (twistType == QUDA_TWIST_GAMMA5_INVERSE)
      flops += (504ll + 504ll + 48ll) * in.Volume();
    else
      flops += (504ll + 48ll) * in.Volume();
  }


  // Public method to apply the twist
  void DiracTwistedClover::TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_DIRECT, parity);
  }

  void DiracTwistedClover::Dslash(ColorSpinorField &, const ColorSpinorField &, QudaParity) const
  {
    // this would really just be a Wilson dslash (not actually instantiated at present)
    errorQuda("Not implemented");
  }

  void DiracTwistedClover::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity,
                                      const ColorSpinorField &x, const double &k) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // k * D * in + (A + i*2*(mu+tm_rho)*kappa*gamma_5) *x
      // tm_rho is a Hasenbusch mass preconditioning parameter applied just like a twisted mass
      // but *not* the inverse of M_ee or M_oo
      ApplyTwistedClover(out, in, *gauge, *clover, k, 2 * (mu + tm_rho) * kappa, x, parity, dagger, commDim, profile);
      // wilson + chiral twist + clover
      flops += (1320ll + 48ll + 504ll) * in.Volume();

    } else {
      // k * D * in + (A + i*2*mu*kappa*gamma_5 * tau_3 - 2 * kappa * epsilon * tau_1 ) * x
      ApplyNdegTwistedClover(out, in, *gauge, *clover, k, 2 * mu * kappa, -2 * kappa * epsilon, x, parity, dagger,
                             commDim, profile);
      // wilson + chiral twist + flavour twist + clover
      flops += (1320ll + 48ll + 48ll + 504ll) * in.Volume();
    }
  }

  // apply full operator
  void DiracTwistedClover::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d", in.TwistFlavor());
    }

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // (-kappa * D + A + i*2*mu*kappa*gamma_5 ) * in
      ApplyTwistedClover(out, in, *gauge, *clover, -kappa, 2.0 * kappa * mu, in, QUDA_INVALID_PARITY, dagger, commDim,
                         profile);
      // wilson + chiral twist + clover
      flops += (1320ll + 48ll + 504ll) * in.Volume();
    } else {
      // (-kappa * D + A + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1) * in
      ApplyNdegTwistedClover(out, in, *gauge, *clover, -kappa, 2 * kappa * mu, -2 * kappa * epsilon, in,
                             QUDA_INVALID_PARITY, dagger, commDim, profile);
      // wilson + chiral twist + flavor twist + clover
      flops += (1320ll + 48ll + 48ll + 504ll) * in.Volume();
    }
  }

  void DiracTwistedClover::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(in);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedClover::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                   ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracTwistedClover::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
  {
    // do nothing
  }

  void DiracTwistedClover::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    clover->prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE);
  }

  void DiracTwistedClover::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double,
                                          double mu, double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, clover, kappa, mass, a, mu_factor, QUDA_TWISTED_CLOVER_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracTwistedCloverPC &dirac) :
      DiracTwistedClover(dirac),
      reverse(false)
  {
  }

  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracParam &param, const int nDim) :
      DiracTwistedClover(param, nDim),
      reverse(false)
  {
  }

  DiracTwistedCloverPC::~DiracTwistedCloverPC()
  {

  }

  DiracTwistedCloverPC& DiracTwistedCloverPC::operator=(const DiracTwistedCloverPC &dirac)
  {
    if (&dirac != this) {
      DiracTwistedClover::operator=(dirac);
    }
    return *this;
  }

  // Public method to apply the inverse twist
  void DiracTwistedCloverPC::TwistCloverInv(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_INVERSE, parity);
  }

  void DiracTwistedCloverPC::WilsonDslash(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity) const
  {
    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      DiracWilson::DslashXpay(out, in, parity, in, 0.0);
    } else {
      // we need an Ls=plain 2 Wilson dslash, which is exactly what the 4-d preconditioned DWF operator is
      ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, parity, dagger, commDim, profile);
    }
  }

  void DiracTwistedCloverPC::WilsonDslashXpay(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity,
                                              const ColorSpinorField &x, double k) const
  {
    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      DiracWilson::DslashXpay(out, in, parity, x, k);
    } else {
      // we need an Ls=plain 2 Wilson dslash, which is exactly what the 4-d preconditioned DWF operator is
      ApplyDomainWall4D(out, in, *gauge, k, 0.0, nullptr, nullptr, x, parity, dagger, commDim, profile);
    }
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedCloverPC::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d", in.TwistFlavor());

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    if (dagger && symmetric && !reverse) {
      auto tmp = getFieldTmp(in);
      TwistCloverInv(tmp, in, 1 - parity);
      WilsonDslash(out, tmp, parity);
    } else {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
        ApplyTwistedCloverPreconditioned(out, in, *gauge, *clover, 1.0, -2.0 * kappa * mu, false, in, parity, dagger,
                                         commDim, profile);
        flops += (1320ll + 48ll + 504ll) * in.Volume();
      } else {
        ApplyNdegTwistedCloverPreconditioned(out, in, *gauge, *clover, 1.0, -2.0 * kappa * mu, 2.0 * kappa * epsilon,
                                             false, in, parity, dagger, commDim, profile);
        flops += (1320ll + 48ll + 48ll + 504ll) * in.Volume();
      }
    }
  }

  // xpay version of the above
  void DiracTwistedCloverPC::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d", in.TwistFlavor());

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    if (dagger && symmetric && !reverse) {
      auto tmp = getFieldTmp(in);
      TwistCloverInv(tmp, in, 1 - parity);
      WilsonDslashXpay(out, tmp, parity, x, k);
    } else {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
        ApplyTwistedCloverPreconditioned(out, in, *gauge, *clover, k, -2.0 * kappa * mu, true, x, parity, dagger,
                                         commDim, profile);
        flops += (1320ll + 48ll + 504ll) * in.Volume();
      } else {
        ApplyNdegTwistedCloverPreconditioned(out, in, *gauge, *clover, k, -2.0 * kappa * mu, 2.0 * kappa * epsilon,
                                             true, x, parity, dagger, commDim, profile);
        flops += (1320ll + 48ll + 48ll + 504ll) * in.Volume();
      }
    }
  }

  void DiracTwistedCloverPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(in);

    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (!symmetric) { // asymmetric preconditioning
      Dslash(tmp, in, parity[0]);
      DiracTwistedClover::DslashXpay(out, tmp, parity[1], in, kappa2);
    } else if (!dagger) { // symmetric preconditioning
      Dslash(tmp, in, parity[0]);
      DslashXpay(out, tmp, parity[1], in, kappa2);
    } else { // symmetric preconditioning, dagger
      TwistCloverInv(out, in, parity[1]);
      reverse = true;
      Dslash(tmp, out, parity[0]);
      reverse = false;
      WilsonDslashXpay(out, tmp, parity[1], in, kappa2);
    }
  }

  void DiracTwistedCloverPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedCloverPC::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                     ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
      return;
    }

    auto tmp = getFieldTmp(b.Even());

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;

    src = odd_bit ? &(x.Even()) : &(x.Odd());
    sol = odd_bit ? &(x.Odd()) : &(x.Even());

    TwistCloverInv(symmetric ? *src : static_cast<ColorSpinorField &>(tmp), odd_bit ? b.Even() : b.Odd(),
                   odd_bit ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
      WilsonDslashXpay(tmp, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
      WilsonDslashXpay(tmp, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
    } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // src = b_e + k D_eo A_oo^-1 b_o
      WilsonDslashXpay(*src, tmp, QUDA_EVEN_PARITY, b.Even(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // src = b_o + k D_oe A_ee^-1 b_e
      WilsonDslashXpay(*src, tmp, QUDA_ODD_PARITY, b.Odd(), kappa);
    } else {
      errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
    }

    if (symmetric) TwistCloverInv(*src, tmp, odd_bit ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY);

    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want
  }

  void DiracTwistedCloverPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
					 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    checkFullSpinor(x, b);
    auto tmp = getFieldTmp(b.Even());
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;

    if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // x_o = A_oo^-1 (b_o + k D_oe x_e)
      WilsonDslashXpay(tmp, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // x_e = A_ee^-1 (b_e + k D_eo x_o)
      WilsonDslashXpay(tmp, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    } else {
      errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
    }

    TwistCloverInv(odd_bit ? x.Even() : x.Odd(), tmp, odd_bit ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY);
  }

  void DiracTwistedCloverPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double,
                                            double mu, double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = -2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, clover, kappa, mass, a, -mu_factor, QUDA_TWISTED_CLOVERPC_DIRAC, matpcType);
  }

  void DiracTwistedCloverPC::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric) {
      clover->prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE);
    } else {
      clover->prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE, parity[0]);
      clover->prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE, parity[1]);
    }
  }
} // namespace quda
