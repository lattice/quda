#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

#define NEW_DSLASH

namespace quda {

  DiracTwistedClover::DiracTwistedClover(const DiracParam &param, const int nDim) :
    DiracWilson(param, nDim),
    mu(param.mu),
    epsilon(param.epsilon),
    clover(*(param.clover))
  {
  }

  DiracTwistedClover::DiracTwistedClover(const DiracTwistedClover &dirac) :
    DiracWilson(dirac),
    mu(dirac.mu),
    epsilon(dirac.epsilon),
    clover(dirac.clover)
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

    if (out.Volume() != clover.VolumeCB())
      errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out.Volume(), clover.VolumeCB());
  }

  // Protected method for applying twist
  void DiracTwistedClover::twistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const QudaTwistGamma5Type twistType, const int parity) const
  {
    checkParitySpinor(out, in);
    ApplyTwistClover(out, in, clover, kappa, mu, 0.0, parity, dagger, twistType);

    if (twistType == QUDA_TWIST_GAMMA5_INVERSE) flops += 1056ll*in.Volume();
    else flops += 552ll*in.Volume();
  }


  // Public method to apply the twist
  void DiracTwistedClover::TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_DIRECT, parity);
  }

  void DiracTwistedClover::Dslash(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity) const
  {
    // this would really just be a Wilson dslash (not actually instantiated at present)
    errorQuda("Not implemented");
  }

  void DiracTwistedClover::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity,
                                      const ColorSpinorField &x, const double &k) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // k * D * in + (1 + i*2*mu*kappa*gamma_5) *x
      ApplyTwistedClover(out, in, *gauge, clover, k, 2 * mu * kappa, x, parity, dagger, commDim, profile);
      flops += (1320ll + 552ll) * in.Volume();

    } else {
      errorQuda("Non-degenerate operator is not implemented");
    }
  }

  void DiracTwistedClover::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d", in.TwistFlavor());
    }

    ApplyTwistedClover(
        out, in, *gauge, clover, -kappa, 2.0 * kappa * mu, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
    flops += (1320ll + 552ll) * in.Volume();
  }

  void DiracTwistedClover::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
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

  void DiracTwistedClover::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracTwistedClover::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    clover.prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE);
  }

  void DiracTwistedClover::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
					  double kappa, double mass, double mu, double mu_factor) const {
    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, &clover, kappa, a, mu_factor, QUDA_TWISTED_CLOVER_DIRAC, QUDA_MATPC_INVALID);
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
      bool reset = newTmp(&tmp2, in);
      TwistCloverInv(*tmp2, in, 1 - parity);
      DiracWilson::Dslash(out, *tmp2, parity);
      deleteTmp(&tmp2, reset);
    } else {
      ApplyTwistedCloverPreconditioned(out, in, *gauge, clover, 1.0, -2.0 * kappa * mu, false, in, parity, dagger,
                                       commDim, profile);
      flops += (1320ll + 552ll) * in.Volume();
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
      bool reset = newTmp(&tmp2, in);
      TwistCloverInv(*tmp2, in, 1 - parity);
      DiracWilson::DslashXpay(out, *tmp2, parity, x, k);
      deleteTmp(&tmp2, reset);
    } else {
      ApplyTwistedCloverPreconditioned(
          out, in, *gauge, clover, k, -2.0 * kappa * mu, true, x, parity, dagger, commDim, profile);
      flops += (1320ll + 552ll) * in.Volume();
    }
  }

  void DiracTwistedCloverPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    bool reset = newTmp(&tmp1, in);

    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (!symmetric) { // asymmetric preconditioning
        Dslash(*tmp1, in, parity[0]);
        DiracTwistedClover::DslashXpay(out, *tmp1, parity[1], in, kappa2);
      } else if (!dagger) { // symmetric preconditioning
        Dslash(*tmp1, in, parity[0]);
        DslashXpay(out, *tmp1, parity[1], in, kappa2);
      } else { // symmetric preconditioning, dagger
        TwistCloverInv(out, in, parity[1]);
        reverse = true;
        Dslash(*tmp1, out, parity[0]);
        reverse = false;
        DiracWilson::DslashXpay(out, *tmp1, parity[1], in, kappa2);
      }
    } else { //Twist doublet
      errorQuda("Non-degenerate operator is not implemented");
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
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

    bool reset = newTmp(&tmp1, b.Even());

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;

    src = odd_bit ? &(x.Even()) : &(x.Odd());
    sol = odd_bit ? &(x.Odd()) : &(x.Even());

    TwistCloverInv(symmetric ? *src : *tmp1, odd_bit ? b.Even() : b.Odd(), odd_bit ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY);

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
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }

    } else { // doublet:
      errorQuda("Non-degenerate operator is not implemented");
    } // end of doublet

    if (symmetric) TwistCloverInv(*src, *tmp1, odd_bit ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY);

    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
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
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }
    } else { // twist doublet:
      errorQuda("Non-degenerate operator is not implemented");
    } // end of twist doublet...

    TwistCloverInv(odd_bit ? x.Even() : x.Odd(), *tmp1, odd_bit ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY);
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
					    double kappa, double mass, double mu, double mu_factor) const {
    double a = -2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, &clover, kappa, a, -mu_factor, QUDA_TWISTED_CLOVERPC_DIRAC, matpcType);
  }

  void DiracTwistedCloverPC::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric) {
      clover.prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE);
    } else {
      clover.prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE, parity[0]);
      clover.prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE, parity[1]);
    }
  }
} // namespace quda
