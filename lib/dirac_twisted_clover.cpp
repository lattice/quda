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

  void DiracTwistedClover::checkParitySpinor(cvector_ref<const ColorSpinorField> &out,
                                             cvector_ref<const ColorSpinorField> &in) const
  {
    Dirac::checkParitySpinor(out, in);

    for (auto i = 0u; i < out.size(); i++) {
      if (out[i].TwistFlavor() == QUDA_TWIST_SINGLET) {
        if (out[i].Volume() != clover->VolumeCB())
          errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out[i].Volume(),
                    clover->VolumeCB());
      } else {
        if (out[i].Volume() / 2 != clover->VolumeCB())
          errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out[i].Volume(),
                    clover->VolumeCB());
      }
    }
  }

  // Protected method for applying twist
  void DiracTwistedClover::twistedCloverApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                              QudaTwistGamma5Type twistType, QudaParity parity) const
  {
    checkParitySpinor(out, in);
    ApplyTwistClover(out, in, *clover, kappa, mu, epsilon, parity, dagger, twistType);
  }


  // Public method to apply the twist
  void DiracTwistedClover::TwistClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                       QudaParity parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_DIRECT, parity);
  }

  void DiracTwistedClover::Dslash(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity) const
  {
    // this would really just be a Wilson dslash (not actually instantiated at present)
    errorQuda("Not implemented");
  }

  void DiracTwistedClover::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // k * D * in + (A + i*2*(mu+tm_rho)*kappa*gamma_5) *x
      // tm_rho is a Hasenbusch mass preconditioning parameter applied just like a twisted mass
      // but *not* the inverse of M_ee or M_oo
      ApplyTwistedClover(out, in, *gauge, *clover, k, 2 * (mu + tm_rho) * kappa, x, parity, dagger, commDim.data,
                         profile);
    } else {
      // k * D * in + (A + i*2*mu*kappa*gamma_5 * tau_3 - 2 * kappa * epsilon * tau_1 ) * x
      ApplyNdegTwistedClover(out, in, *gauge, *clover, k, 2 * mu * kappa, -2 * kappa * epsilon, x, parity, dagger,
                             commDim.data, profile);
    }
  }

  // apply full operator
  void DiracTwistedClover::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d", in.TwistFlavor());
    }

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      // (-kappa * D + A + i*2*mu*kappa*gamma_5 ) * in
      ApplyTwistedClover(out, in, *gauge, *clover, -kappa, 2.0 * kappa * mu, in, QUDA_INVALID_PARITY, dagger,
                         commDim.data, profile);
    } else {
      // (-kappa * D + A + i*2*mu*kappa*gamma_5*tau_3 - 2*epsilon*kappa*tau_1) * in
      ApplyNdegTwistedClover(out, in, *gauge, *clover, -kappa, 2 * kappa * mu, -2 * kappa * epsilon, in,
                             QUDA_INVALID_PARITY, dagger, commDim.data, profile);
    }
  }

  void DiracTwistedClover::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedClover::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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

  void DiracTwistedClover::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                       const QudaSolutionType) const
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
  void DiracTwistedCloverPC::TwistCloverInv(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                            QudaParity parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_INVERSE, parity);
  }

  void DiracTwistedCloverPC::WilsonDslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                          QudaParity parity) const
  {
    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      DiracWilson::DslashXpay(out, in, parity, in, 0.0);
    } else {
      // we need an Ls=plain 2 Wilson dslash, which is exactly what the 4-d preconditioned DWF operator is
      for (auto i = 0u; i < in.size(); i++)
        ApplyDomainWall4D(out[i], in[i], *gauge, 0.0, 0.0, nullptr, nullptr, in[i], parity, dagger, commDim.data,
                          profile);
    }
  }

  void DiracTwistedCloverPC::WilsonDslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                              QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      DiracWilson::DslashXpay(out, in, parity, x, k);
    } else {
      // we need an Ls=plain 2 Wilson dslash, which is exactly what the 4-d preconditioned DWF operator is
      for (auto i = 0u; i < in.size(); i++)
        ApplyDomainWall4D(out[i], in[i], *gauge, k, 0.0, nullptr, nullptr, x[i], parity, dagger, commDim.data, profile);
    }
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedCloverPC::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                    QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d", in.TwistFlavor());

    if (dagger && symmetric && !reverse) {
      auto tmp = getFieldTmp(out);
      TwistCloverInv(tmp, in, (QudaParity)(1 - parity));
      WilsonDslash(out, tmp, parity);
    } else {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
        ApplyTwistedCloverPreconditioned(out, in, *gauge, *clover, 1.0, -2.0 * kappa * mu, false, in, parity, dagger,
                                         commDim.data, profile);
      } else {
        ApplyNdegTwistedCloverPreconditioned(out, in, *gauge, *clover, 1.0, -2.0 * kappa * mu, 2.0 * kappa * epsilon,
                                             false, in, parity, dagger, commDim.data, profile);
      }
    }
  }

  // xpay version of the above
  void DiracTwistedCloverPC::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                        QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor())
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d", in.TwistFlavor());

    if (dagger && symmetric && !reverse) {
      auto tmp = getFieldTmp(out);
      TwistCloverInv(tmp, in, (QudaParity)(1 - parity));
      WilsonDslashXpay(out, tmp, parity, x, k);
    } else {
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
        ApplyTwistedCloverPreconditioned(out, in, *gauge, *clover, k, -2.0 * kappa * mu, true, x, parity, dagger,
                                         commDim.data, profile);
      } else {
        ApplyNdegTwistedCloverPreconditioned(out, in, *gauge, *clover, k, -2.0 * kappa * mu, 2.0 * kappa * epsilon,
                                             true, x, parity, dagger, commDim.data, profile);
      }
    }
  }

  void DiracTwistedCloverPC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(out);

    if (!symmetric) { // asymmetric preconditioning
      Dslash(tmp, in, other_parity);
      DiracTwistedClover::DslashXpay(out, tmp, this_parity, in, kappa2);
    } else if (!dagger) { // symmetric preconditioning
      Dslash(tmp, in, other_parity);
      DslashXpay(out, tmp, this_parity, in, kappa2);
    } else { // symmetric preconditioning, dagger
      TwistCloverInv(out, in, this_parity);
      reverse = true;
      Dslash(tmp, out, other_parity);
      reverse = false;
      WilsonDslashXpay(out, tmp, this_parity, in, kappa2);
    }
  }

  void DiracTwistedCloverPC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracTwistedCloverPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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
    for (auto i = 0u; i < b.size(); i++) {
      src[i] = x[i][other_parity].create_alias();
      sol[i] = x[i][this_parity].create_alias();

      TwistCloverInv(!symmetric ? static_cast<ColorSpinorField &>(tmp) : src[i], b[i][other_parity], other_parity);

      if (symmetric) {
        // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
        WilsonDslashXpay(tmp, src[i], this_parity, b[i][this_parity], kappa);
      } else {
        // src = b_e + k D_eo A_oo^-1 b_o
        WilsonDslashXpay(src[i], tmp, this_parity, b[i][this_parity], kappa);
      }

      if (symmetric) TwistCloverInv(src[i], tmp, this_parity);
    }
  }

  void DiracTwistedCloverPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                         const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    auto tmp = getFieldTmp(x[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);
      // x_o = A_oo^-1 (b_o + k D_oe x_e)
      WilsonDslashXpay(tmp, x[i][this_parity], other_parity, b[i][other_parity], kappa);
      TwistCloverInv(x[i][other_parity], tmp, other_parity);
    }
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

    if (symmetric) {
      clover->prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE);
    } else {
      clover->prefetch(mem_space, stream, CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE, other_parity);
      clover->prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE, this_parity);
    }
  }
} // namespace quda
