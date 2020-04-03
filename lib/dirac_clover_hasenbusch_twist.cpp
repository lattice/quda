#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracCloverHasenbuschTwist::DiracCloverHasenbuschTwist(const DiracParam &param) : DiracClover(param), mu(param.mu) {}

  DiracCloverHasenbuschTwist::DiracCloverHasenbuschTwist(const DiracCloverHasenbuschTwist &dirac) :
    DiracClover(dirac),
    mu(dirac.mu)
  {
  }

  DiracCloverHasenbuschTwist::~DiracCloverHasenbuschTwist() {}

  DiracCloverHasenbuschTwist &DiracCloverHasenbuschTwist::operator=(const DiracCloverHasenbuschTwist &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      clover = dirac.clover;
      mu = dirac.mu;
    }
    return *this;
  }

  void DiracCloverHasenbuschTwist::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool asymmetric = (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) || (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC);

    if (!asymmetric) {
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        ApplyWilsonCloverHasenbuschTwist(out.Even(), in.Odd(), *gauge, clover, -kappa, mu, in.Even(), QUDA_EVEN_PARITY,
                                         dagger, commDim, profile);
        ApplyWilsonClover(out.Odd(), in.Even(), *gauge, clover, -kappa, in.Odd(), QUDA_ODD_PARITY, dagger, commDim,
                          profile);
      } else {
        ApplyWilsonClover(out.Even(), in.Odd(), *gauge, clover, -kappa, in.Even(), QUDA_EVEN_PARITY, dagger, commDim,
                          profile);
        ApplyWilsonCloverHasenbuschTwist(out.Odd(), in.Even(), *gauge, clover, -kappa, mu, in.Odd(), QUDA_ODD_PARITY,
                                         dagger, commDim, profile);
      }

      // 2 c/b applies of DiracClover + (1-imu gamma_5 A)psi_{!p}
      flops += 2 * 1872ll * in.VolumeCB() + (48ll + 504ll) * in.VolumeCB();
    } else {
      if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        ApplyWilsonClover(out.Even(), in.Odd(), *gauge, clover, -kappa, in.Even(), QUDA_EVEN_PARITY, dagger, commDim,
                          profile);
        ApplyTwistedClover(out.Odd(), in.Even(), *gauge, clover, -kappa, mu, in.Odd(), QUDA_ODD_PARITY, dagger, commDim,
                          profile);
      } else {
        ApplyTwistedClover(out.Even(), in.Odd(), *gauge, clover, -kappa, mu, in.Even(), QUDA_EVEN_PARITY, dagger,
                          commDim, profile);
        ApplyWilsonClover(out.Odd(), in.Even(), *gauge, clover, -kappa, in.Odd(), QUDA_ODD_PARITY, dagger, commDim,
                          profile);
      }
      // 2 c/b applies of DiracClover + (1-imu gamma_5)psi_{!p}
      flops += 2 * 1872ll * in.VolumeCB() + 48ll * in.VolumeCB();
    }
  }

  void DiracCloverHasenbuschTwist::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracCloverHasenbuschTwist::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa,
                                                  double mass, double mu, double mu_factor) const
  {
    // double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    // CoarseOp(Y, X, T, *gauge, &clover, kappa, a, mu_factor, QUDA_CLOVER_DIRAC, QUDA_MATPC_INVALID);
    errorQuda("Not Yet Implemented");
  }

  /* **********************************************
   * DiracCloverHasenbuschTwistPC Starts Here
   * ********************************************* */

  DiracCloverHasenbuschTwistPC::DiracCloverHasenbuschTwistPC(const DiracParam &param) :
    DiracCloverPC(param),
    mu(param.mu)
  {
  }

  DiracCloverHasenbuschTwistPC::DiracCloverHasenbuschTwistPC(const DiracCloverHasenbuschTwistPC &dirac) :
    DiracCloverPC(dirac),
    mu(dirac.mu)
  {
  }

  DiracCloverHasenbuschTwistPC::~DiracCloverHasenbuschTwistPC() {}

  DiracCloverHasenbuschTwistPC &DiracCloverHasenbuschTwistPC::operator=(const DiracCloverHasenbuschTwistPC &dirac)
  {
    if (&dirac != this) {
      DiracCloverPC::operator=(dirac);
      mu = dirac.mu;
    }
    return *this;
  }

  // xpay version of the above
  void DiracCloverHasenbuschTwistPC::DslashXpayTwistClovInv(ColorSpinorField &out, const ColorSpinorField &in,
                                                            const QudaParity parity, const ColorSpinorField &x,
                                                            const double &k, const double &b) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilsonCloverHasenbuschTwistPCClovInv(out, in, *gauge, clover, k, b, x, parity, dagger, commDim, profile);

    // DiracCloverPC.DslashXPay -/+ mu ( i gamma_5 ) A
    flops += (1872ll + 48ll + 504ll) * in.Volume();
  }

  // xpay version of the above
  void DiracCloverHasenbuschTwistPC::DslashXpayTwistNoClovInv(ColorSpinorField &out, const ColorSpinorField &in,
                                                              const QudaParity parity, const ColorSpinorField &x,
                                                              const double &k, const double &b) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilsonCloverHasenbuschTwistPCNoClovInv(out, in, *gauge, clover, k, b, x, parity, dagger, commDim, profile);

    //    DiracCloverPC.DslashXPay -/+ mu ( i gamma_5 )
    flops += (1872ll + 48) * in.Volume();
  }

  // Apply the even-odd preconditioned clover-improved Dirac operator
  void DiracCloverHasenbuschTwistPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa * kappa;
    bool reset1 = newTmp(&tmp1, in);

    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (!symmetric) {
      // No need to change order of calls for dagger
      // because the asymmetric operator is actually symmetric
      // A_oo -D_oe A^{-1}_ee D_eo -> A_oo -D^\dag_oe A^{-1}_ee D^\dag_eo
      // the pieces in Dslash and DslashXPay respect the dagger

      // DiracCloverHasenbuschTwistPC::Dslash applies A^{-1}Dslash
      Dslash(*tmp1, in, parity[0]);

      // applies (A + imu*g5 - kappa^2 D)-
      ApplyTwistedClover(out, *tmp1, *gauge, clover, kappa2, mu, in, parity[1], dagger, commDim, profile);
      flops += 1872ll * in.Volume();
    } else if (!dagger) { // symmetric preconditioning
      // We need two cases because M = 1-ADAD and M^\dag = 1-D^\dag A D^dag A
      // where A is actually a clover inverse.

      // This is the non-dag case: AD
      Dslash(*tmp1, in, parity[0]);

      // Then x + AD (AD)
      DslashXpayTwistClovInv(out, *tmp1, parity[1], in, kappa2, mu);
    } else { // symmetric preconditioning, dagger
      // This is the dagger: 1 - DADA
      //  i) Apply A
      CloverInv(out, in, parity[1]);
      // ii) Apply A D => ADA
      Dslash(*tmp1, out, parity[0]);
      // iii) Apply  x + D(ADA)
      DslashXpayTwistNoClovInv(out, *tmp1, parity[1], in, kappa2, mu);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracCloverHasenbuschTwistPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    // and for multi-gpu the input and output fields cannot alias
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracCloverHasenbuschTwistPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa,
                                                    double mass, double mu, double mu_factor) const
  {
    // double a = - 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    // CoarseOp(Y, X, T, *gauge, &clover, kappa, a, -mu_factor,QUDA_CLOVERPC_DIRAC, matpcType);
    errorQuda("Not yet implemented\n");
  }

} // namespace quda
