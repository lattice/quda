#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracClover::DiracClover(const DiracParam &param) : DiracWilson(param), clover(param.clover) {}

  DiracClover::DiracClover(const DiracClover &dirac) : DiracWilson(dirac), clover(dirac.clover) {}

  DiracClover::~DiracClover() { }

  DiracClover& DiracClover::operator=(const DiracClover &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      clover = dirac.clover;
    }
    return *this;
  }

  void DiracClover::checkParitySpinor(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    Dirac::checkParitySpinor(out, in);

    if (out.Volume() != clover->VolumeCB()) {
      errorQuda("Parity spinor volume %lu doesn't match clover checkboard volume %lu", out.Volume(), clover->VolumeCB());
    }
  }

  /** Applies the operator (A + k D) */
  void DiracClover::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (useDistancePC()) {
      ApplyWilsonCloverDistance(out, in, *gauge, *clover, k, distance_pc_alpha0, distance_pc_t0, x, parity, dagger,
                                commDim.data, profile);
    } else {
      ApplyWilsonClover(out, in, *gauge, *clover, k, x, parity, dagger, commDim.data, profile);
    }
  }

  // Public method to apply the clover term only
  void DiracClover::Clover(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    ApplyClover(out, in, *clover, false, parity);
  }

  void DiracClover::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (useDistancePC()) {
      ApplyWilsonCloverDistance(out, in, *gauge, *clover, -kappa, distance_pc_alpha0, distance_pc_t0, in,
                                QUDA_INVALID_PARITY, dagger, commDim.data, profile);
    } else {
      ApplyWilsonClover(out, in, *gauge, *clover, -kappa, in, QUDA_INVALID_PARITY, dagger, commDim.data, profile);
    }
  }

  void DiracClover::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(in);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracClover::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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

  void DiracClover::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                const QudaSolutionType) const
  {
  }

  void DiracClover::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                   double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, clover, kappa, mass, a, mu_factor, QUDA_CLOVER_DIRAC, QUDA_MATPC_INVALID);
  }

  void DiracClover::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    clover->prefetch(mem_space, stream, CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE);
  }

  /*******
   * DiracCloverPC Starts here
   *******/
  DiracCloverPC::DiracCloverPC(const DiracParam &param) : 
    DiracClover(param)
  {
    // For the preconditioned operator, we need to check that the inverse of the clover term is present
    if (!clover->Inverse() && !clover::dynamic_inverse()) errorQuda("Clover inverse required for DiracCloverPC");
  }

  DiracCloverPC::DiracCloverPC(const DiracCloverPC &dirac) : DiracClover(dirac) { }

  DiracCloverPC::~DiracCloverPC() { }

  DiracCloverPC& DiracCloverPC::operator=(const DiracCloverPC &dirac)
  {
    if (&dirac != this) {
      DiracClover::operator=(dirac);
    }
    return *this;
  }

  // Public method
  void DiracCloverPC::CloverInv(ColorSpinorField &out, const ColorSpinorField &in, 
				const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    ApplyClover(out, in, *clover, true, parity);
  }

  // apply hopping term, then clover: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (A_ee^-1 D^dagger_eo) or (A_oo^-1 D^dagger_oe)
  // NOTE - this isn't Dslash dagger since order should be reversed!
  void DiracCloverPC::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			     const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (useDistancePC()) {
      ApplyWilsonCloverPreconditionedDistance(out, in, *gauge, *clover, 0.0, distance_pc_alpha0, distance_pc_t0, in,
                                              parity, dagger, commDim.data, profile);
    } else {
      ApplyWilsonCloverPreconditioned(out, in, *gauge, *clover, 0.0, in, parity, dagger, commDim.data, profile);
    }
  }

  // xpay version of the above
  void DiracCloverPC::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				 const QudaParity parity, const ColorSpinorField &x,
				 const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (useDistancePC()) {
      ApplyWilsonCloverPreconditionedDistance(out, in, *gauge, *clover, k, distance_pc_alpha0, distance_pc_t0, x,
                                              parity, dagger, commDim.data, profile);
    } else {
      ApplyWilsonCloverPreconditioned(out, in, *gauge, *clover, k, x, parity, dagger, commDim.data, profile);
    }
  }

  // Apply the even-odd preconditioned clover-improved Dirac operator
  void DiracCloverPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(in);

    if (!symmetric) {

      // No need to change order of calls for dagger
      // because the asymmetric operator is actually symmetric
      // A_oo -D_oe A^{-1}_ee D_eo -> A_oo -D^\dag_oe A^{-1}_ee D^\dag_eo
      // the pieces in Dslash and DslashXPay respect the dagger

      // DiracCloverPC::Dslash applies A^{-1}Dslash
      Dslash(tmp, in, other_parity);
      // DiracClover::DslashXpay applies (A - kappa^2 D)
      DiracClover::DslashXpay(out, tmp, this_parity, in, kappa2);
    } else if (!dagger) { // symmetric preconditioning
      // We need two cases because M = 1-ADAD and M^\dag = 1-D^\dag A D^dag A
      // where A is actually a clover inverse.

      // This is the non-dag case: AD
      Dslash(tmp, in, other_parity);

      // Then x + AD (AD)
      DslashXpay(out, tmp, this_parity, in, kappa2);
    } else { // symmetric preconditioning, dagger

      // This is the dagger: 1 - DADA
      //  i) Apply A
      CloverInv(out, in, this_parity);
      // ii) Apply A D => ADA
      Dslash(tmp, out, other_parity);
      // iii) Apply  x + D(ADA)
      DiracWilson::DslashXpay(out, tmp, this_parity, in, kappa2);
    }
  }

  void DiracCloverPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    // and for multi-gpu the input and output fields cannot alias
    auto tmp = getFieldTmp(in);

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracCloverPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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
    auto tmp = getFieldTmp(b[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      if (symmetric) {
        // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
        src[i] = x[i][other_parity].create_alias();
        CloverInv(src[i], b[i][other_parity], other_parity);
        DiracWilson::DslashXpay(tmp, src[i], this_parity, b[i][this_parity], kappa);
        CloverInv(src[i], tmp, this_parity);
        sol[i] = x[i][this_parity].create_alias();
      } else {
        // src = b_e + k D_eo A_oo^-1 b_o
        src[i] = x[i][other_parity].create_alias();
        CloverInv(tmp, b[i][other_parity], other_parity); // safe even when tmp = b.odd
        DiracWilson::DslashXpay(src[i], tmp, this_parity, b[this_parity], kappa);
        sol[i] = x[i][this_parity].create_alias();
      }
    }
  }

  void DiracCloverPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    auto tmp = getFieldTmp(b[0].Even());
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);
      // x_o = A_oo^-1 (b_o + k D_oe x_e)
      DiracWilson::DslashXpay(tmp, x[i][this_parity], other_parity, b[i][other_parity], kappa);
      CloverInv(x[i][other_parity], tmp, other_parity);
    }
  }

  void DiracCloverPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                     double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = - 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, T, *gauge, clover, kappa, mass, a, -mu_factor, QUDA_CLOVERPC_DIRAC, matpcType);
  }

  void DiracCloverPC::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
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
