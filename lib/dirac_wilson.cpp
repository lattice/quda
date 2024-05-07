#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracWilson::DiracWilson(const DiracParam &param) : Dirac(param) { }

  DiracWilson::DiracWilson(const DiracWilson &dirac) : Dirac(dirac) { }

  // hack (for DW and TM operators)
  DiracWilson::DiracWilson(const DiracParam &param, const int) : Dirac(param) { }

  DiracWilson::~DiracWilson() { }

  DiracWilson& DiracWilson::operator=(const DiracWilson &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
    }
    return *this;
  }

  void DiracWilson::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, 0.0, in, parity, dagger, commDim.data, profile);
  }

  void DiracWilson::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, k, x, parity, dagger, commDim.data, profile);
  }

  void DiracWilson::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    ApplyWilson(out, in, *gauge, -kappa, in, QUDA_INVALID_PARITY, dagger, commDim.data, profile);
  }

  void DiracWilson::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracWilson::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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

  void DiracWilson::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                const QudaSolutionType) const
  {
    // do nothing
  }

  void DiracWilson::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                   double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Wilson-type operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CloverField *c = nullptr;
    CoarseOp(Y, X, T, *gauge, c, kappa, mass, a, mu_factor, QUDA_WILSON_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracWilsonPC::DiracWilsonPC(const DiracParam &param)
    : DiracWilson(param)
  {

  }

  DiracWilsonPC::DiracWilsonPC(const DiracWilsonPC &dirac) : DiracWilson(dirac) { }

  DiracWilsonPC::~DiracWilsonPC()
  {

  }

  DiracWilsonPC& DiracWilsonPC::operator=(const DiracWilsonPC &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
    }
    return *this;
  }

  void DiracWilsonPC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(out);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(tmp, in, QUDA_ODD_PARITY);
      DslashXpay(out, tmp, QUDA_EVEN_PARITY, in, kappa2);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(tmp, in, QUDA_EVEN_PARITY);
      DslashXpay(out, tmp, QUDA_ODD_PARITY, in, kappa2);
    } else {
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }
  }

  void DiracWilsonPC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracWilsonPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
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
      DslashXpay(x[i][other_parity], b[i][other_parity], this_parity, b[i][this_parity], kappa);
      src[i] = x[i][other_parity].create_alias();
      sol[i] = x[i][this_parity].create_alias();
    }
  }

  void DiracWilsonPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    // create full solution
    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);
      // x_o = b_o + k D_oe x_e
      DslashXpay(x[i][other_parity], x[i][this_parity], other_parity, b[i][other_parity], kappa);
    }
  }

} // namespace quda
