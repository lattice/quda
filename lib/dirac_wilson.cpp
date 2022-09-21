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

  void DiracWilson::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, 0.0, in, parity, dagger, commDim, profile);
    flops += 1320ll*in.Volume();
  }

  void DiracWilson::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                               const ColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, k, x, parity, dagger, commDim, profile);
    flops += 1368ll*in.Volume();
  }

  void DiracWilson::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    ApplyWilson(out, in, *gauge, -kappa, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
    flops += 1368ll * in.Volume();
  }

  void DiracWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracWilson::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x, ColorSpinorField &b,
                            const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracWilson::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
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

  void DiracWilsonPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(in);

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

  void DiracWilsonPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracWilsonPC::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x, ColorSpinorField &b,
                              const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {
      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	// src = b_e + k D_eo b_o
	DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
	src = &(x.Odd());
	sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	// src = b_o + k D_oe b_e
	DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
	src = &(x.Even());
	sol = &(x.Odd());
      } else {
	errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracWilsonPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    // create full solution

    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = b_o + k D_oe x_e
      DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = b_e + k D_eo x_o
      DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    } else {
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }
  }

} // namespace quda
