#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>
#include <dslash_quda.h>

namespace quda {

  GaugeLaplace::GaugeLaplace(const DiracParam &param) :  Dirac(param) { }

  GaugeLaplace::GaugeLaplace(const GaugeLaplace &laplace) :  Dirac(laplace) { }

  GaugeLaplace::~GaugeLaplace() { }

  GaugeLaplace& GaugeLaplace::operator=(const GaugeLaplace &laplace)
  {
    if (&laplace != this) Dirac::operator=(laplace);
    return *this;
  }

  void GaugeLaplace::Dslash(ColorSpinorField &out, const ColorSpinorField &in,  const QudaParity parity) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for directions with a derivative
    for (int i = 0; i < 4; i++) {
      comm_dim[i] = comm_dim_partitioned(i);
      if (laplace3D == i) comm_dim[i] = 0;
    }
    ApplyLaplace(out, in, *gauge, laplace3D, 1.0, 1.0, in, parity, dagger, comm_dim, profile);
    flops += 1320ll*in.Volume(); // FIXME
  }

  void GaugeLaplace::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for directions with a derivative
    for (int i = 0; i < 4; i++) {
      comm_dim[i] = comm_dim_partitioned(i);
      if (laplace3D == i) comm_dim[i] = 0;
    }
    ApplyLaplace(out, in, *gauge, laplace3D, k, 1.0, x, parity, dagger, comm_dim, profile);
    flops += 1368ll*in.Volume(); // FIXME
  }

  void GaugeLaplace::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, -kappa);
  }

  void GaugeLaplace::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void GaugeLaplace::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b, 
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void GaugeLaplace::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    // do nothing
  }

  GaugeLaplacePC::GaugeLaplacePC(const DiracParam &param) : GaugeLaplace(param) { }

  GaugeLaplacePC::GaugeLaplacePC(const GaugeLaplacePC &dirac) : GaugeLaplace(dirac) { }

  GaugeLaplacePC::~GaugeLaplacePC() { }

  GaugeLaplacePC& GaugeLaplacePC::operator=(const GaugeLaplacePC &laplace)
  {
    if (&laplace != this) GaugeLaplace::operator=(laplace);
    return *this;
  }

  void GaugeLaplacePC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;

    bool reset = newTmp(&tmp1, in);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for GaugeLaplacePC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void GaugeLaplacePC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void GaugeLaplacePC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			      ColorSpinorField &x, ColorSpinorField &b, 
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
	errorQuda("MatPCType %d not valid for GaugeLaplacePC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void GaugeLaplacePC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = b_o + k D_oe x_e
      DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = b_e + k D_eo x_o
      DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    } else {
      errorQuda("MatPCType %d not valid for GaugeLaplacePC", matpcType);
    }
  }

} // namespace quda
