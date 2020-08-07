#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
//#include <multigrid.h> //Not yet...

namespace quda {

  DiracOverlapWilson::DiracOverlapWilson(const DiracParam &param) : DiracWilson(param) { }

  DiracOverlapWilson::DiracOverlapWilson(const DiracOverlapWilson &dirac) : DiracWilson(dirac) { }

  // hack (for DW and TM operators)
  DiracOverlapWilson::DiracOverlapWilson(const DiracParam &param, const int nDims) : DiracWilson(param) { } 

  DiracOverlapWilson::~DiracOverlapWilson() { }

  DiracOverlapWilson& DiracOverlapWilson::operator=(const DiracOverlapWilson &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
    }
    return *this;
  }

  void DiracOverlapWilson::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			   const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, 0.0, in, parity, dagger, commDim, profile);
    flops += 1320ll*in.Volume();
  }

  void DiracOverlapWilson::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyWilson(out, in, *gauge, k, x, parity, dagger, commDim, profile);
    flops += 1368ll*in.Volume();
  }

  void DiracOverlapWilson::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    
    ApplyWilson(out, in, *gauge, -kappa, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
    flops += 1368ll * in.Volume();
  }

  void DiracOverlapWilson::H(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    
    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);
    
    M(*tmp1, in);
    gamma5(out, *tmp1);
    deleteTmp(&tmp1, reset);
  }

  void DiracOverlapWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracOverlapWilson::Hprepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				    ColorSpinorField &x, ColorSpinorField &b, 
				    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracOverlapWilson::Hreconstruct(ColorSpinorField &x, const ColorSpinorField &b,
					const QudaSolutionType solType) const
  {
    // do nothing
  }
  
  void DiracOverlapWilson::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				   ColorSpinorField &x, ColorSpinorField &b, 
				   const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }
    
    src = &b;
    sol = &x;
  }
  
  void DiracOverlapWilson::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    // do nothing
  }

  
  DiracOverlapWilsonPC::DiracOverlapWilsonPC(const DiracParam &param)
    : DiracOverlapWilson(param)
  {

  }

  DiracOverlapWilsonPC::DiracOverlapWilsonPC(const DiracOverlapWilsonPC &dirac) 
    : DiracOverlapWilson(dirac)
  {

  }

  DiracOverlapWilsonPC::~DiracOverlapWilsonPC()
  {

  }

  DiracOverlapWilsonPC& DiracOverlapWilsonPC::operator=(const DiracOverlapWilsonPC &dirac)
  {
    if (&dirac != this) {
      DiracOverlapWilson::operator=(dirac);
    }
    return *this;
  }

  void DiracOverlapWilsonPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
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
      errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracOverlapWilsonPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracOverlapWilsonPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
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
	errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracOverlapWilsonPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
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
      errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
    }
  }

} // namespace quda
