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

//    ApplyOverlapWilson(out, in, *gauge, es, size, rho, rho, 0.0, in, parity, dagger, commDim, profile);
    flops += 1320ll*in.Volume();
  }

  void DiracOverlapWilson::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

//    ApplyOverlapWilson(out, in, *gauge, es, size, 1+k*rho, k*rho, 0.0, x, parity, dagger, commDim, profile);
    flops += 0ll*in.Volume();
  }

  void DiracOverlapWilson::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    //OVERLAP_TODO

    flops += 0ll * in.Volume();
  }

  void DiracOverlapWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    //OVERLAP_TODO
    
    deleteTmp(&tmp1, reset);
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

} // namespace quda
