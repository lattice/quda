#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

namespace quda {

  namespace wilson {
#include <dslash_init.cuh>
  }

  DiracWilson::DiracWilson(const DiracParam &param) : Dirac(param)
    { 
      wilson::initConstants(*param.gauge, profile);
    }

  DiracWilson::DiracWilson(const DiracWilson &dirac) : Dirac(dirac)
    { 
      wilson::initConstants(*dirac.gauge, profile);
    }

  DiracWilson::DiracWilson(const DiracParam &param, const int nDims) : Dirac(param)
  { 
    wilson::initConstants(*param.gauge, profile);
    
  }//temporal hack (for DW and TM operators) 

  DiracWilson::~DiracWilson() { }

  DiracWilson& DiracWilson::operator=(const DiracWilson &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
    }
    return *this;
  }

  void DiracWilson::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			   const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) {
      wilsonDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, 
		       &static_cast<const cudaColorSpinorField&>(in), parity, dagger, 0, 0.0, commDim, profile);
    } else {
      errorQuda("Not supported");
    }

    flops += 1320ll*in.Volume();
  }

  void DiracWilson::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (checkLocation(out, in, x) == QUDA_CUDA_FIELD_LOCATION) {
      wilsonDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, 
		       &static_cast<const cudaColorSpinorField&>(in), parity, dagger, 
		       &static_cast<const cudaColorSpinorField&>(x), k, commDim, profile);
    } else {
      errorQuda("Not supported");
    }

    flops += 1368ll*in.Volume();
  }

  void DiracWilson::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    ColorSpinorField *In = &const_cast<ColorSpinorField&>(in);
    if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
      ColorSpinorParam param(in);
      param.location = QUDA_CUDA_FIELD_LOCATION;
      param.fieldOrder =  param.precision == QUDA_DOUBLE_PRECISION ? QUDA_FLOAT2_FIELD_ORDER :
	(param.nSpin == 4 ? QUDA_FLOAT4_FIELD_ORDER : QUDA_FLOAT2_FIELD_ORDER);
      param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      In = ColorSpinorField::Create(param);
      *In = in;
    }

    ColorSpinorField *Out = &out;
    if (out.Location() == QUDA_CPU_FIELD_LOCATION) {
      ColorSpinorParam param(out);
      param.location = QUDA_CUDA_FIELD_LOCATION;
      param.fieldOrder =  param.precision == QUDA_DOUBLE_PRECISION ? QUDA_FLOAT2_FIELD_ORDER :
	(param.nSpin == 4 ? QUDA_FLOAT4_FIELD_ORDER : QUDA_FLOAT2_FIELD_ORDER);
      param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      Out = ColorSpinorField::Create(param);
    }

    checkFullSpinor(*Out, *In);
    DslashXpay(Out->Odd(), In->Even(), QUDA_ODD_PARITY, In->Odd(), -kappa);
    DslashXpay(Out->Even(), In->Odd(), QUDA_EVEN_PARITY, In->Even(), -kappa);

    if (in.Location() == QUDA_CPU_FIELD_LOCATION) delete In;
    if (out.Location() == QUDA_CPU_FIELD_LOCATION) {
      out = *Out;
      delete Out;
    }
  }

  void DiracWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracWilson::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b, 
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracWilson::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    // do nothing
  }

  /* Creates the coarse grid dirac operator
  Takes: multigrid transfer class, which knows
  about the coarse grid blocking, as well as
  having prolongate and restrict member functions
  
  Returns: Color matrices Y[0..2*dim] corresponding
  to the coarse grid operator.  The first 2*dim
  matrices correspond to the forward/backward
  hopping terms on the coarse grid.  Y[2*dim] is
  the color matrix that is diagonal on the coarse
  grid
  */
  void DiracWilson::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
				   double kappa, double mass, double mu, double mu_factor) const {
    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    cudaCloverField *c = NULL;
    CoarseOp(Y, X, T, *gauge, c, kappa, a, mu_factor, QUDA_WILSON_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracWilsonPC::DiracWilsonPC(const DiracParam &param)
    : DiracWilson(param)
  {

  }

  DiracWilsonPC::DiracWilsonPC(const DiracWilsonPC &dirac) 
    : DiracWilson(dirac)
  {

  }

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

    bool reset = newTmp(&tmp1, in);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracWilsonPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracWilsonPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
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
	errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracWilsonPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
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
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }
  }

} // namespace quda
