#include <dirac_quda.h>
#include <blas_quda.h>

DiracWilson::DiracWilson(const DiracParam &param)
  : Dirac(param)
{

}

DiracWilson::DiracWilson(const DiracWilson &dirac) 
  : Dirac(dirac)
{

}

DiracWilson::~DiracWilson()
{

}

DiracWilson& DiracWilson::operator=(const DiracWilson &dirac)
{
  if (&dirac != this) {
    Dirac::operator=(dirac);
    tmp1 = dirac.tmp1;
  }
  return *this;
}

void DiracWilson::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);
  
  dslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
	     0, 0, 0, out.volume, out.length, in.Precision());

  flops += 1320*in.volume;
}

void DiracWilson::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const QudaParity parity, const cudaColorSpinorField &x,
			     const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);

  dslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, x.v, x.norm, k, 
	     out.volume, out.length, in.Precision());

  flops += (1320+48)*in.volume;
}

void DiracWilson::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa);
}

void DiracWilson::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  } else {
    checkFullSpinor(*tmp1, in);
  }

  M(*tmp1, in);
  Mdag(out, *tmp1);

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
}

void DiracWilson::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracWilson::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
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
    tmp1 = dirac.tmp1;
  }
  return *this;
}

void DiracWilsonPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  }

  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY);
    DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY);
    DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
  } else {
    errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
  }

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
  
}

void DiracWilsonPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  M(out, in);
  Mdag(out, out);
}

void DiracWilsonPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
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

void DiracWilsonPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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
