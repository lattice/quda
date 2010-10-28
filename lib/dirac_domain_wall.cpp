#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

DiracDomainWall::DiracDomainWall(const DiracParam &param)
  : DiracWilson(param)
{

}

DiracDomainWall::DiracDomainWall(const DiracDomainWall &dirac) 
  : DiracWilson(dirac)
{

}

DiracDomainWall::~DiracDomainWall()
{

}

DiracDomainWall& DiracDomainWall::operator=(const DiracDomainWall &dirac)
{

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
  }

  return *this;
}

void DiracDomainWall::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0, in.X(4));
  checkParitySpinor(in, out);
  
  domainWallDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 0, 0, 
		       mass, 0, out.volume, out.length, in.Precision());

  int Ls = in.X(4);
  long long unsigned int bulk = (Ls-2)*(in.volume/Ls);
  long long unsigned int wall = 2*in.volume/Ls;
  flops += 1320*(long long int)in.volume + 96*bulk + 120*wall;
}

void DiracDomainWall::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const QudaParity parity, const cudaColorSpinorField &x,
			     const double &k) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0, in.X(4));
  checkParitySpinor(in, out);

  domainWallDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, x.v, x.norm, 
		       mass, k, out.volume, out.length, in.Precision());

  int Ls = in.X(4);
  long long unsigned int bulk = (Ls-2)*(in.volume/Ls);
  long long unsigned int wall = 2*in.volume/Ls;
  flops += (1320+48)*(long long int)in.volume + 96*bulk + 120*wall;
}

void DiracDomainWall::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa);
}

void DiracDomainWall::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;

  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  }

  M(*tmp1, in);
  Mdag(out, *tmp1);

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }

}

void DiracDomainWall::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracDomainWall::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
}

DiracDomainWallPC::DiracDomainWallPC(const DiracParam &param)
  : DiracDomainWall(param)
{

}

DiracDomainWallPC::DiracDomainWallPC(const DiracDomainWallPC &dirac) 
  : DiracDomainWall(dirac)
{

}

DiracDomainWallPC::~DiracDomainWallPC()
{

}

DiracDomainWallPC& DiracDomainWallPC::operator=(const DiracDomainWallPC &dirac)
{
  if (&dirac != this) {
    DiracDomainWall::operator=(dirac);
    tmp1 = dirac.tmp1;
  }

  return *this;
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void DiracDomainWallPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
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
    errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
  }

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }

}

void DiracDomainWallPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  M(out, in);
  Mdag(out, out);
}

void DiracDomainWallPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
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
      errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
    }
    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want
  }

}

void DiracDomainWallPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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
    errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
  }
}

