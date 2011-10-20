#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <tune_quda.h>

DiracDomainWall::DiracDomainWall(const DiracParam &param) : 
  DiracWilson(param), m5(param.m5), kappa5(0.5/(5.0 + m5))
{
  for (int i=0; i<5; i++) {
    blockDslash[i] = dim3(64, 1, 1);
    blockDslashXpay[i] = dim3(64, 1, 1);
  }
}

DiracDomainWall::DiracDomainWall(const DiracDomainWall &dirac) : 
  DiracWilson(dirac), m5(dirac.m5), kappa5(0.5/(5.0 + m5))
{
  for (int i=0; i<5; i++) {
    blockDslash[i] = dirac.blockDslash[i];
    blockDslashXpay[i] = dirac.blockDslashXpay[i];
  }
}

DiracDomainWall::~DiracDomainWall()
{

}

DiracDomainWall& DiracDomainWall::operator=(const DiracDomainWall &dirac)
{

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    m5 = dirac.m5;
    kappa5 = dirac.kappa5;

    for (int i=0; i<5; i++) {
      blockDslash[i] = dirac.blockDslash[i];
      blockDslashXpay[i] = dirac.blockDslashXpay[i];
    }
  }

  return *this;
}

// Find the best block size parameters for the Dslash and DslashXpay kernels
void DiracDomainWall::Tune(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			   const cudaColorSpinorField &x) {

  setDslashTuning(QUDA_TUNE_YES);

  { // Tune Dslash
    TuneDiracDomainWallDslash dslashTune(*this, out, in);
    dslashTune.Benchmark(blockDslash[0]);
    for (int i=0; i<4; i++) 
      if (commDimPartitioned(i)) dslashTune.Benchmark(blockDslash[i+1]);
  }

  { // Tune DslashXpay
    TuneDiracDomainWallDslashXpay dslashXpayTune(*this, out, in, x);
    dslashXpayTune.Benchmark(blockDslashXpay[0]);
    for (int i=0; i<4; i++) 
      if (commDimPartitioned(i)) dslashXpayTune.Benchmark(blockDslashXpay[i+1]);
  }

  setDslashTuning(QUDA_TUNE_NO);
}

void DiracDomainWall::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const QudaParity parity) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  if (!initDomainWall) initDomainWallConstants(in.X(4));
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);
  
  domainWallDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, blockDslash);

  long long Ls = in.X(4);
  long long bulk = (Ls-2)*(in.Volume()/Ls);
  long long wall = 2*in.Volume()/Ls;
  flops += 1320LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
}

void DiracDomainWall::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				 const QudaParity parity, const cudaColorSpinorField &x,
				 const double &k) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  if (!initDomainWall) initDomainWallConstants(in.X(4));
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  domainWallDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, blockDslashXpay);

  long long Ls = in.X(4);
  long long bulk = (Ls-2)*(in.Volume()/Ls);
  long long wall = 2*in.Volume()/Ls;
  flops += (1320LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
}

void DiracDomainWall::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa5);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa5);
}

void DiracDomainWall::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);

  bool reset = newTmp(&tmp1, in);

  M(*tmp1, in);
  Mdag(out, *tmp1);

  deleteTmp(&tmp1, reset);
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
  }

  return *this;
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void DiracDomainWallPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
  double kappa2 = -kappa5*kappa5;

  bool reset = newTmp(&tmp1, in);

  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY);
    DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY);
    DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
  } else {
    errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
  }

  deleteTmp(&tmp1, reset);
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
      DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
      src = &(x.Odd());
      sol = &(x.Even());
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // src = b_o + k D_oe b_e
      DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
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
    DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // x_e = b_e + k D_eo x_o
    DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
  } else {
    errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
  }
}

