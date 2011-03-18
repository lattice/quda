#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

#include <tune_quda.h>

DiracWilson::DiracWilson(const DiracParam &param) : 
  Dirac(param), blockDslash(64, 1, 1), blockDslashXpay(64, 1, 1), 
  blockDslashFace(64, 1, 1), blockDslashXpayFace(64, 1, 1),
  face(param.gauge->X, 4, 12, 1, param.gauge->precision)
{

}

DiracWilson::DiracWilson(const DiracWilson &dirac) : Dirac(dirac), 
  blockDslash(64, 1, 1), blockDslashXpay(64, 1, 1), blockDslashFace(64, 1, 1), blockDslashXpayFace(64, 1, 1),
  face(dirac.face)
{

}

DiracWilson::~DiracWilson()
{

}

DiracWilson& DiracWilson::operator=(const DiracWilson &dirac)
{
  if (&dirac != this) {
    Dirac::operator=(dirac);
    blockDslash = dirac.blockDslash;
    blockDslashXpay = dirac.blockDslashXpay;
    blockDslashFace = dirac.blockDslashFace;
    blockDslashXpayFace = dirac.blockDslashXpayFace;
    face = dirac.face;
  }
  return *this;
}

void DiracWilson::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

  dslashCuda(&out, gauge, &in, parity, dagger, 0, 0.0, blockDslash, blockDslashFace);

  flops += 1320ll*in.volume;
}

void DiracWilson::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const QudaParity parity, const cudaColorSpinorField &x,
			     const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

  dslashCuda(&out, gauge, &in, parity, dagger, &x, k, blockDslashXpay, blockDslashXpayFace);

  flops += 1368ll*in.volume;
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

  bool reset = newTmp(&tmp1, in);
  checkFullSpinor(*tmp1, in);

  M(*tmp1, in);
  Mdag(out, *tmp1);

  deleteTmp(&tmp1, reset);
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

// Find the best block size parameters for the Dslash and DslashXpay kernels
void DiracWilson::Tune(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		       const cudaColorSpinorField &x) {
  setDslashTuning(QUDA_TUNE_YES);

  { // Tune Dslash
    TuneDiracWilsonDslash dslashTune(*this, out, in);
    dslashTune.Benchmark(blockDslash);
#ifdef OVERLAP_COMMS
    dslashTune.Benchmark(blockDslashFace);
#endif
  }

  { // Tune DslashXpay
    TuneDiracWilsonDslashXpay dslashXpayTune(*this, out, in, x);
    dslashXpayTune.Benchmark(blockDslashXpay);
#ifdef OVERLAP_COMMS
    dslashXpayTune.Benchmark(blockDslashXpayFace);
#endif
  }

  setDslashTuning(QUDA_TUNE_NO);

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

void DiracWilsonPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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
