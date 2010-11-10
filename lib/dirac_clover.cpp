#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

DiracClover::DiracClover(const DiracParam &param)
  : DiracWilson(param), clover(*(param.clover))
{

}

DiracClover::DiracClover(const DiracClover &dirac) 
  : DiracWilson(dirac), clover(dirac.clover)
{

}

DiracClover::~DiracClover()
{

}

DiracClover& DiracClover::operator=(const DiracClover &dirac)
{

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    clover = dirac.clover;
  }

  return *this;
}

void DiracClover::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const FullClover &clover) const
{
  Dirac::checkParitySpinor(out, in);

  if (out.Volume() != clover.even.volume) {
    errorQuda("Spinor volume %d doesn't match even clover volume %d",
	      out.Volume(), clover.even.volume);
  }
  if (out.Volume() != clover.odd.volume) {
    errorQuda("Spinor volume %d doesn't match odd clover volume %d",
	      out.Volume(), clover.odd.volume);
  }

}

// Protected method, also used for applying cloverInv
void DiracClover::cloverApply(cudaColorSpinorField &out, const FullClover &clover, 
			      const cudaColorSpinorField &in, const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, clover.even.stride);
  checkParitySpinor(in, out, clover);

  cloverCuda(out.v, out.norm, gauge, clover, in.v, in.norm, parity, 
	     in.volume, in.length, in.Precision());

  flops += 504*in.volume;
}

// Public method to apply the clover term only
void DiracClover::Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
{
  cloverApply(out, clover, in, parity);
}

// FIXME: create kernel to eliminate tmp
void DiracClover::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;

  bool reset = false;
  if (!tmp2) {
    tmp2 = new cudaColorSpinorField(in.Even(), param); // only create if necessary
    reset = true;
  }

  Clover(*tmp2, in.Odd(), QUDA_ODD_PARITY);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp2, -kappa);
  Clover(*tmp2, in.Even(), QUDA_EVEN_PARITY);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp2, -kappa);

  if (reset) {
    delete tmp2;
    tmp2 = 0;
  }
}

void DiracClover::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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

void DiracClover::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracClover::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
}

DiracCloverPC::DiracCloverPC(const DiracParam &param)
  : DiracClover(param), cloverInv(*(param.cloverInv))
{

}

DiracCloverPC::DiracCloverPC(const DiracCloverPC &dirac) 
  : DiracClover(dirac), cloverInv(dirac.clover)
{

}

DiracCloverPC::~DiracCloverPC()
{

}

DiracCloverPC& DiracCloverPC::operator=(const DiracCloverPC &dirac)
{
  if (&dirac != this) {
    DiracClover::operator=(dirac);
    cloverInv = dirac.cloverInv;
    tmp1 = dirac.tmp1;
    tmp2 = dirac.tmp2;
  }

  return *this;
}

// Public method
void DiracCloverPC::CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			      const QudaParity parity) const
{
  cloverApply(out, cloverInv, in, parity);
}

// apply hopping term, then clover: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (A_ee^-1 D^dagger_eo) or (A_oo^-1 D^dagger_oe)
// NOTE - this isn't Dslash dagger since order should be reversed!
void DiracCloverPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			   const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);
  checkParitySpinor(in, out, cloverInv);

  cloverDslashCuda(out.v, out.norm, gauge, cloverInv, in.v, in.norm, parity, dagger, 
		   0, 0, 0.0, out.volume, out.length, in.Precision());

  flops += (1320+504)*in.volume;
}

// xpay version of the above
void DiracCloverPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			       const QudaParity parity, const cudaColorSpinorField &x,
			       const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);
  checkParitySpinor(in, out, cloverInv);

  cloverDslashCuda(out.v, out.norm, gauge, cloverInv, in.v, in.norm, parity, dagger, 
		   x.v, x.norm, k, out.volume, out.length, in.Precision());

  flops += (1320+504+48)*in.volume;
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void DiracCloverPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  // FIXME: For asymmetric, a "DslashCxpay" kernel would improve performance.
  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  }

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY);
    Clover(out, in, QUDA_EVEN_PARITY);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY);
    Clover(out, in, QUDA_ODD_PARITY);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, out, kappa2);
  } else if (!dagger) { // symmetric preconditioning
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("Invalid matpcType");
    }
  } else { // symmetric preconditioning, dagger
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      CloverInv(out, in, QUDA_EVEN_PARITY); 
      Dslash(*tmp1, out, QUDA_ODD_PARITY);
      DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      CloverInv(out, in, QUDA_ODD_PARITY); 
      Dslash(*tmp1, out, QUDA_EVEN_PARITY);
      DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
    }
  }
  
  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
}

void DiracCloverPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  // need extra temporary because of symmetric preconditioning dagger
  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;

  bool reset = false;
  if (!tmp2) {
    tmp2 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  }

  M(*tmp2, in);
  Mdag(out, *tmp2);

  if (reset) {
    delete tmp2;
    tmp2 = 0;
  }
}

void DiracCloverPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol, 
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
			    const QudaSolutionType solType) const
{
  // we desire solution to preconditioned system
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    src = &b;
    sol = &x;
    return;
  }

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(b.Even(), param); // only create if necessary
    reset = true;
  }
  
  // we desire solution to full system
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
    src = &(x.Odd());
    CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
    CloverInv(*src, *tmp1, QUDA_EVEN_PARITY);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
    src = &(x.Even());
    CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
    CloverInv(*src, *tmp1, QUDA_ODD_PARITY);
    sol = &(x.Odd());
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    src = &(x.Odd());
    CloverInv(*tmp1, b.Odd(), QUDA_ODD_PARITY); // safe even when *tmp1 = b.odd
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    src = &(x.Even());
    CloverInv(*tmp1, b.Even(), QUDA_EVEN_PARITY); // safe even when *tmp1 = b.even
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
    sol = &(x.Odd());
  } else {
    errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }

}

void DiracCloverPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    return;
  }

  checkFullSpinor(x, b);

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(b.Even(), param); // only create if necessary
    reset = true;
  }

  // create full solution

  if (matpcType == QUDA_MATPC_EVEN_EVEN ||
      matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // x_o = A_oo^-1 (b_o + k D_oe x_e)
    DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    CloverInv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
      matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)
    DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    CloverInv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
  } else {
    errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
  }

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
}

