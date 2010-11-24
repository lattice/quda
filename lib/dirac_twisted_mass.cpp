#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

DiracTwistedMass::DiracTwistedMass(const DiracParam &param)
  : DiracWilson(param), mu(param.mu)
{

}

DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) 
  : DiracWilson(dirac), mu(dirac.mu)
{

}

DiracTwistedMass::~DiracTwistedMass()
{

}

DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
{
  if (&dirac != this) {
    DiracWilson::operator=(dirac);
  }
  return *this;
}

// Protected method for applying twist
void DiracTwistedMass::twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const QudaTwistGamma5Type twistType) const {
  
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);

  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);

  double flavor_mu = in.twistFlavor * mu;
  
  twistGamma5Cuda(out.v, out.norm, in.v, in.norm, dagger, kappa, flavor_mu, 
  		  in.volume, in.length, in.precision, twistType);
}


// Public method to apply the twist
void DiracTwistedMass::Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const {
  twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
}

void DiracTwistedMass::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);

  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID) {
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);
  }

  // We can elimiate this temporary at the expense of more kernels (like clover)
  bool reset = newTmp(&tmp2, in.Even());

  Twist(*tmp2, in.Odd());
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp2, -kappa);
  Twist(*tmp2, in.Even());
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp2, -kappa);

  deleteTmp(&tmp2, reset);

}

void DiracTwistedMass::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  bool reset = newTmp(&tmp1, in);

  M(*tmp1, in);
  Mdag(out, *tmp1);

  deleteTmp(&tmp1, reset);
}

void DiracTwistedMass::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracTwistedMass::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
}

DiracTwistedMassPC::DiracTwistedMassPC(const DiracParam &param)
  : DiracTwistedMass(param)
{

}

DiracTwistedMassPC::DiracTwistedMassPC(const DiracTwistedMassPC &dirac) 
  : DiracTwistedMass(dirac)
{

}

DiracTwistedMassPC::~DiracTwistedMassPC()
{

}

DiracTwistedMassPC& DiracTwistedMassPC::operator=(const DiracTwistedMassPC &dirac)
{
  if (&dirac != this) {
    DiracTwistedMass::operator=(dirac);
  }
  return *this;
}

// Public method to apply the inverse twist
void DiracTwistedMassPC::TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const {
  twistedApply(out, in, QUDA_TWIST_GAMMA5_INVERSE);
}

// apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
void DiracTwistedMassPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);
  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);

  if (!dagger) {// broken for dagger since order isn't changed
    double flavor_mu = in.twistFlavor * mu;
    twistedMassDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
    			  0, 0, kappa, flavor_mu, 0.0, out.volume, out.length, in.Precision());
    flops += (1320+72)*in.volume;
  } else { // safe to use tmp2 here which may alias in
    bool reset = newTmp(&tmp2, in);

    TwistInv(*tmp2, in);
    DiracWilson::Dslash(out, *tmp2, parity);

    flops += 72*in.volume;

    // if the pointers alias, undo the twist
    if (tmp2->v == in.v) Twist(*tmp2, *tmp2); 

    deleteTmp(&tmp2, reset);
  }

}

// xpay version of the above
void DiracTwistedMassPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				    const QudaParity parity, const cudaColorSpinorField &x,
				    const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);
  checkSpinorAlias(out, x);

  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);
  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);  

  if (!dagger) {
    double flavor_mu = in.twistFlavor * mu;
    twistedMassDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
  			x.v, x.norm, kappa, flavor_mu, k, out.volume, out.length, in.Precision());
    flops += (1320+96)*in.volume;
  } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
    bool reset = newTmp(&tmp1, in);

    TwistInv(*tmp1, in);
    DiracWilson::Dslash(out, *tmp1, parity);
    xpayCuda(x, k, out);
    flops += 96*in.volume;

    // if the pointers alias, undo the twist
    if (tmp1->v == in.v) Twist(*tmp1, *tmp1); 

    deleteTmp(&tmp1, reset);
  }

}

void DiracTwistedMassPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  bool reset = newTmp(&tmp1, in);

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY);
    Twist(out, in);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY);
    Twist(out, in);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, out, kappa2);
  } else { // symmetric preconditioning
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("Invalid matpcType");
    }
  }

  deleteTmp(&tmp1, reset);

}

void DiracTwistedMassPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  // need extra temporary because of symmetric preconditioning dagger
  bool reset = newTmp(&tmp2, in);

  M(*tmp2, in);
  Mdag(out, *tmp2);

  deleteTmp(&tmp2, reset);
}

void DiracTwistedMassPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
			    const QudaSolutionType solType) const
{
  // we desire solution to preconditioned system
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    src = &b;
    sol = &x;
    return;
  }

  bool reset = newTmp(&tmp1, b.Even());
  
  // we desire solution to full system
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
    src = &(x.Odd());
    TwistInv(*src, b.Odd());
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
    TwistInv(*src, *tmp1);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
    src = &(x.Even());
    TwistInv(*src, b.Even());
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
    TwistInv(*src, *tmp1);
    sol = &(x.Odd());
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    src = &(x.Odd());
    TwistInv(*tmp1, b.Odd()); // safe even when *tmp1 = b.odd
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    src = &(x.Even());
    TwistInv(*tmp1, b.Even()); // safe even when *tmp1 = b.even
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
    sol = &(x.Odd());
  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

  deleteTmp(&tmp1, reset);
}

void DiracTwistedMassPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    return;
  }				

  checkFullSpinor(x, b);
  bool reset = newTmp(&tmp1, b.Even());

  // create full solution
  
  if (matpcType == QUDA_MATPC_EVEN_EVEN ||
      matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // x_o = A_oo^-1 (b_o + k D_oe x_e)
    DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    TwistInv(x.Odd(), *tmp1);
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	     matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)
    DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    TwistInv(x.Even(), *tmp1);
  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }
  
  deleteTmp(&tmp1, reset);
}
