#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

DiracTwistedMass::DiracTwistedMass(const DiracParam &param) : DiracWilson(param), mu(param.mu) { }

DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) : DiracWilson(dirac), mu(dirac.mu) { }

DiracTwistedMass::~DiracTwistedMass() { }

DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
{
  if (&dirac != this) {
    DiracWilson::operator=(dirac);
  }
  return *this;
}

// Protected method for applying twist
void DiracTwistedMass::twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const QudaTwistGamma5Type twistType) const
{
  checkParitySpinor(out, in);
  
  if (!initDslash) initDslashConstants(gauge, in.Stride());

  if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

  double flavor_mu = in.TwistFlavor() * mu;
  
  twistGamma5Cuda(&out, &in, dagger, kappa, flavor_mu, twistType);

  flops += 24*in.Volume();
}


// Public method to apply the twist
void DiracTwistedMass::Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
}

void DiracTwistedMass::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  if (in.TwistFlavor() != out.TwistFlavor()) 
    errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

  if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
    errorQuda("Twist flavor not set %d\n", in.TwistFlavor());
  }

  // We can eliminate this temporary at the expense of more kernels (like clover)
  cudaColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
  if (tmp2) {
    if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
    else tmp = tmp2;
  }
  bool reset = newTmp(&tmp, in.Even());

  Twist(*tmp, in.Odd());
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp, -kappa);
  Twist(*tmp, in.Even());
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp, -kappa);

  deleteTmp(&tmp, reset);

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

DiracTwistedMassPC::DiracTwistedMassPC(const DiracParam &param) : DiracTwistedMass(param)
{

}

DiracTwistedMassPC::DiracTwistedMassPC(const DiracTwistedMassPC &dirac) : DiracTwistedMass(dirac) { }

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
void DiracTwistedMassPC::TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  twistedApply(out, in, QUDA_TWIST_GAMMA5_INVERSE);
}

// apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
void DiracTwistedMassPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  if (in.TwistFlavor() != out.TwistFlavor()) 
    errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
  if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

  if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    double flavor_mu = in.TwistFlavor() * mu;
    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda
    twistedMassDslashCuda(&out, gauge, &in, parity, dagger, 0, kappa, flavor_mu, 0.0, commDim);
    flops += (1320+72)*in.Volume();
  } else { // safe to use tmp2 here which may alias in
    bool reset = newTmp(&tmp2, in);

    TwistInv(*tmp2, in);
    DiracWilson::Dslash(out, *tmp2, parity);

    flops += 72*in.Volume();

    // if the pointers alias, undo the twist
    if (tmp2->V() == in.V()) Twist(*tmp2, *tmp2); 

    deleteTmp(&tmp2, reset);
  }

}

// xpay version of the above
void DiracTwistedMassPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				    const QudaParity parity, const cudaColorSpinorField &x,
				    const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.Stride());
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  if (in.TwistFlavor() != out.TwistFlavor()) 
    errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
  if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.TwistFlavor());  

  if (!dagger) {
    double flavor_mu = in.TwistFlavor() * mu;
    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda
    twistedMassDslashCuda(&out, gauge, &in, parity, dagger, &x, kappa, 
			  flavor_mu, k, commDim);
    flops += (1320+96)*in.Volume();
  } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
    bool reset = newTmp(&tmp1, in);

    TwistInv(*tmp1, in);
    DiracWilson::Dslash(out, *tmp1, parity);
    xpayCuda((cudaColorSpinorField&)x, k, out);
    flops += 96*in.Volume();

    // if the pointers alias, undo the twist
    if (tmp1->V() == in.V()) Twist(*tmp1, *tmp1); 

    deleteTmp(&tmp1, reset);
  }

}

void DiracTwistedMassPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  bool reset = newTmp(&tmp1, in);

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY); // fused kernel
    Twist(out, in);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY); // fused kernel
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
