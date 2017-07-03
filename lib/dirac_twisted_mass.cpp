#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

namespace quda {

  namespace twisted {
#include <dslash_init.cuh>
  }

  namespace ndegtwisted {
#include <dslash_init.cuh>
  }

  namespace dslash_aux {
#include <dslash_init.cuh>
  }

  DiracTwistedMass::DiracTwistedMass(const DiracParam &param, const int nDim) 
    : DiracWilson(param, nDim), mu(param.mu), epsilon(param.epsilon) 
  { 
    twisted::initConstants(*param.gauge,profile);
    ndegtwisted::initConstants(*param.gauge,profile);
    dslash_aux::initConstants(*param.gauge,profile);
  }

  DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) 
    : DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon) 
  { 
    twisted::initConstants(*dirac.gauge,profile);
    ndegtwisted::initConstants(*dirac.gauge,profile);
    dslash_aux::initConstants(*dirac.gauge,profile);
  }

  DiracTwistedMass::~DiracTwistedMass() { }

  DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
    }
    return *this;
  }

  // Protected method for applying twist
  void DiracTwistedMass::twistedApply(ColorSpinorField &out, const ColorSpinorField &in,
				      const QudaTwistGamma5Type twistType) const
  {
    checkParitySpinor(out, in);

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) {
	twistGamma5Cuda(&static_cast<cudaColorSpinorField&>(out), 
			&static_cast<const cudaColorSpinorField&>(in),
			dagger, kappa, mu, 0.0, twistType);
      } else {
	errorQuda("Not implemented");
      }
      flops += 24ll*in.Volume();
    } else {
      errorQuda("DiracTwistedMass::twistedApply method for flavor doublet is not implemented..\n");  
    }
	
    flops += 24ll*in.Volume();
  }


  // Public method to apply the twist
  void DiracTwistedMass::Twist(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
  }

  void DiracTwistedMass::TwistedDslash(ColorSpinorField &out, const ColorSpinorField &in,
				       QudaParity parity, QudaTwistDslashType twistDslashType,
				       double a, double b, double c, double d) const {
    twistedMassDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger,
			  0, twistDslashType, a, b, c, d, commDim, profile);
  }

  void DiracTwistedMass::TwistedDslashXpay(ColorSpinorField &out, const ColorSpinorField &in,
					   const ColorSpinorField &x, QudaParity parity,
					   QudaTwistDslashType twistDslashType,
					   double a, double b, double c, double d) const {
    twistedMassDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger,
			  &static_cast<const cudaColorSpinorField&>(x),
			  twistDslashType, a, b, c, d, commDim, profile);
  }
  
  void DiracTwistedMass::NdegTwistedDslash(ColorSpinorField &out, const ColorSpinorField &in,
					   QudaParity parity, QudaTwistDslashType twistDslashType,
					   double a, double b, double c, double d) const {
    ndegTwistedMassDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger,
			  0, twistDslashType, a, b, c, d, commDim, profile);
  }

  void DiracTwistedMass::NdegTwistedDslashXpay(ColorSpinorField &out, const ColorSpinorField &in,
					       const ColorSpinorField &x, QudaParity parity,
					       QudaTwistDslashType twistDslashType,
					       double a, double b, double c, double d) const {
    ndegTwistedMassDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger,
			  &static_cast<const cudaColorSpinorField&>(x),
			  twistDslashType, a, b, c, d, commDim, profile);
  }

  void DiracTwistedMass::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());
    }

    // We can eliminate this temporary at the expense of more kernels (like clover)
    ColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
    if (tmp2) {
      if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
      else tmp = tmp2;
    }
    bool reset = newTmp(&tmp, in.Even());

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = 2.0 * kappa * mu;//for direct twist (must be daggered separately)  

      TwistedDslashXpay(out.Odd(), in.Even(), in.Odd(), QUDA_ODD_PARITY, 
			QUDA_DEG_DSLASH_TWIST_XPAY, a, -kappa, 0.0, 0.0);

      TwistedDslashXpay(out.Even(), in.Odd(), in.Even(), QUDA_EVEN_PARITY,
			QUDA_DEG_DSLASH_TWIST_XPAY, a, -kappa, 0.0, 0.0);

      flops += (1320ll+72ll)*in.Volume();
    } else {
      double a = -2.0 * kappa * mu; //for twist 
      double b = -2.0 * kappa * epsilon;//for twist
      NdegTwistedDslashXpay(out.Odd(), in.Even(), in.Odd(), QUDA_ODD_PARITY,
			    QUDA_NONDEG_DSLASH, a, b, 1.0, -kappa);

      NdegTwistedDslashXpay(out.Even(), in.Odd(), in.Even(), QUDA_EVEN_PARITY,
			    QUDA_NONDEG_DSLASH, a, b, 1.0, -kappa);
				  
      flops += (1320ll+72ll+24ll)*in.Volume();//??
    }
    deleteTmp(&tmp, reset);
  }

  void DiracTwistedMass::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMass::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				 ColorSpinorField &x, ColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracTwistedMass::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				     const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracTwistedMass::createCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T, double kappa, double mu, double mu_factor) const {
    double a = 2.0 * kappa * mu;
    cudaCloverField *c = NULL;
    CoarseOp(Y, X, Xinv, Yhat, T, *gauge, c, kappa, a, mu_factor, QUDA_TWISTED_MASS_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracTwistedMassPC::DiracTwistedMassPC(const DiracTwistedMassPC &dirac) : DiracTwistedMass(dirac) { }

  DiracTwistedMassPC::DiracTwistedMassPC(const DiracParam &param, const int nDim) : DiracTwistedMass(param, nDim){ }

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
  void DiracTwistedMassPC::TwistInv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    twistedApply(out, in,  QUDA_TWIST_GAMMA5_INVERSE);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo A_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedMassPC::Dslash
  (ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * mu;  //for invert twist (not daggered)
      double b = 1.0 / (1.0 + a*a);                     //for invert twist
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	TwistedDslash(out, in, parity, QUDA_DEG_DSLASH_TWIST_INV, a, b, 0.0, 0.0);
	flops += 1392ll*in.Volume();
      } else { 
	TwistedDslash(out, in, parity, QUDA_DEG_TWIST_INV_DSLASH, a, b, 0.0, 0.0);
        flops += 1392ll*in.Volume();
      }
    } else {//TWIST doublet :
      double a = 2.0 * kappa * mu;  
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a*a - b*b);//!    
    
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        NdegTwistedDslash(out, in, parity, QUDA_NONDEG_DSLASH, a, b, c, 0.0);
        flops += (1320ll+120ll)*in.Volume();//per flavor 1320+16*6(rotation per flavor)+24 (scaling per flavor)
      } else {
        ColorSpinorField *doubletTmp=0; 
        bool reset = newTmp(&doubletTmp, in);
    
        twistGamma5Cuda(static_cast<cudaColorSpinorField*>(doubletTmp),
			&static_cast<const cudaColorSpinorField&>(in), dagger, a, b, c, QUDA_TWIST_GAMMA5_INVERSE);//note a -> -a      

	// this is just a vectorized Wilson dslash
        NdegTwistedDslash(out, *doubletTmp, parity, QUDA_NONDEG_DSLASH, 0.0, 0.0, 1.0, 0.0);

        flops += 1440ll*in.Volume();//as for the asymmetric case

        deleteTmp(&doubletTmp, reset);
      }
    }
  }

  // xpay version of the above
  void DiracTwistedMassPC::DslashXpay
  (ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * mu;  //for invert twist
      double b = k / (1.0 + a*a);                     //for invert twist 
      if (!dagger) {
        TwistedDslashXpay(out, in, x, parity, QUDA_DEG_DSLASH_TWIST_INV, a, b, 0.0, 0.0);
        flops += 1416ll*in.Volume();
      } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
        TwistedDslashXpay(out, in, x, parity, QUDA_DEG_TWIST_INV_DSLASH, a, b, 0.0, 0.0);
        flops += 1416ll*in.Volume();
      }
    } else {//TWIST_DOUBLET:
      double a = 2.0 * kappa * mu;  
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a*a - b*b); 
		
      if (!dagger) {	
        c *= k;//(-kappa*kappa)	  
        NdegTwistedDslashXpay(out, in, x, parity, QUDA_NONDEG_DSLASH, a, b, c, 0.0);
        flops += 1464ll*in.Volume();
      } else {
        ColorSpinorField *doubletTmp=0; 
        bool reset = newTmp(&doubletTmp, in);
        twistGamma5Cuda(static_cast<cudaColorSpinorField*>(doubletTmp),
			&static_cast<const cudaColorSpinorField&>(in),
			dagger, -a, b, c, QUDA_TWIST_GAMMA5_INVERSE);//note a -> -a

	// this is just a vectorized Wilson dslash
        NdegTwistedDslashXpay(out, *doubletTmp, x, parity, QUDA_NONDEG_DSLASH, 0.0, 0.0, k, 0.0);
        flops += 1464ll*in.Volume();
        deleteTmp(&doubletTmp, reset);	  
      }
    }
  }

  void DiracTwistedMassPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;

    bool reset = newTmp(&tmp1, in);

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
	  DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
	  DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
      } else {//asymmetric preconditioning 
        double a = 2.0 * kappa * mu;
        if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
          TwistedDslashXpay(out, *tmp1, in, QUDA_EVEN_PARITY, QUDA_DEG_DSLASH_TWIST_XPAY, a, kappa2, 0.0, 0.0);
          flops += (1320ll+96ll)*in.Volume();	 
        } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
          TwistedDslashXpay(out, *tmp1, in, QUDA_ODD_PARITY, QUDA_DEG_DSLASH_TWIST_XPAY, a, kappa2, 0.0, 0.0);
          flops += (1320ll+96ll)*in.Volume();
        }else { // symmetric preconditioning
          errorQuda("Invalid matpcType");
        }
      }
    } else { //Twist doublet
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	Dslash(*tmp1, in, QUDA_ODD_PARITY);
	DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
      } else if (matpcType == QUDA_MATPC_ODD_ODD){
	Dslash(*tmp1, in, QUDA_EVEN_PARITY); // fused kernel
	DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2);
      } else {// asymmetric preconditioning
	//Parameter for invert twist (note the implemented operator: c*(1 - i *a * gamma_5 tau_3 + b * tau_1)):
        //double a = !dagger ? -2.0 * kappa * mu : 2.0 * kappa * mu;  
        double a = -2.0 * kappa * mu;  
        double b = -2.0 * kappa * epsilon;
        double c = 1.0;
	
        if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
          NdegTwistedDslashXpay(out, *tmp1, in, QUDA_EVEN_PARITY, QUDA_NONDEG_DSLASH, a, b, c, kappa2);
          flops += (1464ll)*in.Volume();	 
        } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
          Dslash(*tmp1, in, QUDA_EVEN_PARITY); // fused kernel
          NdegTwistedDslashXpay(out, *tmp1, in, QUDA_ODD_PARITY, QUDA_NONDEG_DSLASH, a, b, c, kappa2);
          flops += (1464ll)*in.Volume();
        } 
      }
    }
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMassPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracTwistedMassPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				   ColorSpinorField &x, ColorSpinorField &b, 
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
    if(b.TwistFlavor() == QUDA_TWIST_SINGLET) {  
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
    } else {//doublet:
      // we desire solution to preconditioned system

      double a = 2.0 * kappa * mu;  
      double bb = 2.0 * kappa * epsilon;
  
      double d = (1.0 + a*a - bb*bb);
      if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
      double c = 1.0 / d;
 
      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = A_ee^-1(b_e + k D_eo A_oo^-1 b_o)
        src = &(x.Odd());
	
	twistGamma5Cuda(static_cast<cudaColorSpinorField*>(src),
			&static_cast<const cudaColorSpinorField&>(b.Odd()),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack
        NdegTwistedDslashXpay(*tmp1, *src, b.Even(), QUDA_EVEN_PARITY, QUDA_NONDEG_DSLASH,  0.0, 0.0, kappa, 0.0);
	twistGamma5Cuda(static_cast<cudaColorSpinorField*>(src),
			static_cast<cudaColorSpinorField*>(tmp1),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack

        sol = &(x.Even()); 
        
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)    
        src = &(x.Even());
	
	twistGamma5Cuda(static_cast<cudaColorSpinorField*>(src),
			&static_cast<const cudaColorSpinorField&>(b.Even()),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack
        NdegTwistedDslashXpay(*tmp1, *src, b.Odd(), QUDA_ODD_PARITY, QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0);
	twistGamma5Cuda(static_cast<cudaColorSpinorField*>(src),
			static_cast<cudaColorSpinorField*>(tmp1),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack
    
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        src = &(x.Odd());

	twistGamma5Cuda(static_cast<cudaColorSpinorField*>(tmp1),
			&static_cast<const cudaColorSpinorField&>(b.Odd()),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack
        NdegTwistedDslashXpay(*src, *tmp1, b.Even(), QUDA_EVEN_PARITY, QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0);

        sol = &(x.Even());
    
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        src = &(x.Even());

        twistGamma5Cuda(static_cast<cudaColorSpinorField*>(tmp1),
			&static_cast<const cudaColorSpinorField&>(b.Even()),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT); // temporal hack

        NdegTwistedDslashXpay(*src, *tmp1, b.Odd(), QUDA_ODD_PARITY, QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0);
    
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }
    }//end of doublet
    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }
  
  void DiracTwistedMassPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    checkFullSpinor(x, b);
    bool reset = newTmp(&tmp1, b.Even());

    // create full solution
    if(b.TwistFlavor() == QUDA_TWIST_SINGLET) {    
      if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
        TwistInv(x.Odd(), *tmp1);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||   matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)
        DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
        TwistInv(x.Even(), *tmp1);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }
    } else { //twist doublet:
      double a = 2.0 * kappa * mu;  
      double bb = 2.0 * kappa * epsilon;
  
      double d = (1.0 + a*a - bb*bb);
      if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
      double c = 1.0 / d;
 
      if (matpcType == QUDA_MATPC_EVEN_EVEN ||  matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        NdegTwistedDslashXpay(*tmp1, x.Even(), b.Odd(), QUDA_ODD_PARITY, QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0);
        twistGamma5Cuda(&static_cast<cudaColorSpinorField&>(x.Odd()),
			static_cast<cudaColorSpinorField*>(tmp1),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||  matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)  
	
        NdegTwistedDslashXpay(*tmp1, x.Odd(), b.Even(), QUDA_EVEN_PARITY, QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0);
        twistGamma5Cuda(&static_cast<cudaColorSpinorField&>(x.Even()),
			static_cast<cudaColorSpinorField*>(tmp1),
			dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);      
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }    
    }//end of twist doublet...
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedMassPC::createCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T, double kappa, double mu, double mu_factor) const {
    double a = -2.0 * kappa * mu;
    cudaCloverField *c = NULL;
    CoarseOp(Y, X, Xinv, Yhat, T, *gauge, c, kappa, a, -mu_factor, QUDA_TWISTED_MASSPC_DIRAC, matpcType);
  }
} // namespace quda
