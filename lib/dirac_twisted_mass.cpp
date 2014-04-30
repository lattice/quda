#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

namespace quda {

  int DiracTwistedMass::initTMFlag = 0;//set to 1 for parity spinors, and 2 for full spinors 

  DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) : DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon) { }

  DiracTwistedMass::DiracTwistedMass(const DiracParam &param, const int nDim) : DiracWilson(param, nDim), mu(param.mu), epsilon(param.epsilon) { }

  DiracTwistedMass::~DiracTwistedMass() { }

  DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
    }
    return *this;
  }

  void DiracTwistedMass::initConstants(const cudaColorSpinorField &a) const {

    if (a.SiteSubset() == QUDA_PARITY_SITE_SUBSET && initTMFlag != 1) {
      int flavor_stride = (a.TwistFlavor() != QUDA_TWIST_PLUS || a.TwistFlavor() != QUDA_TWIST_MINUS) ? a.VolumeCB()/2 : a.VolumeCB();
      initSpinorConstants(a, profile);
      initTwistedMassConstants(flavor_stride, profile);
      initTMFlag = 1;
    } else if (a.SiteSubset() == QUDA_FULL_SITE_SUBSET && initTMFlag != 2) {
      int flavor_stride = (a.TwistFlavor() != QUDA_TWIST_PLUS || a.TwistFlavor() != QUDA_TWIST_MINUS) ? a.VolumeCB()/4 : a.VolumeCB()/2;
      initSpinorConstants(a, profile);
      initTwistedMassConstants(flavor_stride, profile);
      initTMFlag = 2;
    }

  }


  // Protected method for applying twist

  void DiracTwistedMass::twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in,
				      const QudaTwistGamma5Type twistType) const
  {
    checkParitySpinor(out, in);
    initConstants(in);

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS) {
      double flavor_mu = in.TwistFlavor() * mu;
      twistGamma5Cuda(&out, &in, dagger, kappa, flavor_mu, 0.0, twistType);
      flops += 24ll*in.Volume();
    } else {
      errorQuda("DiracTwistedMass::twistedApply method for flavor doublet is not implemented..\n");  
    }
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

    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda      
    initConstants(in);
  
    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = 2.0 * kappa * in.TwistFlavor() * mu;//for direct twist (must be daggered separately)  
      twistedMassDslashCuda(&out.Odd(), gauge, &in.Even(), QUDA_ODD_PARITY, dagger, &in.Odd(), QUDA_DEG_DSLASH_TWIST_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      twistedMassDslashCuda(&out.Even(), gauge, &in.Odd(), QUDA_EVEN_PARITY, dagger, &in.Even(), QUDA_DEG_DSLASH_TWIST_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      flops += (1320ll+72ll)*in.Volume();
    } else {
      double a = -2.0 * kappa * mu; //for twist 
      double b = -2.0 * kappa * epsilon;//for twist
      twistedMassDslashCuda(&out.Odd(), gauge, &in.Even(), QUDA_ODD_PARITY, dagger, &in.Odd(), QUDA_NONDEG_DSLASH, a, b, 1.0, -kappa, commDim, profile);      
      twistedMassDslashCuda(&out.Even(), gauge, &in.Odd(), QUDA_EVEN_PARITY, dagger, &in.Even(), QUDA_NONDEG_DSLASH, a, b, 1.0, -kappa, commDim, profile);      

      flops += (1320ll+72ll+24ll)*in.Volume();//??
    }
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
  void DiracTwistedMassPC::TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    twistedApply(out, in, QUDA_TWIST_GAMMA5_INVERSE);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedMassPC::Dslash
  (cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    initConstants(in);
  
    if (in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = -2.0 * kappa * in.TwistFlavor() * mu;  //for invert twist (not daggered)
      double b = 1.0 / (1.0 + a*a);                     //for invert twist
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	twistedMassDslashCuda(&out, gauge, &in, parity, dagger, 0, QUDA_DEG_DSLASH_TWIST_INV, a, b, 0.0, 0.0, commDim, profile);
	flops += 1392ll*in.Volume();
      } else { 
	twistedMassDslashCuda(&out, gauge, &in, parity, dagger, 0, QUDA_DEG_TWIST_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);	
        flops += 1392ll*in.Volume();
      }
    } else {//TWIST doublet :
      double a = 2.0 * kappa * mu;  
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a*a - b*b);//!    
    
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        twistedMassDslashCuda(&out, gauge, &in, parity, dagger, 0, QUDA_NONDEG_DSLASH, a, b, c, 0.0, commDim, profile); 
        flops += (1320ll+120ll)*in.Volume();//per flavor 1320+16*6(rotation per flavor)+24 (scaling per flavor)
      } else {
        cudaColorSpinorField *doubletTmp=0; 
        bool reset = newTmp(&doubletTmp, in);
    
        twistGamma5Cuda(doubletTmp, &in, dagger, -a, b, c, QUDA_TWIST_GAMMA5_INVERSE);//note a -> -a      
        twistedMassDslashCuda(&out, gauge, doubletTmp, parity, dagger, 0, QUDA_NONDEG_DSLASH, 0.0, 0.0, 1.0, 0.0, commDim, profile);      //merge this!

        flops += 1440ll*in.Volume();//as for the asymmetric case

        deleteTmp(&doubletTmp, reset);
      }
    }
  }

  // xpay version of the above
  void DiracTwistedMassPC::DslashXpay
  (cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const cudaColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

   setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
   initConstants(in);  
  
    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = -2.0 * kappa * in.TwistFlavor() * mu;  //for invert twist
      double b = k / (1.0 + a*a);                     //for invert twist 
      if (!dagger) {
        twistedMassDslashCuda(&out, gauge, &in, parity, dagger, &x, QUDA_DEG_DSLASH_TWIST_INV, a, b, 0.0, 0.0, commDim, profile);
        flops += 1416ll*in.Volume();
      } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
        twistedMassDslashCuda(&out, gauge, &in, parity, dagger, &x, QUDA_DEG_TWIST_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);
        flops += 1416ll*in.Volume();
      }
    } else {//TWIST_DOUBLET:
      double a = 2.0 * kappa * mu;  
      double b = 2.0 * kappa * epsilon;
      double c = 1.0 / (1.0 + a*a - b*b); 
		
      if (!dagger) {	
        c *= k;//(-kappa*kappa)	  
        twistedMassDslashCuda(&out, gauge, &in, parity, dagger, &x, QUDA_NONDEG_DSLASH, a, b, c, 0.0, commDim, profile);
        flops += 1464ll*in.Volume();
      } else {
        cudaColorSpinorField *doubletTmp=0; 
        bool reset = newTmp(&doubletTmp, in);
        twistGamma5Cuda(doubletTmp, &in, dagger, -a, b, c, QUDA_TWIST_GAMMA5_INVERSE);//note a -> -a
        twistedMassDslashCuda(&out, gauge, doubletTmp, parity, dagger, &x, QUDA_NONDEG_DSLASH, 0.0, 0.0, k, 0.0, commDim, profile);
        flops += 1464ll*in.Volume();
        deleteTmp(&doubletTmp, reset);	  
      }
    }
  }

  void DiracTwistedMassPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;

    bool reset = newTmp(&tmp1, in);

    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
	  DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
	  DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
      } else {//asymmetric preconditioning 
        double a = 2.0 * kappa * in.TwistFlavor() * mu;
        if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
          twistedMassDslashCuda(&out, gauge, tmp1, QUDA_EVEN_PARITY, dagger, &in, QUDA_DEG_DSLASH_TWIST_XPAY, a, kappa2, 0.0, 0.0, commDim, profile); 
          flops += (1320ll+96ll)*in.Volume();	 
        } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
          twistedMassDslashCuda(&out, gauge, tmp1, QUDA_ODD_PARITY, dagger, &in, QUDA_DEG_DSLASH_TWIST_XPAY, a, kappa2, 0.0, 0.0, commDim, profile);
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
          twistedMassDslashCuda(&out, gauge, tmp1, QUDA_EVEN_PARITY, dagger, &in, QUDA_NONDEG_DSLASH, a, b, c, kappa2, commDim, profile); 
          flops += (1464ll)*in.Volume();	 
        } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
          Dslash(*tmp1, in, QUDA_EVEN_PARITY); // fused kernel
          twistedMassDslashCuda(&out, gauge, tmp1, QUDA_ODD_PARITY, dagger, &in, QUDA_NONDEG_DSLASH, a, b, c, kappa2, commDim, profile);	 
          flops += (1464ll)*in.Volume();
        } 
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
    if(b.TwistFlavor() == QUDA_TWIST_PLUS || b.TwistFlavor() == QUDA_TWIST_MINUS){  
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
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
	initConstants(*src);
	
        twistGamma5Cuda(src, &b.Odd(), dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack! 
        twistedMassDslashCuda(tmp1, gauge, src, QUDA_EVEN_PARITY, dagger, &b.Even(), QUDA_NONDEG_DSLASH,  0.0, 0.0, kappa, 0.0, commDim, profile);
        twistGamma5Cuda(src, tmp1, dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack!

        sol = &(x.Even()); 
        
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)    
        src = &(x.Even());
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
        initConstants(*src);
	
        twistGamma5Cuda(src, &b.Even(), dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack!                     
        twistedMassDslashCuda(tmp1, gauge, src, QUDA_ODD_PARITY, dagger, &b.Odd(),  QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0, commDim, profile); 
        twistGamma5Cuda(src, tmp1, dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack!
    
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        src = &(x.Odd());
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
	initConstants(*src);

        twistGamma5Cuda(tmp1, &b.Odd(), dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack!                           
        twistedMassDslashCuda(src, gauge, tmp1, QUDA_EVEN_PARITY, dagger, &b.Even(), QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0, commDim, profile);       

        sol = &(x.Even());
    
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        src = &(x.Even());
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
	initConstants(*src);

        twistGamma5Cuda(tmp1, &b.Even(), dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);//temporal hack!                           
        twistedMassDslashCuda(src, gauge, tmp1, QUDA_ODD_PARITY, dagger, &b.Odd(), QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0, commDim, profile);       
    
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }
    }//end of doublet
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
    if(b.TwistFlavor() == QUDA_TWIST_PLUS || b.TwistFlavor() == QUDA_TWIST_MINUS){    
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
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
	initConstants(x.Even());

        twistedMassDslashCuda(tmp1, gauge, &x.Even(), QUDA_ODD_PARITY, dagger, &b.Odd(), QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0, commDim, profile);             
        twistGamma5Cuda(&x.Odd(), tmp1, dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);
 
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||  matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)  
        setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
	initConstants(x.Odd());
	
        twistedMassDslashCuda(tmp1, gauge, &x.Odd(), QUDA_EVEN_PARITY, dagger, &b.Even(), QUDA_NONDEG_DSLASH, 0.0, 0.0, kappa, 0.0, commDim, profile);                   
        twistGamma5Cuda(&x.Even(), tmp1, dagger, a, bb, c, QUDA_TWIST_GAMMA5_DIRECT);      
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
      }    
    }//end of twist doublet...
    deleteTmp(&tmp1, reset);
  }
} // namespace quda
