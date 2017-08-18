#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>

namespace quda {

  namespace twistedclover {
#include <dslash_init.cuh>
  }

  namespace dslash_aux {
#include <dslash_init.cuh>
  }

  DiracTwistedClover::DiracTwistedClover(const DiracParam &param, const int nDim) 
    : DiracWilson(param, nDim), mu(param.mu), epsilon(param.epsilon), clover(*(param.clover))
  {
    twistedclover::initConstants(*param.gauge,profile);
  }

  DiracTwistedClover::DiracTwistedClover(const DiracTwistedClover &dirac) 
    : DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon), clover(dirac.clover)
  {
    twistedclover::initConstants(*dirac.gauge,profile);
  }

  DiracTwistedClover::~DiracTwistedClover() { }

  DiracTwistedClover& DiracTwistedClover::operator=(const DiracTwistedClover &dirac)
  {
    if (&dirac != this)
      {
	DiracWilson::operator=(dirac);
	clover = dirac.clover;
      }

    return *this;
  }

  void DiracTwistedClover::checkParitySpinor(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    Dirac::checkParitySpinor(out, in);

    if (out.Volume() != clover.VolumeCB())
      errorQuda("Parity spinor volume %d doesn't match clover checkboard volume %d", out.Volume(), clover.VolumeCB());
  }

  // Protected method for applying twist
  void DiracTwistedClover::twistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const QudaTwistGamma5Type twistType, const int parity) const
  {
    checkParitySpinor(out, in);
    ApplyTwistClover(out, in, clover, kappa, mu, 0.0, parity, dagger, twistType);

    if (twistType == QUDA_TWIST_GAMMA5_INVERSE) flops += 1056ll*in.Volume();
    else flops += 552ll*in.Volume();
  }


  // Public method to apply the twist
  void DiracTwistedClover::TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_DIRECT, parity);
  }

  void DiracTwistedClover::M(ColorSpinorField &out, const ColorSpinorField &in) const
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

    FullClover *cs = new FullClover(clover, false);
    FullClover *cI = new FullClover(clover, true);

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET){
      double a = 2.0 * kappa * mu;//for direct twist (must be daggered separately)  
      twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out.Odd()),
			      *gauge, cs, cI, &static_cast<const cudaColorSpinorField&>(in.Even()),
			      QUDA_ODD_PARITY, dagger, &static_cast<const cudaColorSpinorField&>(in.Odd()),
			      QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out.Even()),
			      *gauge, cs, cI, &static_cast<const cudaColorSpinorField&>(in.Odd()),
			      QUDA_EVEN_PARITY, dagger, &static_cast<const cudaColorSpinorField&>(in.Even()),
			      QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      flops += (1320ll+552ll)*in.Volume();
    } else {
      errorQuda("Non-deg twisted clover not implemented yet");
    }
    deleteTmp(&tmp, reset);
    delete cs;
    delete cI;
  }

  void DiracTwistedClover::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedClover::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				   ColorSpinorField &x, ColorSpinorField &b, 
				   const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracTwistedClover::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracTwistedClover::createCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T, double kappa, double mu, double mu_factor) const {
    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, Xinv, Yhat, T, *gauge, &clover, kappa, a, mu_factor, QUDA_TWISTED_CLOVER_DIRAC, QUDA_MATPC_INVALID);
  }

  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracTwistedCloverPC &dirac) : DiracTwistedClover(dirac) { }

  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracParam &param, const int nDim) : DiracTwistedClover(param, nDim){ }

  DiracTwistedCloverPC::~DiracTwistedCloverPC()
  {

  }

  DiracTwistedCloverPC& DiracTwistedCloverPC::operator=(const DiracTwistedCloverPC &dirac)
  {
    if (&dirac != this) {
      DiracTwistedClover::operator=(dirac);
    }
    return *this;
  }

  // Public method to apply the inverse twist
  void DiracTwistedCloverPC::TwistCloverInv(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const
  {
    twistedCloverApply(out, in, QUDA_TWIST_GAMMA5_INVERSE, parity);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedCloverPC::Dslash
  (ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    FullClover *cs = new FullClover(clover, false);
    FullClover *cI = new FullClover(clover, true);

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * mu;  //for invert twist (not daggered)
      double b = 1.;// / (1.0 + a*a);                     //for invert twist 
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, cs, cI,
				&static_cast<const cudaColorSpinorField&>(in), parity, dagger, 0,
				QUDA_DEG_DSLASH_CLOVER_TWIST_INV, a, b, 0.0, 0.0, commDim, profile);

	flops += 2376ll*in.Volume();
      } else {	
	twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, cs, cI,
				&static_cast<const cudaColorSpinorField&>(in), parity, dagger, 0,
				QUDA_DEG_CLOVER_TWIST_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);
        flops += 1320ll*in.Volume();
      }
    } else {//TWIST doublet :
      errorQuda("Non-degenerate DiracTwistedCloverPC is not implemented \n");
    }
    delete cs;
    delete cI;
  }

  // xpay version of the above
  void DiracTwistedCloverPC::DslashXpay
  (ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    FullClover *cs = new FullClover(clover, false);
    FullClover *cI = new FullClover(clover, true);

    if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      double a = -2.0 * kappa * in.TwistFlavor() * mu;  //for invert twist
      //      double b = k / (1.0 + a*a);                     //for invert twist	NO HABRÍA QUE APLICAR CLOVER_TWIST_INV???
      double b = k;                     //for invert twist	NO HABRÍA QUE APLICAR CLOVER_TWIST_INV???
      if (!dagger) {
	twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, cs, cI,
				&static_cast<const cudaColorSpinorField&>(in), parity, dagger,
				&static_cast<const cudaColorSpinorField&>(x),
				QUDA_DEG_DSLASH_CLOVER_TWIST_INV, a, b, 0.0, 0.0, commDim, profile);

        flops += 2400ll*in.Volume();
      } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
	twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, cs, cI,
				&static_cast<const cudaColorSpinorField&>(in), parity, dagger,
				&static_cast<const cudaColorSpinorField&>(x),
				QUDA_DEG_CLOVER_TWIST_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);

        flops += 1344ll*in.Volume();
      }
    } else {//TWIST_DOUBLET:
      errorQuda("Non-degenerate DiracTwistedCloverPC is not implemented \n");
    }
    delete cs;
    delete cI;
  }

  void DiracTwistedCloverPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;
    bool reset = newTmp(&tmp1, in);

    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    FullClover *cs = new FullClover(clover, false);
    FullClover *cI = new FullClover(clover, true);

    if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
      if (symmetric) {
	if (dagger) {
	  TwistCloverInv(*tmp1, in, parity[1]);
	  Dslash(out, *tmp1, parity[0]);
	  TwistCloverInv(*tmp1, out, parity[0]);
	  DslashXpay(out, *tmp1, parity[1], in, kappa2);
	} else {
	  Dslash(*tmp1, in, parity[0]);
	  DslashXpay(out, *tmp1, parity[1], in, kappa2);
	}
      } else { // asymmetric preconditioning
        double a = 2.0 * kappa * in.TwistFlavor() * mu;
	Dslash(*tmp1, in, parity[0]);
	twistedCloverDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, cs, cI,
				static_cast<cudaColorSpinorField*>(tmp1), parity[1], dagger,
				&static_cast<const cudaColorSpinorField&>(in),
				QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY, a, kappa2, 0.0, 0.0, commDim, profile);

	flops += (1320ll+96ll)*in.Volume();
      }
    } else { //Twist doublet
      errorQuda("Non-degenerate DiracTwistedCloverPC is not implemented \n");
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracTwistedCloverPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
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
        TwistCloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
        TwistCloverInv(*src, *tmp1, QUDA_EVEN_PARITY);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
        src = &(x.Even());
        TwistCloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
        TwistCloverInv(*src, *tmp1, QUDA_ODD_PARITY);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        src = &(x.Odd());
        TwistCloverInv(*tmp1, b.Odd(), QUDA_ODD_PARITY); // safe even when *tmp1 = b.odd
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        src = &(x.Even());
        TwistCloverInv(*tmp1, b.Even(), QUDA_EVEN_PARITY); // safe even when *tmp1 = b.even
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }
    } else {//doublet:
      errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }//end of doublet
    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }
  
  void DiracTwistedCloverPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
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
        TwistCloverInv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||   matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)
        DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
        TwistCloverInv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }
    } else { //twist doublet:
      errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }//end of twist doublet...
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::createCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T, double kappa, double mu, double mu_factor) const {
    double a = -2.0 * kappa * mu * T.Vectors().TwistFlavor();
    CoarseOp(Y, X, Xinv, Yhat, T, *gauge, &clover, kappa, a, -mu_factor, QUDA_TWISTED_CLOVERPC_DIRAC, matpcType);
  }
} // namespace quda
