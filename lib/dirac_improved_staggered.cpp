#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracParam &param)
    : Dirac(param), fatGauge(*(param.fatGauge)), longGauge(*(param.longGauge)) { }

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracImprovedStaggered &dirac)
    : Dirac(dirac), fatGauge(dirac.fatGauge), longGauge(dirac.longGauge) { }

  DiracImprovedStaggered::~DiracImprovedStaggered() { }

  DiracImprovedStaggered& DiracImprovedStaggered::operator=(const DiracImprovedStaggered &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      fatGauge = dirac.fatGauge;
      longGauge = dirac.longGauge;
    }
    return *this;
  }

  void DiracImprovedStaggered::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) {
      errorQuda("Staggered dslash requires 5-d fermion fields");
    }

    if (in.Precision() != out.Precision()) {
      errorQuda("Input and output spinor precisions don't match in dslash_quda");
    }

    if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not single parity, in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    }

    if ((out.Volume()/out.X(4) != 2*fatGauge.VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	(out.Volume()/out.X(4) != fatGauge.VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), fatGauge.VolumeCB());
    }
  }

  void DiracImprovedStaggered::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    ApplyImprovedStaggered(out, in, fatGauge, longGauge, 0., in, parity, dagger, commDim, profile);
    flops += 1146ll*in.Volume();
  }

  void DiracImprovedStaggered::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {    
    checkParitySpinor(in, out);

    // Need to catch the zero mass case.
    if (k == 0.0) {
      // There's a sign convention difference for Dslash vs DslashXpay, which is
      // triggered by looking for k == 0. We need to hack around this.
      if (dagger == QUDA_DAG_YES) {
        ApplyImprovedStaggered(out, in, fatGauge, longGauge, 0., x, parity, QUDA_DAG_NO, commDim, profile);
      } else {
        ApplyImprovedStaggered(out, in, fatGauge, longGauge, 0., x, parity, QUDA_DAG_YES, commDim, profile);
      }
      flops += 1146ll * in.Volume();
    } else {
      ApplyImprovedStaggered(out, in, fatGauge, longGauge, k, x, parity, dagger, commDim, profile);
      flops += 1158ll * in.Volume();
    }
  }

  // Full staggered operator
  void DiracImprovedStaggered::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    // Need to flip sign via dagger convention if mass == 0.
    if (mass == 0.0) {
      if (dagger == QUDA_DAG_YES) {
        ApplyImprovedStaggered(out, in, fatGauge, longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
      } else {
        ApplyImprovedStaggered(out, in, fatGauge, longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
      }
      flops += 1146ll * in.Volume();
    } else {
      ApplyImprovedStaggered(out, in, fatGauge, longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      flops += 1158ll * in.Volume();
    }
  }

  void DiracImprovedStaggered::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);
  
    //even
    Dslash(tmp1->Even(), in.Even(), QUDA_ODD_PARITY);  
    DslashXpay(out.Even(), tmp1->Even(), QUDA_EVEN_PARITY, in.Even(), 4*mass*mass);
  
    //odd
    Dslash(tmp1->Even(), in.Odd(), QUDA_EVEN_PARITY);  
    DslashXpay(out.Odd(), tmp1->Even(), QUDA_ODD_PARITY, in.Odd(), 4*mass*mass);    

    deleteTmp(&tmp1, reset);
  }

  void DiracImprovedStaggered::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			       ColorSpinorField &x, ColorSpinorField &b, 
			       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;  
  }

  void DiracImprovedStaggered::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				   const QudaSolutionType solType) const
  {
    // do nothing
  }

  void DiracImprovedStaggered::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa,
                                              double mass, double mu, double mu_factor) const
  {
    StaggeredCoarseOp(Y, X, T, fatGauge, mass, QUDA_ASQTAD_DIRAC, QUDA_MATPC_INVALID);
  }

  void DiracImprovedStaggered::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    fatGauge.prefetch(mem_space, stream);
    longGauge.prefetch(mem_space, stream);
  }

  DiracImprovedStaggeredPC::DiracImprovedStaggeredPC(const DiracParam &param)
    : DiracImprovedStaggered(param)
  {

  }

  DiracImprovedStaggeredPC::DiracImprovedStaggeredPC(const DiracImprovedStaggeredPC &dirac) 
    : DiracImprovedStaggered(dirac)
  {

  }

  DiracImprovedStaggeredPC::~DiracImprovedStaggeredPC()
  {

  }

  DiracImprovedStaggeredPC& DiracImprovedStaggeredPC::operator=(const DiracImprovedStaggeredPC &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
    }
 
    return *this;
  }

  // Unlike with clover, for ex, we don't need a custom Dslash or DslashXpay.
  // That's because the convention for preconditioned staggered is to
  // NOT divide out the factor of "2m", i.e., for the even system we invert
  // (4m^2 - D_eo D_oe), not (1 - (1/(4m^2)) D_eo D_oe).

  void DiracImprovedStaggeredPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);
  
    QudaParity parity = QUDA_INVALID_PARITY;
    QudaParity other_parity = QUDA_INVALID_PARITY;
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      parity = QUDA_EVEN_PARITY;
      other_parity = QUDA_ODD_PARITY;
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      parity = QUDA_ODD_PARITY;
      other_parity = QUDA_EVEN_PARITY;
    } else {
      errorQuda("Invalid matpcType(%d) in function\n", matpcType);    
    }

    // Convention note: Dslash applies D_eo, DslashXpay applies 4m^2 - D_oe!
    // Note the minus sign convention in the Xpay version.
    // This applies equally for the e <-> o permutation.

    Dslash(*tmp1, in, other_parity);  
    DslashXpay(out, *tmp1, parity, in, 4*mass*mass);

    deleteTmp(&tmp1, reset);
  }

  void DiracImprovedStaggeredPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    errorQuda("MdagM is no longer defined for DiracImprovedStaggeredPC. Use M instead.\n");
    /*
    // need extra temporary because for multi-gpu the input
    // and output fields cannot alias
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    M(out, *tmp2); // doesn't need to be Mdag b/c M is normal!
    deleteTmp(&tmp2, reset);
    */
  }

  void DiracImprovedStaggeredPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				 ColorSpinorField &x, ColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {

    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
      return;
    }
  
    // we desire solution to full system.
    // See sign convention comment in DiracStaggeredPC::M().
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {

      // With the convention given in DiracStaggered::M(),
      // the source is src = 2m b_e + D_eo b_o
      // But remember, DslashXpay actually applies
      // -D_eo. Flip the sign on 2m to compensate, and
      // then flip the overall sign.
      src = &(x.Odd());
      DslashXpay(*src, b.Odd(), QUDA_EVEN_PARITY, b.Even(), -2*mass);
      blas::ax(-1.0, *src);
      sol = &(x.Even());
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // See above, permute e <-> o
      src = &(x.Even());
      DslashXpay(*src, b.Even(), QUDA_ODD_PARITY, b.Odd(), -2*mass);
      blas::ax(-1.0, *src);
      sol = &(x.Odd());
    } else {
      errorQuda("MatPCType %d not valid for DiracStaggeredPC", matpcType);
    }

    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want 
  }

  void DiracImprovedStaggeredPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				     const QudaSolutionType solType) const
  {

    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }

    checkFullSpinor(x, b);

    // create full solution
    // See sign convention comment in DiracStaggeredPC::M()
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      
      // With the convention given in DiracStaggered::M(),
      // the reconstruct is x_o = 1/(2m) (b_o + D_oe x_e)
      // But remember: DslashXpay actually applies -D_oe, 
      // so just like above we need to flip the sign
      // on b_o. We then correct this by applying an additional
      // minus sign when we rescale by 2m.
      DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), -1.0);
      blas::ax(-0.5/mass, x.Odd());
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // See above, permute e <-> o
      DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), -1.0);
      blas::ax(-0.5/mass, x.Even());
    } else {
      errorQuda("MatPCType %d not valid for DiracStaggeredPC", matpcType);
    }
  }

} // namespace quda
