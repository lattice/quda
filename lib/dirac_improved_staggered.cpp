#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  namespace improvedstaggered {
#include <dslash_init.cuh>
  }

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracParam &param) : 
    Dirac(param), fatGauge(*(param.fatGauge)), longGauge(*(param.longGauge)), 
    face1(param.fatGauge->X(), 4, 6, 3, param.fatGauge->Precision()),
    face2(param.fatGauge->X(), 4, 6, 3, param.fatGauge->Precision()) 
    //FIXME: this may break mixed precision multishift solver since may not have fatGauge initializeed yet
  {
    improvedstaggered::initConstants(*param.gauge, profile);    
    improvedstaggered::initStaggeredConstants(fatGauge, longGauge, profile);
  }

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracImprovedStaggered &dirac) 
  : Dirac(dirac), fatGauge(dirac.fatGauge), longGauge(dirac.longGauge), face1(dirac.face1), face2(dirac.face2)
  {
    improvedstaggered::initConstants(*dirac.gauge, profile);
    improvedstaggered::initStaggeredConstants(fatGauge, longGauge, profile);
  }

  DiracImprovedStaggered::~DiracImprovedStaggered() { }

  DiracImprovedStaggered& DiracImprovedStaggered::operator=(const DiracImprovedStaggered &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      fatGauge = dirac.fatGauge;
      longGauge = dirac.longGauge;
      face1 = dirac.face1;
      face2 = dirac.face2;
    }
    return *this;
  }

  void DiracImprovedStaggered::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Precision() != out.Precision()) {
      errorQuda("Input and output spinor precisions don't match in dslash_quda");
    }

    if (in.Stride() != out.Stride()) {
      errorQuda("Input %d and output %d spinor strides don't match in dslash_quda", in.Stride(), out.Stride());
    }

    if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not single parity, in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    }

    if ((out.Volume() != 2*fatGauge.VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	(out.Volume() != fatGauge.VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), fatGauge.VolumeCB());
    }
  }


  void DiracImprovedStaggered::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			      const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    if (Location(out, in) == QUDA_CUDA_FIELD_LOCATION) {
      improvedstaggered::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
      improvedStaggeredDslashCuda(&static_cast<cudaColorSpinorField&>(out), fatGauge, longGauge,
				  &static_cast<const cudaColorSpinorField&>(in), parity, 
				  dagger, 0, 0, commDim, profile);
    } else {
      errorQuda("Not supported");
    }  

    flops += 1146ll*in.Volume();
  }

  void DiracImprovedStaggered::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				  const QudaParity parity, const ColorSpinorField &x,
				  const double &k) const
  {    
    checkParitySpinor(in, out);

    if (Location(out, in, x) == QUDA_CUDA_FIELD_LOCATION) {
      improvedstaggered::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
      improvedStaggeredDslashCuda(&static_cast<cudaColorSpinorField&>(out), fatGauge, longGauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger, 
			  &static_cast<const cudaColorSpinorField&>(x), k, commDim, profile);
    } else {
      errorQuda("Not supported");
    }  

    flops += 1158ll*in.Volume();
  }

  // Full staggered operator
  void DiracImprovedStaggered::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), 2*mass);  
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), 2*mass);
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

  /* Creates the coarse grid dirac operator
  Takes: multigrid transfer class, which knows
  about the coarse grid blocking, as well as
  having prolongate and restrict member functions
  
  Returns: Color matrices Y[0..2*dim] corresponding
  to the coarse grid operator.  The first 2*dim
  matrices correspond to the forward/backward
  hopping terms on the coarse grid.  X[2*dim] is
  the color matrix that is diagonal on the coarse
  grid , this one is trivial but let's keep it for the moment
  */

  void DiracImprovedStaggered::createCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T) const {
    CoarseKSOp(Y, X, Xinv, Yhat, T, &fatGauge, &longGauge,  2*mass, QUDA_ASQTAD_DIRAC, QUDA_MATPC_INVALID);//
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

  void DiracImprovedStaggeredPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    errorQuda("DiracImprovedStaggeredPC::M() is not implemented\n");
  }

  void DiracImprovedStaggeredPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
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
    Dslash(*tmp1, in, other_parity);  
    DslashXpay(out, *tmp1, parity, in, 4*mass*mass);

    deleteTmp(&tmp1, reset);
  }

  void DiracImprovedStaggeredPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				 ColorSpinorField &x, ColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {//PC staggered multigrid version:
      // we desire solution to full system. It's a bit hacky : 1) compute -2m*be-D_eo b_o and 2) apply -1
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // src = 2m*b_e + D_eo b_o
        DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), -2.0*mass);
        blas::ax(-1.0 , x.Odd());
        src = &(x.Odd());
        sol = &(x.Even());
      } else if ( matpcType == QUDA_MATPC_ODD_ODD ) { 
      // src = 2m*b_o + D_oe b_e
        DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), -2.0*mass);
        blas::ax(-1.0 , x.Even());
        src = &(x.Even());
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracImprovedStaggeredPC", matpcType);
      }
    }  
  }

  void DiracImprovedStaggeredPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				     const QudaSolutionType solType) const
  {

    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {

      return;//do nothing

    } else {//hack: 1) compute -1.0*be-D_eo b_o and 2) apply - (1 / 2m)
      checkFullSpinor(x, b);
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = 1.0 / 2m ( b_o + D_oe x_e)
        DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), -1.0);
        blas::ax(-1.0 / (2.0*mass), x.Odd());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = 1.0 / 2m ( b_e + D_eo x_o)
        DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), -1.0);
        blas::ax(-1.0 / (2.0*mass), x.Even());
      } else {
        errorQuda("MatPCType %d not valid for DiracImprovedStaggeredPC", matpcType);
      }
    }
  }

} // namespace quda
