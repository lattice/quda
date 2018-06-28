#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracStaggered::DiracStaggered(const DiracParam &param) : Dirac(param) { }

  DiracStaggered::DiracStaggered(const DiracStaggered &dirac) : Dirac(dirac) { }

  DiracStaggered::~DiracStaggered() { }

  DiracStaggered& DiracStaggered::operator=(const DiracStaggered &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
    }
    return *this;
  }

  void DiracStaggered::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) {
      errorQuda("Staggered dslash requires 5-d fermion fields");
    }

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

    if ((out.Volume()/out.X(4) != 2*gauge->VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	(out.Volume()/out.X(4) != gauge->VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), gauge->VolumeCB());
    }
  }


  void DiracStaggered::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			      const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) {
      staggeredDslashCuda(&static_cast<cudaColorSpinorField&>(out), 
			  *gauge, &static_cast<const cudaColorSpinorField&>(in), parity, 
			  dagger, 0, 0, commDim, profile);
    } else {
      errorQuda("Not supported");
    }

    flops += 570ll*in.Volume();
  }

  void DiracStaggered::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				  const QudaParity parity, const ColorSpinorField &x,
				  const double &k) const
  {    
    checkParitySpinor(in, out);

    if (checkLocation(out, in, x) == QUDA_CUDA_FIELD_LOCATION) {
      staggeredDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			  &static_cast<const cudaColorSpinorField&>(in), parity, dagger, 
			  &static_cast<const cudaColorSpinorField&>(x), k, commDim, profile);
    } else {
      errorQuda("Not supported");
    }  

    flops += 582ll*in.Volume();
  }

  // Full staggered operator
  void DiracStaggered::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), 2*mass);  
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), 2*mass);
  }

  void DiracStaggered::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
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

  void DiracStaggered::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			       ColorSpinorField &x, ColorSpinorField &b, 
			       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;  
  }

  void DiracStaggered::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
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
  hopping terms on the coarse grid.  Y[2*dim] is
  the color matrix that is diagonal on the coarse
  grid
  */
  void DiracStaggered::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
           double kappa, double mass, double mu, double mu_factor) const {
    StaggeredCoarseOp(Y, X, T, *gauge, mass, QUDA_STAGGERED_DIRAC, QUDA_MATPC_INVALID);
  }


  DiracStaggeredPC::DiracStaggeredPC(const DiracParam &param)
    : DiracStaggered(param)
  {

  }

  DiracStaggeredPC::DiracStaggeredPC(const DiracStaggeredPC &dirac) 
    : DiracStaggered(dirac)
  {

  }

  DiracStaggeredPC::~DiracStaggeredPC()
  {

  }

  DiracStaggeredPC& DiracStaggeredPC::operator=(const DiracStaggeredPC &dirac)
  {
    if (&dirac != this) {
      DiracStaggered::operator=(dirac);
    }
 
    return *this;
  }

  void DiracStaggeredPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    errorQuda("DiracStaggeredPC::M() is not implemented\n");
  }

  void DiracStaggeredPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
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

  void DiracStaggeredPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				 ColorSpinorField &x, ColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {
    src = &b;
    sol = &x;  
  }

  void DiracStaggeredPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				     const QudaSolutionType solType) const
  {
    // do nothing
  }



} // namespace quda
