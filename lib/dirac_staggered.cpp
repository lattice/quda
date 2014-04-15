#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracStaggered::DiracStaggered(const DiracParam &param) : 
    Dirac(param), face1(param.gauge->X(), 4, 6, 1, param.gauge->Precision()), face2(param.gauge->X(), 4, 6, 1, param.gauge->Precision())
    //FIXME: this may break mixed precision multishift solver since may not have fatGauge initializeed yet
  {
  }

  DiracStaggered::DiracStaggered(const DiracStaggered &dirac) 
  : Dirac(dirac), face1(dirac.face1), face2(dirac.face2)
  {
  }

  DiracStaggered::~DiracStaggered() { }

  DiracStaggered& DiracStaggered::operator=(const DiracStaggered &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      face1 = dirac.face1;
      face2 = dirac.face2;
    }
    return *this;
  }

  void DiracStaggered::checkParitySpinor(const cudaColorSpinorField &in, const cudaColorSpinorField &out) const
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
  }


  void DiracStaggered::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			      const QudaParity parity) const
  {
    checkParitySpinor(in, out);

    initSpinorConstants(in, profile);
    setFace(face1, face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    staggeredDslashCuda(&out, gauge, &in, parity, dagger, 0, 0, commDim, profile);
  
    flops += 654ll*in.Volume();
  }

  void DiracStaggered::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				  const QudaParity parity, const cudaColorSpinorField &x,
				  const double &k) const
  {    
    checkParitySpinor(in, out);

    initSpinorConstants(in, profile);
    setFace(face1, face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    staggeredDslashCuda(&out, gauge, &in, parity, dagger, &x, k, commDim, profile);
  
    flops += 666ll*in.Volume();
  }

  // Full staggered operator
  void DiracStaggered::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in.Even());

    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp1, 2*mass);  
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp1, 2*mass);
  
    deleteTmp(&tmp1, reset);
  }

  void DiracStaggered::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);
  
    cudaColorSpinorField* mytmp = dynamic_cast<cudaColorSpinorField*>(&(tmp1->Even()));
    cudaColorSpinorField* ineven = dynamic_cast<cudaColorSpinorField*>(&(in.Even()));
    cudaColorSpinorField* inodd = dynamic_cast<cudaColorSpinorField*>(&(in.Odd()));
    cudaColorSpinorField* outeven = dynamic_cast<cudaColorSpinorField*>(&(out.Even()));
    cudaColorSpinorField* outodd = dynamic_cast<cudaColorSpinorField*>(&(out.Odd()));
  
    //even
    Dslash(*mytmp, *ineven, QUDA_ODD_PARITY);  
    DslashXpay(*outeven, *mytmp, QUDA_EVEN_PARITY, *ineven, 4*mass*mass);
  
    //odd
    Dslash(*mytmp, *inodd, QUDA_EVEN_PARITY);  
    DslashXpay(*outodd, *mytmp, QUDA_ODD_PARITY, *inodd, 4*mass*mass);    

    deleteTmp(&tmp1, reset);
  }

  void DiracStaggered::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			       cudaColorSpinorField &x, cudaColorSpinorField &b, 
			       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;  
  }

  void DiracStaggered::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				   const QudaSolutionType solType) const
  {
    // do nothing
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

  void DiracStaggeredPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    errorQuda("DiracStaggeredPC::M() is not implemented\n");
  }

  void DiracStaggeredPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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

  void DiracStaggeredPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
				 cudaColorSpinorField &x, cudaColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {
    src = &b;
    sol = &x;  
  }

  void DiracStaggeredPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				     const QudaSolutionType solType) const
  {
    // do nothing
  }

} // namespace quda
