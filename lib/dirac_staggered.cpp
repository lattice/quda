#include <dirac_quda.h>
#include <blas_quda.h>

DiracStaggeredPC::DiracStaggeredPC(const DiracParam &param)
    : Dirac(param), fatGauge(param.fatGauge), longGauge(param.longGauge) {

}

DiracStaggeredPC::DiracStaggeredPC(const DiracStaggeredPC &dirac) 
  : Dirac(dirac) {

}

DiracStaggeredPC::~DiracStaggeredPC() {

}

DiracStaggeredPC& DiracStaggeredPC::operator=(const DiracStaggeredPC &dirac) {

  if (&dirac != this) {
    Dirac::operator=(dirac);
    tmp1=dirac.tmp1;
  }
 
  return *this;
}



void DiracStaggeredPC::checkParitySpinor(const cudaColorSpinorField &in, const cudaColorSpinorField &out) {

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

  if ((out.Volume() != 2*fatGauge->volume && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
      (out.Volume() != fatGauge->volume && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), fatGauge->volume);
  }

}


void DiracStaggeredPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity, const QudaDagType dagger) {

  if (!initDslash) {
    initDslashConstants(*fatGauge, in.Stride(), 0);
  }
    checkParitySpinor(in, out);
    
    staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, dagger, 
			0, 0, 0, out.volume, out.length, in.Precision());
    
    flops += 1187*in.volume;
}

void DiracStaggeredPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity, const QudaDagType dagger,
				const cudaColorSpinorField &x, const double &k) {
    
    if (!initDslash){
	initDslashConstants(*fatGauge, in.Stride(), 0);
    }
    checkParitySpinor(in, out);
    
    staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, dagger, x.v, x.norm, k, 
			out.volume, out.length, in.Precision());
    
    flops += (1187+12)*in.volume;
}

void DiracStaggeredPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType dagger) 
{
  errorQuda("M function in DiracStaggeredPC is not implemented\n");
}

void DiracStaggeredPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) 
{
  
  if (!initDslash){
    initDslashConstants(*fatGauge, in.Stride(), 0);
  }
  
  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = false;
  }
  
  QudaParity parity;
  QudaParity other_parity;
  if (matpcType == QUDA_MATPC_EVEN_EVEN){
    parity = QUDA_EVEN_PARITY;
    other_parity = QUDA_ODD_PARITY;
  }else if (matpcType == QUDA_MATPC_ODD_ODD){
    parity = QUDA_ODD_PARITY;
    other_parity = QUDA_EVEN_PARITY;
  }else{
    errorQuda("Invalid matpcType(%d) in function\n", matpcType);    
  }
  
  QudaDagType dagger = QUDA_DAG_NO;
  
  Dslash(*tmp1, in, other_parity, dagger);  
  DslashXpay(out, *tmp1, parity, dagger, in, 4*mass*mass);

  if (reset) {
      delete tmp1;
      tmp1 = 0;
  }
}

void DiracStaggeredPC::Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			     cudaColorSpinorField &x, cudaColorSpinorField &b, 
			     const QudaSolutionType solutionType, const QudaDagType dagger) 
{
  src = &b;
  sol = &x;  
}

void DiracStaggeredPC::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solutionType, const QudaDagType dagger) {
  // do nothing
}




DiracStaggered::DiracStaggered(const DiracParam &param)
    : Dirac(param), fatGauge(param.fatGauge), longGauge(param.longGauge) {

}

DiracStaggered::DiracStaggered(const DiracStaggered &dirac) 
  : Dirac(dirac) {

}

DiracStaggered::~DiracStaggered() {

}

DiracStaggered& DiracStaggered::operator=(const DiracStaggered &dirac) {

  if (&dirac != this) {
    Dirac::operator=(dirac);
    tmp1=dirac.tmp1;
  }
 
  return *this;
}



void DiracStaggered::checkParitySpinor(const cudaColorSpinorField &in, const cudaColorSpinorField &out) {

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

  if ((out.Volume() != 2*fatGauge->volume && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
      (out.Volume() != fatGauge->volume && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), fatGauge->volume);
  }

}


void DiracStaggered::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity, const QudaDagType dagger) {

  if (!initDslash) {
    initDslashConstants(*fatGauge, in.Stride(), 0);
  }
    checkParitySpinor(in, out);
    
    staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, dagger, 
			0, 0, 0, out.volume, out.length, in.Precision());
    
    flops += 1187*in.volume;
}

void DiracStaggered::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity, const QudaDagType dagger,
				const cudaColorSpinorField &x, const double &k) {
    
    if (!initDslash){
	initDslashConstants(*fatGauge, in.Stride(), 0);
    }
    checkParitySpinor(in, out);
    
    staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, dagger, x.v, x.norm, k, 
			out.volume, out.length, in.Precision());
    
    flops += (1187+12)*in.volume;
}

void DiracStaggered::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType dagger) 
{
  errorQuda("M function in DiracStaggeredPC is not implemented\n");  
}

void DiracStaggered::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) 
{
  
  if (!initDslash){
    initDslashConstants(*fatGauge, in.Stride(), 0);
  }
  
  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = false;
  }
  
 

  QudaDagType dagger = QUDA_DAG_NO;
  
  cudaColorSpinorField* mytmp = dynamic_cast<cudaColorSpinorField*>(tmp1->even);
  cudaColorSpinorField* ineven = dynamic_cast<cudaColorSpinorField*>(in.even);
  cudaColorSpinorField* inodd = dynamic_cast<cudaColorSpinorField*>(in.odd);
  cudaColorSpinorField* outeven = dynamic_cast<cudaColorSpinorField*>(out.even);
  cudaColorSpinorField* outodd = dynamic_cast<cudaColorSpinorField*>(out.odd);
  
  //even
  Dslash(*mytmp, *ineven, QUDA_ODD_PARITY, dagger);  
  DslashXpay(*outeven, *mytmp, QUDA_EVEN_PARITY, dagger, *ineven, 4*mass*mass);
  
  //odd
  Dslash(*mytmp, *inodd, QUDA_EVEN_PARITY, dagger);  
  DslashXpay(*outodd, *mytmp, QUDA_ODD_PARITY, dagger, *inodd, 4*mass*mass);    
  
  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
}

void DiracStaggered::Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			     cudaColorSpinorField &x, cudaColorSpinorField &b, 
			     const QudaSolutionType solutionType, const QudaDagType dagger) 
{
  src = &b;
  sol = &x;  
}

void DiracStaggered::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				 const QudaSolutionType solutionType, const QudaDagType dagger) {
  // do nothing
}

