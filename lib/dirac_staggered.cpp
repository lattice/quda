#include <dirac_quda.h>
#include <blas_quda.h>
#include <tune_quda.h>

DiracStaggered::DiracStaggered(const DiracParam &param) : Dirac(param),
  blockDslash(64, 1, 1), blockDslashXpay(64, 1, 1), blockDslashFace(64, 1, 1), blockDslashXpayFace(64, 1, 1),
  fatGauge(param.fatGauge), longGauge(param.longGauge)
{

}

DiracStaggered::DiracStaggered(const DiracStaggered &dirac) : Dirac(dirac),
  blockDslash(64, 1, 1), blockDslashXpay(64, 1, 1), blockDslashFace(64, 1, 1), blockDslashXpayFace(64, 1, 1),
  fatGauge(dirac.fatGauge), longGauge(dirac.longGauge)
{

}

DiracStaggered::~DiracStaggered()
{

}

DiracStaggered& DiracStaggered::operator=(const DiracStaggered &dirac)
{
  if (&dirac != this) {
    Dirac::operator=(dirac);
    blockDslash = dirac.blockDslash;
    blockDslashXpay = dirac.blockDslashXpay;
    blockDslashFace = dirac.blockDslashFace;
    blockDslashXpayFace = dirac.blockDslashXpayFace;
    fatGauge = dirac.fatGauge;
    longGauge = dirac.longGauge;
  }
 
  return *this;
}

// Find the best block size parameters for the Dslash and DslashXpay kernels
void DiracStaggered::Tune(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const cudaColorSpinorField &x) {
  setDslashTuning(QUDA_TUNE_YES);

  { // Tune Dslash
    TuneDiracStaggeredDslash dslashTune(*this, out, in);
    dslashTune.Benchmark(blockDslash);
  }

  { // Tune DslashXpay
    TuneDiracStaggeredDslashXpay dslashXpayTune(*this, out, in, x);
    dslashXpayTune.Benchmark(blockDslashXpay);
  }

  setDslashTuning(QUDA_TUNE_NO);
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

  if ((out.Volume() != 2*fatGauge->volume && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
      (out.Volume() != fatGauge->volume && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
      errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), fatGauge->volume);
  }
}


void DiracStaggered::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity) const
{
  if (!initDslash) {
    initDslashConstants(*fatGauge, in.Stride());
    initStaggeredConstants(*fatGauge, *longGauge);
  }
  checkParitySpinor(in, out);
    
  staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, dagger, 
		      0, 0, 0, out.volume, out.bytes, out.norm_bytes, in.Precision(), 
		      blockDslash, blockDslashFace);
    
  flops += 1146*in.volume;
}

void DiracStaggered::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity, const cudaColorSpinorField &x,
				const double &k) const
{    
  if (!initDslash){
    initDslashConstants(*fatGauge, in.Stride());
    initStaggeredConstants(*fatGauge, *longGauge);
  }
  checkParitySpinor(in, out);
  
  staggeredDslashCuda(out.v, out.norm, *fatGauge, *longGauge, in.v, in.norm, parity, 
		      dagger, x.v, x.norm, k, out.volume, out.bytes, out.norm_bytes, 
		      in.Precision(), blockDslashXpay, blockDslashXpayFace);
  
  flops += (1146+12)*in.volume;
}

void DiracStaggered::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  errorQuda("DiracStaggered::M() is not implemented");  
}

void DiracStaggered::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  
  if (!initDslash){
    initDslashConstants(*fatGauge, in.Stride());
    initStaggeredConstants(*fatGauge, *longGauge);
  }
  
  bool reset = newTmp(&tmp1, in);
  
  cudaColorSpinorField* mytmp = dynamic_cast<cudaColorSpinorField*>(tmp1->even);
  cudaColorSpinorField* ineven = dynamic_cast<cudaColorSpinorField*>(in.even);
  cudaColorSpinorField* inodd = dynamic_cast<cudaColorSpinorField*>(in.odd);
  cudaColorSpinorField* outeven = dynamic_cast<cudaColorSpinorField*>(out.even);
  cudaColorSpinorField* outodd = dynamic_cast<cudaColorSpinorField*>(out.odd);
  
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
  if (!initDslash){
    initDslashConstants(*fatGauge, in.Stride());
    initStaggeredConstants(*fatGauge, *longGauge);
  }
  
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




