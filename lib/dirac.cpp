#include <dirac_quda.h>
#include <dslash_quda.h>

Dirac::Dirac(const DiracParam &param) 
  : gauge(*(param.gauge)), kappa(param.kappa), mass(param.mass), matpcType(param.matpcType), 
    flops(0), tmp1(param.tmp1), tmp2(param.tmp2) {

}

Dirac::Dirac(const Dirac &dirac) 
  : gauge(dirac.gauge), kappa(dirac.kappa), matpcType(dirac.matpcType), 
    flops(0), tmp1(dirac.tmp1), tmp2(dirac.tmp2) {

}

Dirac::~Dirac() {

}

Dirac& Dirac::operator=(const Dirac &dirac) {

  if(&dirac != this) {
    gauge = dirac.gauge;
    kappa = dirac.kappa;
    matpcType = dirac.matpcType;
    flops = 0;
    tmp1 = dirac.tmp1;
    tmp2 = dirac.tmp2;
  }

  return *this;

}

void Dirac::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) {

  if (in.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS || 
      out.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) {
    errorQuda("Cuda Dirac operator requires UKQCD basis, out = %d, in = %d", 
	      out.GammaBasis(), in.GammaBasis());
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

  if ((out.Volume() != 2*gauge.volume && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
      (out.Volume() != gauge.volume && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
    errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), gauge.volume);
  }

}

void Dirac::checkFullSpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) {
   if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() != QUDA_FULL_SITE_SUBSET) {
    errorQuda("ColorSpinorFields are not full fields, in = %d, out = %d", 
	      in.SiteSubset(), out.SiteSubset());
  } 
}

// Dirac operator factory
Dirac* Dirac::create(const DiracParam &param) {
  
  if (param.type == QUDA_WILSON_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracWilson operator\n");
    return new DiracWilson(param);
  } else if (param.type == QUDA_WILSONPC_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracWilsonPC operator\n");
    return new DiracWilsonPC(param);
  } else if (param.type == QUDA_CLOVER_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracClover operator\n");
    return new DiracClover(param);
  } else if (param.type == QUDA_CLOVERPC_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracCloverPC operator\n");
    return new DiracCloverPC(param);
  } else if (param.type == QUDA_ASQTAD_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracStaggered operator\n");
    return new DiracStaggered(param);
  } else if (param.type == QUDA_ASQTADPC_DIRAC) {
    if (param.verbose >= QUDA_VERBOSE) printfQuda("Creating a DiracStaggeredPC operator\n");
    return new DiracStaggeredPC(param);    
  } else {
    return 0;
  }

}
