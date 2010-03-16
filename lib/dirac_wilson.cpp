#include <dirac.h>
#include <blas_quda.h>

DiracWilson::DiracWilson(const DiracParam &param)
  : Dirac(param) {

}

DiracWilson::DiracWilson(const DiracWilson &dirac) 
  : Dirac(dirac) {

}

DiracWilson::~DiracWilson() {

}

DiracWilson& DiracWilson::operator=(const DiracWilson &dirac) {

  if (&dirac != this) {
    Dirac::operator=(dirac);
  }

  return *this;
}

void DiracWilson::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const int parity, const QudaDagType dagger) {

  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);

  dslashCuda(out.v, gauge, in.v, parity, dagger, out.volume, 
	     out.length, out.norm, in.norm, in.Precision());

  flops += 1320*in.volume;
}

void DiracWilson::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const int parity, const QudaDagType dagger,
			     const cudaColorSpinorField &x, const double &k) {

  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);

  dslashXpayCuda(out.v, gauge, in.v, parity, dagger, x.v, k, out.volume, out.length, 
		 out.norm, in.norm, x.norm, in.Precision());

  flops += (1320+48)*in.volume;
}

void DiracWilson::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType dagger) {
  checkFullSpinor(out, in);
  DslashXpay(out.Odd(), in.Even(), 1, dagger, in.Odd(), -kappa);
  DslashXpay(out.Even(), in.Odd(), 0, dagger, in.Even(), -kappa);
}

void DiracWilson::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) {
  checkFullSpinor(out, in);
  ColorSpinorParam param;
  param.create = QUDA_NULL_CREATE;
  cudaColorSpinorField tmp(in, param);
  M(tmp, in, QUDA_DAG_NO);
  M(out, tmp, QUDA_DAG_YES);
}

void DiracWilson::Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
			  const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
			  const QudaSolutionType solutionType, const QudaDagType dagger) {
  ColorSpinorParam param;
  param.create = QUDA_REFERENCE_CREATE;

  src = cudaColorSpinorField(b, param);
  sol = cudaColorSpinorField(x, param);
}

void DiracWilson::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solutionType, const QudaDagType dagger) {
  // do nothing
}

DiracWilsonPC::DiracWilsonPC(const DiracParam &param)
  : DiracWilson(param), tmp(*(param.tmp)) {

}

DiracWilsonPC::DiracWilsonPC(const DiracWilsonPC &dirac) 
  : DiracWilson(dirac), tmp(dirac.tmp) {

}

DiracWilsonPC::~DiracWilsonPC() {

}

DiracWilsonPC& DiracWilsonPC::operator=(const DiracWilsonPC &dirac) {

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    tmp = dirac.tmp;
  }

  return *this;
}

void DiracWilsonPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaDagType dagger) {
  double kappa2 = -kappa*kappa;
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    Dslash(tmp, in, 1, dagger);
    DslashXpay(out, tmp, 0, dagger, in, kappa2); 
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    Dslash(tmp, in, 0, dagger);
    DslashXpay(out, tmp, 1, dagger, in, kappa2); 
  } else {
    errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
  }
  
}

void DiracWilsonPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in){
  M(out, in, QUDA_DAG_NO);
  M(out, out, QUDA_DAG_YES);
}

void DiracWilsonPC::Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
			    const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
			    const QudaSolutionType solType, const QudaDagType dagger) {

  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    DiracWilson::Prepare(src, sol, x, b, solType, dagger);
  }

  ColorSpinorParam param;
  param.create = QUDA_REFERENCE_CREATE;

  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = b_e + k D_eo b_o
    DslashXpay(src, b.Odd(), 0, dagger, b.Even(), kappa);
    sol = cudaColorSpinorField(x.Even(), param);    
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = b_o + k D_oe b_e
    DslashXpay(src, b.Even(), 1, dagger, b.Odd(), kappa);
    sol = cudaColorSpinorField(x.Odd(), param);    
  } else {
    errorQuda("MatPCType %d not valid for DiracWilson", matpcType);
  }

}

void DiracWilsonPC::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType, const QudaDagType dagger) {

  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    DiracWilson::Reconstruct(x, b, solType, dagger);
  }				

  checkFullSpinor(x, b);
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // x_o = b_o + k D_oe x_e
    DslashXpay(x.Odd(), x.Even(), 1, dagger, b.Odd(), kappa);
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // x_e = b_e + k D_eo x_o
    DslashXpay(x.Even(), x.Odd(), 0, dagger, b.Even(), kappa);
  } else {
    errorQuda("MatPCType %d not valid for DiracWilson", matpcType);
  }
}

