#include <dirac_quda.h>
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
    tmp1=dirac.tmp1;
  }

  return *this;
}

void DiracWilson::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const QudaParity parity, const QudaDagType dagger) {

  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);

  dslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
	     0, 0, 0, out.volume, out.length, in.Precision());

  flops += 1320*in.volume;
}

void DiracWilson::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			     const QudaParity parity, const QudaDagType dagger,
			     const cudaColorSpinorField &x, const double &k) {

  if (!initDslash) initDslashConstants(gauge, in.Stride(), 0);
  checkParitySpinor(in, out);

  dslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, x.v, x.norm, k, 
	     out.volume, out.length, in.Precision());

  flops += (1320+48)*in.volume;
}

void DiracWilson::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType dagger) {
  checkFullSpinor(out, in);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, dagger, in.Odd(), -kappa);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, dagger, in.Even(), -kappa);
}

void DiracWilson::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) {
  checkFullSpinor(out, in);

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  } else {
    checkFullSpinor(*tmp1, in);
  }

  M(*tmp1, in, QUDA_DAG_NO);
  M(out, *tmp1, QUDA_DAG_YES);

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
}

void DiracWilson::Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solutionType, const QudaDagType dagger) {
  ColorSpinorParam param;

  src = &b;
  sol = &x;
}

void DiracWilson::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solutionType, const QudaDagType dagger) {
  // do nothing
}

DiracWilsonPC::DiracWilsonPC(const DiracParam &param)
  : DiracWilson(param) {

}

DiracWilsonPC::DiracWilsonPC(const DiracWilsonPC &dirac) 
  : DiracWilson(dirac) {

}

DiracWilsonPC::~DiracWilsonPC() {

}

DiracWilsonPC& DiracWilsonPC::operator=(const DiracWilsonPC &dirac) {

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    tmp1 = dirac.tmp1;
  }

  return *this;
}

void DiracWilsonPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaDagType dagger) {
  double kappa2 = -kappa*kappa;

  ColorSpinorParam param;
  param.create = QUDA_NULL_FIELD_CREATE;
  bool reset = false;
  if (!tmp1) {
    tmp1 = new cudaColorSpinorField(in, param); // only create if necessary
    reset = true;
  }

  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY, dagger);
    DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, dagger, in, kappa2); 
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY, dagger);
    DslashXpay(out, *tmp1, QUDA_ODD_PARITY, dagger, in, kappa2); 
  } else {
    errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
  }

  if (reset) {
    delete tmp1;
    tmp1 = 0;
  }
  
}

void DiracWilsonPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in){
  M(out, in, QUDA_DAG_NO);
  M(out, out, QUDA_DAG_YES);
}

void DiracWilsonPC::Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
			    const QudaSolutionType solType, const QudaDagType dagger) {

  // we desire solution to preconditioned system
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    DiracWilson::Prepare(src, sol, x, b, solType, dagger);
    return;
  }

  // we desire solution to full system
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = b_e + k D_eo b_o
    DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, dagger, b.Even(), kappa);
    src = &(x.Odd());
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = b_o + k D_oe b_e
    DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, dagger, b.Odd(), kappa);
    src = &(x.Even());
    sol = &(x.Odd());
  } else {
    errorQuda("MatPCType %d not valid for DiracWilson", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

}

void DiracWilsonPC::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType, const QudaDagType dagger) {

  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    return DiracWilson::Reconstruct(x, b, solType, dagger);
  }				

  // create full solution

  checkFullSpinor(x, b);
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // x_o = b_o + k D_oe x_e
    DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, dagger, b.Odd(), kappa);
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // x_e = b_e + k D_eo x_o
    DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, dagger, b.Even(), kappa);
  } else {
    errorQuda("MatPCType %d not valid for DiracWilson", matpcType);
  }
}

