#include <dirac.h>
#include <iostream>

DiracClover::DiracClover(const DiracParam &param)
  : DiracWilson(param), clover(*(param.clover)) {

}

DiracClover::DiracClover(const DiracClover &dirac) 
  : DiracWilson(dirac), clover(dirac.clover) {

}

DiracClover::~DiracClover() {

}

DiracClover& DiracClover::operator=(const DiracClover &dirac) {

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    clover = dirac.clover;
  }

  return *this;
}

void DiracClover::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const FullClover &clover) {
  Dirac::checkParitySpinor(out, in);

  if (out.Volume() != clover.even.volume) {
    errorQuda("Spinor volume %d doesn't match even clover volume %d",
	      out.Volume(), clover.even.volume);
  }
  if (out.Volume() != clover.odd.volume) {
    errorQuda("Spinor volume %d doesn't match odd clover volume %d",
	      out.Volume(), clover.odd.volume);
  }

#if (__CUDA_ARCH__ != 130)
  if ((clover.even.precision == QUDA_DOUBLE_PRECISION) ||
      (clover.odd.precision == QUDA_DOUBLE_PRECISION)) {
    errorQuda("Double precision not supported on this GPU");
  }
#endif

}

// Protected method, also used for applying cloverInv
void DiracClover::cloverApply(cudaColorSpinorField &out, const FullClover &clover, 
			      const cudaColorSpinorField &in, const int parity) {

  if (!initDslash) initDslashConstants(gauge, in.stride, clover.even.stride);
  checkParitySpinor(in, out, clover);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    cloverDCuda((double2*)out.v, gauge, clover, (double2*)in.v, parity, out.volume, out.length);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    cloverSCuda((float4*)out.v, gauge, clover, (float4*)in.v, parity, out.volume, out.length);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    cloverHCuda((short4*)out.v, (float*)out.norm, gauge, clover, (short4*)in.v, 
		(float*)in.norm, parity, out.volume, out.length);
  }
  checkCudaError();

  flops += 504*in.volume;
}

// Public method to apply the clover term only
void DiracClover::Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			 const int parity) {
  cloverApply(out, clover, in, parity);
}

// FIXME: create kernel to eliminate tmp
void DiracClover::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType dagger) {
  checkFullSpinor(out, in);

  ColorSpinorParam param;
  param.create = QUDA_NULL_CREATE;
  cudaColorSpinorField tmp(in.Even(), param);

  Clover(tmp, in.Odd(), 1);
  DslashXpay(out.Odd(), in.Even(), 1, dagger, tmp, -kappa);
  Clover(tmp, in.Even(), 0);
  DslashXpay(out.Even(), in.Odd(), 0, dagger, tmp, -kappa);
}

void DiracClover::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) {
  checkFullSpinor(out, in);
  ColorSpinorParam param;
  param.create = QUDA_NULL_CREATE;
  cudaColorSpinorField tmp(in, param);
  M(tmp, in, QUDA_DAG_NO);
  M(out, tmp, QUDA_DAG_YES);
}

void DiracClover::Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
			  const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
			  const QudaSolutionType solType, const QudaDagType dagger) {
  ColorSpinorParam param;
  param.create = QUDA_REFERENCE_CREATE;

  src = cudaColorSpinorField(b, param);
  sol = cudaColorSpinorField(x, param);
}

void DiracClover::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType, const QudaDagType dagger) {
  // do nothing
}

DiracCloverPC::DiracCloverPC(const DiracParam &param)
  : DiracClover(param), cloverInv(*(param.cloverInv)), tmp(*(param.tmp)) {

}

DiracCloverPC::DiracCloverPC(const DiracCloverPC &dirac) 
  : DiracClover(dirac), cloverInv(dirac.clover), tmp(dirac.tmp)  {

}

DiracCloverPC::~DiracCloverPC() {

}

DiracCloverPC& DiracCloverPC::operator=(const DiracCloverPC &dirac) {

  if (&dirac != this) {
    DiracClover::operator=(dirac);
    cloverInv = dirac.cloverInv;
    tmp = dirac.tmp;
  }

  return *this;
}

// Public method
void DiracCloverPC::CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			      const int parity) {
  cloverApply(out, cloverInv, in, parity);
}

// apply hopping term, then clover: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (A_ee^-1 D^dagger_eo) or (A_oo^-1 D^dagger_oe)
void DiracCloverPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			   const int parity, const QudaDagType dagger) {

  if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);
  checkParitySpinor(in, out, cloverInv);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    cloverDslashDCuda((double2*)out.v, gauge, cloverInv, (double2*)in.v, parity, dagger,
		      out.volume, out.length);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    cloverDslashSCuda((float4*)out.v, gauge, cloverInv, (float4*)in.v, parity, dagger,
		      out.volume, out.length);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    cloverDslashHCuda((short4*)out.v, (float*)out.norm, gauge, cloverInv, (short4*)in.v, 
		      (float*)in.norm, parity, dagger, out.volume, out.length);
  }
  checkCudaError();

  flops += (1320+504)*in.volume;
}

// xpay version of the above
void DiracCloverPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			       const int parity, const QudaDagType dagger, 
			       const cudaColorSpinorField &x, const double &k) {

  if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);
  checkParitySpinor(in, out, cloverInv);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    cloverDslashXpayDCuda((double2*)out.v, gauge, cloverInv, (double2*)in.v, parity, 
			  dagger, (double2*)x.v, k, out.volume, out.length);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    cloverDslashXpaySCuda((float4*)out.v, gauge, cloverInv, (float4*)in.v, parity, 
			  dagger, (float4*)x.v, k, out.volume, out.length);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    cloverDslashXpayHCuda((short4*)out.v, (float*)out.norm, gauge, cloverInv, 
			  (short4*)in.v, (float*)in.norm, parity, dagger, 
			  (short4*)x.v, (float*)x.norm, k, out.volume, out.length);
  }
  checkCudaError();

  flops += (1320+504+48)*in.volume;
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void DiracCloverPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaDagType dagger) {
  double kappa2 = -kappa*kappa;

  // FIXME: For asymmetric, a "DslashCxpay" kernel would improve performance.

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(tmp, in, 1, dagger);
    Clover(out, in, 0);
    DiracWilson::DslashXpay(out, tmp, 0, dagger, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(tmp, in, 0, dagger);
    Clover(out, in, 1);
    DiracWilson::DslashXpay(out, tmp, 1, dagger, out, kappa2);
  } else if (!dagger) { // symmetric preconditioning
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(tmp, in, 1, dagger);
      DslashXpay(out, tmp, 0, dagger, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(tmp, in, 0, dagger);
      DslashXpay(out, tmp, 1, dagger, in, kappa2); 
    } else {
      errorQuda("Invalid matpcType");
    }
  } else { // symmetric preconditioning, dagger
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      CloverInv(out, in, 0); 
      Dslash(tmp, out, 1, dagger);
      DiracWilson::DslashXpay(out, tmp, 0, dagger, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      CloverInv(out, in, 1); 
      Dslash(tmp, out, 0, dagger);
      DiracWilson::DslashXpay(out, tmp, 1, dagger, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
    }
  }

  
}

void DiracCloverPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) {
  // need extra temporary because of symmetric preconditioning dagger
  ColorSpinorParam param;
  param.create = QUDA_NULL_CREATE;
  cudaColorSpinorField tmp2(tmp, param);
  M(tmp2, in, QUDA_DAG_NO);
  M(out, tmp2, QUDA_DAG_YES);
}

void DiracCloverPC::Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol, 
			    const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
			    const QudaSolutionType solType, const QudaDagType dagger) {

  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    DiracClover::Prepare(src, sol, x, b, solType, dagger);
    return;
  }

  ColorSpinorParam param;
  param.create = QUDA_REFERENCE_CREATE;

  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
    CloverInv(src, b.Odd(), 1);
    DiracWilson::DslashXpay(tmp, src, 0, dagger, b.Even(), kappa);
    CloverInv(src, tmp, 0);
    sol = cudaColorSpinorField(x.Even(), param);
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
    CloverInv(src, b.Even(), 0);
    DiracWilson::DslashXpay(tmp, src, 1, dagger, b.Odd(), kappa);
    CloverInv(src, tmp, 1);
    sol = cudaColorSpinorField(x.Odd(), param);
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    CloverInv(tmp, b.Odd(), 1); // safe even when tmp = b.odd
    DiracWilson::DslashXpay(src, tmp, 0, dagger, b.Even(), kappa);
    sol = cudaColorSpinorField(x.Even(), param);
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    CloverInv(tmp, b.Even(), 0); // safe even when tmp = b.even
    DiracWilson::DslashXpay(src, tmp, 1, dagger, b.Odd(), kappa);
    sol = cudaColorSpinorField(x.Odd(), param);
  } else {
    errorQuda("MatPCType %d not valid for DiracClover", matpcType);
  }

}

void DiracCloverPC::Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType, const QudaDagType dagger) {

  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    DiracClover::Reconstruct(x, b, solType, dagger);
    return;
  }

  checkFullSpinor(x, b);

  if (matpcType == QUDA_MATPC_EVEN_EVEN ||
      matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // x_o = A_oo^-1 (b_o + k D_oe x_e)
    DiracWilson::DslashXpay(tmp, x.Even(), 1, dagger, b.Odd(), kappa);
    CloverInv(x.Odd(), tmp, 1);
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
      matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)
    DiracWilson::DslashXpay(tmp, x.Odd(), 0, dagger, b.Even(), kappa);
    CloverInv(x.Even(), tmp, 0);
  } else {
    errorQuda("MatPCType %d not valid for DiracClover", matpcType);
  }

}

