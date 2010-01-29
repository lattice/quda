#include <dirac.h>
#include <dslash_quda.h>

Dirac::Dirac(const DiracParam &param) 
  : gauge(*(param.gauge)), kappa(param.kappa), matpcType(param.matpcType), flops(0) {

}

Dirac::Dirac(const Dirac &dirac) 
  : gauge(dirac.gauge), kappa(dirac.kappa), matpcType(dirac.matpcType), flops(0) {

}

Dirac::~Dirac() {

}

Dirac& Dirac::operator=(const Dirac &dirac) {

  if(&dirac != this) {
    gauge = dirac.gauge;
    kappa = dirac.kappa;
    matpcType = dirac.matpcType;
    flops = 0;
  }

  return *this;

}

void Dirac::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) {
  if (in.gammaBasis() != QUDA_UKQCD_BASIS || out.gammaBasis() != QUDA_UKQCD_BASIS) {
    errorQuda("Cuda Dirac operator requires UKQCD basis, out = %d, in = %d", 
	      out.gammaBasis(), in.gammaBasis());
  }

  if (in.Precision() != out.Precision()) {
    errorQuda("Input and output spinor precisions don't match in dslash_quda");
  }

  if (in.Stride() != out.Stride()) {
    errorQuda("Input and output spinor strides don't match in dslash_quda");
  }

  if (in.fieldSubset() != QUDA_PARITY_FIELD_SUBSET || out.fieldSubset() != QUDA_PARITY_FIELD_SUBSET) {
    errorQuda("ColorSpinorFields are not single parity, in = %d, out = %d", 
	      in.fieldSubset(), out.fieldSubset());
  }

  if (out.Volume() != gauge.volume) {
    errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), gauge.volume);
  }

#if (__CUDA_ARCH__ != 130)
  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    errorQuda("Double precision not supported on this GPU");
  }

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    errorQuda("Double precision not supported on this GPU");
  }
#endif
}

void Dirac::checkFullSpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) {
   if (in.fieldSubset() != QUDA_FULL_FIELD_SUBSET || out.fieldSubset() != QUDA_FULL_FIELD_SUBSET) {
    errorQuda("ColorSpinorFields are not full fields, in = %d, out = %d", 
	      in.fieldSubset(), out.fieldSubset());
  } 
}

// Dirac operator factory
Dirac* Dirac::create(const DiracParam &param) {

  if (param.type == QUDA_WILSON_DIRAC) {
    return new DiracWilson(param);
  } else if (param.type == QUDA_WILSONPC_DIRAC) {
    return new DiracWilsonPC(param);
  } else if (param.type == QUDA_CLOVER_DIRAC) {
    return new DiracClover(param);
  } else if (param.type == QUDA_CLOVERPC_DIRAC) {
    return new DiracCloverPC(param);
  } else {
    errorQuda("Dirac Operator not implemented");
    return 0;
  }

}
