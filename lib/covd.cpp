#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <contractQuda.h>

namespace quda {

  namespace covdev {
#include <dslash_init.cuh>
  }

  CovD::CovD(cudaGaugeField *gauge, TimeProfile &prof) : 
    gauge(gauge), profile(&prof)
    { 
      covdev::initConstants(*gauge, prof);
    }

  CovD::~CovD() { }

  void CovD::Apply(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		       const QudaParity parity, const int mu)
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    covDev(&out, *gauge, &in, parity, mu, *profile);

    flops += 144ll*in.Volume();
  }

  void CovD::M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int mu)
  {
    checkFullSpinor(out, in);

    Apply(out.Odd(),  in.Even(), QUDA_ODD_PARITY,  mu);
    Apply(out.Even(), in.Odd(),  QUDA_EVEN_PARITY, mu);
  }

  void CovD::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    if (in.Nspin() != 4) {
      errorQuda("Only Wilson-like fermions supported at this moment");
    }

    if ( (in.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS || out.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) && 
	 in.Nspin() == 4) {
      errorQuda("CUDA Dirac operator requires UKQCD basis, out = %d, in = %d", 
		out.GammaBasis(), in.GammaBasis());
    }

    if (in.Precision() != out.Precision()) {
      errorQuda("Input precision %d and output spinor precision %d don't match in dslash_quda",
		in.Precision(), out.Precision());
    }

    if (in.Stride() != out.Stride()) {
      errorQuda("Input %d and output %d spinor strides don't match in dslash_quda", 
		in.Stride(), out.Stride());
    }

    if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not single parity: in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    }

    if (out.Ndim() != 5) {
      if ((out.Volume() != gauge->Volume() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ||
	  (out.Volume() != gauge->VolumeCB() && out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ) {
	errorQuda("Spinor volume %d doesn't match gauge volume %d", out.Volume(), gauge->VolumeCB());
      }
    } else {
      errorQuda("Domain Wall fermions not supported yet");
    }
  }

  void CovD::checkFullSpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() != QUDA_FULL_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not full fields: in = %d, out = %d", 
		in.SiteSubset(), out.SiteSubset());
    } 
  }

  void CovD::checkSpinorAlias(const cudaColorSpinorField &a, const cudaColorSpinorField &b) const {
    if (a.V() == b.V()) errorQuda("Aliasing pointers");
  }

} // namespace quda
