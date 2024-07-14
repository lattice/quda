#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>
#include <dslash_quda.h>

namespace quda {

  GaugeCovDev::GaugeCovDev(const DiracParam &param) :  Dirac(param), covdev_mu(param.covdev_mu) { }

  GaugeCovDev::GaugeCovDev(const GaugeCovDev &covDev) :  Dirac(covDev), covdev_mu(covDev.covdev_mu) { }

  GaugeCovDev::~GaugeCovDev() { }

  GaugeCovDev& GaugeCovDev::operator=(const GaugeCovDev &covDev)
  {
    if (&covDev != this) Dirac::operator=(covDev);
    covdev_mu = covDev.covdev_mu;
    return *this;
  }

  void GaugeCovDev::DslashCD(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             QudaParity parity, int mu) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for mu derivative (FIXME - only communicate in the given direction)
    comm_dim[mu % 4] = comm_dim_partitioned(mu % 4);
    ApplyCovDev(out, in, *gauge, mu, parity, dagger, comm_dim, profile);
  }

  void GaugeCovDev::MCD(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const int mu) const
  {
    checkFullSpinor(out, in);
    DslashCD(out, in, QUDA_INVALID_PARITY, mu);
  }

  void GaugeCovDev::MdagMCD(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const int mu) const
  {
    auto tmp = getFieldTmp(out);

    MCD(tmp, in, mu);
    MCD(out, tmp, (mu+4)%8);
  }

  void GaugeCovDev::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           QudaParity parity) const
  {
    DslashCD(out, in, parity, covdev_mu);
  }

  void GaugeCovDev::DslashXpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity,
                               cvector_ref<const ColorSpinorField> &, double) const
  {
    //do nothing
  }

  void GaugeCovDev::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    MCD(out, in, covdev_mu);
  }

  void GaugeCovDev::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    MdagMCD(out, in, covdev_mu);
  }

  void GaugeCovDev::prepare(cvector_ref<ColorSpinorField> &, cvector_ref<ColorSpinorField> &,
                            cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                            const QudaSolutionType) const
  {
    // do nothing
  }

  void GaugeCovDev::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                const QudaSolutionType) const
  {
    // do nothing
  }

} // namespace quda
