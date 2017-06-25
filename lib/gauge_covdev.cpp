#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>
#include <stencil.h>

namespace quda {

  GaugeCovDev::GaugeCovDev(const DiracParam &param) :  Dirac(param) { }

  GaugeCovDev::GaugeCovDev(const GaugeCovDev &covDev) :  Dirac(covDev) { }

  GaugeCovDev::~GaugeCovDev() { }

  GaugeCovDev& GaugeCovDev::operator=(const GaugeCovDev &covDev)
  {
    if (&covDev != this) Dirac::operator=(covDev);
    return *this;
  }

  void GaugeCovDev::Dslash(ColorSpinorField &out, const ColorSpinorField &in,  const QudaParity parity, const int mu) const
  {
    checkSpinorAlias(in, out);

    ApplyCovDev(out, in, *gauge, parity, mu);

    flops += 1320ll*in.Volume(); // FIXME
  }

  void GaugeCovDev::M(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const
  {
    checkFullSpinor(out, in);
    Dslash(out, in, QUDA_INVALID_PARITY, mu);
  }

  void GaugeCovDev::MdagM(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const
  {
    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in, mu);
    M(out, *tmp1, (mu+4)%8);

    deleteTmp(&tmp1, reset);
  }

} // namespace quda
