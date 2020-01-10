#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>
#include <dslash_quda.h>

namespace quda {

  GaugeCovDev::GaugeCovDev(const DiracParam &param) :  Dirac(param) { }

  GaugeCovDev::GaugeCovDev(const GaugeCovDev &covDev) :  Dirac(covDev) { }

  GaugeCovDev::~GaugeCovDev() { }

  GaugeCovDev& GaugeCovDev::operator=(const GaugeCovDev &covDev)
  {
    if (&covDev != this) Dirac::operator=(covDev);
    return *this;
  }

  void GaugeCovDev::DslashCD(ColorSpinorField &out, const ColorSpinorField &in,  const QudaParity parity, const int mu) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for mu derivative (FIXME - only communicate in the given direction)
    comm_dim[mu % 4] = comm_dim_partitioned(mu % 4);
    ApplyCovDev(out, in, *gauge, mu, parity, dagger, comm_dim, profile);
    flops += 1320ll*in.Volume(); // FIXME
  }

  void GaugeCovDev::MCD(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const
  {
    checkFullSpinor(out, in);
    DslashCD(out, in, QUDA_INVALID_PARITY, mu);
  }

  void GaugeCovDev::MdagMCD(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const
  {
    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    MCD(*tmp1, in, mu);
    MCD(out, *tmp1, (mu+4)%8);

    deleteTmp(&tmp1, reset);
  }

  void GaugeCovDev::Dslash(ColorSpinorField &out, const ColorSpinorField &in,  const QudaParity parity) const
  {
    //do nothing
  }

  void GaugeCovDev::DslashXpay(ColorSpinorField &out, 
			       const ColorSpinorField &in, 
			       const QudaParity parity, 
			       const ColorSpinorField &x,
			       const double &k) const
  {
    //do nothing
  }

  void GaugeCovDev::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    //do nothing
  }

  void GaugeCovDev::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    //do nothing
  }
  
  void GaugeCovDev::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b, 
			    const QudaSolutionType solType) const
  {
    //do nothing
  }

  void GaugeCovDev::reconstruct(ColorSpinorField &x, 
				const ColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    //do nothing
  }
  
} // namespace quda
