#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <multigrid.h>
#include <dslash_quda.h>

namespace quda {

  GaugeLaplace::GaugeLaplace(const DiracParam &param) :  Dirac(param) { }

  GaugeLaplace::GaugeLaplace(const GaugeLaplace &laplace) :  Dirac(laplace) { }

  GaugeLaplace::~GaugeLaplace() { }

  GaugeLaplace& GaugeLaplace::operator=(const GaugeLaplace &laplace)
  {
    if (&laplace != this) Dirac::operator=(laplace);
    return *this;
  }

  void GaugeLaplace::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                            QudaParity parity) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for directions with a derivative
    for (int i = 0; i < 4; i++) {
      comm_dim[i] = comm_dim_partitioned(i);
      if (laplace3D == i) comm_dim[i] = 0;
    }
    ApplyLaplace(out, in, *gauge, laplace3D, 1.0, 1.0, in, parity, comm_dim, profile);
  }

  void GaugeLaplace::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    checkSpinorAlias(in, out);

    int comm_dim[4] = {};
    // only switch on comms needed for directions with a derivative
    for (int i = 0; i < 4; i++) {
      comm_dim[i] = comm_dim_partitioned(i);
      if (laplace3D == i) comm_dim[i] = 0;
    }
    ApplyLaplace(out, in, *gauge, laplace3D, k, 1.0, x, parity, comm_dim, profile);
  }

  void GaugeLaplace::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, -kappa);
  }

  void GaugeLaplace::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void GaugeLaplace::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                             cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      sol[i] = x[i].create_alias();
    }
  }

  void GaugeLaplace::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                 const QudaSolutionType) const
  {
    // do nothing
  }

  GaugeLaplacePC::GaugeLaplacePC(const DiracParam &param) : GaugeLaplace(param)
  {
    for (auto i = 0; i < 4; i++)
      if (laplace3D == i) errorQuda("Cannot use 3-d operator with e/o preconditioning");
  }

  GaugeLaplacePC::GaugeLaplacePC(const GaugeLaplacePC &dirac) : GaugeLaplace(dirac) { }

  GaugeLaplacePC::~GaugeLaplacePC() { }

  GaugeLaplacePC& GaugeLaplacePC::operator=(const GaugeLaplacePC &laplace)
  {
    if (&laplace != this) GaugeLaplace::operator=(laplace);
    return *this;
  }

  void GaugeLaplacePC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    double kappa2 = -kappa*kappa;
    auto tmp = getFieldTmp(out);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(tmp, in, QUDA_ODD_PARITY);
      DslashXpay(out, tmp, QUDA_EVEN_PARITY, in, kappa2);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(tmp, in, QUDA_EVEN_PARITY);
      DslashXpay(out, tmp, QUDA_ODD_PARITY, in, kappa2);
    } else {
      errorQuda("MatPCType %d not valid for GaugeLaplacePC", matpcType);
    }
  }

  void GaugeLaplacePC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void GaugeLaplacePC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                               cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                               const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      for (auto i = 0u; i < b.size(); i++) {
        src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
        sol[i] = x[i].create_alias();
      }
      return;
    }

    // we desire solution to full system
    for (auto i = 0u; i < b.size(); i++) {
      // src = b_e + k D_eo b_o
      DslashXpay(x[i][other_parity], b[i][other_parity], this_parity, b[i][this_parity], kappa);
      src[i] = x[i][other_parity].create_alias();
      sol[i] = x[i][this_parity].create_alias();
    }
  }

  void GaugeLaplacePC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                   const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    for (auto i = 0u; i < b.size(); i++) {
      // create full solution
      checkFullSpinor(x[i], b[i]);
      DslashXpay(x[i][other_parity], x[i][this_parity], other_parity, b[i][other_parity], kappa);
    }
  }

} // namespace quda
