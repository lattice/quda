#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <staggered_kd_build_xinv.h>

namespace quda
{

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracParam &param) :
    DiracImprovedStaggered(param),
    Xinv(param.xInvKD),
    parent_dirac_type(param.dirac == nullptr ? QUDA_INVALID_DIRAC : param.dirac->getDiracType())
  {
  }

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracImprovedStaggeredKD &dirac) :
    DiracImprovedStaggered(dirac), Xinv(dirac.Xinv), parent_dirac_type(dirac.parent_dirac_type)
  {
  }

  DiracImprovedStaggeredKD::~DiracImprovedStaggeredKD() { }

  DiracImprovedStaggeredKD &DiracImprovedStaggeredKD::operator=(const DiracImprovedStaggeredKD &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
      Xinv = dirac.Xinv;
      parent_dirac_type = dirac.parent_dirac_type;
    }
    return *this;
  }

  void DiracImprovedStaggeredKD::Dslash(ColorSpinorField &, const ColorSpinorField &, const QudaParity) const
  {
    errorQuda("The improved staggered Kahler-Dirac operator does not have a single parity form");
  }

  void DiracImprovedStaggeredKD::DslashXpay(ColorSpinorField &, const ColorSpinorField &, const QudaParity,
                                            const ColorSpinorField &, const double &) const
  {
    errorQuda("The improved staggered Kahler-Dirac operator does not have a single parity form");
  }

  // Full staggered operator
  void DiracImprovedStaggeredKD::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // Due to the staggered convention, the staggered part is applying
    // (  2m     -D_eo ) (x_e) = (b_e)
    // ( -D_oe   2m    ) (x_o) = (b_o)
    // ... but under the hood we need to catch the zero mass case.

    checkFullSpinor(out, in);

    auto tmp = getFieldTmp(in);

    if (dagger == QUDA_DAG_NO) {

      if (mass == 0.) {
        ApplyImprovedStaggered(tmp, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim,
                               profile);
        flops += 1146ll * in.Volume();
      } else {
        ApplyImprovedStaggered(tmp, in, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim,
                               profile);
        flops += 1158ll * in.Volume();
      }

      ApplyStaggeredKahlerDiracInverse(out, tmp, *Xinv, false);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

    } else { // QUDA_DAG_YES

      ApplyStaggeredKahlerDiracInverse(tmp, in, *Xinv, true);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

      if (mass == 0.) {
        ApplyImprovedStaggered(out, tmp, *fatGauge, *longGauge, 0., tmp, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim,
                               profile);
        flops += 1146ll * in.Volume();
      } else {
        ApplyImprovedStaggered(out, tmp, *fatGauge, *longGauge, 2. * mass, tmp, QUDA_INVALID_PARITY, dagger, commDim,
                               profile);
        flops += 1158ll * in.Volume();
      }
    }
  }

  void DiracImprovedStaggeredKD::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    auto tmp = getFieldTmp(in);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracImprovedStaggeredKD::KahlerDiracInv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    ApplyStaggeredKahlerDiracInverse(out, in, *Xinv, dagger == QUDA_DAG_YES);
  }

  void DiracImprovedStaggeredKD::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                         ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracImprovedStaggeredKD::prepareSpecialMG(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                                  ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    checkFullSpinor(x, b);

    // need to modify rhs
    auto tmp = getFieldTmp(b);
    KahlerDiracInv(tmp, b);

    // if we're preconditioning the Schur op, we need to rescale by the mass
    if (parent_dirac_type == QUDA_ASQTAD_DIRAC) {
      b = tmp;
    } else if (parent_dirac_type == QUDA_ASQTADPC_DIRAC) {
      b = tmp;
      blas::ax(0.5 / mass, b);
    } else
      errorQuda("Unexpected parent Dirac type %d", parent_dirac_type);

    sol = &x;
    src = &b;
  }

  void DiracImprovedStaggeredKD::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracImprovedStaggeredKD::reconstructSpecialMG(ColorSpinorField &, const ColorSpinorField &,
                                                      const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracImprovedStaggeredKD::updateFields(cudaGaugeField *, cudaGaugeField *fat_gauge_in,
                                              cudaGaugeField *long_gauge_in, CloverField *)
  {
    Dirac::updateFields(fat_gauge_in, nullptr, nullptr, nullptr);
    fatGauge = fat_gauge_in;
    longGauge = long_gauge_in;
  }

  void DiracImprovedStaggeredKD::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double, double mass,
                                                double, double, bool allow_truncation) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Staggered KD operators only support aggregation coarsening");
    StaggeredCoarseOp(Y, X, T, *fatGauge, *longGauge, *Xinv, mass, allow_truncation, QUDA_ASQTADKD_DIRAC,
                      QUDA_MATPC_INVALID);
  }

  void DiracImprovedStaggeredKD::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    DiracImprovedStaggered::prefetch(mem_space, stream);
    if (Xinv != nullptr) Xinv->prefetch(mem_space, stream);
  }

} // namespace quda
