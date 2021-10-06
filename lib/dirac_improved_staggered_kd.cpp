#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <staggered_kd_build_xinv.h>

namespace quda
{

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracParam &param) :
    DiracImprovedStaggered(param), Xinv(param.xInvKD)
  {
  }

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracImprovedStaggeredKD &dirac) :
    DiracImprovedStaggered(dirac), Xinv(dirac.Xinv)
  {
  }

  DiracImprovedStaggeredKD::~DiracImprovedStaggeredKD() { }

  DiracImprovedStaggeredKD &DiracImprovedStaggeredKD::operator=(const DiracImprovedStaggeredKD &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
      Xinv = dirac.Xinv;
    }
    return *this;
  }

  void DiracImprovedStaggeredKD::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) { errorQuda("Staggered dslash requires 5-d fermion fields"); }

    if (in.Precision() != out.Precision()) {
      errorQuda("Input and output spinor precisions don't match in dslash_quda");
    }

    if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not full parity, in = %d, out = %d", in.SiteSubset(), out.SiteSubset());
    }

    if (out.Volume() / out.X(4) != 2 * gauge->VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), gauge->VolumeCB());
    }
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

    // TODO: add left vs right precond

    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp2, in);

    bool right_block_precond = false;

    if (right_block_precond) {
      if (dagger == QUDA_DAG_NO) {
        // K-D op is right-block preconditioned
        ApplyStaggeredKahlerDiracInverse(*tmp2, in, *Xinv, false);
        flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block
        if (mass == 0.) {
          ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 0., *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_YES,
                                 commDim, profile);
          flops += 1146ll * in.Volume();
        } else {
          ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 2. * mass, *tmp2, QUDA_INVALID_PARITY, dagger,
                                 commDim, profile);
          flops += 1158ll * in.Volume();
        }
      } else { // QUDA_DAG_YES

        if (mass == 0.) {
          ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim,
                                 profile);
          flops += 1146ll * in.Volume();
        } else {
          ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim,
                                 profile);
          flops += 1158ll * in.Volume();
        }
        ApplyStaggeredKahlerDiracInverse(out, *tmp2, *Xinv, true);
        flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block
      }
    } else { // left preconditioned
      if (dagger == QUDA_DAG_NO) {

        if (mass == 0.) {
          ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim,
                                 profile);
          flops += 1146ll * in.Volume();
        } else {
          ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim,
                                 profile);
          flops += 1158ll * in.Volume();
        }

        ApplyStaggeredKahlerDiracInverse(out, *tmp2, *Xinv, false);
        flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

      } else { // QUDA_DAG_YES

        ApplyStaggeredKahlerDiracInverse(*tmp2, in, *Xinv, true);
        flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

        if (mass == 0.) {
          ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 0., *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_NO,
                                 commDim, profile);
          flops += 1146ll * in.Volume();
        } else {
          ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 2. * mass, *tmp2, QUDA_INVALID_PARITY, dagger,
                                 commDim, profile);
          flops += 1158ll * in.Volume();
        }
      }
    }

    deleteTmp(&tmp2, reset);
  }

  void DiracImprovedStaggeredKD::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
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
    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    checkFullSpinor(x, b);

    bool right_block_precond = false;

    if (right_block_precond) {
      // need to modify the solution
      src = &b;
      sol = &x;
    } else {
      // need to modify rhs
      bool reset = newTmp(&tmp1, b);

      KahlerDiracInv(*tmp1, b);
      b = *tmp1;

      deleteTmp(&tmp1, reset);
      sol = &x;
      src = &b;
    }
  }

  void DiracImprovedStaggeredKD::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracImprovedStaggeredKD::reconstructSpecialMG(ColorSpinorField &x, const ColorSpinorField &b,
                                                      const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?

    checkFullSpinor(x, b);

    bool right_block_precond = false;

    if (right_block_precond) {
      bool reset = newTmp(&tmp1, b.Even());

      KahlerDiracInv(*tmp1, x);
      x = *tmp1;

      deleteTmp(&tmp1, reset);
    }
    // nothing required for left block preconditioning
  }

  void DiracImprovedStaggeredKD::updateFields(cudaGaugeField *, cudaGaugeField *fat_gauge_in,
                                              cudaGaugeField *long_gauge_in, cudaCloverField *)
  {
    Dirac::updateFields(fat_gauge_in, nullptr, nullptr, nullptr);
    fatGauge = fat_gauge_in;
    longGauge = long_gauge_in;

    // Recompute Xinv (I guess we should do that here?)
    BuildStaggeredKahlerDiracInverse(*Xinv, *fatGauge, mass);
  }

  void DiracImprovedStaggeredKD::createCoarseOp(GaugeField &, GaugeField &, const Transfer &, double, double, double,
                                                double) const
  {
    errorQuda("Staggered KD operators do not support MG coarsening yet");

    // if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
    //  errorQuda("Staggered KD operators only support aggregation coarsening");
    // StaggeredCoarseOp(Y, X, T, *fatGauge, Xinv, mass, QUDA_ASQTADKD_DIRAC, QUDA_MATPC_INVALID);
  }

  void DiracImprovedStaggeredKD::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    DiracImprovedStaggered::prefetch(mem_space, stream);
    if (Xinv != nullptr) Xinv->prefetch(mem_space, stream);
  }

} // namespace quda
