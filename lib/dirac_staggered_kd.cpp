#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <staggered_kd_build_xinv.h>

namespace quda
{

  DiracStaggeredKD::DiracStaggeredKD(const DiracParam &param) : DiracStaggered(param), Xinv(param.xInvKD), parent_dirac_type(param.dirac == nullptr ? QUDA_INVALID_DIRAC : param.dirac->getDiracType()) { }

  DiracStaggeredKD::DiracStaggeredKD(const DiracStaggeredKD &dirac) : DiracStaggered(dirac), Xinv(dirac.Xinv), parent_dirac_type(dirac.parent_dirac_type) { }

  DiracStaggeredKD::~DiracStaggeredKD() { }

  DiracStaggeredKD &DiracStaggeredKD::operator=(const DiracStaggeredKD &dirac)
  {
    if (&dirac != this) {
      DiracStaggered::operator=(dirac);
      Xinv = dirac.Xinv;
      parent_dirac_type = dirac.parent_dirac_type;
    }
    return *this;
  }

  void DiracStaggeredKD::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
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

  void DiracStaggeredKD::Dslash(ColorSpinorField &, const ColorSpinorField &, const QudaParity) const
  {
    errorQuda("The staggered Kahler-Dirac operator does not have a single parity form");
  }

  void DiracStaggeredKD::DslashXpay(ColorSpinorField &, const ColorSpinorField &, const QudaParity,
                                    const ColorSpinorField &, const double &) const
  {
    errorQuda("The staggered Kahler-Dirac operator does not have a single parity form");
  }

  // Full staggered operator
  void DiracStaggeredKD::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // Due to the staggered convention, the staggered part is applying
    // (  2m     -D_eo ) (x_e) = (b_e)
    // ( -D_oe   2m    ) (x_o) = (b_o)
    // ... but under the hood we need to catch the zero mass case.

    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp2, in);

    if (dagger == QUDA_DAG_NO) {

      if (mass == 0.) {
        ApplyStaggered(*tmp2, in, *gauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim, profile);
        flops += 570ll * in.Volume();
      } else {
        ApplyStaggered(*tmp2, in, *gauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
        flops += 582ll * in.Volume();
      }
      ApplyStaggeredKahlerDiracInverse(out, *tmp2, *Xinv, false);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

    } else { // QUDA_DAG_YES

      ApplyStaggeredKahlerDiracInverse(*tmp2, in, *Xinv, true);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block

      if (mass == 0.) {
        ApplyStaggered(out, *tmp2, *gauge, 0., *tmp2, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim, profile);
        flops += 570ll * in.Volume();
      } else {
        ApplyStaggered(out, *tmp2, *gauge, 2. * mass, *tmp2, QUDA_INVALID_PARITY, dagger, commDim, profile);
        flops += 582ll * in.Volume();
      }
    }

    deleteTmp(&tmp2, reset);
  }

  void DiracStaggeredKD::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracStaggeredKD::KahlerDiracInv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    ApplyStaggeredKahlerDiracInverse(out, in, *Xinv, dagger == QUDA_DAG_YES);
  }

  void DiracStaggeredKD::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                 ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    sol = &x;
    src = &b;
  }

  void DiracStaggeredKD::prepareSpecialMG(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                          ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    checkFullSpinor(x, b);

    // need to modify rhs
    bool reset = newTmp(&tmp1, b);

    KahlerDiracInv(*tmp1, b);

    // if we're preconditioning the Schur op, we need to rescale by the mass
    // parent could be an ASQTAD operator if we've enabled dropping the long links
    if (parent_dirac_type == QUDA_STAGGERED_DIRAC || parent_dirac_type == QUDA_ASQTAD_DIRAC) {
      b = *tmp1;
    } else if (parent_dirac_type == QUDA_STAGGEREDPC_DIRAC || parent_dirac_type == QUDA_ASQTADPC_DIRAC) {
      b = *tmp1;
      blas::ax(0.5 / mass, b);
    } else
      errorQuda("Unexpected parent Dirac type %d", parent_dirac_type);

    deleteTmp(&tmp1, reset);
    sol = &x;
    src = &b;
  }

  void DiracStaggeredKD::reconstruct(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracStaggeredKD::reconstructSpecialMG(ColorSpinorField &, const ColorSpinorField &, const QudaSolutionType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracStaggeredKD::updateFields(cudaGaugeField *gauge_in, cudaGaugeField *, cudaGaugeField *, CloverField *)
  {
    Dirac::updateFields(gauge_in, nullptr, nullptr, nullptr);
  }

  void DiracStaggeredKD::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double, double mass, double, double, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Staggered KD operators only support aggregation coarsening");

    // Irrelivant for naive staggered
    constexpr bool allow_drop_long = false;

    StaggeredCoarseOp(Y, X, T, *gauge, *gauge, *Xinv, mass, allow_drop_long, QUDA_STAGGEREDKD_DIRAC, QUDA_MATPC_INVALID);
  }

  void DiracStaggeredKD::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    DiracStaggered::prefetch(mem_space, stream);
    if (Xinv != nullptr) Xinv->prefetch(mem_space, stream);
  }

} // namespace quda
