#include <string.h>
#include <algorithm>

#include "multigrid.h"
#include "tune_quda.h"
#include "transfer.h"
#include "blas_quda.h"
#include "dslash_quda.h"

namespace quda
{

  DiracCoarsePV::DiracCoarsePV(const DiracParam &param, bool gpu_setup) :
    DiracCoarse(param, gpu_setup),
    m5(param.m5),
    kappa5(0.5 / (5.0 + m5)),
    Ls(param.Ls),
    parent_dwf(param.parent_dwf),
    mass_pv(1.0)
  {
    prepareMobiusCoefficients(param);
  }

  DiracCoarsePV::DiracCoarsePV(const DiracParam &param, std::shared_ptr<GaugeField> Y_h, std::shared_ptr<GaugeField> X_h,
                               std::shared_ptr<GaugeField> Xinv_h, std::shared_ptr<GaugeField> Yhat_h,
                               std::shared_ptr<GaugeField> Y_d, std::shared_ptr<GaugeField> X_d,
                               std::shared_ptr<GaugeField> Xinv_d, std::shared_ptr<GaugeField> Yhat_d) :
    DiracCoarse(param, Y_h, X_h, Xinv_h, Yhat_h, Y_d, X_d, Xinv_d, Yhat_d),
    m5(param.m5),
    kappa5(0.5 / (5.0 + m5)),
    Ls(param.Ls),
    parent_dwf(param.parent_dwf),
    mass_pv(1.0)
  {
    prepareMobiusCoefficients(param);
  }

  DiracCoarsePV::DiracCoarsePV(const DiracCoarse &dirac, const DiracParam &param) :
    DiracCoarse(dirac, param),
    m5(param.m5),
    kappa5(0.5 / (5.0 + m5)),
    Ls(param.Ls),
    parent_dwf(param.parent_dwf),
    mass_pv(1.0)
  {
    prepareMobiusCoefficients(param);
  }

  void DiracCoarsePV::Dslash(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity) const
  {
    errorQuda("The coarse PV operator does not have a single parity form");
  }

  void DiracCoarsePV::DslashXpay(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, QudaParity,
                                 cvector_ref<const ColorSpinorField> &, double) const
  {
    errorQuda("The coarse PV operator does not have a single parity form");
  }

  void DiracCoarsePV::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    if (out.size() != 1 || in.size() != 1)
      errorQuda("DiracCoarsePV does not support multi-rhs yet; out.size == %lu , in.size == %lu", out.size(), in.size());

    if (out.X(4) != Ls) errorQuda("Unexpected fourth dimension for out = %d, expected %d", out.X(4), Ls);
    if (in.X(4) != Ls) errorQuda("Unexpected fourth dimension for in = %d, expected %d", in.X(4), Ls);

    auto tmp = getFieldTmp<ColorSpinorField>(out[0]);

    // these can be used to get rid of the redundant split/join
    ApplyMDwf(tmp, in);
    ApplyPVDagger(out, tmp);
  }

  void DiracCoarsePV::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    if (out.size() != 1 || in.size() != 1)
      errorQuda("DiracCoarsePV does not support multi-rhs yet; out.size == %lu , in.size == %lu", out.size(), in.size());

    if (out.X(4) != Ls) errorQuda("Unexpected fourth dimension for out = %d, expected %d", out.X(4), Ls);
    if (in.X(4) != Ls) errorQuda("Unexpected fourth dimension for in = %d, expected %d", in.X(4), Ls);

    auto tmp = getFieldTmp<ColorSpinorField>(out[0]);

    //printfQuda("Calling DiracCoarsePV::MdagM\n");

    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracCoarsePV::ApplyPVDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    if (out.size() != 1 || in.size() != 1)
      errorQuda("DiracCoarsePV does not support multi-rhs yet; out.size == %lu , in.size == %lu", out.size(), in.size());

    if (out.X(4) != Ls) errorQuda("Unexpected fourth dimension for out = %d, expected %d", out.X(4), Ls);
    if (in.X(4) != Ls) errorQuda("Unexpected fourth dimension for in = %d, expected %d", in.X(4), Ls);

    if (parent_dwf != QUDA_DOMAIN_WALL_4D_DIRAC)
      errorQuda("Only the coarse DomainWall4DPV operator is supported for now");

    ColorSpinorParam csParam(out[0]);
    csParam.nDim = 4;
    csParam.x[4] = 1;
    csParam.create = QUDA_NULL_FIELD_CREATE;

    auto in_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto out_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto chiral_plus = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto chiral_minus = getFieldTmp<ColorSpinorField>(Ls, csParam);

    // split rhs
    Split5DTo4DFields(in_4d, in[0]);

    // This bit is spiritually equivalent to the DWF call:
    // ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, dagger, commDim.data,
    //                   profile);

    // DiracCoarse::Mdag(out_4d, in_4d); // this ends up calling DiracCoarsePV::Mdag
    flipDagger();
    DiracCoarse::M(out_4d, in_4d);
    flipDagger();
    blas::axpy(-1.0, in_4d, out_4d);
    blas::ax(-0.5 / kappa, out_4d); // undo the kappa baked into DiracCoarse

    // This next block is spiritually equivalent to the DWF call:
    // ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, Dslash5Type::DSLASH5_DWF);
    // the only difference between dagger and non-dagger is the direction of the projector

    ApplyCoarseChiralProj(chiral_plus, in_4d, +1);  // for the forwards direction
    ApplyCoarseChiralProj(chiral_minus, in_4d, -1); // for the direction direction
    for (int s = 0; s < Ls; s++) {
      // forwards direction
      blas::axpy((s == Ls - 1) ? -mass_pv : 1, chiral_plus[(s + 1) % Ls], out_4d[s]);
      // backwards direction
      blas::axpy((s == 0) ? -mass_pv : 1, chiral_minus[(s + Ls - 1) % Ls], out_4d[s]);
    }

    // This last bit is spiritually equivalent to the call:
    // blas::xpay(in, -kappa5, out);

    blas::xpay(in_4d, -2.0 * kappa5, out_4d);

    Join4DTo5DField(out[0], out_4d);
  }

  void DiracCoarsePV::ApplyMDwf(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    if (out.size() != 1 || in.size() != 1)
      errorQuda("DiracCoarsePV does not support multi-rhs yet; out.size == %lu , in.size == %lu", out.size(), in.size());

    if (out.X(4) != Ls) errorQuda("Unexpected fourth dimension for out = %d, expected %d", out.X(4), Ls);
    if (in.X(4) != Ls) errorQuda("Unexpected fourth dimension for in = %d, expected %d", in.X(4), Ls);

    if (parent_dwf != QUDA_DOMAIN_WALL_4D_DIRAC)
      errorQuda("Only the coarse DomainWall4DPV operator is supported for now");

    ColorSpinorParam csParam(out[0]);
    csParam.nDim = 4;
    csParam.x[4] = 1;
    csParam.create = QUDA_NULL_FIELD_CREATE;

    auto in_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto out_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto chiral_plus = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto chiral_minus = getFieldTmp<ColorSpinorField>(Ls, csParam);

    // split rhs
    Split5DTo4DFields(in_4d, in[0]);

    // This bit is spiritually equivalent to the DWF call:
    // ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, dagger, commDim.data,
    //                   profile);

    DiracCoarse::M(out_4d, in_4d);
    blas::axpy(-1.0, in_4d, out_4d);
    blas::ax(-0.5 / kappa, out_4d); // undo the kappa baked into DiracCoarse

    // This next block is spiritually equivalent to the DWF call:
    // ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, Dslash5Type::DSLASH5_DWF);

    ApplyCoarseChiralProj(chiral_plus, in_4d, +1);  // for the backwards direction
    ApplyCoarseChiralProj(chiral_minus, in_4d, -1); // for the forwards direction
    for (int s = 0; s < Ls; s++) {
      // forwards direction
      blas::axpy((s == Ls - 1) ? -mass : 1, chiral_minus[(s + 1) % Ls], out_4d[s]);
      // backwards direction
      blas::axpy((s == 0) ? -mass : 1, chiral_plus[(s + Ls - 1) % Ls], out_4d[s]);
    }

    // This last bit is spiritually equivalent to the call:
    // blas::xpay(in, -kappa5, out);

    blas::xpay(in_4d, -2.0 * kappa5, out_4d);

    Join4DTo5DField(out[0], out_4d);
  }

  void DiracCoarsePV::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                              cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                              const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    create_alias(src, b);
    create_alias(sol, x);
  }

  void DiracCoarsePV::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                  const QudaSolutionType) const
  {
    /* do nothing */
  }

  bool DiracCoarsePV::hermitian() const { return (mass_pv == mass); }

  // Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarsePV::createCoarseOp(GaugeField &, GaugeField &, const Transfer &, double, double, double, double,
                                     bool) const
  {
    errorQuda("DiracCoarsePV does not define createCoarseOp, construct a DiracCoarse first instead");
  }

  void DiracCoarsePV::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    DiracCoarse::prefetch(mem_space, stream);
  }

  void DiracCoarsePV::prepareMobiusCoefficients(const DiracParam &param)
  {
    if (parent_dwf == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
      memcpy(b_5, param.b_5, sizeof(Complex) * param.Ls);
      memcpy(c_5, param.c_5, sizeof(Complex) * param.Ls);

      double b = b_5[0].real();
      double c = c_5[0].real();
      mobius_kappa_b = 0.5 / (b * (m5 + 4.) + 1.);
      mobius_kappa_c = 0.5 / (c * (m5 + 4.) - 1.);

      mobius_kappa = mobius_kappa_b / mobius_kappa_c;

      // check if doing zMobius
      for (int i = 0; i < Ls; i++) {
        if (b_5[i].imag() != 0.0 || c_5[i].imag() != 0.0
            || (i < Ls - 1 && (b_5[i] != b_5[i + 1] || c_5[i] != c_5[i + 1]))) {
          zMobius = true;
        }
      }

      if (zMobius) {
        logQuda(QUDA_VERBOSE, "%s: Detected variable or complex cofficients: using zMobius\n", __func__);
      } else {
        logQuda(QUDA_VERBOSE, "%s: Detected fixed real cofficients: using regular Mobius\n", __func__);
      }

      if (zMobius) { errorQuda("zMobius has NOT been fully tested in QUDA"); }
    }
  }
} // namespace quda
