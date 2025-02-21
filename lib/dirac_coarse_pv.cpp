#include <string.h>
#include <multigrid.h>
#include <tune_quda.h>
#include <algorithm>
#include <transfer.h>
#include <blas_quda.h>

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

  void DiracCoarsePV::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             QudaParity parity) const
  {
    errorQuda("The coarse PV operator does not have a single parity form");
  }

  void DiracCoarsePV::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                 QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    errorQuda("The coarse PV operator does not have a single parity form");
  }

  void DiracCoarsePV::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarsePV::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracCoarsePV::ApplyPVDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarsePV::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                              cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                              const QudaSolutionType solType) const
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
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
    errorQuda("DiracCoarsePV has not been implemented yet");
  }

  bool DiracCoarsePV::hermitian() const { return (mass_pv == mass); }

  // Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarsePV::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                     double mu_factor, bool) const
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
