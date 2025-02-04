#include <string.h>
#include <multigrid.h>
#include <tune_quda.h>
#include <algorithm>
#include <transfer.h>
#include <blas_quda.h>

namespace quda
{

  DiracCoarsePV::DiracCoarsePV(const DiracParam &param, bool gpu_setup) : DiracCoarse(param, gpu_setup)
  {
    /* do nothing */
    errorQuda("DiracCoarsePV has not been implemented yet");
  }

  DiracCoarsePV::DiracCoarsePV(const DiracParam &param, std::shared_ptr<GaugeField> Y_h, std::shared_ptr<GaugeField> X_h,
                               std::shared_ptr<GaugeField> Xinv_h, std::shared_ptr<GaugeField> Yhat_h,
                               std::shared_ptr<GaugeField> Y_d, std::shared_ptr<GaugeField> X_d,
                               std::shared_ptr<GaugeField> Xinv_d, std::shared_ptr<GaugeField> Yhat_d) :
    DiracCoarse(param, Y_h, X_h, Xinv_h, Yhat_h, Y_d, X_d, Xinv_d, Yhat_d)
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
  }

  DiracCoarsePV::DiracCoarsePV(const DiracCoarse &dirac, const DiracParam &param) : DiracCoarse(dirac, param)
  {
    /* do nothing */
    errorQuda("DiracCoarsePV has not been implemented yet");
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
    errorQuda("DiracCoarsePV::createCoarseOp has not been implemented yet");

    /*if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Coarse operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    if (checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
      initializeLazy(QUDA_CPU_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Y_h), *(this->X_h), *(this->Xinv_h), kappa, mass, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional);
    } else {
      initializeLazy(QUDA_CUDA_FIELD_LOCATION);
      if (Y.Order() != X.Order()) { errorQuda("Y/X order mismatch in createCoarseOp: %d %d\n", Y.Order(), X.Order()); }
      bool use_mma = Y.Order() == QUDA_MILC_GAUGE_ORDER;
      CoarseCoarseOp(Y, X, T, *(this->Y_d), *(this->X_d), *(this->Xinv_d), kappa, mass, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional, use_mma);
    }*/
  }

  void DiracCoarsePV::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    errorQuda("DiracCoarsePV has not been implemented yet");
    Dirac::prefetch(mem_space, stream);
    if (Y_d) Y_d->prefetch(mem_space, stream);
    if (X_d) X_d->prefetch(mem_space, stream);
  }
} // namespace quda
