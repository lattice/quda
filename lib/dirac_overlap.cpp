#include <util_quda.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda
{
  /**
   * Apply the overlap overlap
   * out = D * in
   * If m is not zero, then
   * out = m * x + (1 - m) * D * in
   * D is defined as 0.5 * (1 + \gamma_5 sign(\gamma_5 M)) where M is the Wilson operator
   */
  void ApplyOverlap(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                    OverlapKernel &O, double m, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                    const int *comm_override, TimeProfile &profile)
  {
    auto in_def = getFieldTmp(out);
    auto b1 = getFieldTmp(out);
    auto b2 = getFieldTmp(out);
    auto Mb1 = getFieldTmp(out);
    auto Ab1 = getFieldTmp(out);

    cvector_ref<ColorSpinorField> &evecs = O.evecs;
    cvector<double> &evals = O.evals;
    const double remez_order = O.remez_order[0];
    cvector<double> &remez_coeff = O.remez_coeff[0];
    const double lambda_max = (1.0 + 8.0 * O.kappa);
    const double epsilon = O.epsilon;

    /**
     * Apply (1 - m) * 0.5 directly to the input
     */
    if (dagger) {
      blas::axy((1 - m) * 0.5, in, out);
      gamma5(in_def, out);
    } else {
      blas::axy((1 - m) * 0.5, in, in_def);
      gamma5(out, in_def);
    }

    /**
     * \gamma_5 sign(\gamma_5 M) for small eigenvalues
     * Define the eigenvalues and eigenvectors \gamma_5 M v_i = \lambda_i v_i
     * ==> \gamma_5 \sum_i sign(\lambda_i) |v_i><v_i|
     */
    std::vector<quda::Complex> alpha(evecs.size() * in_def.size());
    blas::block::cDotProduct(alpha, evecs, in_def);
    for (auto &v : alpha) { v *= -1; }
    blas::block::caxpy(alpha, evecs, in_def);
    for (size_t i = 0; i < evecs.size(); i++) {
      for (size_t j = 0; j < in_def.size(); ++j) { alpha[i * in_def.size() + j] *= -evals[i] / abs(evals[i]); }
    }
    blas::block::caxpy(alpha, evecs, out);
    if (!dagger) { gamma5(out, out); }

    /**
     * \gamma_5 sign(\gamma_5 M) for large eigenvalues
     * Define the Chebyshev polynomial approximation P(x) ~ x^{-1/2}
     * ==> M P(M^\dagger M)
     * Here M is the normalized Wilson operator which has the maximum eigenvalue 1
     */
    blas::zero(b1);
    blas::zero(b2);
    for (int k = remez_order; k >= 0; --k) {
      ApplyWilson(Mb1, b1, U, -O.kappa, b1, parity, false, comm_override, profile);
      ApplyWilson(Ab1, Mb1, U, -O.kappa, Mb1, parity, true, comm_override, profile);
      blas::axpby(-(1.0 + epsilon) / (1.0 - epsilon), b1, 2.0 / (1.0 - epsilon) / (lambda_max * lambda_max), Ab1);
      if (k > 0) {
        blas::axpbypczw(remez_coeff[k], in_def, 2.0, Ab1, -1.0, b2, b2);
      } else {
        blas::axpbypczw(remez_coeff[0], in_def, 1.0, Ab1, -1.0, b2, b2);
      }
      std::swap(b1, b2);
    }
    ApplyWilson(Mb1, b1, U, -O.kappa, b1, parity, false, comm_override, profile);
    if (dagger) { gamma5(Mb1, Mb1); }
    if (m == 0.0) {
      blas::axpbyz(1.0 / lambda_max, Mb1, 1.0, out, out);
    } else {
      blas::axpbypczw(m, x, 1.0 / lambda_max, Mb1, 1.0, out, out);
    }
  }

  DiracOverlap::DiracOverlap(const DiracParam &param) : Dirac(param), overlap_kernel(param.overlap_kernel) { }

  DiracOverlap::DiracOverlap(const DiracOverlap &dirac) : Dirac(dirac), overlap_kernel(dirac.overlap_kernel) { }

  DiracOverlap::~DiracOverlap() { }

  DiracOverlap &DiracOverlap::operator=(const DiracOverlap &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      overlap_kernel = dirac.overlap_kernel;
    }
    return *this;
  }

  void DiracOverlap::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                            QudaParity parity) const
  {
    ApplyOverlap(out, in, *gauge, *overlap_kernel, 0.0, in, parity, dagger, commDim.data, profile);
  }

  // Defined as k * x + (1 - k) * D * in
  void DiracOverlap::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    ApplyOverlap(out, in, *gauge, *overlap_kernel, k, x, parity, dagger, commDim.data, profile);
  }

  // Defined as m + (1 - m) D
  void DiracOverlap::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, mass);
  }

  // Defined as m^2 + (1 - m^2) DdagD
  void DiracOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    Dslash(tmp, in, QUDA_INVALID_PARITY);
    flipDagger();
    DslashXpay(out, tmp, QUDA_INVALID_PARITY, in, mass * mass);
    flipDagger();
  }

  // Defined as m^2 + (1 - m^2) P DdagD P where P = (1 +- gamma_5) / 2
  // For overlap dslash P DdagD P = P D P
  void DiracOverlap::MdagMChiral(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                 QudaChirality chirality) const
  {
    ColorSpinorParam param(in[0]);
    param.nSpin = 4;
    param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    param.mem_type = QUDA_MEMORY_DEVICE; // TODO: Hack for eigensolver in the host memory
    param.setPrecision(param.Precision(), param.Precision(), true);
    auto in_tmp = getFieldTmp<ColorSpinorField>(in.size(), param);
    auto out_tmp = getFieldTmp<ColorSpinorField>(out.size(), param);

    for (size_t i = 0; i < in.size(); i++) { spinorChiralReconstruct(in_tmp[i], in[i], chirality); }
    DslashXpay(out_tmp, in_tmp, QUDA_INVALID_PARITY, in_tmp, mass * mass);
    for (size_t i = 0; i < out.size(); i++) { spinorChiralProject(out[i], out_tmp[i], chirality); }
  }

  void DiracOverlap::prepare(cvector_ref<ColorSpinorField> &out, cvector_ref<ColorSpinorField> &in,
                             cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    create_alias(in, b);
    create_alias(out, x);
  }

  void DiracOverlap::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    if (solType == QUDA_MAT_SOLUTION) {
      // We actually apply (1 - D) x'
      // x = -1 / (1 - m) * b + 1 / (1 - m) * x'
      // x' = M^{-1} * b = (m + (1 - m) D)^{-1} * b
      blas::axpby(-1.0 / (1.0 - mass), b, 1.0 / (1.0 - mass), x);
    } else if (solType == QUDA_MATDAG_MAT_SOLUTION) {
      // We actually apply (1 - DdagD) x'
      // x = -1 / (1 - m^2) * b + 1 / (1 - m^2) * x'
      // x' = (MdagM)^{-1} * b = (m^2 + (1 - m^2) DdagD)^{-1} * b
      blas::axpby(-1.0 / (1.0 - mass * mass), b, 1.0 / (1.0 - mass * mass), x);
    }
  }

  void DiracOverlap::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
  }
} // namespace quda