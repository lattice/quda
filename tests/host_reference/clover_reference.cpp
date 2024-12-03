#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>

#include <util_quda.h>
#include <host_utils.h>
#include <wilson_dslash_reference.h>

/**
 * @brief Apply the clover matrix field
 * @tparam real_t The floating-point type used for the vectors
 * @param[out] out Result field (single parity)
 * @param[in] clover Clover-matrix field (full field)
 * @param[in] in Input field (single parity)
 * @param[in] parity Parity to which we are applying the clover field
 */
template <typename real_t> void cloverReference(real_t *out, const real_t *clover, const real_t *in, int parity)
{
  using complex = std::complex<real_t>;
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2 * (N - 1) * N / 2;

#pragma omp parallel for
  for (int i = 0; i < Vh; i++) {
    const complex *In = reinterpret_cast<const complex *>(&in[i * nSpin * nColor * 2]);
    complex *Out = reinterpret_cast<complex *>(&out[i * nSpin * nColor * 2]);

    for (int chi = 0; chi < nSpin / 2; chi++) {
      const real_t *D = &clover[((parity * Vh + i) * 2 + chi) * chiralBlock];
      const complex *L = reinterpret_cast<const complex *>(&D[N]);

      for (int s_col = 0; s_col < nSpin / 2; s_col++) { // 2 spins per chiral block
        for (int c_col = 0; c_col < nColor; c_col++) {
          const int col = s_col * nColor + c_col;
          const int Col = chi * N + col;
          Out[Col] = 0.0;

          for (int s_row = 0; s_row < nSpin / 2; s_row++) { // 2 spins per chiral block
            for (int c_row = 0; c_row < nColor; c_row++) {
              const int row = s_row * nColor + c_row;
              const int Row = chi * N + row;

              if (row == col) {
                Out[Col] += D[row] * In[Row];
              } else if (col < row) {
                int k = N * (N - 1) / 2 - (N - col) * (N - col - 1) / 2 + row - col - 1;
                Out[Col] += conj(L[k]) * In[Row];
              } else if (row < col) {
                int k = N * (N - 1) / 2 - (N - row) * (N - row - 1) / 2 + col - row - 1;
                Out[Col] += L[k] * In[Row];
              }
            }
          }
        }
      }
    }
  }
}

void apply_clover(void *out, const void *clover, const void *in, int parity, QudaPrecision precision)
{

  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
    cloverReference(static_cast<double *>(out), static_cast<const double *>(clover), static_cast<const double *>(in),
                    parity);
    break;
  case QUDA_SINGLE_PRECISION:
    cloverReference(static_cast<float *>(out), static_cast<const float *>(clover), static_cast<const float *>(in),
                    parity);
    break;
  default: errorQuda("Unsupported precision %d", precision);
  }
}

void clover_dslash(void *out, const void *const *gauge, const void *clover, const void *in, int parity, int dagger,
                   QudaPrecision precision, QudaGaugeParam &param)
{
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  wil_dslash(tmp, gauge, in, parity, dagger, precision, param);
  apply_clover(out, clover, tmp, parity, precision);

  host_free(tmp);
}

// Apply the even-odd preconditioned Wilson-clover operator
void clover_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inv, const void *in,
                  double kappa, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param)
{
  double kappa2 = -kappa * kappa;
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  switch (matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(tmp, gauge, in, 1, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 1, precision);
      wil_dslash(tmp, gauge, out, 0, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 0, precision);
    } else {
      apply_clover(tmp, clover_inv, in, 0, precision);
      wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
      apply_clover(tmp, clover_inv, out, 1, precision);
      wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 1, precision);
    wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 0, precision);
    xpay(tmp, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(tmp, gauge, in, 0, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 0, precision);
      wil_dslash(tmp, gauge, out, 1, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 1, precision);
    } else {
      apply_clover(tmp, clover_inv, in, 1, precision);
      wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
      apply_clover(tmp, clover_inv, out, 0, precision);
      wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 0, precision);
    wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 1, precision);
    xpay(tmp, kappa2, out, Vh * spinor_site_size, precision);
    break;
  default: errorQuda("Unsupoorted matpc=%d", matpc_type);
  }

  host_free(tmp);
}

// Apply the full Wilson-clover operator
void clover_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, int dagger,
                QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  void *tmp = safe_malloc(V * spinor_site_size * precision);

  const void *inEven = in;
  const void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmpEven = tmp;
  void *tmpOdd = (char *)tmp + Vh * spinor_site_size * precision;

  // Odd part
  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  apply_clover(tmpOdd, clover, inOdd, 1, precision);

  // Even part
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);
  apply_clover(tmpEven, clover, inEven, 0, precision);

  // lastly apply the kappa term
  xpay(tmp, -kappa, out, V * spinor_site_size, precision);

  host_free(tmp);
}

/**
 * @brief Apply a twist, out = tmpH - i*a*gamma_5 *in
 *
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor which the twist is applied to
 * @param[in] tmpH Host input spinor which is accumulated into output
 * @param[in] clover Host input clover
 * @param[in] a Scaling factor a
 * @param[in] precision Single or double precision
 */
void applyTwist(void *out, const void *in, const void *tmpH, double a, QudaPrecision precision)
{
  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
#pragma omp parallel for
    for (int i = 0; i < Vh; i++)
      for (int s = 0; s < 4; s++) {
        double a5 = ((s / 2) ? -1.0 : +1.0) * a;
        for (int c = 0; c < 3; c++) {
          ((double *)out)[i * 24 + s * 6 + c * 2 + 0]
            = ((double *)tmpH)[i * 24 + s * 6 + c * 2 + 0] - a5 * ((double *)in)[i * 24 + s * 6 + c * 2 + 1];
          ((double *)out)[i * 24 + s * 6 + c * 2 + 1]
            = ((double *)tmpH)[i * 24 + s * 6 + c * 2 + 1] + a5 * ((double *)in)[i * 24 + s * 6 + c * 2 + 0];
        }
      }
    break;
  case QUDA_SINGLE_PRECISION:
#pragma omp parallel for
    for (int i = 0; i < Vh; i++)
      for (int s = 0; s < 4; s++) {
        float a5 = ((s / 2) ? -1.0 : +1.0) * a;
        for (int c = 0; c < 3; c++) {
          ((float *)out)[i * 24 + s * 6 + c * 2 + 0]
            = ((float *)tmpH)[i * 24 + s * 6 + c * 2 + 0] - a5 * ((float *)in)[i * 24 + s * 6 + c * 2 + 1];
          ((float *)out)[i * 24 + s * 6 + c * 2 + 1]
            = ((float *)tmpH)[i * 24 + s * 6 + c * 2 + 1] + a5 * ((float *)in)[i * 24 + s * 6 + c * 2 + 0];
        }
      }
    break;
  default: errorQuda("Unsupported precision %d", precision);
  }
}

/**
 * @brief Apply the single-parity clover then twist, out = x - i*a*gamma_5 Clov *in
 *
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor which the clover twist is applied to
 * @param[in] x Host input spinor which is accumulated into output
 * @param[in] clover Host input clover
 * @param[in] a Scaling factor a
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] parity 0 for D_ee, 1 for D_oo
 * @param[in] precision Single or double precision
 */
void twistClover(void *out, const void *in, const void *x, const void *clover, double a, int dagger, int parity,
                 QudaPrecision precision)
{
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  // tmp1 = Clov in
  apply_clover(tmp, clover, in, parity, precision);
  applyTwist(out, tmp, x, (dagger ? -a : a), precision);

  host_free(tmp);
}

/**
 * @brief Apply the single-parity degenerate twisted clover term, (C + i*a*gamma_5)/(C^2 + a^2)
 *
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor
 * @param[in] parity 0 for D_ee, 1 for D_oo
 * @param[in] twist Twist type dictating whether or not the twist or inverse twist is being applied
 * @param[in] precision Single or double precision
 */
void twistCloverGamma5(void *out, const void *in, const void *clover, const void *clover_inverse, int dagger,
                       double kappa, double mu, QudaTwistFlavorType flavor, int parity, QudaTwistGamma5Type twist,
                       QudaPrecision precision)
{
  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  double a = 0.0;

  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu * flavor;

    if (dagger) a *= -1.0;

    apply_clover(tmp1, clover, in, parity, precision);
    applyTwist(out, in, tmp1, a, precision);
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    a = -2.0 * kappa * mu * flavor;

    if (dagger) a *= -1.0;

    apply_clover(tmp1, clover, in, parity, precision);
    applyTwist(tmp2, in, tmp1, a, precision);
    apply_clover(out, clover_inverse, tmp2, parity, precision);
  } else {
    errorQuda("Twist type %d not defined", twist);
  }

  host_free(tmp2);
  host_free(tmp1);
}

/**
 * @brief Apply the single-parity non-degenerate twisted clover term
 *
 *       (A + i*mu*gamma_5*tau3 - epsilon*tau1) for QUDA_TWIST_GAMMA5_DIRECT
 * and   (A - i*mu*gamma_5*tau3 + epsilon*tau1)/(A^2 + mu^2 - epsilon^2) for QUDA_TWIST_GAMMA5_INVERSE
 *
 * @param[out] out1 Host output rhs for flavor 0
 * @param[out] out2 Host output rhs for flavor 1
 * @param[in] in1 Host input spinor for flavor 0
 * @param[in] in2 Host input spinor for flavor 1
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] parity 0 for D_ee, 1 for D_oo
 * @param[in] twist Twist type dictating whether or not the twist or inverse twist is being applied
 * @param[in] precision Single or double precision
 */
void ndegTwistCloverGamma5(void *out1, void *out2, const void *in1, const void *in2, const void *clover,
                           const void *clover_inverse, const int dagger, const double kappa, const double mu,
                           const double epsilon, const int parity, QudaTwistGamma5Type twist, QudaPrecision precision)
{
  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  double a = 0.0, b = 0.0;

  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu;
    b = -2.0 * kappa * epsilon;

    if (dagger) a *= -1.0;

    // apply_clover zeroes its output
    apply_clover(tmp1, clover, in1, parity, precision);
    apply_clover(tmp2, clover, in2, parity, precision);
    // out = tmp + (i 2 kappa mu gamma_5 tau_3) * in
    applyTwist(out1, in1, tmp1, a, precision);
    applyTwist(out2, in2, tmp2, -a, precision);
    // out += (-epsilon tau_1) * in
    axpy(b, in2, out1, Vh * spinor_site_size, precision);
    axpy(b, in1, out2, Vh * spinor_site_size, precision);
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    void *tmptmp1 = safe_malloc(Vh * spinor_site_size * precision);
    void *tmptmp2 = safe_malloc(Vh * spinor_site_size * precision);

    a = -2.0 * kappa * mu;
    b = 2.0 * kappa * epsilon;

    if (dagger) a *= -1.0;

    // tmp = C * in
    apply_clover(tmp1, clover, in1, parity, precision);
    apply_clover(tmp2, clover, in2, parity, precision);
    // out = tmp - (i 2 kappa mu gamma_5 tau_3) * in
    applyTwist(tmptmp1, in1, tmp1, a, precision);
    applyTwist(tmptmp2, in2, tmp2, -a, precision);
    // tmptmp += +(epsilon * tau_1) * in
    axpy(b, in2, tmptmp1, Vh * spinor_site_size, precision);
    axpy(b, in1, tmptmp2, Vh * spinor_site_size, precision);

    // out = (A - i 2 kappa mu gamma5 tau3 + epsilon tau1)/(A^2 + mu^2 - epsilon^2)
    apply_clover(out1, clover_inverse, tmptmp1, parity, precision);
    apply_clover(out2, clover_inverse, tmptmp2, parity, precision);

    host_free(tmptmp1);
    host_free(tmptmp2);
  } else {
    errorQuda("Twist type %d not defined", twist);
  }

  host_free(tmp2);
  host_free(tmp1);
}

void tmc_dslash(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
                double kappa, double mu, QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int parity, int dagger,
                QudaPrecision precision, const QudaGaugeParam &param)
{
  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    wil_dslash(tmp1, gauge, in, parity, dagger, precision, param);
    twistCloverGamma5(out, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE,
                      precision);
  } else {
    if (dagger) {
      twistCloverGamma5(tmp1, in, clover, clover_inverse, dagger, kappa, mu, flavor, 1 - parity,
                        QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out, gauge, tmp1, parity, dagger, precision, param);
    } else {
      wil_dslash(tmp1, gauge, in, parity, dagger, precision, param);
      twistCloverGamma5(out, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
    }
  }

  host_free(tmp2);
  host_free(tmp1);
}

// Apply the full twisted-clover operator
void tmc_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
             QudaTwistFlavorType flavor, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  void *tmp = safe_malloc(V * spinor_site_size * precision);

  const void *inEven = in;
  const void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmpEven = tmp;
  void *tmpOdd = (char *)tmp + Vh * spinor_site_size * precision;

  // Odd part
  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  twistCloverGamma5(tmpOdd, inOdd, clover, NULL, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // Even part
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);
  twistCloverGamma5(tmpEven, inEven, clover, NULL, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // lastly apply the kappa term
  xpay(tmp, -kappa, out, V * spinor_site_size, precision);

  host_free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void tmc_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
               double kappa, double mu, QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int dagger,
               QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  double kappa2 = -kappa * kappa;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  switch (matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(tmp2, gauge, tmp1, 0, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
    } else {
      twistCloverGamma5(out, in, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(tmp1, gauge, out, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE,
                      precision);
    wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, in, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_DIRECT,
                      precision);
    xpay(tmp2, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(tmp2, gauge, tmp1, 1, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
    } else {
      twistCloverGamma5(out, in, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(tmp1, gauge, out, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE,
                        precision);
      wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, clover_inverse, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE,
                      precision);
    wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp1, in, clover, clover_inverse, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_DIRECT,
                      precision);
    xpay(tmp1, kappa2, out, Vh * spinor_site_size, precision);
    break;
  default: errorQuda("Unsupported matpc=%d", matpc_type);
  }

  host_free(tmp2);
  host_free(tmp1);
}

// apply the full non-degenerate twisted-clover operator
void tmc_ndeg_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
                  double epsilon, int dagger_bit, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  // V-4d volume and Vh=V/2, see tests/utils/host_utils.cpp -> setDims()
  const void *inEven1 = in;
  const void *inEven2 = (char *)inEven1 + precision * Vh * spinor_site_size;

  const void *inOdd1 = (char *)inEven2 + precision * Vh * spinor_site_size;
  const void *inOdd2 = (char *)inOdd1 + precision * Vh * spinor_site_size;

  void *outEven1 = out;
  void *outEven2 = (char *)outEven1 + precision * Vh * spinor_site_size;

  void *outOdd1 = (char *)outEven2 + precision * Vh * spinor_site_size;
  void *outOdd2 = (char *)outOdd1 + precision * Vh * spinor_site_size;

  void *tmpEven1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmpEven2 = safe_malloc(Vh * spinor_site_size * precision);

  void *tmpOdd1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmpOdd2 = safe_malloc(Vh * spinor_site_size * precision);

  // full dslash operator:
  wil_dslash(outOdd1, gauge, inEven1, 1, dagger_bit, precision, gauge_param);
  wil_dslash(outOdd2, gauge, inEven2, 1, dagger_bit, precision, gauge_param);
  // apply the twisted clover term
  ndegTwistCloverGamma5(tmpEven1, tmpEven2, inEven1, inEven2, clover, NULL, dagger_bit, kappa, mu, epsilon, 0,
                        QUDA_TWIST_GAMMA5_DIRECT, precision);

  wil_dslash(outEven1, gauge, inOdd1, 0, dagger_bit, precision, gauge_param);
  wil_dslash(outEven2, gauge, inOdd2, 0, dagger_bit, precision, gauge_param);
  // apply the twisted clover term
  ndegTwistCloverGamma5(tmpOdd1, tmpOdd2, inOdd1, inOdd2, clover, NULL, dagger_bit, kappa, mu, epsilon, 1,
                        QUDA_TWIST_GAMMA5_DIRECT, precision);

  // combine
  xpay(tmpOdd1, -kappa, outOdd1, Vh * spinor_site_size, precision);
  xpay(tmpOdd2, -kappa, outOdd2, Vh * spinor_site_size, precision);

  xpay(tmpEven1, -kappa, outEven1, Vh * spinor_site_size, precision);
  xpay(tmpEven2, -kappa, outEven2, Vh * spinor_site_size, precision);

  host_free(tmpOdd1);
  host_free(tmpOdd2);
  //
  host_free(tmpEven1);
  host_free(tmpEven2);
}

// dagger_bit && (QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || QUDA_MATPC_ODD_ODD_ASYMMETRIC)
//
//   M_{ee}^{-1}^\dagger (D_{eo})^\dagger M_{oo}^{-1}^\dagger  (parity == 0)
//   M_{oo}^{-1}^\dagger (D_{oe})^\dagger M_{ee}^{-1}^\dagger  (parity == 1)
//
// dagger_bit && (QUDA_MATPC_EVEN_EVEN || QUDA_MATPC_ODD_ODD)
//
//   (D_{oe})^\dagger M_{oo}^{-1}^\dagger (parity == 0)
//   (D_{eo})^\dagger M_{ee}^{-1}^\dagger (parity == 1)
//
// !dagger_bit
//
//   M_{ee}^{-1} D_{eo} (parity == 0)
//   M_{oo}^{-1} D_{oe} (parity == 1)
//
void tmc_ndeg_dslash(void *out, const void *const *gauge, const void *clover, const void *clover_inverse,
                     const void *in, double kappa, double mu, double epsilon, QudaMatPCType matpc_type, int parity,
                     int dagger_bit, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  // V-4d volume and Vh=V/2, see tests/utils/host_utils.cpp -> setDims()
  const void *in1 = in;
  const void *in2 = (char *)in1 + precision * Vh * spinor_site_size;

  void *out1 = out;
  void *out2 = (char *)out1 + precision * Vh * spinor_site_size;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  if (dagger_bit) {
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, clover_inverse, dagger_bit, kappa, mu, epsilon, 1 - parity,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      void *tmptmp1 = safe_malloc(Vh * spinor_site_size * precision);
      void *tmptmp2 = safe_malloc(Vh * spinor_site_size * precision);
      wil_dslash(tmptmp1, gauge, in1, parity, dagger_bit, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, in2, parity, dagger_bit, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, clover_inverse, dagger_bit, kappa, mu, epsilon,
                            parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
      host_free(tmptmp1);
      host_free(tmptmp2);
    } else {
      ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, clover_inverse, dagger_bit, kappa, mu, epsilon, 1 - parity,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out1, gauge, tmp1, parity, dagger_bit, precision, gauge_param);
      wil_dslash(out2, gauge, tmp2, parity, dagger_bit, precision, gauge_param);
    }
  } else {
    wil_dslash(tmp1, gauge, in1, parity, dagger_bit, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, parity, dagger_bit, precision, gauge_param);
    ndegTwistCloverGamma5(out1, out2, tmp1, tmp2, clover, clover_inverse, dagger_bit, kappa, mu, epsilon, parity,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
  }
  host_free(tmp1);
  host_free(tmp2);
}

// Apply the even-odd preconditioned non-degenerate twisted clover Dirac operator
void tmc_ndeg_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
                    double kappa, double mu, double epsilon, QudaMatPCType matpc_type, int dagger,
                    QudaPrecision precision, const QudaGaugeParam &gauge_param)
{

  double kappa2 = -kappa * kappa;

  const void *in1 = in;
  const void *in2 = (char *)in1 + precision * Vh * spinor_site_size;

  void *out1 = out;
  void *out2 = (char *)out1 + precision * Vh * spinor_site_size;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmptmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmptmp2 = safe_malloc(Vh * spinor_site_size * precision);

  switch (matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(out1, gauge, in1, 1, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, in2, 1, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmp1, tmp2, out1, out2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmptmp1, gauge, tmp1, 0, dagger, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, tmp2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      ndegTwistCloverGamma5(out1, out2, in1, in2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out1, 1, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, out2, 1, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out1, gauge, tmptmp1, 0, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, tmptmp2, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, 2 * Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in1, 1, dagger, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, 1, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out1, gauge, tmptmp1, 0, dagger, precision, gauge_param);
    wil_dslash(out2, gauge, tmptmp2, 0, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                          QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp1, kappa2, out1, Vh * spinor_site_size, precision);
    xpay(tmp2, kappa2, out2, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(out1, gauge, in1, 0, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, in2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmp1, tmp2, out1, out2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmptmp1, gauge, tmp1, 1, dagger, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, tmp2, 1, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      ndegTwistCloverGamma5(out1, out2, in1, in2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out1, 0, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, out2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out1, gauge, tmptmp1, 1, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, tmptmp2, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, 2 * Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in1, 0, dagger, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, 0, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, clover_inverse, dagger, kappa, mu, epsilon, 0,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out1, gauge, tmptmp1, 1, dagger, precision, gauge_param);
    wil_dslash(out2, gauge, tmptmp2, 1, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, clover_inverse, dagger, kappa, mu, epsilon, 1,
                          QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp1, kappa2, out1, Vh * spinor_site_size, precision);
    xpay(tmp2, kappa2, out2, Vh * spinor_site_size, precision);
    break;
  default: errorQuda("Unsupported matpc=%d", matpc_type);
  }

  host_free(tmp2);
  host_free(tmp1);
  host_free(tmptmp1);
  host_free(tmptmp2);
}

// Apply the full twisted-clover operator
//   for now   [  A             -k D            ]
//             [ -k D    A(1 - i mu gamma_5 A)  ]

void clover_ht_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
                   int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param, QudaMatPCType matpc_type)
{
  // out = CloverMat in
  clover_mat(out, gauge, clover, in, kappa, dagger, precision, gauge_param);

  bool asymmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  const void *inEven = in;
  const void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;

  if (asymmetric) {
    // Unprec op for asymmetric prec op:
    // apply a simple twist

    // out_parity = out_parity -/+ i mu gamma_5
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // out_e = out_e  -/+ i mu gamma5 in_e
      applyTwist(outEven, inEven, outEven, (dagger ? -mu : mu), precision);

    } else {
      // out_o = out_o  -/+ i mu gamma5 in_o
      applyTwist(outOdd, inOdd, outOdd, (dagger ? -mu : mu), precision);
    }
  } else {

    // Symmetric case:  - i mu gamma_5 A^2 psi_in
    void *tmp = safe_malloc(Vh * spinor_site_size * precision);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {

      // tmp = A_ee in_e
      apply_clover(tmp, clover, inEven, 0, precision);

      // two factors of 2 for two clover applications => (1/4) mu
      // out_e = out_e -/+ i gamma_5 mu A_ee (A_ee) in_ee
      twistClover(outEven, tmp, outEven, clover, 0.25 * mu, dagger, 0, precision);

    } else {
      apply_clover(tmp, clover, inOdd, 1, precision);

      // two factors of 2 for two clover applications => (1/4) mu
      // out_e = out_e -/+ i gamma_5 mu A (A_ee)
      twistClover(outOdd, tmp, outOdd, clover, 0.25 * mu, dagger, 1, precision);
    }
    host_free(tmp);
  }
}

// Apply the even-odd preconditioned Dirac operator
void clover_ht_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse,
                     const void *in, double kappa, double mu, QudaMatPCType matpc_type, int dagger,
                     QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  clover_matpc(out, gauge, clover, clover_inverse, in, kappa, matpc_type, dagger, precision, gauge_param);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    twistClover(out, in, out, clover, 0.5 * mu, dagger, (matpc_type == QUDA_MATPC_EVEN_EVEN ? 0 : 1), precision);
  } else {
    applyTwist(out, in, out, (dagger ? -mu : mu), precision);
  }
}
