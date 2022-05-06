#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>

#include <util_quda.h>
#include <host_utils.h>
#include <wilson_dslash_reference.h>

/**
   @brief Apply the clover matrix field
   @param[out] out Result field (single parity)
   @param[in] clover Clover-matrix field (full field)
   @param[in] in Input field (single parity)
   @param[in] parity Parity to which we are applying the clover field
 */
template <typename sFloat, typename cFloat> void cloverReference(sFloat *out, cFloat *clover, sFloat *in, int parity)
{
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2 * (N - 1) * N / 2;

  for (int i = 0; i < Vh; i++) {
    std::complex<sFloat> *In = reinterpret_cast<std::complex<sFloat> *>(&in[i * nSpin * nColor * 2]);
    std::complex<sFloat> *Out = reinterpret_cast<std::complex<sFloat> *>(&out[i * nSpin * nColor * 2]);

    for (int chi = 0; chi < nSpin / 2; chi++) {
      cFloat *D = &clover[((parity * Vh + i) * 2 + chi) * chiralBlock];
      std::complex<cFloat> *L = reinterpret_cast<std::complex<cFloat> *>(&D[N]);

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

void apply_clover(void *out, void *clover, void *in, int parity, QudaPrecision precision)
{

  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
    cloverReference(static_cast<double *>(out), static_cast<double *>(clover), static_cast<double *>(in), parity);
    break;
  case QUDA_SINGLE_PRECISION:
    cloverReference(static_cast<float *>(out), static_cast<float *>(clover), static_cast<float *>(in), parity);
    break;
  default: errorQuda("Unsupported precision %d", precision);
  }
}

void clover_dslash(void *out, void **gauge, void *clover, void *in, int parity, int dagger, QudaPrecision precision,
                   QudaGaugeParam &param)
{
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  wil_dslash(tmp, gauge, in, parity, dagger, precision, param);
  apply_clover(out, clover, tmp, parity, precision);

  host_free(tmp);
}

// Apply the even-odd preconditioned Wilson-clover operator
void clover_matpc(void *out, void **gauge, void *clover, void *clover_inv, void *in, double kappa,
                  QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param)
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
void clover_mat(void *out, void **gauge, void *clover, void *in, double kappa, int dagger, QudaPrecision precision,
                QudaGaugeParam &gauge_param)
{

  void *tmp = safe_malloc(V * spinor_site_size * precision);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
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

void applyTwist(void *out, void *in, void *tmpH, double a, QudaPrecision precision)
{
  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
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

// out = x - i*a*gamma_5 Clov *in  =
void twistClover(void *out, void *in, void *x, void *clover, const double a, int dagger, int parity,
                 QudaPrecision precision)
{
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  // tmp1 = Clov in
  apply_clover(tmp, clover, in, parity, precision);
  applyTwist(out, tmp, x, (dagger ? -a : a), precision);

  host_free(tmp);
}

// Apply (C + i*a*gamma_5)/(C^2 + a^2)
void twistCloverGamma5(void *out, void *in, void *clover, void *cInv, const int dagger, const double kappa,
                       const double mu, const QudaTwistFlavorType flavor, const int parity, QudaTwistGamma5Type twist,
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
    apply_clover(out, cInv, tmp2, parity, precision);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  host_free(tmp2);
  host_free(tmp1);
}

// Apply (A + i*mu*gamma_5*tau3 - epsilon*tau1) for QUDA_TWIST_GAMMA5_DIRECT
// and   (A - i*mu*gamma_5*tau3 + epsilon*tau1)/(A^2 + mu^2 - epsilon^2) for QUDA_TWIST_GAMMA5_INVERSE
void ndegTwistCloverGamma5(void *out1, void *out2, void *in1, void *in2, void *clover, void *cInv, const int dagger,
                           const double kappa, const double mu, const double epsilon, const int parity,
                           QudaTwistGamma5Type twist, QudaPrecision precision)
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
    apply_clover(out1, cInv, tmptmp1, parity, precision);
    apply_clover(out2, cInv, tmptmp2, parity, precision);

    host_free(tmptmp1);
    host_free(tmptmp2);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  host_free(tmp2);
  host_free(tmp1);
}

void tmc_dslash(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu,
                QudaTwistFlavorType flavor, int parity, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                QudaGaugeParam &param)
{
  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  if (dagger) {
    twistCloverGamma5(tmp1, in, clover, cInv, dagger, kappa, mu, flavor, 1 - parity, QUDA_TWIST_GAMMA5_INVERSE,
                      precision);
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      wil_dslash(tmp2, gauge, tmp1, parity, dagger, precision, param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      wil_dslash(out, gauge, tmp1, parity, dagger, precision, param);
    }
  } else {
    wil_dslash(tmp1, gauge, in, parity, dagger, precision, param);
    twistCloverGamma5(out, tmp1, clover, cInv, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
  }

  host_free(tmp2);
  host_free(tmp1);
}

// Apply the full twisted-clover operator
void tmc_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, QudaTwistFlavorType flavor,
             int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param)
{

  void *tmp = safe_malloc(V * spinor_site_size * precision);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
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
void tmc_matpc(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu,
               QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
               QudaGaugeParam &gauge_param)
{

  double kappa2 = -kappa * kappa;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  switch (matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp2, gauge, tmp1, 0, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      twistCloverGamma5(out, in, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, in, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp2, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp2, gauge, tmp1, 1, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      twistCloverGamma5(out, in, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp1, in, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp1, kappa2, out, Vh * spinor_site_size, precision);
    break;
  default: errorQuda("Unsupported matpc=%d", matpc_type);
  }

  host_free(tmp2);
  host_free(tmp1);
}

// apply the full non-degenerate twisted-clover operator
void tmc_ndeg_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, double epsilon,
                  int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param)
{
  // V-4d volume and Vh=V/2, see tests/utils/host_utils.cpp -> setDims()
  void *inEven1 = in;
  void *inEven2 = (char *)inEven1 + precision * Vh * spinor_site_size;

  void *inOdd1 = (char *)inEven2 + precision * Vh * spinor_site_size;
  void *inOdd2 = (char *)inOdd1 + precision * Vh * spinor_site_size;

  void *outEven1 = out;
  void *outEven2 = (char *)outEven1 + precision * Vh * spinor_site_size;

  void *outOdd1 = (char *)outEven2 + precision * Vh * spinor_site_size;
  void *outOdd2 = (char *)outOdd1 + precision * Vh * spinor_site_size;

  void *tmpEven1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmpEven2 = safe_malloc(Vh * spinor_site_size * precision);

  void *tmpOdd1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmpOdd2 = safe_malloc(Vh * spinor_site_size * precision);

  // full dslash operator:
  wil_dslash(outOdd1, gauge, inEven1, 1, daggerBit, precision, gauge_param);
  wil_dslash(outOdd2, gauge, inEven2, 1, daggerBit, precision, gauge_param);
  // apply the twisted clover term
  ndegTwistCloverGamma5(tmpEven1, tmpEven2, inEven1, inEven2, clover, NULL, daggerBit, kappa, mu, epsilon, 0,
                        QUDA_TWIST_GAMMA5_DIRECT, precision);

  wil_dslash(outEven1, gauge, inOdd1, 0, daggerBit, precision, gauge_param);
  wil_dslash(outEven2, gauge, inOdd2, 0, daggerBit, precision, gauge_param);
  // apply the twisted clover term
  ndegTwistCloverGamma5(tmpOdd1, tmpOdd2, inOdd1, inOdd2, clover, NULL, daggerBit, kappa, mu, epsilon, 1,
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

// daggerBit && (QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || QUDA_MATPC_ODD_ODD_ASYMMETRIC)
//
//   M_{ee}^{-1}^\dagger (D_{eo})^\dagger M_{oo}^{-1}^\dagger  (oddBit == 0)
//   M_{oo}^{-1}^\dagger (D_{oe})^\dagger M_{ee}^{-1}^\dagger  (oddBit == 1)
//
// daggerBit && (QUDA_MATPC_EVEN_EVEN || QUDA_MATPC_ODD_ODD)
//
//   (D_{oe})^\dagger M_{oo}^{-1}^\dagger (oddBit == 0)
//   (D_{eo})^\dagger M_{ee}^{-1}^\dagger (oddBit == 1)
//
// !daggerBit
//
//   M_{ee}^{-1} D_{eo} (oddBit == 0)
//   M_{oo}^{-1} D_{oe} (oddBit == 1)
//
void tmc_ndeg_dslash(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu,
                     double epsilon, int oddBit, QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision,
                     QudaGaugeParam &gauge_param)
{
  // V-4d volume and Vh=V/2, see tests/utils/host_utils.cpp -> setDims()
  void *in1 = in;
  void *in2 = (char *)in1 + precision * Vh * spinor_site_size;

  void *out1 = out;
  void *out2 = (char *)out1 + precision * Vh * spinor_site_size;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  if (daggerBit) {
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, cInv, daggerBit, kappa, mu, epsilon, 1 - oddBit,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      void *tmptmp1 = safe_malloc(Vh * spinor_site_size * precision);
      void *tmptmp2 = safe_malloc(Vh * spinor_site_size * precision);
      wil_dslash(tmptmp1, gauge, tmp1, oddBit, daggerBit, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, tmp2, oddBit, daggerBit, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, cInv, daggerBit, kappa, mu, epsilon, oddBit,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      host_free(tmptmp1);
      host_free(tmptmp2);
    } else {
      wil_dslash(out1, gauge, tmp1, oddBit, daggerBit, precision, gauge_param);
      wil_dslash(out2, gauge, tmp2, oddBit, daggerBit, precision, gauge_param);
    }
  } else {
    wil_dslash(tmp1, gauge, in1, oddBit, daggerBit, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, oddBit, daggerBit, precision, gauge_param);
    ndegTwistCloverGamma5(out1, out2, tmp1, tmp2, clover, cInv, daggerBit, kappa, mu, epsilon, oddBit,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
  }
  host_free(tmp1);
  host_free(tmp2);
}

// Apply the even-odd preconditioned non-degenerate twisted clover Dirac operator
void tmc_ndeg_matpc(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu, double epsilon,
                    QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param)
{

  double kappa2 = -kappa * kappa;

  void *in1 = in;
  void *in2 = (char *)in1 + precision * Vh * spinor_site_size;

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
      ndegTwistCloverGamma5(tmp1, tmp2, out1, out2, clover, cInv, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmptmp1, gauge, tmp1, 0, dagger, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, tmp2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, cInv, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      ndegTwistCloverGamma5(out1, out2, in1, in2, clover, cInv, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out1, 1, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, out2, 1, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, cInv, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out1, gauge, tmptmp1, 0, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, tmptmp2, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, 2 * Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in1, 1, dagger, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, 1, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, cInv, dagger, kappa, mu, epsilon, 1,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out1, gauge, tmptmp1, 0, dagger, precision, gauge_param);
    wil_dslash(out2, gauge, tmptmp2, 0, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, cInv, dagger, kappa, mu, epsilon, 0, QUDA_TWIST_GAMMA5_DIRECT,
                          precision);
    xpay(tmp1, kappa2, out1, Vh * spinor_site_size, precision);
    xpay(tmp2, kappa2, out2, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(out1, gauge, in1, 0, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, in2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmp1, tmp2, out1, out2, clover, cInv, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmptmp1, gauge, tmp1, 1, dagger, precision, gauge_param);
      wil_dslash(tmptmp2, gauge, tmp2, 1, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(out1, out2, tmptmp1, tmptmp2, clover, cInv, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      ndegTwistCloverGamma5(out1, out2, in1, in2, clover, cInv, dagger, kappa, mu, epsilon, 1,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out1, 0, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, out2, 0, dagger, precision, gauge_param);
      ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, cInv, dagger, kappa, mu, epsilon, 0,
                            QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out1, gauge, tmptmp1, 1, dagger, precision, gauge_param);
      wil_dslash(out2, gauge, tmptmp2, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, 2 * Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in1, 0, dagger, precision, gauge_param);
    wil_dslash(tmp2, gauge, in2, 0, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmptmp1, tmptmp2, tmp1, tmp2, clover, cInv, dagger, kappa, mu, epsilon, 0,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out1, gauge, tmptmp1, 1, dagger, precision, gauge_param);
    wil_dslash(out2, gauge, tmptmp2, 1, dagger, precision, gauge_param);
    ndegTwistCloverGamma5(tmp1, tmp2, in1, in2, clover, cInv, dagger, kappa, mu, epsilon, 1, QUDA_TWIST_GAMMA5_DIRECT,
                          precision);
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

void cloverHasenbuchTwist_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, int dagger,
                              QudaPrecision precision, QudaGaugeParam &gauge_param, QudaMatPCType matpc_type)
{

  // out = CloverMat in
  clover_mat(out, gauge, clover, in, kappa, dagger, precision, gauge_param);

  bool asymmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
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
void cloverHasenbuschTwist_matpc(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu,
                                 QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                                 QudaGaugeParam &gauge_param)
{

  clover_matpc(out, gauge, clover, cInv, in, kappa, matpc_type, dagger, precision, gauge_param);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    twistClover(out, in, out, clover, 0.5 * mu, dagger, (matpc_type == QUDA_MATPC_EVEN_EVEN ? 0 : 1), precision);
  } else {
    applyTwist(out, in, out, (dagger ? -mu : mu), precision);
  }
}
