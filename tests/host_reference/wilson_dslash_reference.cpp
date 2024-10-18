#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gauge_field.h>
#include <color_spinor_field.h>

#include "host_utils.h"
#include "index_utils.hpp"
#include "util_quda.h"
#include "wilson_dslash_reference.h"
#include "dslash_reference.h"
#include "gamma_reference.h"

using namespace quda;

//
// dslashReference()
//
// if parity is zero: calculate odd parity spinor elements (using even parity spinor)
// if parity is one:  calculate even parity spinor elements
//
// if dagger is zero: perform ordinary dslash operator
// if dagger is one:  perform hermitian conjugate of dslash
//

/**
 * @brief Perform a Wilson dslash operation on a spinor field
 *
 * @tparam real_t The floating-point type used for the computation.
 * @param[out] res The result of the Dslash operation.
 * @param[in] gaugeFull The full gauge field.
 * @param[in] ghostGauge The ghost gauge field for multi-GPU computations.
 * @param[in] spinorField The input spinor field.
 * @param[in] fwdSpinor The forward ghost region of the spinor field
 * @param[in] backSpinor The backward ghost region of the spinor field
 * @param[in] parity The parity of the dslash (0 for even, 1 for odd).
 * @param[in] dagger Whether to apply the original or the Hermitian conjugate operator
 */
template <typename real_t>
void dslashReference(real_t *res, const real_t *const *gaugeFull, const real_t *const *ghostGauge,
                     const real_t *spinorField, const real_t *const *fwdSpinor, const real_t *const *backSpinor,
                     int parity, int dagger)
{
#pragma omp parallel for
  for (auto i = 0lu; i < Vh * spinor_site_size; i++) res[i] = 0.0;

  const real_t *gaugeEven[4], *gaugeOdd[4];
  const real_t *ghostGaugeEven[4] = {nullptr, nullptr, nullptr, nullptr};
  const real_t *ghostGaugeOdd[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int dir = 0; dir < 4; dir++) {
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;

    if (is_multi_gpu()) {
      ghostGaugeEven[dir] = ghostGauge[dir];
      ghostGaugeOdd[dir] = ghostGauge[dir] + (faceVolume[dir] / 2) * gauge_site_size;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < Vh; i++) {

    for (int dir = 0; dir < 8; dir++) {
      const real_t *gauge = gaugeLink(i, dir, parity, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);
      const real_t *spinor = spinorNeighbor(i, dir, parity, spinorField, fwdSpinor, backSpinor, 1, 1);

      real_t projectedSpinor[spinor_site_size], gaugedSpinor[spinor_site_size];
      int projIdx = 2 * (dir / 2) + (dir + dagger) % 2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);

      for (int s = 0; s < 4; s++) {
        if (dir % 2 == 0)
          su3Mul(&gaugedSpinor[s * (3 * 2)], gauge, &projectedSpinor[s * (3 * 2)]);
        else
          su3Tmul(&gaugedSpinor[s * (3 * 2)], gauge, &projectedSpinor[s * (3 * 2)]);
      }

      sum(&res[i * spinor_site_size], &res[i * spinor_site_size], gaugedSpinor, spinor_site_size);
    }
  }
}

void wil_dslash(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                const QudaGaugeParam &gauge_param)
{
  GaugeFieldParam gauge_field_param(gauge_param, (void *)gauge);
  gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  gauge_field_param.location = QUDA_CPU_FIELD_LOCATION;
  GaugeField cpu(gauge_field_param);
  void *ghostGauge[4] = {cpu.Ghost()[0].data(), cpu.Ghost()[1].data(), cpu.Ghost()[2].data(), cpu.Ghost()[3].data()};

  // Get spinor ghost fields
  // First wrap the input spinor into a ColorSpinorField
  ColorSpinorParam csParam;
  csParam.location = QUDA_CPU_FIELD_LOCATION;
  csParam.v = (void *)in;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d = 0; d < 4; d++) csParam.x[d] = Z[d];
  csParam.setPrecision(precision);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.x[0] /= 2;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.pc_type = QUDA_4D_PC;

  ColorSpinorField inField(csParam);

  { // Now do the exchange
    QudaParity otherParity = QUDA_INVALID_PARITY;
    if (parity == QUDA_EVEN_PARITY)
      otherParity = QUDA_ODD_PARITY;
    else if (parity == QUDA_ODD_PARITY)
      otherParity = QUDA_EVEN_PARITY;
    else
      errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
    const int nFace = 1;

    inField.exchangeGhost(otherParity, nFace, dagger);
  }
  void **fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
  void **back_nbr_spinor = inField.backGhostFaceBuffer;

  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference((double *)out, (double **)gauge, (double **)ghostGauge, (double *)in, (double **)fwd_nbr_spinor,
                    (double **)back_nbr_spinor, parity, dagger);
  } else {
    dslashReference((float *)out, (float **)gauge, (float **)ghostGauge, (float *)in, (float **)fwd_nbr_spinor,
                    (float **)back_nbr_spinor, parity, dagger);
  }
}

// applies b*(1 + i*a*gamma_5)
template <typename real_t>
void twistGamma5(real_t *out, const real_t *in, int dagger, real_t kappa, real_t mu, QudaTwistFlavorType flavor, int V,
                 QudaTwistGamma5Type twist)
{
  real_t a = 0.0, b = 0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu * flavor;         // mu already includes the flavor
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu * flavor;
    b = 1.0 / (1.0 + a * a);
  } else {
    errorQuda("Twist type %d not defined", twist);
  }

  if (dagger) a *= -1.0;

#pragma omp parallel for
  for (int i = 0; i < V; i++) {
    real_t tmp[24];
    for (int s = 0; s < 4; s++)
      for (int c = 0; c < 3; c++) {
        real_t a5 = ((s / 2) ? -1.0 : +1.0) * a;
        tmp[s * 6 + c * 2 + 0] = b * (in[i * 24 + s * 6 + c * 2 + 0] - a5 * in[i * 24 + s * 6 + c * 2 + 1]);
        tmp[s * 6 + c * 2 + 1] = b * (in[i * 24 + s * 6 + c * 2 + 1] + a5 * in[i * 24 + s * 6 + c * 2 + 0]);
      }

    for (int j = 0; j < 24; j++) out[i * 24 + j] = tmp[j];
  }
}

void twist_gamma5(void *out, const void *in, int dagger, double kappa, double mu, QudaTwistFlavorType flavor, int V,
                  QudaTwistGamma5Type twist, QudaPrecision precision)
{

  if (precision == QUDA_DOUBLE_PRECISION) {
    twistGamma5((double *)out, (double *)in, dagger, kappa, mu, flavor, V, twist);
  } else {
    twistGamma5((float *)out, (float *)in, dagger, (float)kappa, (float)mu, flavor, V, twist);
  }
}

void tm_dslash(void *out, const void *const *gauge, const void *in_, double kappa, double mu,
               QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param)
{
  // in some cases, for simplicity, in is modified in place.
  void *in = (void *)in_;

  if (dagger && (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD))
    twist_gamma5(in, in, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);

  wil_dslash(out, gauge, in, parity, dagger, precision, gauge_param);

  if (!dagger
      || (dagger && (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC))) {
    twist_gamma5(out, out, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
  } else {
    twist_gamma5(in, in, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  }
}

void wil_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger, QudaPrecision precision,
             const QudaGaugeParam &gauge_param)
{
  const void *inEven = in;
  const void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;

  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V * spinor_site_size, precision);
}

void tm_mat(void *out, const void *const *gauge, const void *in, double kappa, double mu, QudaTwistFlavorType flavor,
            int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  const void *inEven = in;
  const void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmp = safe_malloc(V * spinor_site_size * precision);

  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);

  // apply the twist term to the full lattice
  twist_gamma5(tmp, in, dagger, kappa, mu, flavor, V, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // combine
  xpay(tmp, -kappa, (double *)out, V * spinor_site_size, precision);

  host_free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void wil_matpc(void *outEven, const void *const *gauge, const void *inEven, double kappa, QudaMatPCType matpc_type,
               int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  // FIXME: remove once reference clover is finished
  // full dslash operator
  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 1, dagger, precision, gauge_param);
    wil_dslash(outEven, gauge, tmp, 0, dagger, precision, gauge_param);
  } else {
    wil_dslash(tmp, gauge, inEven, 0, dagger, precision, gauge_param);
    wil_dslash(outEven, gauge, tmp, 1, dagger, precision, gauge_param);
  }

  // lastly apply the kappa term
  double kappa2 = -kappa * kappa;
  xpay(inEven, kappa2, outEven, Vh * spinor_site_size, precision);

  host_free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void tm_matpc(void *outEven, const void *const *gauge, const void *inEven_, double kappa, double mu,
              QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
              const QudaGaugeParam &gauge_param)
{
  // for optimization reasons, inEven gets flipped "in-place" and then it's undone later
  void *inEven = (void *)inEven_;

  void *tmp = safe_malloc(Vh * spinor_site_size * precision);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 1, dagger, precision, gauge_param);
    twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(outEven, gauge, tmp, 0, dagger, precision, gauge_param);
    twist_gamma5(tmp, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  } else if (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 0, dagger, precision, gauge_param);
    twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(outEven, gauge, tmp, 1, dagger, precision, gauge_param);
    twist_gamma5(tmp, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  } else if (!dagger) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      wil_dslash(tmp, gauge, inEven, 1, dagger, precision, gauge_param);
      twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 0, dagger, precision, gauge_param);
      twist_gamma5(outEven, outEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      wil_dslash(tmp, gauge, inEven, 0, dagger, precision, gauge_param);
      twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 1, dagger, precision, gauge_param);
      twist_gamma5(outEven, outEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      twist_gamma5(inEven, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp, gauge, inEven, 1, dagger, precision, gauge_param);
      twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 0, dagger, precision, gauge_param);
      twist_gamma5(inEven, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      twist_gamma5(inEven, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp, gauge, inEven, 0, dagger, precision, gauge_param);
      twist_gamma5(tmp, tmp, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 1, dagger, precision, gauge_param);
      twist_gamma5(inEven, inEven, dagger, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision); // undo
    }
  }
  // lastly apply the kappa term
  double kappa2 = -kappa * kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    xpay(inEven, kappa2, outEven, Vh * spinor_site_size, precision);
  } else {
    xpay(tmp, kappa2, outEven, Vh * spinor_site_size, precision);
  }

  host_free(tmp);
}

//----- for non-degenerate dslash only----
template <typename real_t>
void ndegTwistGamma5(real_t *out1, real_t *out2, const real_t *in1, const real_t *in2, const int dagger,
                     const real_t kappa, const real_t mu, const real_t epsilon, const int V, QudaTwistGamma5Type twist)
{
  real_t a = 0.0, b = 0.0, d = 0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu;
    b = -2.0 * kappa * epsilon;
    d = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu;
    b = 2.0 * kappa * epsilon;
    d = 1.0 / (1.0 + a * a - b * b);
  } else {
    errorQuda("Twist type %d not defined", twist);
  }

  if (dagger) a *= -1.0;

#pragma omp parallel for
  for (int i = 0; i < V; i++) {
    real_t tmp1[24];
    real_t tmp2[24];
    for (int s = 0; s < 4; s++)
      for (int c = 0; c < 3; c++) {
        real_t a5 = ((s / 2) ? -1.0 : +1.0) * a;
        tmp1[s * 6 + c * 2 + 0] = d
          * (in1[i * 24 + s * 6 + c * 2 + 0] - a5 * in1[i * 24 + s * 6 + c * 2 + 1] + b * in2[i * 24 + s * 6 + c * 2 + 0]);
        tmp1[s * 6 + c * 2 + 1] = d
          * (in1[i * 24 + s * 6 + c * 2 + 1] + a5 * in1[i * 24 + s * 6 + c * 2 + 0] + b * in2[i * 24 + s * 6 + c * 2 + 1]);
        tmp2[s * 6 + c * 2 + 0] = d
          * (in2[i * 24 + s * 6 + c * 2 + 0] + a5 * in2[i * 24 + s * 6 + c * 2 + 1] + b * in1[i * 24 + s * 6 + c * 2 + 0]);
        tmp2[s * 6 + c * 2 + 1] = d
          * (in2[i * 24 + s * 6 + c * 2 + 1] - a5 * in2[i * 24 + s * 6 + c * 2 + 0] + b * in1[i * 24 + s * 6 + c * 2 + 1]);
      }
    for (int j = 0; j < 24; j++) out1[i * 24 + j] = tmp1[j], out2[i * 24 + j] = tmp2[j];
  }
}

void ndeg_twist_gamma5(void *outf1, void *outf2, const void *inf1, const void *inf2, const int dagger,
                       const double kappa, const double mu, const double epsilon, const int Vf,
                       QudaTwistGamma5Type twist, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    ndegTwistGamma5((double *)outf1, (double *)outf2, (double *)inf1, (double *)inf2, dagger, kappa, mu, epsilon, Vf,
                    twist);
  } else // single precision dslash
  {
    ndegTwistGamma5((float *)outf1, (float *)outf2, (float *)inf1, (float *)inf2, dagger, (float)kappa, (float)mu,
                    (float)epsilon, Vf, twist);
  }
}

void tm_ndeg_dslash(void *out, const void *const *gauge, const void *in_, double kappa, double mu, double epsilon,
                    QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
                    const QudaGaugeParam &gauge_param)
{
  // for optimization reasons, in gets flipped "in-place" and then it's undone later
  void *in = (void *)in_;

  void *out1 = out;
  void *out2 = (char *)out1 + Vh * spinor_site_size * precision;

  void *in1 = in;
  void *in2 = (char *)in1 + Vh * spinor_site_size * precision;

  if (dagger && (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD))
    ndeg_twist_gamma5(in1, in2, in1, in2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);

  wil_dslash(out1, gauge, in1, parity, dagger, precision, gauge_param);
  wil_dslash(out2, gauge, in2, parity, dagger, precision, gauge_param);

  if (!dagger || (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)) {
    ndeg_twist_gamma5(out1, out2, out1, out2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
  }
}

void tm_ndeg_matpc(void *outEven, const void *const *gauge, const void *inEven, double kappa, double mu, double epsilon,
                   QudaMatPCType matpc_type, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param)
{
  void *outEven1 = outEven;
  void *outEven2 = (char *)outEven1 + Vh * spinor_site_size * precision;

  const void *inEven1 = inEven;
  const void *inEven2 = (char *)inEven1 + Vh * spinor_site_size * precision;

  void *tmp1 = safe_malloc(Vh * spinor_site_size * precision);
  void *tmp2 = safe_malloc(Vh * spinor_site_size * precision);

  double kappa2 = -kappa * kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    if (!dagger) {
      if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
        wil_dslash(tmp1, gauge, inEven1, 1, dagger, precision, gauge_param);
        wil_dslash(tmp2, gauge, inEven2, 1, dagger, precision, gauge_param);
        ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
        wil_dslash(outEven1, gauge, tmp1, 0, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 0, dagger, precision, gauge_param);
        ndeg_twist_gamma5(outEven1, outEven2, outEven1, outEven2, dagger, kappa, mu, epsilon, Vh,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
      } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
        wil_dslash(tmp1, gauge, inEven1, 0, dagger, precision, gauge_param);
        wil_dslash(tmp2, gauge, inEven2, 0, dagger, precision, gauge_param);
        ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
        wil_dslash(outEven1, gauge, tmp1, 1, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 1, dagger, precision, gauge_param);
        ndeg_twist_gamma5(outEven1, outEven2, outEven1, outEven2, dagger, kappa, mu, epsilon, Vh,
                          QUDA_TWIST_GAMMA5_INVERSE, precision);
      }
    } else {
      if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
        ndeg_twist_gamma5(tmp1, tmp2, inEven1, inEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE,
                          precision);
        wil_dslash(outEven1, gauge, tmp1, 1, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 1, dagger, precision, gauge_param);
        ndeg_twist_gamma5(tmp1, tmp2, outEven1, outEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE,
                          precision);
        wil_dslash(outEven1, gauge, tmp1, 0, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 0, dagger, precision, gauge_param);
      } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
        ndeg_twist_gamma5(tmp1, tmp2, inEven1, inEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE,
                          precision);
        wil_dslash(outEven1, gauge, tmp1, 0, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 0, dagger, precision, gauge_param);
        ndeg_twist_gamma5(tmp1, tmp2, outEven1, outEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE,
                          precision);
        wil_dslash(outEven1, gauge, tmp1, 1, dagger, precision, gauge_param);
        wil_dslash(outEven2, gauge, tmp2, 1, dagger, precision, gauge_param);
      }
    }
    xpay(inEven1, kappa2, outEven1, Vh * spinor_site_size, precision);
    xpay(inEven2, kappa2, outEven2, Vh * spinor_site_size, precision);
  } else if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      wil_dslash(tmp1, gauge, inEven1, 1, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 1, dagger, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 0, dagger, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 0, dagger, precision, gauge_param);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      wil_dslash(tmp1, gauge, inEven1, 0, dagger, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 0, dagger, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 1, dagger, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 1, dagger, precision, gauge_param);
    }
    ndeg_twist_gamma5(tmp1, tmp2, inEven1, inEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp1, kappa2, outEven1, Vh * spinor_site_size, precision);
    xpay(tmp2, kappa2, outEven2, Vh * spinor_site_size, precision);
  }

  host_free(tmp1);
  host_free(tmp2);
}

void tm_ndeg_mat(void *out, const void *const *gauge, const void *in, double kappa, double mu, double epsilon,
                 int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param)
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
  wil_dslash(outOdd1, gauge, inEven1, 1, dagger, precision, gauge_param);
  wil_dslash(outOdd2, gauge, inEven2, 1, dagger, precision, gauge_param);

  wil_dslash(outEven1, gauge, inOdd1, 0, dagger, precision, gauge_param);
  wil_dslash(outEven2, gauge, inOdd2, 0, dagger, precision, gauge_param);

  // apply the twist term
  ndeg_twist_gamma5(tmpEven1, tmpEven2, inEven1, inEven2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT,
                    precision);
  ndeg_twist_gamma5(tmpOdd1, tmpOdd2, inOdd1, inOdd2, dagger, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT,
                    precision);
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

// End of nondeg TM
