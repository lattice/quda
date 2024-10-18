#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include <gauge_field.h>
#include <color_spinor_field.h>

#include "host_utils.h"
#include "index_utils.hpp"
#include "util_quda.h"
#include "domain_wall_dslash_reference.h"
#include "dslash_reference.h"
#include "gamma_reference.h"

using namespace quda;

/**
 * @brief Apply the 4-d Dslash (Wilson) to all fifth dimensional slices for a 4-d data layout
 *
 * @tparam type Domain wall preconditioning type (4 or 5 dimensions)
 * @tparam real_t The floating-point type used for the computation.
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] ghostGauge The ghost gauge field for multi-GPU computations.
 * @param[in] in Host input spinor
 * @param[in] fwdSpinor The forward ghost region of the spinor field
 * @param[in] backSpinor The backward ghost region of the spinor field
 * @param[in] parity The parity of the dslash (0 for even, 1 for odd).
 * @param[in] dagger Whether to apply the original or the Hermitian conjugate operator
 */
template <QudaPCType type, typename real_t>
void dslashReference_4d(real_t *out, const real_t *const *gauge, real_t const *const *ghostGauge, const real_t *in,
                        const real_t *const *fwdSpinor, const real_t *const *backSpinor, int parity, int dagger)
{
#pragma omp parallel for
  for (auto i = 0lu; i < V5h * spinor_site_size; i++) out[i] = 0.0;

  const real_t *gaugeEven[4], *gaugeOdd[4];
  const real_t *ghostGaugeEven[4], *ghostGaugeOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    gaugeEven[dir] = gauge[dir];
    gaugeOdd[dir] = gauge[dir] + Vh * gauge_site_size;

    ghostGaugeEven[dir] = is_multi_gpu() ? ghostGauge[dir] : nullptr;
    ghostGaugeOdd[dir] = is_multi_gpu() ? ghostGauge[dir] + (faceVolume[dir] / 2) * gauge_site_size : nullptr;
  }

  for (int xs = 0; xs < Ls; xs++) {
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      int sp_idx = i + Vh * xs;
      for (int dir = 0; dir < 8; dir++) {
        int gaugeOddBit = (xs % 2 == 0 || type == QUDA_4D_PC) ? parity : (parity + 1) % 2;

        const real_t *gauge = gaugeLink(i, dir, gaugeOddBit, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);
        const real_t *spinor = spinorNeighbor_5d<type>(sp_idx, dir, parity, in, fwdSpinor, backSpinor, 1, 1);

        real_t projectedSpinor[spinor_site_size], gaugedSpinor[spinor_site_size];
        int projIdx = 2 * (dir / 2) + (dir + dagger) % 2;
        multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);

        for (int s = 0; s < 4; s++) {
          if (dir % 2 == 0)
            su3Mul(&gaugedSpinor[s * (3 * 2)], gauge, &projectedSpinor[s * (3 * 2)]);
          else
            su3Tmul(&gaugedSpinor[s * (3 * 2)], gauge, &projectedSpinor[s * (3 * 2)]);
        }
        sum(&out[sp_idx * (4 * 3 * 2)], &out[sp_idx * (4 * 3 * 2)], gaugedSpinor, 4 * 3 * 2);
      }
    }
  }
}

/**
 * @brief Performs a linear combination of vectors with gamma_+ or gamma_- projection
 *
 * @tparam plus If true, use gamma_+; if false, use gamma_-
 * @tparam real_t The floating-point type used for the vectors
 * @param[out] z The output vector
 * @param[in] a The scaling factor for vector x
 * @param[in] x The first input vector
 * @param[in] b The scaling factor for vector y
 * @param[in] y The second input vector
 * @param[in] idx_cb_4d The 4D checkerboard index
 * @param[in] s The fifth dimension index for vector x and z
 * @param[in] sp The fifth dimension index for vector y
 */
template <bool plus, class real_t> // plus = true -> gamma_+; plus = false -> gamma_-
void axpby_ssp_project(real_t *z, real_t a, const real_t *x, real_t b, const real_t *y, int idx_cb_4d, int s, int sp)
{
  // z_s = a*x_s + b*\gamma_+/-*y_sp
  // Will use the DeGrand-Rossi/CPS basis, where gamma5 is diagonal:
  // +1   0
  //  0  -1
  for (int spin = (plus ? 0 : 2); spin < (plus ? 2 : 4); spin++) {
    for (int color_comp = 0; color_comp < 6; color_comp++) {
      z[(s * Vh + idx_cb_4d) * 24 + spin * 6 + color_comp] = a * x[(s * Vh + idx_cb_4d) * 24 + spin * 6 + color_comp]
        + b * y[(sp * Vh + idx_cb_4d) * 24 + spin * 6 + color_comp];
    }
  }
  for (int spin = (plus ? 2 : 0); spin < (plus ? 4 : 2); spin++) {
    for (int color_comp = 0; color_comp < 6; color_comp++) {
      z[(s * Vh + idx_cb_4d) * 24 + spin * 6 + color_comp] = a * x[(s * Vh + idx_cb_4d) * 24 + spin * 6 + color_comp];
    }
  }
}

/**
 * @brief Apply the Ls dimension portion (m5) of EOFA Mobius dslash
 *
 * @tparam real_t The floating-point type used for the vectors
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] mferm Domain wall fermion mass
 * @param[in] m5 Domain wall bulk fermion mass
 * @param[in] b "b" value for Mobius
 * @param[in] c "c" value for Mobius
 * @param[in] mq1 EOFA parameter mq1
 * @param[in] mq2 EOFA parameter mq2
 * @param[in] mq3 EOFA parameter mq3
 * @param[in] eofa_pm EOFA parameter eofa_pm
 * @param[in] eofa_shift EOFA parameter eofa_shift
 * @param[in] precision Single or double precision
 */
template <typename real_t>
void mdw_eofa_m5_ref(real_t *out, const real_t *in, int parity, int dagger, real_t mferm, real_t m5, real_t b, real_t c,
                     real_t mq1, real_t mq2, real_t mq3, int eofa_pm, real_t eofa_shift)
{
  real_t alpha = b + c;
  real_t eofa_norm = alpha * (mq3 - mq2) * std::pow(alpha + 1., 2 * Ls)
    / (std::pow(alpha + 1., Ls) + mq2 * std::pow(alpha - 1., Ls))
    / (std::pow(alpha + 1., Ls) + mq3 * std::pow(alpha - 1., Ls));

  real_t kappa = 0.5 * (c * (4. + m5) - 1.) / (b * (4. + m5) + 1.);

  constexpr int spinor_size = 4 * 3 * 2;
#pragma omp parallel for
  for (int i = 0; i < V5h; i++) {
    for (int one_site = 0; one_site < 24; one_site++) { out[i * spinor_size + one_site] = 0.; }
    for (int dir = 8; dir < 10; dir++) {
      // Calls for an extension of the original function.
      // 8 is forward hop, which wants P_+, 9 is backward hop,
      // which wants P_-.  Dagger reverses these.
      const real_t *spinor = spinorNeighbor_5d<QUDA_4D_PC>(i, dir, parity, in);
      real_t projectedSpinor[spinor_size];
      int projIdx = 2 * (dir / 2) + (dir + dagger) % 2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      // J  Need a conditional here for s=0 and s=Ls-1.
      int X = fullLatticeIndex_5d_4dpc(i, parity);
      int xs = X / (Z[3] * Z[2] * Z[1] * Z[0]);

      if ((xs == 0 && dir == 9) || (xs == Ls - 1 && dir == 8)) {
        ax(projectedSpinor, -mferm, projectedSpinor, spinor_size);
      }
      sum(&out[i * spinor_size], &out[i * spinor_size], projectedSpinor, spinor_size);
    }
    // 1 + kappa*D5
    axpby((real_t)1., &in[i * spinor_size], kappa, &out[i * spinor_size], spinor_size);
  }

  // Initialize
  std::vector<real_t> shift_coeffs(Ls);

  // Construct Mooee_shift
  real_t N = (eofa_pm ? 1.0 : -1.0) * (2.0 * eofa_shift * eofa_norm)
    * (std::pow(alpha + 1.0, Ls) + mq1 * std::pow(alpha - 1.0, Ls));

  // For the kappa preconditioning
  int idx = 0;
  N *= 1. / (b * (m5 + 4.) + 1.);
  for (int s = 0; s < Ls; s++) {
    idx = eofa_pm ? (s) : (Ls - 1 - s);
    shift_coeffs[idx] = N * std::pow(-1.0, s) * std::pow(alpha - 1.0, s) / std::pow(alpha + 1.0, Ls + s + 1);
  }

  // The eofa part.
#pragma omp parallel for
  for (int idx_cb_4d = 0; idx_cb_4d < Vh; idx_cb_4d++) {
    for (int s = 0; s < Ls; s++) {
      if (dagger == 0) {
        if (eofa_pm) {
          axpby_ssp_project<true>(out, (real_t)1., out, shift_coeffs[s], in, idx_cb_4d, s, Ls - 1);
        } else {
          axpby_ssp_project<false>(out, (real_t)1., out, shift_coeffs[s], in, idx_cb_4d, s, 0);
        }
      } else {
        if (eofa_pm) {
          axpby_ssp_project<true>(out, (real_t)1., out, shift_coeffs[s], in, idx_cb_4d, Ls - 1, s);
        } else {
          axpby_ssp_project<false>(out, (real_t)1., out, shift_coeffs[s], in, idx_cb_4d, 0, s);
        }
      }
    }
  }
}

void mdw_eofa_m5(void *out, const void *in, int parity, int dagger, double mferm, double m5, double b, double c,
                 double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    mdw_eofa_m5_ref<double>((double *)out, (double *)in, parity, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm,
                            eofa_shift);
  } else {
    mdw_eofa_m5_ref<float>((float *)out, (float *)in, parity, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm,
                           eofa_shift);
  }
  return;
}

/**
 * @brief Apply the Ls dimension portion (m5) of the domain wall dslash in a 4-d data layout
 *
 * @tparam type Domain wall preconditioning type (4 or 5 dimensions)
 * @tparam zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 * @tparam real_t The floating-point type used for the vectors
 * @param[in,out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_ee, 1 for D_oo
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] mferm Domain wall fermion mass
 */
template <QudaPCType type, bool zero_initialize = false, typename real_t>
void dslashReference_5th(real_t *out, const real_t *in, int parity, int dagger, real_t mferm)
{
#pragma omp parallel for
  for (int i = 0; i < V5h; i++) {
    if (zero_initialize)
      for (int one_site = 0; one_site < 24; one_site++) out[i * (4 * 3 * 2) + one_site] = 0.0;
    for (int dir = 8; dir < 10; dir++) {
      // Calls for an extension of the original function.
      // 8 is forward hop, which wants P_+, 9 is backward hop,
      // which wants P_-.  Dagger reverses these.
      const real_t *spinor = spinorNeighbor_5d<type>(i, dir, parity, in);
      real_t projectedSpinor[4 * 3 * 2];
      int projIdx = 2 * (dir / 2) + (dir + dagger) % 2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      // J  Need a conditional here for s=0 and s=Ls-1.
      int X = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, parity) : fullLatticeIndex_5d_4dpc(i, parity);
      int xs = X / (Z[3] * Z[2] * Z[1] * Z[0]);

      if ((xs == 0 && dir == 9) || (xs == Ls - 1 && dir == 8)) {
        ax(projectedSpinor, (real_t)(-mferm), projectedSpinor, 4 * 3 * 2);
      }
      sum(&out[i * (4 * 3 * 2)], &out[i * (4 * 3 * 2)], projectedSpinor, 4 * 3 * 2);
    }
  }
}

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the domain wall dslash in a 4-d data layout
 *
 * @tparam real_t The floating-point type used for the vectors
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe (unused)
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] mferm Domain wall fermion mass
 * @param[in] kappa Kappa values for each 5th dimension slice
 */
template <typename real_t>
void dslashReference_5th_inv(real_t *out, const real_t *in, int, int dagger, real_t mferm, const double *kappa)
{
  double *inv_Ftr = (double *)safe_malloc(Ls * sizeof(real_t));
  double *Ftr = (double *)safe_malloc(Ls * sizeof(real_t));
  for (int xs = 0; xs < Ls; xs++) {
    inv_Ftr[xs] = 1.0 / (1.0 + pow(2.0 * kappa[xs], Ls) * mferm);
    Ftr[xs] = -2.0 * kappa[xs] * mferm * inv_Ftr[xs];
    for (int i = 0; i < Vh; i++) { memcpy(&out[24 * (i + Vh * xs)], &in[24 * (i + Vh * xs)], 24 * sizeof(real_t)); }
  }
  if (dagger == 0) {
    // s = 0
    for (int i = 0; i < Vh; i++) {
      ax(&out[12 + 24 * (i + Vh * (Ls - 1))], (real_t)(inv_Ftr[0]), &in[12 + 24 * (i + Vh * (Ls - 1))], 12);
    }

    // s = 1 ... ls-2
    for (int xs = 0; xs <= Ls - 2; ++xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((real_t)(2.0 * kappa[xs]), &out[24 * (i + Vh * xs)], &out[24 * (i + Vh * (xs + 1))], 12);
        axpy((real_t)Ftr[xs], &out[12 + 24 * (i + Vh * xs)], &out[12 + 24 * (i + Vh * (Ls - 1))], 12);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] *= 2.0 * kappa[tmp_s];
    }
    for (int xs = 0; xs < Ls; xs++) { Ftr[xs] = -pow(2.0 * kappa[xs], Ls - 1) * mferm * inv_Ftr[xs]; }
    // s = ls-2 ... 0
    for (int xs = Ls - 2; xs >= 0; --xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((real_t)Ftr[xs], &out[24 * (i + Vh * (Ls - 1))], &out[24 * (i + Vh * xs)], 12);
        axpy((real_t)(2.0 * kappa[xs]), &out[12 + 24 * (i + Vh * (xs + 1))], &out[12 + 24 * (i + Vh * xs)], 12);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] /= 2.0 * kappa[tmp_s];
    }
    // s = ls -1
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax(&out[24 * (i + Vh * (Ls - 1))], (real_t)(inv_Ftr[Ls - 1]), &out[24 * (i + Vh * (Ls - 1))], 12);
    }
  } else {
    // s = 0
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax(&out[24 * (i + Vh * (Ls - 1))], (real_t)(inv_Ftr[0]), &in[24 * (i + Vh * (Ls - 1))], 12);
    }

    // s = 1 ... ls-2
    for (int xs = 0; xs <= Ls - 2; ++xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((real_t)Ftr[xs], &out[24 * (i + Vh * xs)], &out[24 * (i + Vh * (Ls - 1))], 12);
        axpy((real_t)(2.0 * kappa[xs]), &out[12 + 24 * (i + Vh * xs)], &out[12 + 24 * (i + Vh * (xs + 1))], 12);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] *= 2.0 * kappa[tmp_s];
    }
    for (int xs = 0; xs < Ls; xs++) { Ftr[xs] = -pow(2.0 * kappa[xs], Ls - 1) * mferm * inv_Ftr[xs]; }
    // s = ls-2 ... 0
    for (int xs = Ls - 2; xs >= 0; --xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((real_t)(2.0 * kappa[xs]), &out[24 * (i + Vh * (xs + 1))], &out[24 * (i + Vh * xs)], 12);
        axpy((real_t)Ftr[xs], &out[12 + 24 * (i + Vh * (Ls - 1))], &out[12 + 24 * (i + Vh * xs)], 12);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] /= 2.0 * kappa[tmp_s];
    }
    // s = ls -1
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax(&out[12 + 24 * (i + Vh * (Ls - 1))], (real_t)(inv_Ftr[Ls - 1]), &out[12 + 24 * (i + Vh * (Ls - 1))], 12);
    }
  }
  host_free(inv_Ftr);
  host_free(Ftr);
}

/**
 * @brief Compute the power of a complex number with an integer exponent.
 *
 * This function calculates the power of a complex number `x` raised to an integer exponent `y`.
 * It uses the C++ standard library's `std::pow` function to perform the calculation,
 * but ensures that the input and output types are compatible with the C complex type `sComplex`.
 *
 * @tparam sComplex The type of the C99 complex number.
 * @param[in] x The complex number to raise to a power.
 * @param[in] y The integer exponent to raise `x` to.
 * @return The complex number `x` raised to the power `y`.
 */
template <typename sComplex> sComplex cpow(const sComplex &x, int y)
{
  static_assert(sizeof(sComplex) == sizeof(Complex), "C and C++ complex type sizes do not match");
  // note that C++ standard explicitly calls out that casting between C and C++ complex is legal
  const Complex x_ = reinterpret_cast<const Complex &>(x);
  Complex z_ = std::pow(x_, y);
  sComplex z = reinterpret_cast<sComplex &>(z_);
  return z;
}

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the Mobius dslash in a 4-d data layout
 *
 * @tparam real_t The floating-point type used for the vectors
 * @tparam sComplex The C99 complex floating point type used for the vectors
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe (unused)
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] mferm Domain wall fermion mass
 * @param[in] kappa Kappa values for each 5th dimension slice
 */
template <typename real_t, typename sComplex>
void mdslashReference_5th_inv(real_t *out, const real_t *in, int, int dagger, real_t mferm, const sComplex *kappa)
{
  sComplex *inv_Ftr = (sComplex *)safe_malloc(Ls * sizeof(sComplex));
  sComplex *Ftr = (sComplex *)safe_malloc(Ls * sizeof(sComplex));
  for (int xs = 0; xs < Ls; xs++) {
    inv_Ftr[xs] = 1.0 / (1.0 + cpow(2.0 * kappa[xs], Ls) * mferm);
    Ftr[xs] = -2.0 * kappa[xs] * mferm * inv_Ftr[xs];
    for (int i = 0; i < Vh; i++) { memcpy(&out[24 * (i + Vh * xs)], &in[24 * (i + Vh * xs)], 24 * sizeof(real_t)); }
  }
  if (dagger == 0) {
    // s = 0
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax((sComplex *)&out[12 + 24 * (i + Vh * (Ls - 1))], inv_Ftr[0], (sComplex *)&in[12 + 24 * (i + Vh * (Ls - 1))], 6);
    }

    // s = 1 ... ls-2
    for (int xs = 0; xs <= Ls - 2; ++xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((2.0 * kappa[xs]), (sComplex *)&out[24 * (i + Vh * xs)], (sComplex *)&out[24 * (i + Vh * (xs + 1))], 6);
        axpy(Ftr[xs], (sComplex *)&out[12 + 24 * (i + Vh * xs)], (sComplex *)&out[12 + 24 * (i + Vh * (Ls - 1))], 6);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] *= 2.0 * kappa[tmp_s];
    }
    for (int xs = 0; xs < Ls; xs++) Ftr[xs] = -cpow(2.0 * kappa[xs], Ls - 1) * mferm * inv_Ftr[xs];

    // s = ls-2 ... 0
    for (int xs = Ls - 2; xs >= 0; --xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy(Ftr[xs], (sComplex *)&out[24 * (i + Vh * (Ls - 1))], (sComplex *)&out[24 * (i + Vh * xs)], 6);
        axpy((2.0 * kappa[xs]), (sComplex *)&out[12 + 24 * (i + Vh * (xs + 1))],
             (sComplex *)&out[12 + 24 * (i + Vh * xs)], 6);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] /= 2.0 * kappa[tmp_s];
    }
    // s = ls -1
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax((sComplex *)&out[24 * (i + Vh * (Ls - 1))], inv_Ftr[Ls - 1], (sComplex *)&out[24 * (i + Vh * (Ls - 1))], 6);
    }
  } else {
    // s = 0
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax((sComplex *)&out[24 * (i + Vh * (Ls - 1))], inv_Ftr[0], (sComplex *)&in[24 * (i + Vh * (Ls - 1))], 6);
    }

    // s = 1 ... ls-2
    for (int xs = 0; xs <= Ls - 2; ++xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy(Ftr[xs], (sComplex *)&out[24 * (i + Vh * xs)], (sComplex *)&out[24 * (i + Vh * (Ls - 1))], 6);
        axpy((2.0 * kappa[xs]), (sComplex *)&out[12 + 24 * (i + Vh * xs)],
             (sComplex *)&out[12 + 24 * (i + Vh * (xs + 1))], 6);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] *= 2.0 * kappa[tmp_s];
    }
    for (int xs = 0; xs < Ls; xs++) Ftr[xs] = -cpow(2.0 * kappa[xs], Ls - 1) * mferm * inv_Ftr[xs];

    // s = ls-2 ... 0
    for (int xs = Ls - 2; xs >= 0; --xs) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        axpy((2.0 * kappa[xs]), (sComplex *)&out[24 * (i + Vh * (xs + 1))], (sComplex *)&out[24 * (i + Vh * xs)], 6);
        axpy(Ftr[xs], (sComplex *)&out[12 + 24 * (i + Vh * (Ls - 1))], (sComplex *)&out[12 + 24 * (i + Vh * xs)], 6);
      }
      for (int tmp_s = 0; tmp_s < Ls; tmp_s++) Ftr[tmp_s] /= 2.0 * kappa[tmp_s];
    }
    // s = ls -1
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      ax((sComplex *)&out[12 + 24 * (i + Vh * (Ls - 1))], inv_Ftr[Ls - 1],
         (sComplex *)&out[12 + 24 * (i + Vh * (Ls - 1))], 6);
    }
  }
  host_free(inv_Ftr);
  host_free(Ftr);
}

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the EOFA Mobius dslash
 *
 * @tparam real_t The floating-point type used for the vectors
 * @param[out] out Host output rhs
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] mferm Domain wall fermion mass
 * @param[in] m5 Domain wall bulk fermion mass
 * @param[in] b "b" value for Mobius
 * @param[in] c "c" value for Mobius
 * @param[in] mq1 EOFA parameter mq1
 * @param[in] mq2 EOFA parameter mq2
 * @param[in] mq3 EOFA parameter mq3
 * @param[in] eofa_pm EOFA parameter eofa_pm
 * @param[in] eofa_shift EOFA parameter eofa_shift
 */
template <typename real_t>
void mdw_eofa_m5inv_ref(real_t *out, const real_t *in, int parity, int dagger, real_t mferm, real_t m5, real_t b,
                        real_t c, real_t mq1, real_t mq2, real_t mq3, int eofa_pm, real_t eofa_shift)
{
  real_t alpha = b + c;
  real_t eofa_norm = alpha * (mq3 - mq2) * std::pow(alpha + 1., 2 * Ls)
    / (std::pow(alpha + 1., Ls) + mq2 * std::pow(alpha - 1., Ls))
    / (std::pow(alpha + 1., Ls) + mq3 * std::pow(alpha - 1., Ls));
  real_t kappa5 = (c * (4. + m5) - 1.) / (b * (4. + m5) + 1.); // alpha = b+c

  using sComplex = double _Complex;

  std::vector<sComplex> kappa_array(Ls, -0.5 * kappa5);
  std::vector<real_t> eofa_u(Ls);
  std::vector<real_t> eofa_x(Ls);
  std::vector<real_t> eofa_y(Ls);

  mdslashReference_5th_inv(out, in, parity, dagger, mferm, kappa_array.data());

  real_t N = (eofa_pm ? +1. : -1.) * (2. * eofa_shift * eofa_norm)
    * (std::pow(alpha + 1., Ls) + mq1 * std::pow(alpha - 1., Ls)) / (b * (m5 + 4.) + 1.);

  // Here the signs are somewhat mixed:
  // There is one -1 from N for eofa_pm = minus, thus the u_- here is actually -u_- in the document
  // It turns out this actually simplies things.
  for (int s = 0; s < Ls; s++) {
    eofa_u[eofa_pm ? s : Ls - 1 - s] = N * std::pow(-1., s) * std::pow(alpha - 1., s) / std::pow(alpha + 1., Ls + s + 1);
  }

  real_t sherman_morrison_fac;

  real_t factor = -kappa5 * mferm;
  if (eofa_pm) {
    // eofa_pm = plus
    // Computing x
    eofa_x[0] = eofa_u[0];
    for (int s = Ls - 1; s > 0; s--) {
      eofa_x[0] -= factor * eofa_u[s];
      factor *= -kappa5;
    }
    eofa_x[0] /= 1. + factor;
    for (int s = 1; s < Ls; s++) { eofa_x[s] = eofa_x[s - 1] * (-kappa5) + eofa_u[s]; }
    // Computing y
    eofa_y[Ls - 1] = 1. / (1. + factor);
    sherman_morrison_fac = eofa_x[Ls - 1];
    for (int s = Ls - 1; s > 0; s--) { eofa_y[s - 1] = eofa_y[s] * (-kappa5); }
  } else {
    // eofa_pm = minus
    // Computing x
    eofa_x[Ls - 1] = eofa_u[Ls - 1];
    for (int s = 0; s < Ls - 1; s++) {
      eofa_x[Ls - 1] -= factor * eofa_u[s];
      factor *= -kappa5;
    }
    eofa_x[Ls - 1] /= 1. + factor;
    for (int s = Ls - 1; s > 0; s--) { eofa_x[s - 1] = eofa_x[s] * (-kappa5) + eofa_u[s - 1]; }
    // Computing y
    eofa_y[0] = 1. / (1. + factor);
    sherman_morrison_fac = eofa_x[0];
    for (int s = 1; s < Ls; s++) { eofa_y[s] = eofa_y[s - 1] * (-kappa5); }
  }
  sherman_morrison_fac = -0.5 / (1. + sherman_morrison_fac); // 0.5 for the spin project factor

  // The EOFA stuff
#pragma omp parallel for
  for (int idx_cb_4d = 0; idx_cb_4d < Vh; idx_cb_4d++) {
    for (int s = 0; s < Ls; s++) {
      for (int sp = 0; sp < Ls; sp++) {
        real_t t = 2.0 * sherman_morrison_fac;
        if (dagger == 0) {
          t *= eofa_x[s] * eofa_y[sp];
          if (eofa_pm) {
            axpby_ssp_project<true>(out, (real_t)1., out, t, in, idx_cb_4d, s, sp);
          } else {
            axpby_ssp_project<false>(out, (real_t)1., out, t, in, idx_cb_4d, s, sp);
          }
        } else {
          t *= eofa_y[s] * eofa_x[sp];
          if (eofa_pm) {
            axpby_ssp_project<true>(out, (real_t)1., out, t, in, idx_cb_4d, s, sp);
          } else {
            axpby_ssp_project<false>(out, (real_t)1., out, t, in, idx_cb_4d, s, sp);
          }
        }
      }
    }
  }
}

void mdw_eofa_m5inv(void *out, const void *in, int parity, int dagger, double mferm, double m5, double b, double c,
                    double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    mdw_eofa_m5inv_ref<double>((double *)out, (const double *)in, parity, dagger, mferm, m5, b, c, mq1, mq2, mq3,
                               eofa_pm, eofa_shift);
  } else {
    mdw_eofa_m5inv_ref<float>((float *)out, (const float *)in, parity, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm,
                              eofa_shift);
  }
}

// this actually applies the preconditioned dslash, e.g., D_ee * \psi_e + D_eo * \psi_o or D_oo * \psi_o + D_oe * \psi_e
void dw_dslash(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param, double mferm)
{
  GaugeFieldParam gauge_field_param(gauge_param, (void **)gauge);
  gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  GaugeField cpu(gauge_field_param);
  void *ghostGauge[4] = {cpu.Ghost()[0].data(), cpu.Ghost()[1].data(), cpu.Ghost()[2].data(), cpu.Ghost()[3].data()};

  // Get spinor ghost fields
  // First wrap the input spinor into a ColorSpinorField
  ColorSpinorParam csParam;
  csParam.v = (void *)in;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 5; // for DW dslash
  for (int d = 0; d < 4; d++) csParam.x[d] = Z[d];
  csParam.x[4] = Ls; // 5th dimention
  csParam.setPrecision(precision);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.x[0] /= 2;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.pc_type = QUDA_5D_PC;
  csParam.location = QUDA_CPU_FIELD_LOCATION;

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
    dslashReference_4d<QUDA_5D_PC>((double *)out, (double **)gauge, (double **)ghostGauge, (double *)in,
                                   (double **)fwd_nbr_spinor, (double **)back_nbr_spinor, parity, dagger);
    dslashReference_5th<QUDA_5D_PC>((double *)out, (double *)in, parity, dagger, mferm);
  } else {
    dslashReference_4d<QUDA_5D_PC>((float *)out, (float **)gauge, (float **)ghostGauge, (float *)in,
                                   (float **)fwd_nbr_spinor, (float **)back_nbr_spinor, parity, dagger);
    dslashReference_5th<QUDA_5D_PC>((float *)out, (float *)in, parity, dagger, (float)mferm);
  }
}

void dslash_4_4d(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                 const QudaGaugeParam &gauge_param, double)
{
  GaugeFieldParam gauge_field_param(gauge_param, (void **)gauge);
  gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  GaugeField cpu(gauge_field_param);
  void *ghostGauge[4] = {cpu.Ghost()[0].data(), cpu.Ghost()[1].data(), cpu.Ghost()[2].data(), cpu.Ghost()[3].data()};

  // Get spinor ghost fields
  // First wrap the input spinor into a ColorSpinorField
  ColorSpinorParam csParam;
  csParam.v = (void *)in;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 5; // for DW dslash
  for (int d = 0; d < 4; d++) csParam.x[d] = Z[d];
  csParam.x[4] = Ls; // 5th dimention
  csParam.setPrecision(precision);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.x[0] /= 2;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.pc_type = QUDA_4D_PC;
  csParam.location = QUDA_CPU_FIELD_LOCATION;

  ColorSpinorField inField(csParam);

  { // Now do the exchange
    QudaParity otherParity = QUDA_INVALID_PARITY;
    if (parity == QUDA_EVEN_PARITY)
      otherParity = QUDA_ODD_PARITY;
    else if (parity == QUDA_ODD_PARITY)
      otherParity = QUDA_EVEN_PARITY;
    else
      errorQuda("ERROR: full parity not supported");
    const int nFace = 1;

    inField.exchangeGhost(otherParity, nFace, dagger);
  }
  void **fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
  void **back_nbr_spinor = inField.backGhostFaceBuffer;
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_4d<QUDA_4D_PC>((double *)out, (double **)gauge, (double **)ghostGauge, (double *)in,
                                   (double **)fwd_nbr_spinor, (double **)back_nbr_spinor, parity, dagger);
  } else {
    dslashReference_4d<QUDA_4D_PC>((float *)out, (float **)gauge, (float **)ghostGauge, (float *)in,
                                   (float **)fwd_nbr_spinor, (float **)back_nbr_spinor, parity, dagger);
  }
}

void dw_dslash_5_4d(void *out, const void *const *, const void *in, int parity, int dagger, QudaPrecision precision,
                    const QudaGaugeParam &, double mferm, bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((double *)out, (double *)in, parity, dagger, mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((double *)out, (double *)in, parity, dagger, mferm);
  } else {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((float *)out, (float *)in, parity, dagger, (float)mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((float *)out, (float *)in, parity, dagger, (float)mferm);
  }
}

void dslash_5_inv(void *out, const void *const *, const void *in, int parity, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &, double mferm, double *kappa)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_5th_inv((double *)out, (double *)in, parity, dagger, mferm, kappa);
  } else {
    dslashReference_5th_inv((float *)out, (float *)in, parity, dagger, (float)mferm, kappa);
  }
}

void mdw_dslash_5_inv(void *out, const void *const *, const void *in, int parity, int dagger, QudaPrecision precision,
                      const QudaGaugeParam &, double mferm, const double _Complex *kappa)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    mdslashReference_5th_inv((double *)out, (double *)in, parity, dagger, mferm, kappa);
  } else {
    mdslashReference_5th_inv((float *)out, (float *)in, parity, dagger, (float)mferm, kappa);
  }
}

void mdw_dslash_5(void *out, const void *const *, const void *in, int parity, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &, double mferm, const double _Complex *kappa, bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((double *)out, (double *)in, parity, dagger, mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((double *)out, (double *)in, parity, dagger, mferm);
  } else {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((float *)out, (float *)in, parity, dagger, (float)mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((float *)out, (float *)in, parity, dagger, (float)mferm);
  }
  for (int xs = 0; xs < Ls; xs++) {
    cxpay((char *)in + precision * Vh * spinor_site_size * xs, kappa[xs],
          (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
  }
}

void mdw_dslash_4_pre(void *out, const void *const *, const void *in, int parity, int dagger, QudaPrecision precision,
                      const QudaGaugeParam &, double mferm, const double _Complex *b5, const double _Complex *c5,
                      bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((double *)out, (double *)in, parity, dagger, mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((double *)out, (double *)in, parity, dagger, mferm);
    for (int xs = 0; xs < Ls; xs++) {
      axpby(b5[xs], (double _Complex *)in + Vh * spinor_site_size / 2 * xs, 0.5 * c5[xs],
            (double _Complex *)out + Vh * spinor_site_size / 2 * xs, Vh * spinor_site_size / 2);
    }
  } else {
    if (zero_initialize)
      dslashReference_5th<QUDA_4D_PC, true>((float *)out, (float *)in, parity, dagger, (float)mferm);
    else
      dslashReference_5th<QUDA_4D_PC, false>((float *)out, (float *)in, parity, dagger, (float)mferm);
    for (int xs = 0; xs < Ls; xs++) {
      axpby((float _Complex)(b5[xs]), (float _Complex *)in + Vh * (spinor_site_size / 2) * xs,
            (float _Complex)(0.5 * c5[xs]), (float _Complex *)out + Vh * (spinor_site_size / 2) * xs,
            Vh * spinor_site_size / 2);
    }
  }
}

void dw_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger_bit, QudaPrecision precision,
            const QudaGaugeParam &gauge_param, double mferm)
{

  const void *inEven = in;
  const void *inOdd = (const char *)in + V5h * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + V5h * spinor_site_size * precision;

  dw_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param, mferm);
  dw_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param, mferm);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V5 * spinor_site_size, precision);
}

void dw_4d_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger_bit,
               QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm)
{

  const void *inEven = in;
  const void *inOdd = (char *)in + V5h * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + V5h * spinor_site_size * precision;

  dslash_4_4d(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param, mferm);
  dw_dslash_5_4d(outOdd, gauge, inOdd, 1, dagger_bit, precision, gauge_param, mferm, false);

  dslash_4_4d(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param, mferm);
  dw_dslash_5_4d(outEven, gauge, inEven, 0, dagger_bit, precision, gauge_param, mferm, false);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V5 * spinor_site_size, precision);
}

void mdw_mat(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
             const double _Complex *kappa_c, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param,
             double mferm, const double _Complex *b5, const double _Complex *c5)
{
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);
  double _Complex *kappa5 = (double _Complex *)safe_malloc(Ls * sizeof(double _Complex));

  for (int xs = 0; xs < Ls; xs++) kappa5[xs] = 0.5 * kappa_b[xs] / kappa_c[xs];

  const void *inEven = in;
  const void *inOdd = (const char *)in + V5h * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + V5h * spinor_site_size * precision;

  if (!dagger) {
    mdw_dslash_4_pre(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(outOdd, gauge, tmp, 1, dagger, precision, gauge_param, mferm);
    mdw_dslash_5(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, kappa5, true);
  } else {
    dslash_4_4d(tmp, gauge, inEven, 1, dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(outOdd, gauge, tmp, 0, dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, kappa5, true);
  }

  for (int xs = 0; xs < Ls; xs++) {
    cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, -kappa_b[xs],
          (char *)outOdd + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
  }

  if (!dagger) {
    mdw_dslash_4_pre(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(outEven, gauge, tmp, 0, dagger, precision, gauge_param, mferm);
    mdw_dslash_5(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, kappa5, true);
  } else {
    dslash_4_4d(tmp, gauge, inOdd, 0, dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(outEven, gauge, tmp, 1, dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, kappa5, true);
  }

  for (int xs = 0; xs < Ls; xs++) {
    cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, -kappa_b[xs],
          (char *)outEven + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
  }

  host_free(kappa5);
  host_free(tmp);
}

void mdw_eofa_mat(void *out, const void *const *gauge, const void *in, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param, double mferm, double m5, double b, double c, double mq1,
                  double mq2, double mq3, int eofa_pm, double eofa_shift)
{
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);

  std::vector<double _Complex> b_array(Ls, b);
  std::vector<double _Complex> c_array(Ls, c);

  auto b5 = b_array.data();
  auto c5 = c_array.data();

  auto kappa_b = 0.5 / (b * (4. + m5) + 1.);

  const void *inEven = in;
  const void *inOdd = (char *)in + V5h * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + V5h * spinor_site_size * precision;

  if (!dagger) {
    mdw_dslash_4_pre(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(outOdd, gauge, tmp, 1, dagger, precision, gauge_param, mferm);
    mdw_eofa_m5(tmp, inOdd, 1, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
  } else {
    dslash_4_4d(tmp, gauge, inEven, 1, dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(outOdd, gauge, tmp, 0, dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_eofa_m5(tmp, inOdd, 1, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
  }

  for (int xs = 0; xs < Ls; xs++) {
    cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, -kappa_b,
          (char *)outOdd + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
  }

  if (!dagger) {
    mdw_dslash_4_pre(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(outEven, gauge, tmp, 0, dagger, precision, gauge_param, mferm);
    mdw_eofa_m5(tmp, inEven, 0, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
  } else {
    dslash_4_4d(tmp, gauge, inOdd, 0, dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(outEven, gauge, tmp, 1, dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_eofa_m5(tmp, inEven, 0, dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
  }

  for (int xs = 0; xs < Ls; xs++) {
    cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, -kappa_b,
          (char *)outEven + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
  }

  host_free(tmp);
}
//
void dw_matdagmat(void *out, const void *const *gauge, const void *in, double kappa, int dagger_bit,
                  QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm)
{
  void *tmp = safe_malloc(V5 * spinor_site_size * precision);

  dw_mat(tmp, gauge, in, kappa, dagger_bit, precision, gauge_param, mferm);
  dagger_bit = (dagger_bit == 1) ? 0 : 1;
  dw_mat(out, gauge, tmp, kappa, dagger_bit, precision, gauge_param, mferm);

  host_free(tmp);
}

void dw_matpc(void *out, const void *const *gauge, const void *in, double kappa, QudaMatPCType matpc_type,
              int dagger_bit, QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm)
{
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    dw_dslash(tmp, gauge, in, 1, dagger_bit, precision, gauge_param, mferm);
    dw_dslash(out, gauge, tmp, 0, dagger_bit, precision, gauge_param, mferm);
  } else {
    dw_dslash(tmp, gauge, in, 0, dagger_bit, precision, gauge_param, mferm);
    dw_dslash(out, gauge, tmp, 1, dagger_bit, precision, gauge_param, mferm);
  }

  // lastly apply the kappa term
  double kappa2 = -kappa * kappa;
  xpay(in, kappa2, out, V5h * spinor_site_size, precision);

  host_free(tmp);
}

void dw_4d_matpc(void *out, const void *const *gauge, const void *in, double kappa, QudaMatPCType matpc_type,
                 int dagger_bit, QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm)
{
  double kappa2 = -kappa * kappa;
  double *kappa5 = (double *)safe_malloc(Ls * sizeof(double));
  for (int xs = 0; xs < Ls; xs++) kappa5[xs] = kappa;
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);
  //------------------------------------------
  double *output = (double *)out;
  for (auto k = 0lu; k < V5h * spinor_site_size; k++) output[k] = 0.0;
  //------------------------------------------

  int odd_bit = (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
  bool symmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) ? true : false;
  QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

  if (symmetric && !dagger_bit) {
    dslash_4_4d(tmp, gauge, in, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[1], dagger_bit, precision, gauge_param, mferm, kappa5);
    xpay(in, kappa2, out, V5h * spinor_site_size, precision);
  } else if (symmetric && dagger_bit) {
    dslash_5_inv(tmp, gauge, in, parity[1], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(tmp, gauge, out, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(out, gauge, tmp, parity[1], dagger_bit, precision, gauge_param, mferm);
    xpay(in, kappa2, out, V5h * spinor_site_size, precision);
  } else {
    dslash_4_4d(tmp, gauge, in, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger_bit, precision, gauge_param, mferm);
    xpay(in, kappa2, tmp, V5h * spinor_site_size, precision);
    dw_dslash_5_4d(out, gauge, in, parity[1], dagger_bit, precision, gauge_param, mferm, true);
    xpay(tmp, -kappa, out, V5h * spinor_site_size, precision);
  }
  host_free(tmp);
  host_free(kappa5);
}

void mdw_matpc(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
               const double _Complex *kappa_c, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param, double mferm, const double _Complex *b5, const double _Complex *c5)
{
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);
  double _Complex *kappa5 = (double _Complex *)safe_malloc(Ls * sizeof(double _Complex));
  double _Complex *kappa2 = (double _Complex *)safe_malloc(Ls * sizeof(double _Complex));
  double _Complex *kappa_mdwf = (double _Complex *)safe_malloc(Ls * sizeof(double _Complex));
  for (int xs = 0; xs < Ls; xs++) {
    kappa5[xs] = 0.5 * kappa_b[xs] / kappa_c[xs];
    kappa2[xs] = -kappa_b[xs] * kappa_b[xs];
    kappa_mdwf[xs] = -kappa5[xs];
  }

  int odd_bit = (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
  bool symmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) ? true : false;
  QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

  if (symmetric && !dagger) {
    mdw_dslash_4_pre(tmp, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_5_inv(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)in + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (symmetric && dagger) {
    mdw_dslash_5_inv(tmp, gauge, in, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)in + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (!symmetric && !dagger) {
    mdw_dslash_4_pre(out, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_5_inv(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_5(tmp, gauge, in, parity[0], dagger, precision, gauge_param, mferm, kappa5, true);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (!symmetric && dagger) {
    dslash_4_4d(out, gauge, in, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5(tmp, gauge, in, parity[0], dagger, precision, gauge_param, mferm, kappa5, true);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else {
    errorQuda("Unsupported matpc_type=%d dagger=%d", matpc_type, dagger);
  }

  host_free(tmp);
  host_free(kappa5);
  host_free(kappa2);
  host_free(kappa_mdwf);
}

void mdw_eofa_matpc(void *out, const void *const *gauge, const void *in, QudaMatPCType matpc_type, int dagger,
                    QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm, double m5, double b,
                    double c, double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift)
{
  void *tmp = safe_malloc(V5h * spinor_site_size * precision);

  std::vector<double _Complex> kappa2_array(Ls, -0.25 / (b * (4. + m5) + 1.) / (b * (4. + m5) + 1.));
  std::vector<double _Complex> b_array(Ls, b);
  std::vector<double _Complex> c_array(Ls, c);

  auto kappa2 = kappa2_array.data();
  auto b5 = b_array.data();
  auto c5 = c_array.data();

  int odd_bit = (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
  bool symmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) ? true : false;
  QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

  if (symmetric && !dagger) {
    mdw_dslash_4_pre(tmp, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    mdw_eofa_m5inv(tmp, out, parity[1], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_eofa_m5inv(out, tmp, parity[0], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)in + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (symmetric && dagger) {
    mdw_eofa_m5inv(tmp, in, parity[1], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_eofa_m5inv(out, tmp, parity[0], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)in + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (!symmetric && !dagger) {
    mdw_dslash_4_pre(out, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm);
    mdw_eofa_m5inv(out, tmp, parity[1], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm);
    mdw_eofa_m5(tmp, in, parity[0], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else if (!symmetric && dagger) {
    dslash_4_4d(out, gauge, in, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_eofa_m5inv(out, tmp, parity[0], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_eofa_m5(tmp, in, parity[0], dagger, mferm, m5, b, c, mq1, mq2, mq3, eofa_pm, eofa_shift, precision);
    for (int xs = 0; xs < Ls; xs++) {
      cxpay((char *)tmp + precision * Vh * spinor_site_size * xs, kappa2[xs],
            (char *)out + precision * Vh * spinor_site_size * xs, Vh * spinor_site_size, precision);
    }
  } else {
    errorQuda("Unsupported matpc_type=%d dagger=%d", matpc_type, dagger);
  }

  host_free(tmp);
}

void mdw_mdagm_local(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
                     const double _Complex *kappa_c, QudaMatPCType matpc_type, QudaPrecision precision,
                     const QudaGaugeParam &gauge_param, double mferm, const double _Complex *b5,
                     const double _Complex *c5)
{
  lat_dim_t R;
  for (int d = 0; d < 4; d++) { R[d] = comm_dim_partitioned(d) ? 2 : 0; }

  GaugeField *padded_gauge = createExtendedGauge((void **)gauge, (QudaGaugeParam &)gauge_param, R);

  int padded_V = 1;
  int W[4];
  for (int d = 0; d < 4; d++) {
    W[d] = Z[d] + 2 * R[d];
    padded_V *= Z[d] + 2 * R[d];
  }
  int padded_V5 = padded_V * Ls;
  int padded_Vh = padded_V / 2;
  int padded_V5h = padded_Vh * Ls;

  static_assert(sizeof(char) == 1, "This code assumes sizeof(char) == 1.");

  char *padded_in = (char *)safe_malloc(padded_V5h * spinor_site_size * precision);
  memset(padded_in, 0, padded_V5h * spinor_site_size * precision);
  char *padded_out = (char *)safe_malloc(padded_V5h * spinor_site_size * precision);
  memset(padded_out, 0, padded_V5h * spinor_site_size * precision);
  char *padded_tmp = (char *)safe_malloc(padded_V5h * spinor_site_size * precision);
  memset(padded_tmp, 0, padded_V5h * spinor_site_size * precision);

  char *in_alias = (char *)in;
  char *out_alias = (char *)out;

  for (int s = 0; s < Ls; s++) {
    for (int index_cb_4d = 0; index_cb_4d < Vh; index_cb_4d++) {
      // calculate padded_index_cb_4d
      int x[4];
      coordinate_from_shrinked_index(x, index_cb_4d, Z, R.data, 0); // parity = 0
      int padded_index_cb_4d = index_4d_cb_from_coordinate_4d(x, W);
      // copy data
      memcpy(&padded_in[spinor_site_size * precision * (s * padded_Vh + padded_index_cb_4d)],
             &in_alias[spinor_site_size * precision * (s * Vh + index_cb_4d)], spinor_site_size * precision);
    }
  }

  QudaGaugeParam padded_gauge_param(gauge_param);
  for (int d = 0; d < 4; d++) { padded_gauge_param.X[d] += 2 * R[d]; }

  void *padded_gauge_p[] = {padded_gauge->data(0), padded_gauge->data(1), padded_gauge->data(2), padded_gauge->data(3)};

  // Extend these global variables then restore them
  int V5_old = V5;
  V5 = padded_V5;
  int Vh_old = Vh;
  Vh = padded_Vh;
  int V5h_old = V5h;
  V5h = padded_V5h;
  int Z_old[4];
  for (int d = 0; d < 4; d++) {
    Z_old[d] = Z[d];
    Z[d] = W[d];
  }

  // dagger = 0
  mdw_matpc(padded_tmp, padded_gauge_p, padded_in, kappa_b, kappa_c, matpc_type, 0, precision, padded_gauge_param,
            mferm, b5, c5);
  // dagger = 1
  mdw_matpc(padded_out, padded_gauge_p, padded_tmp, kappa_b, kappa_c, matpc_type, 1, precision, padded_gauge_param,
            mferm, b5, c5);

  // Restore them
  V5 = V5_old;
  Vh = Vh_old;
  V5h = V5h_old;
  for (int d = 0; d < 4; d++) { Z[d] = Z_old[d]; }

  for (int s = 0; s < Ls; s++) {
    for (int index_cb_4d = 0; index_cb_4d < Vh; index_cb_4d++) {
      // calculate padded_index_cb_4d
      int x[4];
      coordinate_from_shrinked_index(x, index_cb_4d, Z, R.data, 0); // parity = 0
      int padded_index_cb_4d = index_4d_cb_from_coordinate_4d(x, W);
      // copy data
      memcpy(&out_alias[spinor_site_size * precision * (s * Vh + index_cb_4d)],
             &padded_out[spinor_site_size * precision * (s * padded_Vh + padded_index_cb_4d)],
             spinor_site_size * precision);
    }
  }

  host_free(padded_in);
  host_free(padded_out);
  host_free(padded_tmp);

  delete padded_gauge;
}
