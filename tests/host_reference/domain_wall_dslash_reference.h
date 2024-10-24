#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

/**
 * @brief Apply the preconditioned 5-d domain wall dslash, e.g., D_ee * \psi_e + D_eo * \psi_o or D_oo * \psi_o + D_oe * \psi_e
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_ee * \psi_e + D_eo * \psi_o, 1 for D_oo * \psi_o + D_oe * \psi_e
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_dslash(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the 4-d Dslash (Wilson) to all fifth dimensional slices for a 4-d data layout
 *
 * @param out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass (unused)
 */
void dslash_4_4d(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                 const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the Ls dimension portion (m5) of the domain wall dslash in a 4-d data layout
 *
 * @param out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass (unused)
 * @param[in] zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void dw_dslash_5_4d(void *out, const void *const *gauge, const void *in, int parity, int dagger,
                    QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm, bool zero_initialize);

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the domain wall dslash in a 4-d data layout
 *
 * @param out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 * @param[in] kappa Kappa values for each 5th dimension slice
 */
void dslash_5_inv(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param, double mferm, double *kappa);

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the Mobius dslash
 *
 * @param out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 * @param[in] kappa Kappa values for each 5th dimension slice
 */
void mdw_dslash_5_inv(void *out, const void *const *gauge, const void *in, int parity, int dagger,
                      QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm,
                      const double _Complex *kappa);

/**
 * @brief Apply the Ls dimension portion (m5) of the Mobius dslash
 *
 * @param[in,out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass (unused)
 * @param[in] kappa Kappa values for each 5th dimension slice
 * @param[in] zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void mdw_dslash_5(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param, double mferm, const double _Complex *kappa, bool zero_initialize);

/**
 * @brief Apply the Ls dimension portion of D_eo/D_oe (i.e., the b + c * D5) for the Mobius dslash
 *
 * @param[in,out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass (unused)
 * @param[in] b5 Array of b5 values for each fifth dimensional slice
 * @param[in] c5 Array of c5 values for each fifth dimensional slice
 * @param[in] zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void mdw_dslash_4_pre(void *out, const void *const *gauge, const void *in, int parity, int dagger,
                      QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm,
                      const double _Complex *b5, const double _Complex *c5, bool zero_initialize);

/**
 * @brief Apply the full-parity 5-d domain wall operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the domain wall operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger, QudaPrecision precision,
            const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the full-parity 4-d data layout domain wall operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the domain wall operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_4d_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger, QudaPrecision precision,
               const QudaGaugeParam &param, double mferm);

/**
 * @brief Apply the full-parity Mobius operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa_b Kappa_b values for the Mobius operator
 * @param[in] kappa_c Kappa_c values for the Mobius operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 * @param[in] b5 Array of b5 values for each fifth dimensional slice
 * @param[in] c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_mat(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
             const double _Complex *kappa_c, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param,
             double mferm, const double _Complex *b5, const double _Complex *c5);

/**
 * @brief Apply the M^dag M for the full-parity 5-d domain wall operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the domain wall operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_matdagmat(void *out, const void *const *gauge, const void *in, double kappa, int dagger,
                  QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric 5-d preconditioned domain wall operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the domain wall operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_matpc(void *out, const void *const *gauge, const void *in, double kappa, QudaMatPCType matpc_type, int dagger,
              QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric 4-d preconditioned domain wall operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the domain wall operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 */
void dw_4d_matpc(void *out, const void *const *gauge, const void *in, double kappa, QudaMatPCType matpc_type,
                 int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Mobius operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa_b Kappa_b values for the Mobius operator
 * @param[in] kappa_c Kappa_c values for the Mobius operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 * @param[in] b5 Array of b5 values for each fifth dimensional slice
 * @param[in] c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_matpc(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
               const double _Complex *kappa_c, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param, double mferm, const double _Complex *b5, const double _Complex *c5);

/**
 * @brief Apply the local portion of the preconditioned M^dag M for the Mobius operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa_b Kappa_b values for the Mobius operator
 * @param[in] kappa_c Kappa_c values for the Mobius operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] mferm Domain wall fermion mass
 * @param[in] b5 Array of b5 values for each fifth dimensional slice
 * @param[in] c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_mdagm_local(void *out, const void *const *gauge, const void *in, const double _Complex *kappa_b,
                     const double _Complex *kappa_c, QudaMatPCType matpc_type, QudaPrecision precision,
                     const QudaGaugeParam &gauge_param, double mferm, const double _Complex *b5,
                     const double _Complex *c5);

/**
 * @brief Apply the Ls dimension portion (m5) of the eofa Mobius dslash
 *
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
void mdw_eofa_m5(void *out, const void *in, int parity, int dagger, double mferm, double m5, double b, double c,
                 double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

/**
 * @brief Apply the inverse of the Ls dimension portion (m5) of the eofa Mobius dslash
 *
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
void mdw_eofa_m5inv(void *out, const void *in, int parity, int dagger, double mferm, double m5, double b, double c,
                    double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

/**
 * @brief Apply the full eofa Mobius matrix
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
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
void mdw_eofa_mat(void *out, const void *const *gauge, const void *in, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param, double mferm, double m5, double b, double c, double mq1,
                  double mq2, double mq3, int eofa_pm, double eofa_shift);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned eofa Mobius matrix
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
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
void mdw_eofa_matpc(void *out, const void *const *gauge, const void *in, QudaMatPCType matpc_type, int dagger,
                    QudaPrecision precision, const QudaGaugeParam &gauge_param, double mferm, double m5, double b,
                    double c, double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift);
