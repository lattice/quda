#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

/**
 * @brief Apply the preconditioned 5-d domain wall dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_dslash(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
               QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the 4-d Dslash to all fifth dimensional slices for a 4-d data layout
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass (unused)
 */
void dslash_4_4d(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                 QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the Ls dimension portion of the domain wall dslash in a 4-d data layout
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass (unused)
 * @param zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void dw_dslash_5_4d(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                    QudaGaugeParam &gauge_param, double mferm, bool zero_initialize);

/**
 * @brief Apply the inverse of the Ls dimension portion of the domain wall dslash in a 4-d data layout
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param kappa Kappa values for each 5th dimension slice
 */
void dslash_5_inv(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                  QudaGaugeParam &gauge_param, double mferm, double *kappa);

/**
 * @brief Apply the inverse of the Ls dimension portion of the Mobius dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param kappa Kappa values for each 5th dimension slice
 */
void mdw_dslash_5_inv(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                      QudaGaugeParam &gauge_param, double mferm, double _Complex *kappa);

/**
 * @brief Apply the Ls dimension portion of the Mobius dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass (unused)
 * @param kappa Kappa values for each 5th dimension slice
 * @param zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void mdw_dslash_5(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                  QudaGaugeParam &gauge_param, double mferm, double _Complex *kappa, bool zero_initialize);

/**
 * @brief Pre-apply b_5 and c_5 parameters for the Mobius dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass (unused)
 * @param b5 Array of b5 values for each fifth dimensional slice
 * @param c5 Array of c5 values for each fifth dimensional slice
 * @param zero_initialize Whether or not to zero initialize or accumulate into the output rhs
 */
void mdw_dslash_4_pre(void *out, void *const *gauge, void *in, int parity, int dagger, QudaPrecision precision,
                      QudaGaugeParam &gauge_param, double mferm, double _Complex *b5, double _Complex *c5,
                      bool zero_initialize);

/**
 * @brief Apply the full-parity 5-d domain wall operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the domain wall operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_mat(void *out, void *const *gauge, void *in, double kappa, int dagger, QudaPrecision precision,
            QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the full-parity 4-d data layout domain wall operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the domain wall operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_4d_mat(void *out, void *const *gauge, void *in, double kappa, int dagger, QudaPrecision precision,
               QudaGaugeParam &param, double mferm);

/**
 * @brief Apply the full-parity Mobius operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa_b Kappa_b values for the Mobius operator
 * @param kappa_c Kappa_c values for the Mobius operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param b5 Array of b5 values for each fifth dimensional slice
 * @param c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_mat(void *out, void *const *gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c, int dagger,
             QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double _Complex *b5,
             double _Complex *c5);

/**
 * @brief Apply the M^dag M for the full-parity 5-d domain wall operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the domain wall operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_matdagmat(void *out, void *const *gauge, void *in, double kappa, int dagger, QudaPrecision precision,
                  QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric 5-d preconditioned domain wall operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the domain wall operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_matpc(void *out, void *const *gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger,
              QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric 4-d preconditioned domain wall operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the domain wall operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 */
void dw_4d_matpc(void *out, void *const *gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger,
                 QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Mobius operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa_b Kappa_b values for the Mobius operator
 * @param kappa_c Kappa_c values for the Mobius operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param b5 Array of b5 values for each fifth dimensional slice
 * @param c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_matpc(void *out, void *const *gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c,
               QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm,
               double _Complex *b5, double _Complex *c5);

/**
 * @brief Apply the local portion of the preconditioned M^dag M for the Mobius operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa_b Kappa_b values for the Mobius operator
 * @param kappa_c Kappa_c values for the Mobius operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param b5 Array of b5 values for each fifth dimensional slice
 * @param c5 Array of c5 values for each fifth dimensional slice
 */
void mdw_mdagm_local(void *out, void *const *gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c,
                     QudaMatPCType matpc_type, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm,
                     double _Complex *b5, double _Complex *c5);

/**
 * @brief Apply the Ls dimension portion of the eofa Mobius dslash
 *
 * @param out Host output rhs
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param mferm Domain wall fermion mass
 * @param m5 Domain wall bulk fermion mass
 * @param b "b" value for Mobius
 * @param c "c" value for Mobius
 * @param mq1 EOFA parameter mq1
 * @param mq2 EOFA parameter mq2
 * @param mq3 EOFA parameter mq3
 * @param eofa_pm EOFA parameter eofa_pm
 * @param eofa_shift EOFA parameter eofa_shift
 * @param precision Single or double precision
 */
void mdw_eofa_m5(void *out, void *in, int parity, int dagger, double mferm, double m5, double b, double c, double mq1,
                 double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

/**
 * @brief Apply the inverse of the Ls dimension portion of the eofa Mobius dslash
 *
 * @param out Host output rhs
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param mferm Domain wall fermion mass
 * @param m5 Domain wall bulk fermion mass
 * @param b "b" value for Mobius
 * @param c "c" value for Mobius
 * @param mq1 EOFA parameter mq1
 * @param mq2 EOFA parameter mq2
 * @param mq3 EOFA parameter mq3
 * @param eofa_pm EOFA parameter eofa_pm
 * @param eofa_shift EOFA parameter eofa_shift
 * @param precision Single or double precision
 */
void mdw_eofa_m5inv(void *out, void *in, int parity, int dagger, double mferm, double m5, double b, double c,
                    double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

/**
 * @brief Apply the full eofa Mobius matrix
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param m5 Domain wall bulk fermion mass
 * @param b "b" value for Mobius
 * @param c "c" value for Mobius
 * @param mq1 EOFA parameter mq1
 * @param mq2 EOFA parameter mq2
 * @param mq3 EOFA parameter mq3
 * @param eofa_pm EOFA parameter eofa_pm
 * @param eofa_shift EOFA parameter eofa_shift
 */
void mdw_eofa_mat(void *out, void *const *gauge, void *in, int dagger, QudaPrecision precision,
                  QudaGaugeParam &gauge_param, double mferm, double m5, double b, double c, double mq1, double mq2,
                  double mq3, int eofa_pm, double eofa_shift);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned eofa Mobius matrix
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param mferm Domain wall fermion mass
 * @param m5 Domain wall bulk fermion mass
 * @param b "b" value for Mobius
 * @param c "c" value for Mobius
 * @param mq1 EOFA parameter mq1
 * @param mq2 EOFA parameter mq2
 * @param mq3 EOFA parameter mq3
 * @param eofa_pm EOFA parameter eofa_pm
 * @param eofa_shift EOFA parameter eofa_shift
 */
void mdw_eofa_matpc(void *out, void *const *gauge, void *in, QudaMatPCType matpc_type, int dagger,
                    QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double m5, double b, double c,
                    double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift);
