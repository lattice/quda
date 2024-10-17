#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

/**
 * @brief Apply even-odd or odd-even component of the Wilson dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void wil_dslash(void *out, const void *const *gauge, const void *in, int parity, int dagger, QudaPrecision precision,
                const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full-parity Wilson dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void wil_mat(void *out, const void *const *gauge, const void *in, double kappa, int dagger, QudaPrecision precision,
             const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Wilson dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void wil_matpc(void *out, const void *const *gauge, const void *in, double kappa, QudaMatPCType matpc_type, int dagger,
               QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the twisted mass dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_dslash(void *out, const void *const *gauge, const void *in, double kappa, double mu, QudaTwistFlavorType flavor,
               QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
               const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity twisted mass operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_mat(void *out, const void *const *gauge, const void *in, double kappa, double mu, QudaTwistFlavorType flavor,
            int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned twisted mass operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_matpc(void *out, const void *const *gauge, const void *in, double kappa, double mu, QudaTwistFlavorType flavor,
              QudaMatPCType matpc_type, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the twisted clover dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_dslash(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
                double kappa, double mu, QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int parity, int dagger,
                QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
             QudaTwistFlavorType flavor, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
               double kappa, double mu, QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int dagger,
               QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the non-degenerate twisted clover dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_ndeg_dslash(void *out, const void *const *gauge, const void *clover, const void *clover_inverse,
                     const void *in, double kappa, double mu, double epsilon, QudaMatPCType matpc_type, int parity,
                     int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity non-degenerate twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_ndeg_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
                  double epsilon, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned non-degenerate twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tmc_ndeg_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse, const void *in,
                    double kappa, double mu, double epsilon, QudaMatPCType matpc_type, int dagger,
                    QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the non-degenerate twisted mass dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_ndeg_dslash(void *out, const void *const *gauge, const void *in, double kappa, double mu, double epsilon,
                    QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
                    const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full-parity non-degenerate twisted mass dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_ndeg_mat(void *out, const void *const *gauge, const void *in, double kappa, double mu, double epsilon,
                 int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned non-degenerate twisted mass operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] epsilon Epsilon parameter for the non-degenerate term
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void tm_ndeg_matpc(void *out, const void *const *gauge, const void *in, double kappa, double mu, double epsilon,
                   QudaMatPCType matpc_type, int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the clover to a single-parity spinor
 *
 * @param[out] out Host output rhs
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] precision Single or double precision
 */
void apply_clover(void *out, const void *clover, const void *in, int parity, QudaPrecision precision);

/**
 * @brief Apply the even-odd or odd-even component of the clover dslash
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] parity 0 for D_eo, 1 for D_oe
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void clover_dslash(void *out, const void *const *gauge, const void *clover, const void *in, int parity, int dagger,
                   QudaPrecision precision, QudaGaugeParam &param);

/**
 * @brief Apply the full parity clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void clover_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, int dagger,
                QudaPrecision precision, const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void clover_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inv, const void *in,
                  double kappa, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                  const QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity Hasenbusch-twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 * @param[in] matpc_type Matrix preconditioning type
 */
void clover_ht_mat(void *out, const void *const *gauge, const void *clover, const void *in, double kappa, double mu,
                   int dagger, QudaPrecision precision, const QudaGaugeParam &gauge_param, QudaMatPCType matpc_type);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Hasenbusch-twisted clover operator
 *
 * @param[out] out Host output rhs
 * @param[in] gauge Gauge links
 * @param[in] clover Host input clover
 * @param[in] clover_inverse Host input clover inverse
 * @param[in] in Host input spinor
 * @param[in] kappa Kappa value for the Wilson operator
 * @param[in] mu Mu parameter for the twist
 * @param[in] matpc_type Matrix preconditioning type
 * @param[in] dagger 0 for the regular operator, 1 for the dagger operator
 * @param[in] precision Single or double precision
 * @param[in] gauge_param Gauge field parameters
 */
void clover_ht_matpc(void *out, const void *const *gauge, const void *clover, const void *clover_inverse,
                     const void *in, double kappa, double mu, QudaMatPCType matpc_type, int dagger,
                     QudaPrecision precision, const QudaGaugeParam &gauge_param);
