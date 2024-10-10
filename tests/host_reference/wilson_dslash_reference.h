#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

/**
 * @brief Apply even-odd or odd-even component of the Wilson dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void wil_dslash(void *out, void **gauge, void *in, int parity, int dagger, QudaPrecision precision,
                QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full-parity Wilson dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void wil_mat(void *out, void **gauge, void *in, double kappa, int dagger, QudaPrecision precision,
             QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Wilson dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void wil_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger,
               QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the twisted mass dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param matpc_type Matrix preconditioning type
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_dslash(void *out, void **gauge, void *in, double kappa, double mu, QudaTwistFlavorType flavor,
               QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity twisted mass operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_mat(void *out, void **gauge, void *in, double kappa, double mu, QudaTwistFlavorType flavor, int dagger,
            QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned twisted mass operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_matpc(void *out, void **gauge, void *in, double kappa, double mu, QudaTwistFlavorType flavor,
              QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the twisted clover dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param matpc_type Matrix preconditioning type
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_dslash(void *out, void **gauge, void *clover, void *clover_inverse, void *in, double kappa, double mu,
                QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
                QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, QudaTwistFlavorType flavor,
             int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_matpc(void *out, void **gauge, void *clover, void *clover_inverse, void *in, double kappa, double mu,
               QudaTwistFlavorType flavor, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
               QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the non-degenerate twisted clover dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param matpc_type Matrix preconditioning type
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_ndeg_dslash(void *out, void **gauge, void *clover, void *clover_inverse, void *in, double kappa, double mu,
                     double epsilon, QudaMatPCType matpc_type, int parity, int dagger, QudaPrecision precision,
                     QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity non-degenerate twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_ndeg_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, double epsilon, int dagger,
                  QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned non-degenerate twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param flavor Twist flavor type dictating whether or not the twist or inverse twist is being applied
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tmc_ndeg_matpc(void *out, void **gauge, void *clover, void *clover_inverse, void *in, double kappa, double mu,
                    double epsilon, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                    QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-odd or odd-even component of the non-degenerate twisted mass dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param matpc_type Matrix preconditioning type
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_ndeg_dslash(void *out, void **gauge, void *in, double kappa, double mu, double epsilon, QudaMatPCType matpc_type,
                    int parity, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full-parity non-degenerate twisted mass dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_ndeg_mat(void *out, void **gauge, void *in, double kappa, double mu, double epsilon, int dagger,
                 QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned non-degenerate twisted mass operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param epsilon Epsilon parameter for the non-degenerate term
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void tm_ndeg_matpc(void *out, void **gauge, void *in, double kappa, double mu, double epsilon, QudaMatPCType matpc_type,
                   int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the clover to a single-parity spinor
 *
 * @param out Host output rhs
 * @param clover Host input clover
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param precision Single or double precision
 */
void apply_clover(void *out, void *clover, void *in, int parity, QudaPrecision precision);

/**
 * @brief Apply the even-odd or odd-even component of the clover dslash
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param in Host input spinor
 * @param parity 0 for D_eo, 1 for D_oe
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void clover_dslash(void *out, void **gauge, void *clover, void *in, int parity, int dagger, QudaPrecision precision,
                   QudaGaugeParam &param);

/**
 * @brief Apply the full parity clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void clover_mat(void *out, void **gauge, void *clover, void *in, double kappa, int dagger, QudaPrecision precision,
                QudaGaugeParam &gauge_param);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void clover_matpc(void *out, void **gauge, void *clover, void *clover_inv, void *in, double kappa,
                  QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);

/**
 * @brief Apply the full parity Hasenbusch-twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 * @param matpc_type Matrix preconditioning type
 */
void clover_ht_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, int dagger,
                   QudaPrecision precision, QudaGaugeParam &gauge_param, QudaMatPCType matpc_type);

/**
 * @brief Apply the even-even or odd-odd symmetric or asymmetric preconditioned Hasenbusch-twisted clover operator
 *
 * @param out Host output rhs
 * @param gauge Gauge links
 * @param clover Host input clover
 * @param clover_inverse Host input clover inverse
 * @param in Host input spinor
 * @param kappa Kappa value for the Wilson operator
 * @param mu Mu parameter for the twist
 * @param matpc_type Matrix preconditioning type
 * @param dagger 0 for the regular operator, 1 for the dagger operator
 * @param precision Single or double precision
 * @param gauge_param Gauge field parameters
 */
void clover_ht_matpc(void *out, void **gauge, void *clover, void *clover_inverse, void *in, double kappa, double mu,
                     QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param);
