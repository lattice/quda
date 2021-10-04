#ifndef _DOMAIN_WALL_DSLASH_REFERENCE_H
#define _DOMAIN_WALL_DSLASH_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

void dw_dslash(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm);

void dslash_4_4d(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm);

void dw_dslash_5_4d(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm, bool zero_initialize);

void dslash_5_inv(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm, double *kappa);

void mdw_dslash_5_inv(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm, double _Complex *kappa);

void mdw_dslash_5(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm, double _Complex *kappa, bool zero_initialize);

void mdw_dslash_4_pre(void *res, void **gaugeFull, void *spinorField, int oddBit, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm, double _Complex *b5, double _Complex *c5, bool zero_initialize);

void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger, QudaPrecision precision, QudaGaugeParam &param,
    double mferm);

void dw_4d_mat(void *out, void **gauge, void *in, double kappa, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm);

void mdw_mat(void *out, void **gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c, int dagger,
    QudaPrecision precision, QudaGaugeParam &param, double mferm, double _Complex *b5, double _Complex *c5);

void dw_matdagmat(void *out, void **gauge, void *in, double kappa, int dagger, QudaPrecision precision,
    QudaGaugeParam &param, double mferm);

void dw_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger,
    QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm);

void dw_4d_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger,
    QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm);

void mdw_matpc(void *out, void **gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c,
    QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm,
    double _Complex *b5, double _Complex *c5);

void mdw_mdagm_local(void *out, void **gauge, void *in, double _Complex *kappa_b, double _Complex *kappa_c,
                     QudaMatPCType matpc_type, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm,
                     double _Complex *b5, double _Complex *c5);
void mdw_eofa_m5(void *res, void *spinorField, int oddBit, int daggerBit, double mferm, double m5, double b, double c,
                 double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

void mdw_eofa_m5inv(void *res, void *spinorField, int oddBit, int daggerBit, double mferm, double m5, double b, double c,
                    double mq1, double mq2, double mq3, int eofa_pm, double eofa_shift, QudaPrecision precision);

void mdw_eofa_mat(void *out, void **gauge, void *in, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param,
                  double mferm, double m5, double b, double c, double mq1, double mq2, double mq3, int eofa_pm,
                  double eofa_shift);

void mdw_eofa_matpc(void *out, void **gauge, void *in, QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                    QudaGaugeParam &gauge_param, double mferm, double m5, double b, double c, double mq1, double mq2,
                    double mq3, int eofa_pm, double eofa_shift);

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
