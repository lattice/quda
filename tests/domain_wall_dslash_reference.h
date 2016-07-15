#ifndef _DOMAIN_WALL_DSLASH_REFERENCE_H
#define _DOMAIN_WALL_DSLASH_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void dw_dslash(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		 QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dslash_4_4d(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		 QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_dslash_5_4d(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		      QudaPrecision precision, QudaGaugeParam &param, double mferm, bool zero_initialize);

  void dslash_5_inv(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		 QudaPrecision precision, QudaGaugeParam &param, double mferm, double *kappa);

  void mdw_dslash_5(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		    QudaPrecision precision, QudaGaugeParam &param, double mferm, double *kappa, bool zero_initialize);

  void mdw_dslash_4_pre(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
			QudaPrecision precision, QudaGaugeParam &param, double mferm, double *b5, double *c5, bool zero_initialize);

  void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_4d_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void mdw_mat(void *out, void **gauge, void *in, double *kappa_b, double *kappa_c, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm, double *b5, double *c5);

  void dw_matdagmat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision,
		QudaGaugeParam &gauge_param, double mferm);

  void dw_4d_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision,
		QudaGaugeParam &gauge_param, double mferm);

  void mdw_matpc(void *out, void **gauge, void *in, double *kappa_b, double *kappa_c, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision,
		QudaGaugeParam &gauge_param, double mferm, double *b5, double *c5);

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
