#ifndef _DOMAIN_WALL_DSLASH_REFERENCE_H
#define _DOMAIN_WALL_DSLASH_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void dw_dslash(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		 QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_matdagmat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm);

  void dw_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision,
		QudaGaugeParam &gauge_param, double mferm);

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
