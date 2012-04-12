#ifndef _DOMAIN_WALL_DSLASH_REFERENCE_H
#define _DOMAIN_WALL_DSLASH_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int Z[4];
  extern int Vh;
  extern int V;

  void setDims(int *, const int);
  
  void dw_dslash(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit, 
		 QudaPrecision precision, QudaGaugeParam &param, double mferm, const int nodes);

  void mat(void *out, void **gauge, void *in, double kappa, int daggerBit,
	   QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm);

  void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm, const int nodes);

  void dw_matdagmat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &param, double mferm, const int nodes);

  void matpc(void *out, void **gauge, void *in, double kappa,
	     QudaMatPCType matpc_type,  int daggerBit,
	     QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm);

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
