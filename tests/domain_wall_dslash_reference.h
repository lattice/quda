#ifndef _DOMAIN_WALL_DSLASH_REFERENCE_H
#define _DOMAIN_WALL_DSLASH_REFERENCE_H

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int Z[4];
  extern int Vh;
  extern int V;

  void setDims(int *, const int);
  
  void dslash(void *res, void **gaugeFull, void *spinorField, 
	      int oddBit, int daggerBit, 
	      QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm);

  void mat(void *out, void **gauge, void *in, double kappa, int daggerBit,
	   QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm);

  void matpc(void *out, void **gauge, void *in, double kappa,
	     QudaMatPCType matpc_type,  int daggerBit,
	     QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm);

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
