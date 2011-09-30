
#ifndef _STAGGERED_QUDA_DSLASH_REF_H
#define _STAGGERED_QUDA_DSLASH_REF_H
#include <blas_reference.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

void setDims(int *);

void staggered_dslash(void *res, void ** fatlink, void** longlink, void *spinorField,
		      int oddBit, int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
void staggered_dslash_mg4dir(cpuColorSpinorField* out, void **fatlink, void** longlink, void** ghost_fatlink, 
			     void** ghost_longlink, cpuColorSpinorField* in, int oddBit, int daggerBit,
			     QudaPrecision sPrecision, QudaPrecision gPrecision);  

void mat(void *out, void **fatlink, void** longlink, void *in, double kappa, int daggerBit,
	 QudaPrecision sPrecision, QudaPrecision gPrecision);

void staggered_matpc(void *out, void **fatlink, void ** longlink, void *in, double kappa, MatPCType matpc_type, 
		     int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
void matdagmat(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
	       QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, QudaParity parity);    
void matdagmat_mg4dir(cpuColorSpinorField* out, void **fatlink, void **longlink, void** ghost_fatlink, void** ghost_longlink,
		      cpuColorSpinorField* in, double mass, int dagger_bit,
		      QudaPrecision sPrecision, QudaPrecision gPrecision, cpuColorSpinorField* tmp, QudaParity parity);

#endif // _QUDA_DLASH_REF_H
