#include <blas_reference.h>

#ifndef _STAGGERED_QUDA_DSLASH_REF_H
#define _STAGGERED_QUDA_DSLASH_REF_H

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int Z[4];
    extern int Vh;
    extern int V;

    typedef enum QudaParity_s {
	QUDA_EVEN,
	QUDA_ODD,
	QUDA_EVENODD
    } QudaParity;

    
    void setDims(int *);
    
    void staggered_dslash(void *res, void ** fatlink, void** longlink, void *spinorField,
		int oddBit, int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
    
    void mat(void *out, void **fatlink, void** longlink, void *in, double kappa, int daggerBit,
	     QudaPrecision sPrecision, QudaPrecision gPrecision);
    
    void staggered_matpc(void *out, void **fatlink, void ** longlink, void *in, double kappa, MatPCType matpc_type, 
			 int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
    void matdagmat_milc(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
			QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, QudaParity parity);    
    void mymatdagmat_milc(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
			  QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, QudaParity parity);    

    

#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_REF_H
