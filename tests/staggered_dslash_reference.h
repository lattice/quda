
#ifndef _STAGGERED_QUDA_DSLASH_REF_H
#define _STAGGERED_QUDA_DSLASH_REF_H
#include <blas_reference.h>
#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int Z[4];
    extern int Vh;
    extern int V;

    typedef enum MyQudaParity_s {
	QUDA_EVEN,
	QUDA_ODD,
	QUDA_EVENODD
    } MyQudaParity;

    
    void setDims(int *);
    
  void staggered_dslash(void *res, void ** fatlink, void** longlink, void *spinorField,
			int oddBit, int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
  void staggered_dslash_mg(void *res, void **fatlink, void** longlink, void* ghost_fatlink, void* ghost_longlink,
			   void *spinorField, void* fwd_nbr_spinor, void* back_nbr_spinor, 
			   int oddBit, int daggerBit,
			   QudaPrecision sPrecision, QudaPrecision gPrecision);
  void staggered_dslash_mg4dir(void *res, void **fatlink, void** longlink, void** ghost_fatlink, void** ghost_longlink,
			       void *spinorField, void** fwd_nbr_spinor, void** back_nbr_spinor,
			       int oddBit, int daggerBit,
			       QudaPrecision sPrecision, QudaPrecision gPrecision);  
  
    void mat(void *out, void **fatlink, void** longlink, void *in, double kappa, int daggerBit,
	     QudaPrecision sPrecision, QudaPrecision gPrecision);
    
    void staggered_matpc(void *out, void **fatlink, void ** longlink, void *in, double kappa, MatPCType matpc_type, 
			 int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
  void matdagmat(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
		 QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, MyQudaParity parity);    
  void matdagmat_mg(void *out, void **fatlink, void* ghost_fatlink, void** longlink, void* ghost_longlink, 
		    void *in, void* fwd_nbr_spinor, void* back_nbr_spinor, double mass, int dagger_bit,
		    QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, MyQudaParity parity);
  
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_REF_H
