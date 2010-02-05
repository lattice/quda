#ifndef _SPINOR_QUDA_H
#define _SPINOR_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  ParitySpinor allocateParitySpinor(int *X, Precision precision, int stride);
  FullSpinor allocateSpinorField(int *X, Precision precision, int stride);
  
  void freeParitySpinor(ParitySpinor spinor);
  void freeSpinorField(FullSpinor spinor);
  void freeSpinorBuffer(void);

  void loadParitySpinor(ParitySpinor, void *spinor, Precision cpu_prec,
			DiracFieldOrder dirac_order);
  void loadSpinorField(FullSpinor, void *spinor, Precision cpu_prec,
		       DiracFieldOrder dirac_order);
  
  void retrieveParitySpinor(void *res, ParitySpinor spinor,
			    Precision cpu_prec, DiracFieldOrder dirac_order);
  void retrieveSpinorField(void *res, FullSpinor spinor,
			   Precision cpu_prec, DiracFieldOrder dirac_order);
  
  void spinorHalfPack(float *c, short *s0, float *f0);
  void spinorHalfUnpack(float *f0, float *c, short *s0);

#ifdef __cplusplus
}
#endif

#endif // _SPINOR_QUDA_H
