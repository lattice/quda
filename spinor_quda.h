#ifndef _QUDA_SPINOR_H
#define _QUDA_SPINOR_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  ParitySpinor allocateParitySpinor(int *X, Precision precision);
  FullSpinor allocateSpinorField(int *X, Precision precision);
  ParityClover allocateParityClover(int length, Precision precision);
  FullClover allocateCloverField(int length, Precision precision);
  
  void freeParitySpinor(ParitySpinor spinor);
  void freeSpinorField(FullSpinor spinor);
  void freeSpinorBuffer();
  void freeParityClover(ParityClover clover);
  void freeCloverField(FullClover clover);

  void loadParitySpinor(ParitySpinor, void *spinor, Precision cpu_prec, DiracFieldOrder dirac_order);
  void loadSpinorField(FullSpinor, void *spinor, Precision cpu_prec, DiracFieldOrder dirac_order);
  
  void retrieveParitySpinor(void *res, ParitySpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order);
  void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order);
  
  void spinorHalfPack(float *c, short *s0, float *f0);
  void spinorHalfUnpack(float *f0, float *c, short *s0);


#ifdef __cplusplus
}
#endif

#endif // _QUDA_SPINOR_H
