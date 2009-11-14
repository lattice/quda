#ifndef _QUDA_SPINOR_H
#define _QUDA_SPINOR_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  // -- spinor_quda.cpp

  ParitySpinor allocateParitySpinor(int *X, Precision precision, int stride);
  FullSpinor allocateSpinorField(int *X, Precision precision, int stride);
  
  void freeParitySpinor(ParitySpinor spinor);
  void freeSpinorField(FullSpinor spinor);
  void freeSpinorBuffer();

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

  // -- clover_quda.cpp

  void allocateParityClover(ParityClover *, int *X, int pad, Precision precision);
  void allocateCloverField(FullClover *, int *X, int pad, Precision precision);

  void freeParityClover(ParityClover *clover);
  void freeCloverField(FullClover *clover);

  void loadParityClover(ParityClover ret, void *clover, Precision cpu_prec,
			CloverFieldOrder clover_order);
  void loadFullClover(FullClover ret, void *clover, Precision cpu_prec,
		      CloverFieldOrder clover_order);
  void loadCloverField(FullClover ret, void *clover, Precision cpu_prec,
		       CloverFieldOrder clover_order);

  /* void createCloverField(FullClover *cudaClover, void *cpuClover, int *X,
                         Precision precision); */

#ifdef __cplusplus
}
#endif

#endif // _QUDA_SPINOR_H
