#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  void allocateParityClover(ParityClover *, int *X, int pad,
			    Precision precision);
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

#endif // _CLOVER_QUDA_H
