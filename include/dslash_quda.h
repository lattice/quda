#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int initDslash;

  int dslashCudaSharedBytes(QudaPrecision spinor_prec, int blockDim);

  void initDslashConstants(FullGauge gauge, int sp_stride, int cl_stride);

  // plain wilson
  
  void dslashCuda(void *out, FullGauge gauge, void *in, int parity, int dagger,
		  int volume, int length, void *outNorm, void *inNorm, const QudaPrecision precision);

  void dslashXpayCuda(void *out, FullGauge gauge, void *in, int parity, int dagger,
		      void *x, double k, int volume, int length, void *outNorm, 
		      void *inNorm, void *xNorm, const QudaPrecision precision);

  // clover dslash

  void cloverDslashCuda(void *out, FullGauge gauge, FullClover cloverInv, void *in, 
			int oddBit, int daggerBit, int volume, int length, 
			void *outNorm, void *inNorm, const QudaPrecision precision);

  void cloverDslashXpayCuda(void *out, FullGauge gauge, FullClover cloverInv, void *in, 
			    int oddBit, int daggerBit, void *x, double a, int volume,
			    int length, void *outNorm, void *inNorm, void *xNorm,
			    const QudaPrecision);

  // solo clover term
  void cloverCuda(void *out, FullGauge gauge, FullClover clover, void *in, int oddBit,
		  int volume, int length, void *outNorm, void *inNorm, const QudaPrecision precision);

#ifdef __cplusplus
}
#endif

#endif // _DLASH_QUDA_H
