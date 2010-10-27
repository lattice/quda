#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  void initCache();

  extern int initDslash;

  int dslashCudaSharedBytes(QudaPrecision spinor_prec, int blockDim);

  void initDslashConstants(FullGauge gauge, int sp_stride, int cl_stride, int Ls=1);
  void initCommonConstants(FullGauge gauge);

  // plain Wilson Dslash  
  void dslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, 
		  const void *inNorm, const int parity, const int dagger, 
		  const void *x, const void *xNorm, const double k,
		  const int volume, const int length, const QudaPrecision precision);
    
  // clover Dslash
  void cloverDslashCuda(void *out, void *outNorm, const FullGauge gauge, 
			const FullClover cloverInv, const void *in, const void *inNorm,
			const int oddBit, const int daggerBit, const void *x, const void *xNorm,
			const double k, const int volume, const int length, const QudaPrecision precision);
    
  // solo clover term
  void cloverCuda(void *out, void *outNorm, const FullGauge gauge, const FullClover clover, 
		  const void *in, const void *inNorm, const int oddBit, const int volume, 
		  const int length, const QudaPrecision precision);

  // domain wall Dslash  
  void domainWallDslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, 
			    const void *inNorm, const int parity, const int dagger, 
			    const void *x, const void *xNorm, const double m_f, const double k,
			    const int volume5d, const int length, const QudaPrecision precision);
    
  // staggered Dslash
  void staggeredDslashCuda(void *out, void *outNorm, const FullGauge fatGauge, FullGauge longGauge, const void *in, 
			   const void *inNorm, const int parity, const int dagger, 
			   const void *x, const void *xNorm, const double k,
			   const int volume, const int length, const QudaPrecision precision);

#ifdef __cplusplus
}
#endif

#endif // _DLASH_QUDA_H
