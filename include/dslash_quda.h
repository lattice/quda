#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <face_quda.h>

void setFace(const FaceBuffer &face, const int stride);

#ifdef __cplusplus
extern "C" {
#endif

  void initCache();

  extern int initDslash;
  extern int initClover;
  extern int initDomainWall;
  extern bool qudaPt0;
  extern bool qudaPtNm1;

  void setDslashTuning(QudaTune tune);

  void initCommonConstants(const FullGauge gauge);
  void initDslashConstants(const FullGauge gauge, const int sp_stride);
  void initCloverConstants(const int cl_stride);
  void initDomainWallConstants(const int Ls);
  void initStaggeredConstants(FullGauge fatgauge, FullGauge longgauge);


  // plain Wilson Dslash  
  void dslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, const void *inNorm,
		  const int oddBit, const int daggerBit, const void *x, const void *xNorm,
		  const double k, const int volume, const size_t bytes, const size_t norm_bytes, 
		  const QudaPrecision precision, const dim3 block, const dim3 blockFace);
    
  // clover Dslash
  void cloverDslashCuda(void *out, void *outNorm, const FullGauge gauge, 
			const FullClover cloverInv, const void *in, const void *inNorm,
			const int oddBit, const int daggerBit, const void *x, const void *xNorm,
			const double k, const int volume, const size_t bytes, const size_t norm_bytes, 
			const QudaPrecision precision, const dim3 block, const dim3 blockFace);
    
  // solo clover term
  void cloverCuda(void *out, void *outNorm, const FullGauge gauge, const FullClover clover, 
		  const void *in, const void *inNorm, const int oddBit, const int volume, 
		  const size_t bytes, const size_t norm_bytes, const QudaPrecision precision,
		  const dim3 block);

  // domain wall Dslash  
  void domainWallDslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, 
			    const void *inNorm, const int parity, const int dagger, 
			    const void *x, const void *xNorm, const double m_f, const double k,
			    const int volume5d, const size_t bytes, const size_t norm_bytes, 
			    const QudaPrecision precision, const dim3 block, const dim3 blockFace);
    
  // staggered Dslash
  void staggeredDslashCuda(void *out, void *outNorm, const FullGauge fatGauge, FullGauge longGauge, const void *in, 
			   const void *inNorm, const int parity, const int dagger, 
			   const void *x, const void *xNorm, const double k,
			   const int volume, const size_t bytes, const size_t norm_bytes, 
			   const QudaPrecision precision, const dim3 block, const dim3 blockFace);

  // twisted mass Dslash  
  void twistedMassDslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, 
			     const void *inNorm, const int parity, const int dagger, 
			     const void *x, const void *xNorm, const double kappa, const double mu,
			     const double a, const int volume, const size_t bytes, const size_t norm_bytes,
			     const QudaPrecision precision, const dim3 block, const dim3 blockFace);

  // solo twist term
  void twistGamma5Cuda(void *out, void *outNorm, const void *in, const void *inNorm,
		       const int dagger, const double kappa, const double mu, const int volume, 
		       const size_t bytes, const size_t norm_bytes, const QudaPrecision precision, 
		       const QudaTwistGamma5Type, const dim3 block);

#ifdef __cplusplus
}
#endif

#endif // _DLASH_QUDA_H
