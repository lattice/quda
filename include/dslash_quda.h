#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <face_quda.h>

void setFace(const FaceBuffer &face);

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
#ifdef __cplusplus
}
#endif

// plain Wilson Dslash  
void wilsonDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in,
		      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
		      const double &k, const dim3 &block, const dim3 &blockFace, const int *commDim);

// clover Dslash
void cloverDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, 
		      const FullClover cloverInv, const cudaColorSpinorField *in, 
		      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
		      const double &k, const dim3 &block, const dim3 &blockFace, const int *commDim);

// solo clover term
void cloverCuda(cudaColorSpinorField *out, const FullGauge gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int oddBit, const dim3 &block, const int *commDim);

// domain wall Dslash  
void domainWallDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in, 
			  const int parity, const int dagger, const cudaColorSpinorField *x, 
			  const double &m_f, const double &k, const dim3 &block, const dim3 &blockFace);

// staggered Dslash    
void staggeredDslashCuda(cudaColorSpinorField *out, const FullGauge fatGauge, FullGauge longGauge,
			 const cudaColorSpinorField *in, const int parity, const int dagger, 
			 const cudaColorSpinorField *x, const double &k,
			 const dim3 &block, const dim3 &blockFace);

// twisted mass Dslash  
void twistedMassDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in,
			   const int parity, const int dagger, const cudaColorSpinorField *x, 
			   const double &kappa, const double &mu, const double &a, 
			   const dim3 &block, const dim3 &blockFace, const int *commDim);

// solo twist term
void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu,
		     const QudaTwistGamma5Type, const dim3 &block);

// face packing routines
void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dim, const QudaDirection dir, const int dagger, 
		    const int parity, const cudaStream_t &stream);

#endif // _DSLASH_QUDA_H
