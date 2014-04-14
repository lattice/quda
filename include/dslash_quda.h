#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <tune_quda.h>
#include <face_quda.h>
#include <gauge_field.h>

namespace quda {

  /**
    @param pack Sets whether to use a kernel to pack the T dimension
    */
  void setKernelPackT(bool pack);

  /**
    @return Whether the T dimension is kernel packed or not
    */
  bool getKernelPackT();

  /**
    @param pack Sets whether to use a kernel to pack twisted spinor
    */
  void setTwistPack(bool pack);

  /**
    @return Whether a kernel requires twisted pack or not
    */
  bool getTwistPack();

  void setFace(const FaceBuffer& face1, const FaceBuffer& face2);

  bool getDslashLaunch();

  void createDslashEvents();
  void destroyDslashEvents();

  void initLatticeConstants(const LatticeField &lat, TimeProfile &profile);
  void initGaugeConstants(const cudaGaugeField &gauge, TimeProfile &profile);
  void initSpinorConstants(const cudaColorSpinorField &spinor, TimeProfile &profile);
  void initDslashConstants(TimeProfile &profile);
  void initCloverConstants (const cudaCloverField &clover, TimeProfile &profile);
  void initStaggeredConstants(const cudaGaugeField &fatgauge, 
      const cudaGaugeField &longgauge, TimeProfile &profile);
  void initTwistedMassConstants(const int flv_stride, TimeProfile &profile);

  // plain Wilson Dslash  
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
      const double &k, const int *commDim, TimeProfile &profile);

  // clover Dslash
  void cloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
      const FullClover cloverInv, const cudaColorSpinorField *in, 
      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
      const double &k, const int *commDim, TimeProfile &profile);

  // clover Dslash
  void asymCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
      const FullClover cloverInv, const cudaColorSpinorField *in, 
      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
      const double &k, const int *commDim, TimeProfile &profile);

  // solo clover term
  void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
      const cudaColorSpinorField *in, const int oddBit);

  // domain wall Dslash  
  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, 
      const int parity, const int dagger, const cudaColorSpinorField *x, 
      const double &m_f, const double &k, const int *commDim, TimeProfile &profile);

  // staggered Dslash    
  void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge,
      const cudaColorSpinorField *in, const int parity, const int dagger, 
      const cudaColorSpinorField *x, const double &k, 
      const int *commDim, TimeProfile &profile);

  // improved staggered Dslash    
  void improvedStaggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, const cudaGaugeField &longGauge,
      const cudaColorSpinorField *in, const int parity, const int dagger, 
      const cudaColorSpinorField *x, const double &k, 
      const int *commDim, TimeProfile &profile);

  // twisted mass Dslash  
  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const   cudaColorSpinorField *in, 
      const int parity, const int dagger, const cudaColorSpinorField *x, const QudaTwistDslashType type,
      const double &kappa, const double &mu, const double &epsilon, const double &k, const int *commDim, TimeProfile &profile);

  // twisted clover Dslash  
  void twistedCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover *clover, const FullClover *cloverInv, 
			     const   cudaColorSpinorField *in, const int parity, const int dagger, const cudaColorSpinorField *x, const QudaTwistCloverDslashType type,
			     const double &kappa, const double &mu, const double &epsilon, const double &k, const int *commDim, TimeProfile &profile);

  // solo twist term
  void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in, const int dagger,     
      const double &kappa, const double &mu, const double &epsilon, 
      const QudaTwistGamma5Type);

  // solo twist clover term
  void twistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in, const int dagger, const double &kappa, const double &mu,
			     const double &epsilon, const QudaTwistGamma5Type twist, const FullClover *clov, const FullClover *clovInv, const int parity);
  // face packing routines
  void packFace(void *ghost_buf, cudaColorSpinorField &in, const int nFace, const int dagger, 
      const int parity, const int dim, const int face_num, const cudaStream_t &stream,
      const double a=0.0, const double b=0.0);

  void packFaceExtended(void *ghost_buf, cudaColorSpinorField &field, const int nFace, const int R[], const int dagger,
      const int parity, const int dim, const int face_num, const cudaStream_t &stream, const bool unpack=false);


  // face packing routines
  void packFace(void *ghost_buf, cudaColorSpinorField &in, FullClover &clov, FullClover &clovInv,
		const int nFace, const int dagger, const int parity, const int dim, const int face_num,
		const cudaStream_t &stream, const double a=0.0);


}

#endif // _DSLASH_QUDA_H
