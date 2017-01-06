#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <tune_quda.h>
#include <face_quda.h>
#include <gauge_field.h>

#include <worker.h>

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
     Sets commDim array used in dslash_pack.cu
   */
  void setPackComms(const int *commDim);

  bool getDslashLaunch();

  void createDslashEvents();
  void destroyDslashEvents();


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

  // Added for 4d EO preconditioning in DWF
  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
			    const int parity, const int dagger, const cudaColorSpinorField *x,
			    const double &m_f, const double &a, const double &b,
			    const int *commDim, const int DS_type, TimeProfile &profile);

  // Added for 4d EO preconditioning in Mobius DWF
  void MDWFDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
		      const int parity, const int dagger, const cudaColorSpinorField *x, const double &m_f, const double &k,
		      const int *commDim, const int DS_type, TimeProfile &profile);

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

  // face packing routines
  void packFace(void *ghost_buf, cudaColorSpinorField &in, bool zero_copy, const int nFace, const int dagger,
		const int parity, const int dim, const int face_num, const cudaStream_t &stream);

  void packFaceExtended(void *ghost_buf, cudaColorSpinorField &field, bool zero_copy, const int nFace, const int R[], const int dagger,
			const int parity, const int dim, const int face_num, const cudaStream_t &stream, const bool unpack=false);

  // face packing routines
  void packFace(void *ghost_buf, cudaColorSpinorField &in, FullClover &clov, FullClover &clovInv,
		const int nFace, const int dagger, const int parity, const int dim, const int face_num,
		const cudaStream_t &stream, const double a=0.0);

  /**
     out = gamma_5 in
     @param out Output field
     @param in Input field
   */
  void gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in);

}

#endif // _DSLASH_QUDA_H
