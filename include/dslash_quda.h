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

  /**
     @brief Apply clover-matrix field to a color-spinor field
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] clover Clover-matrix field
     @param[in] inverse Whether we are applying the inverse or not
     @param[in] Field parity (if color-spinor field is single parity)
  */
  void ApplyClover(ColorSpinorField &out, const ColorSpinorField &in,
		   const CloverField &clover, bool inverse, int parity);

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

  // twisted mass Dslash  
  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const   cudaColorSpinorField *in, 
			     const int parity, const int dagger, const cudaColorSpinorField *x, const QudaTwistDslashType type,
			     const double &kappa, const double &mu, const double &epsilon, const double &k, const int *commDim, TimeProfile &profile);

  // twisted mass Dslash
  void ndegTwistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const   cudaColorSpinorField *in,
				 const int parity, const int dagger, const cudaColorSpinorField *x, const QudaTwistDslashType type,
				 const double &kappa, const double &mu, const double &epsilon, const double &k, const int *commDim,
				 TimeProfile &profile);

  // twisted clover Dslash
  void twistedCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge,
			       const FullClover *clover, const FullClover *cloverInv, const   cudaColorSpinorField *in,
			       const int parity, const int dagger, const cudaColorSpinorField *x, const QudaTwistCloverDslashType type,
			       const double &kappa, const double &mu, const double &epsilon, const double &k, const int *commDim,
			       TimeProfile &profile);

  /**
     @brief Apply the twisted-mass gamma operator to a color-spinor field.
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] d Which gamma matrix we are applying (C counting, so gamma_5 has d=4)
     @param[in] kappa kappa parameter
     @param[in] mu mu parameter
     @param[in] epsilon epsilon parameter
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] twist The type of kernel we are doing
  */
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu,
		       double epsilon, int dagger, QudaTwistGamma5Type type);

  /**
     @brief Apply twisted clover-matrix field to a color-spinor field
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] clover Clover-matrix field
     @param[in] kappa kappa parameter
     @param[in] mu mu parameter
     @param[in] epsilon epsilon parameter
     @param[in] Field parity (if color-spinor field is single parity)
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] twist The type of kernel we are doing
       if (twist == QUDA_TWIST_GAMMA5_DIRECT) apply (Clover + i*a*gamma_5) to the input spinor
       else if (twist == QUDA_TWIST_GAMMA5_INVERSE) apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  */
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist);


  /**
     @brief Dslash face packing routine
     @param[out] ghost_buf Array of packed halos, order is [2*dim+dir]
     @param[in] in Input ColorSpinorField to be packed
     @param[in] location Array of locations where the packed fields are (Device, Host or Remote)
     @param[in] nFace Depth of halo
     @param[in] dagger Whether this is for the dagger operator
     @param[in] parity Field parity
     @param[in] dim Which dimensions we are packing
     @param[in] face_num Are we packing backwards (0), forwards (1) or both directions (2)
     @param[in] stream Which stream are we executing in
     @param[in] a Packing coefficient (twisted-mass only)
     @param[in] b Packing coefficient (twisted-mass only)
  */
  void packFace(void *ghost_buf[2*QUDA_MAX_DIM], cudaColorSpinorField &in, MemoryLocation location[],
		const int nFace, const int dagger, const int parity, const int dim, const int face_num,
		const cudaStream_t &stream, const double a=0.0, const double b=0.0);

  void packFaceExtended(void *ghost_buf[2*QUDA_MAX_DIM], cudaColorSpinorField &field, MemoryLocation location[],
			const int nFace, const int R[], const int dagger, const int parity, const int dim,
			const int face_num, const cudaStream_t &stream, const bool unpack=false);

  /**
     @brief Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
     @param[out] out Output field
     @param[in] in Input field
  */
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in);

}

#endif // _DSLASH_QUDA_H
