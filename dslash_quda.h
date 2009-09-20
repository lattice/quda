#ifndef _QUDA_DSLASH_H
#define _QUDA_DSLASH_H

#include <cuComplex.h>

#include <quda.h>

#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 24 // real numbers per spinor
#define cloverSiteSize 72 // real numbers per block-diagonal clover matrix

#ifdef __cplusplus
extern "C" {
#endif

  extern FullGauge cudaGaugePrecise;
  extern FullGauge cudaGaugeSloppy;

  extern FullClover cudaClover;
  extern FullClover cudaCloverSloppy;

  extern QudaGaugeParam *gauge_param;
  extern QudaInvertParam *invert_param;

// ---------- dslash_quda.cu ----------

  int dslashCudaSharedBytes(Precision spinor_prec, int blockDim);

  // Double precision routines
  void dslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);
  void dslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // Single precision routines
  void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);
  void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // Half precision dslash routines
  void dslashHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);
  void dslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // wrapper to above
  void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
		  int parity, int dagger);
  void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
		      int parity, int dagger, ParitySpinor x, double a);

  // Full Wilson matrix
  void MatCuda(FullSpinor out, FullGauge gauge, FullSpinor in, double kappa,
	       int daggerBit);
  void MatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		 double kappa, ParitySpinor tmp, MatPCType matpc_type,
		 int daggerBit);
  void MatPCDagMatPCCuda(ParitySpinor outEven, FullGauge gauge,
			 ParitySpinor inEven, double kappa, ParitySpinor tmp,
			 MatPCType matpc_type);

  // clover Dslash routines
  void cloverDslashCuda(ParitySpinor out, FullGauge gauge,
			FullClover cloverInv, ParitySpinor in, int parity,
			int dagger);
  void cloverDslashDCuda(ParitySpinor res, FullGauge gauge,
			 FullClover cloverInv, ParitySpinor spinor,
			 int oddBit, int daggerBit);
  void cloverDslashSCuda(ParitySpinor res, FullGauge gauge,
			 FullClover cloverInv, ParitySpinor spinor,
			 int oddBit, int daggerBit);
  void cloverDslashHCuda(ParitySpinor res, FullGauge gauge,
			 FullClover cloverInv, ParitySpinor spinor,
			 int oddBit, int daggerBit);

  void cloverDslashXpayCuda(ParitySpinor out, FullGauge gauge,
			    FullClover cloverInv, ParitySpinor in, int parity,
			    int dagger, ParitySpinor x, double a);
  void cloverDslashXpayDCuda(ParitySpinor res, FullGauge gauge,
			     FullClover cloverInv, ParitySpinor spinor,
			     int oddBit, int daggerBit, ParitySpinor x,
			     double a);
  void cloverDslashXpaySCuda(ParitySpinor res, FullGauge gauge,
			     FullClover cloverInv, ParitySpinor spinor,
			     int oddBit, int daggerBit, ParitySpinor x,
			     double a);
  void cloverDslashXpayHCuda(ParitySpinor res, FullGauge gauge,
			     FullClover cloverInv, ParitySpinor spinor,
			     int oddBit, int daggerBit, ParitySpinor x,
			     double a);

  void cloverMatPCCuda(ParitySpinor out, FullGauge gauge,
		       FullClover cloverInv, ParitySpinor in, double kappa,
		       ParitySpinor tmp, MatPCType matpc_type, int dagger);
  void cloverMatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge,
			       FullClover cloverInv, ParitySpinor in,
			       double kappa, ParitySpinor tmp,
			       MatPCType matpc_type);
  void cloverMatCuda(FullSpinor out, FullGauge gauge, FullClover clover,
		     FullSpinor in, double kappa, ParitySpinor tmp,
		     int dagger);

  // routines for applying the clover term alone
  void cloverCuda(ParitySpinor out, FullGauge gauge, FullClover clover,
		  ParitySpinor in, int parity);
  void cloverDCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		   ParitySpinor spinor, int oddBit);
  void cloverSCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		   ParitySpinor spinor, int oddBit);
  void cloverHCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		   ParitySpinor spinor, int oddBit);
  
  // -- inv_cg_cuda.cpp
  void invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
		    FullGauge gaugeSloppy, ParitySpinor tmp,
		    QudaInvertParam *param);
  
  // -- inv_bicgstab_cuda.cpp
  void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
			  FullGauge gaugeSloppy, ParitySpinor tmp, 
			  QudaInvertParam *param, DagType dag_type);
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_H
