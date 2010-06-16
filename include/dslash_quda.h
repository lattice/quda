#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern unsigned long long dslash_quda_flops;
  extern unsigned long long dslash_quda_bytes;

  void initCache(void);
  int dslashCudaSharedBytes(Precision spinor_prec, int blockDim);

  // Double precision routines
  void dslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslash3DDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // Single precision routines
  void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslash3DSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // Half precision dslash routines
  void dslashHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslash3DHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);

  void dslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double a);

  // wrapper to above
  void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
		  int parity, int dagger);

  void dslash3DCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
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

  void cloverMatPCCuda(ParitySpinor out, FullGauge gauge, FullClover clover,
		       FullClover cloverInv, ParitySpinor in, double kappa,
		       ParitySpinor tmp, MatPCType matpc_type, int dagger);
  void cloverMatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge,
			       FullClover clover, FullClover cloverInv,
			       ParitySpinor in, double kappa, ParitySpinor tmp,
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

#ifdef __cplusplus
}
#endif

#endif // _DLASH_QUDA_H
