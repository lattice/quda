#ifndef _QUDA_DSLASH_H
#define _QUDA_DSLASH_H

#include <cuComplex.h>

#include <quda.h>

#define packed12GaugeSiteSize 12 // real numbers per link, using SU(3) reconstruction
#define packed8GaugeSiteSize 8 // real numbers per link, using SU(3) reconstruction

#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 24 // real numbers per spinor

#define BLOCK_DIM (64) // threads per block
#define GRID_DIM (Nh/BLOCK_DIM) // there are Nh threads in total

#define SPINOR_BYTES (Nh*spinorSiteSize*sizeof(float))

#define PACKED12_GAUGE_BYTES (4*Nh*packed12GaugeSiteSize*sizeof(float))
#define PACKED8_GAUGE_BYTES (4*Nh*packed8GaugeSiteSize*sizeof(float))

#ifdef __cplusplus
extern "C" {
#endif

  extern FullGauge cudaGauge;
  extern FullGauge cudaHGauge;
  extern QudaGaugeParam *gauge_param;
  extern QudaInvertParam *invert_param;

  extern ParityHSpinor hSpinor1;
  extern ParityHSpinor hSpinor2;

// ---------- dslash_quda.cu ----------

  int dslashCudaSharedBytes();
  
  void setCudaGaugeParam();

  // Single precision routines
  void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit);
  void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, float a);

  // Half precision dslash routines
  void dslashHCuda(ParityHSpinor res, FullGauge gauge, ParityHSpinor spinor,
		   int oddBit, int daggerBit);
  void dslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParityHSpinor spinor, 
		       int oddBit, int daggerBit, ParityHSpinor x, float a);

  // wrapper to above
  void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger);

  // Full Wilson matrix
  void MatCuda(FullSpinor out, FullGauge gauge, FullSpinor in, float kappa);
  void MatDagCuda(FullSpinor out, FullGauge gauge, FullSpinor in, float kappa);

  void MatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		 float kappa, ParitySpinor tmp, MatPCType matpc_type);
  void MatPCDagCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		    float kappa, ParitySpinor tmp, MatPCType matpc_type);
  void MatPCDagMatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven,
			 float kappa, ParitySpinor tmp, MatPCType matpc_type);
  
  QudaSumComplex MatPCcDotWXCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
				 float kappa, ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type);
  QudaSumComplex MatPCDagcDotWXCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
				    float kappa, ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type);
  
  // ---------- dslash_reference.cpp ----------
  
  void dslashReference(float *res, float **gauge, float *spinorField, 
		       int oddBit, int daggerBit);
  
  void Mat(float *out, float **gauge, float *in, float kappa);
  void MatDag(float *out, float **gauge, float *in, float kappa);
  void MatDagMat(float *out, float **gauge, float *in, float kappa);
  
  void MatPC(float *out, float **gauge, float *in, float kappa, MatPCType matpc_type);
  void MatPCDag(float *out, float **gauge, float *in, float kappa, MatPCType matpc_type);
  void MatPCDagMatPC(float *out, float **gauge, float *in, float kappa, MatPCType matpc_type);
  
  // -- inv_cg_cuda.cpp
  void invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
		    ParitySpinor tmp, QudaInvertParam *param);
  
// -- inv_bicgstab_cuda.cpp
  void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, FullGauge gaugeSloppy, 
			  FullGauge gaugePrecise, ParitySpinor tmp, 
			  QudaInvertParam *param, DagType dag_type);
  
// ---------- cg_reference.cpp ----------  
void cgReference(float *out, float **gauge, float *in, float kappa, float tol);

#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_H
