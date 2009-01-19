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
  extern QudaGaugeParam *gauge_param;

// ---------- dslash_quda.cu ----------

  void setCudaGaugeParam();

  void dslashCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		  int oddBit, int daggerBit);
  void dslashXpayCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		      int oddBit, int daggerBit, ParitySpinor x, float a);
  int  dslashCudaSharedBytes();
  
  void MatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		 float kappa, ParitySpinor tmp, MatPCType matpc_type);
  void MatPCDagCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		    float kappa, ParitySpinor tmp, MatPCType matpc_type);
  void MatPCDagMatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven,
			 float kappa, ParitySpinor tmp, MatPCType matpc_type);
  
  cuComplex MatPCcDotWXCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
			    float kappa, ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type);
  cuComplex MatPCDagcDotWXCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
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
void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
			ParitySpinor tmp, QudaInvertParam *param, DagType dag_type);
  
// ---------- cg_reference.cpp ----------  
void cgReference(float *out, float **gauge, float *in, float kappa, float tol);

#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_H
