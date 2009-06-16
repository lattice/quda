#include <cuComplex.h>
#include <enum_quda.h>

#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#define REDUCE_DOUBLE 64
#define REDUCE_KAHAN 32

#if (__CUDA_ARCH__ == 130)
#define REDUCE_TYPE REDUCE_DOUBLE
#define QudaSumFloat double
#define QudaSumComplex cuDoubleComplex
#define QudaSumFloat3 double3
#else
#define REDUCE_TYPE REDUCE_KAHAN
#define QudaSumFloat float
#define QudaSumComplex cuComplex
#define QudaSumFloat3 float3
#endif

#ifdef __cplusplus
extern "C" {
#endif

  // ---------- blas_quda.cu ----------
  
  void zeroQuda(ParitySpinor a);
  void copyQuda(ParitySpinor dst, ParitySpinor src);
  
  double axpyNormQuda(double a, ParitySpinor x, ParitySpinor y);
  double sumQuda(ParitySpinor b);
  double normQuda(ParitySpinor b);
  double reDotProductQuda(ParitySpinor a, ParitySpinor b);
  
  void axpbyQuda(double a, ParitySpinor x, double b, ParitySpinor y);
  void axpyQuda(double a, ParitySpinor x, ParitySpinor y);
  void axQuda(double a, ParitySpinor x);
  void xpayQuda(ParitySpinor x, double a, ParitySpinor y);
  void mxpyQuda(ParitySpinor x, ParitySpinor y);
  
  void axpyZpbxQuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b);

  void caxpbyQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y);
  void caxpyQuda(double2 a, ParitySpinor x, ParitySpinor y);
  void cxpaypbzQuda(ParitySpinor, double2 b, ParitySpinor y, double2 c, ParitySpinor z);
  void caxpbypzYmbwQuda(double2, ParitySpinor, double2, ParitySpinor, ParitySpinor, ParitySpinor);

  cuDoubleComplex cDotProductQuda(ParitySpinor, ParitySpinor);
  cuDoubleComplex xpaycDotzyQuda(ParitySpinor x, double a, ParitySpinor y, ParitySpinor z);

  void blasTest();
  void axpbyTest();
  
  double3 cDotProductNormAQuda(ParitySpinor a, ParitySpinor b);
  double3 cDotProductNormBQuda(ParitySpinor a, ParitySpinor b);
  double3 caxpbypzYmbwcDotProductWYNormYQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y, 
					     ParitySpinor z, ParitySpinor w, ParitySpinor u);
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_BLAS_H
