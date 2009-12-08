#include <cuComplex.h>
#include <enum_quda.h>

#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct double3_s {
    double x;
    double y;
    double z;
  } double3;

  // ---------- blas_quda.cu ----------
  
  void zeroCuda(ParitySpinor a);
  void copyCuda(ParitySpinor dst, ParitySpinor src);
  
  double axpyNormCuda(double a, ParitySpinor x, ParitySpinor y);
  double sumCuda(ParitySpinor b);
  double normCuda(ParitySpinor b);
  double reDotProductCuda(ParitySpinor a, ParitySpinor b);
  double xmyNormCuda(ParitySpinor a, ParitySpinor b);
  
  void axpbyCuda(double a, ParitySpinor x, double b, ParitySpinor y);
  void axpyCuda(double a, ParitySpinor x, ParitySpinor y);
  void axCuda(double a, ParitySpinor x);
  void xpyCuda(ParitySpinor x, ParitySpinor y);
  void xpayCuda(ParitySpinor x, double a, ParitySpinor y);
  void mxpyCuda(ParitySpinor x, ParitySpinor y);
  
  void axpyZpbxCuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b);

  void caxpbyCuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y);
  void caxpyCuda(double2 a, ParitySpinor x, ParitySpinor y);
  void cxpaypbzCuda(ParitySpinor, double2 b, ParitySpinor y, double2 c, ParitySpinor z);
  void caxpbypzYmbwCuda(double2, ParitySpinor, double2, ParitySpinor, ParitySpinor, ParitySpinor);

  cuDoubleComplex cDotProductCuda(ParitySpinor, ParitySpinor);
  cuDoubleComplex xpaycDotzyCuda(ParitySpinor x, double a, ParitySpinor y, ParitySpinor z);

  //  void blasTest();
  //  void axpbyTest();
  
  double3 cDotProductNormACuda(ParitySpinor a, ParitySpinor b);
  double3 cDotProductNormBCuda(ParitySpinor a, ParitySpinor b);
  double3 caxpbypzYmbwcDotProductWYNormYQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y, 
					     ParitySpinor z, ParitySpinor w, ParitySpinor u);

  extern unsigned long long blas_quda_flops;
  extern unsigned long long blas_quda_bytes;

#ifdef __cplusplus
}
#endif

#endif // _QUDA_BLAS_H
