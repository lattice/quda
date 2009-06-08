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

// ---------- blas_cuda.cu ----------

void zeroCuda(float* dst, int cnt);
void copyCuda(float* dst, float *src, int len);

void axpbyCuda(float a, float *x, float b, float *y, int len);
void axpyCuda(float a, float *x, float *y, int len);
void axCuda(float a, float *x, int len);
void xpayCuda(float *x, float a, float *y, int len);
void mxpyCuda(float *x, float *y, int len);

void axpyZpbxCuda(float a, float *x, float *y, float *z, float b, int len);
double axpyNormCuda(float a, float *x, float *y, int len);

double sumCuda(float *a, int n);
double normCuda(float *a, int n);
double reDotProductCuda(float *a, float *b, int n);

void blasTest();
void axpbyTest();

void caxpbyCuda(float2 a, float2 *x, float2 b, float2 *y, int len);
void caxpyCuda(float2 a, float2 *x, float2 *y, int len);
void cxpaypbzCuda(float2 *x, float2 b, float2 *y, float2 c, float2 *z, int len);
cuDoubleComplex cDotProductCuda(float2*, float2*, int len);
void caxpbypzYmbwCuda(float2, float2*, float2, float2*, float2*, float2*, int len);
double3 cDotProductNormACuda(float2 *a, float2 *b, int n);
double3 cDotProductNormBCuda(float2 *a, float2 *b, int n);
double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
						 float2 *z, float2 *w, float2 *u, int len);

cuDoubleComplex xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int len);

// ---------- blas_reference.cpp ----------

void zero(float* a, int cnt);
void copy(float* a, float *b, int len);

void ax(float a, float *x, int len);

void axpbyCuda(float a, float *x, float b, float *y, int len);
void axpy(float a, float *x, float *y, int len);
void xpay(float *x, float a, float *y, int len);
void mxpy(float *x, float *y, int len);

float norm(float *vector, int len);
float reDotProduct(float *v1, float *v2, int len);
float imDotProduct(float *v1, float *v2, int len);
double normD(float *vector, int len);
double reDotProductD(float *v1, float *v2, int len);
double imDotProductD(float *v1, float *v2, int len);


#ifdef __cplusplus
}
#endif

#endif // _QUDA_BLAS_H
