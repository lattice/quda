#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <util_quda.h>

#define REDUCE_THREADS 128
#define REDUCE_MAX_BLOCKS 64

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

inline void checkSpinor(ParitySpinor &a, ParitySpinor &b) {
  if (a.precision == QUDA_HALF_PRECISION || b.precision == QUDA_HALF_PRECISION) {
    printf("checkSpinor error, this kernel does not support QUDA_HALF_PRECISION\n");
    exit(-1);
  }

  if (a.precision != b.precision) {
    printf("checkSpinor error, precisions do not match: %d %d\n", a.precision, b.precision);
    exit(-1);
  }

  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

template <typename Float>
void zeroCuda(Float* dst, int len) {
  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  cudaMemset(dst, 0, len*sizeof(Float));
}

void zeroQuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    zeroCuda((double*)a.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    zeroCuda((float*)a.spinor, a.length);
  } else {
    zeroCuda((short*)a.spinor, a.length);
    zeroCuda((float*)a.spinorNorm, a.length/spinorSiteSize);
  }
}

template <typename Float>
void copyCuda(Float* dst, Float *src, int len) {
  cudaMemcpy(dst, src, len*sizeof(Float), cudaMemcpyDeviceToDevice);
}

void copyQuda(ParitySpinor dst, ParitySpinor src) {
  checkSpinor(dst, src);
  if (dst.precision == QUDA_DOUBLE_PRECISION) copyCuda((double*)dst.spinor, (double*)src.spinor, dst.length);
  else copyCuda((float*)dst.spinor, (float*)src.spinor, src.length);
}


template <typename Float>
__global__ void axpbyKernel(Float a, Float *x, Float b, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] = a*x[i] + b*y[i];
    i += gridSize;
  } 
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpbyQuda(double a, ParitySpinor x, double b, ParitySpinor y) {
  checkSpinor(x, y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) 
    axpbyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, b, (double*)y.spinor, x.length);
  else
    axpbyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float)b, (float*)y.spinor, x.length);
}

template <typename Float>
__global__ void axpyKernel(Float a, Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += a*x[i];
    i += gridSize;
  } 
}

// performs the operation y[i] = a*x[i] + y[i]
void axpyQuda(double a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) 
    axpyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, x.length);
  else
    axpyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
}

template <typename Float>
__global__ void xpayKernel(Float *x, Float a, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] = x[i] + a*y[i];
    i += gridSize;
  } 
}

// performs the operation y[i] = x[i] + a*y[i]
void xpayQuda(ParitySpinor x, double a, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) 
    xpayKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, a, (double*)y.spinor, x.length);
  else
    xpayKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float)a, (float*)y.spinor, x.length);
}

template <typename Float>
__global__ void mxpyKernel(Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] -= x[i];
    i += gridSize;
  } 
}

// performs the operation y[i] -= x[i] (minus x plus y)
void mxpyQuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) 
    mxpyKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  else
    mxpyKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
}

template <typename Float>
__global__ void axKernel(Float a, Float *x, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    x[i] *= a;
    i += gridSize;
  } 
}

// performs the operation x[i] = a*x[i]
void axQuda(double a, ParitySpinor x) {
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) 
    axKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, x.length);
  else
    axKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, x.length);
}

template <typename Float2>
__global__ void caxpyKernel(Float2 a, Float2 *x, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z;
    Z.x = x[i].x;
    Z.y = x[i].y;
    y[i].x += a.x*Z.x - a.y*Z.y;
    y[i].y += a.y*Z.x + a.x*Z.y;
    i += gridSize;
  } 
  
}

// performs the operation y[i] += a*x[i]
void caxpyQuda(double2 a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpyKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, (double2*)y.spinor, length);
  } else {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    caxpyKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, (float2*)y.spinor, length);
  }
}

template <typename Float2>
__global__ void caxpbyKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z1, Z2;
    Z1.x = x[i].x;
    Z2.y = x[i].y;
    Z1.x = y[i].x;
    Z2.y = y[i].y;
    y[i].x = a.x*Z1.x + b.x*Z2.x - a.y*Z1.y - b.y*Z2.y;
    y[i].y = a.y*Z1.x + b.y*Z2.x + a.x*Z1.y + b.x*Z2.y;
    i += gridSize;
  } 
  
}

// performs the operation y[i] = c*x[i] + b*y[i]
void caxbyQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbyKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, length);
  } else {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbyKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, length);
  }
}

template <typename Float2>
__global__ void cxpaypbzKernel(Float2 *x, Float2 a, Float2 *y, Float2 b, Float2 *z, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 T1, T2, T3;
    T1.x = x[i].x;
    T1.y = x[i].y;
    T2.x = y[i].x;
    T2.y = y[i].y;
    T3.x = z[i].x;
    T3.y = z[i].y;
    
    T1.x += a.x*T2.x - a.y*T2.y;
    T1.y += a.y*T2.x + a.x*T2.y;
    T1.x += b.x*T3.x - b.y*T3.y;
    T1.y += b.y*T3.x + b.x*T3.y;
    
    z[i].x = T1.x;
    z[i].y = T1.y;
    i += gridSize;
  } 
  
}

// performs the operation z[i] = x[i] + a*y[i] + b*z[i]
void cxpaypbzQuda(ParitySpinor x, double2 a, ParitySpinor y, double2 b, ParitySpinor z) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    cxpaypbzKernel<<<dimGrid, dimBlock>>>((double2*)x.spinor, a, (double2*)y.spinor, b, (double2*)z.spinor, length);
  } else {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    cxpaypbzKernel<<<dimGrid, dimBlock>>>((float2*)x.spinor, af2, (float2*)y.spinor, bf2, (float2*)z.spinor, length);
  }
}

template <typename Float>
__global__ void axpyZpbxKernel(Float a, Float *x, Float *y, Float *z, Float b, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float x_i = x[i];
    y[i] += a*x_i;
    x[i] = z[i] + b*x_i;
    i += gridSize;
  }
}

// performs the operations: {y[i] = a x[i] + y[i]; x[i] = z[i] + b x[i]}
void axpyZpbxQuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpyZpbxKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, (double*)z.spinor, b, x.length);
  } else {
    axpyZpbxKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, (float*)z.spinor, (float)b, x.length);
  }
}

template <typename Float2>
__global__ void caxpbypzYmbwKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, int len) {

  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 X, Y, Z, W;
    X.x = x[i].x;
    X.y = x[i].y;
    Y.x = y[i].x;
    Y.y = y[i].y;
    W.x = w[i].x;
    W.y = w[i].y;
    
    Z.x = a.x*X.x - a.y*X.y;
    Z.y = a.y*X.x + a.x*X.y;
    Z.x += b.x*Y.x - b.y*Y.y;
    Z.y += b.y*Y.x + b.x*Y.y;
    Y.x -= b.x*W.x - b.y*W.y;
    Y.y -= b.y*W.x + b.x*W.y;	
    
    z[i].x += Z.x;
    z[i].y += Z.y;
    y[i].x = Y.x;
    y[i].y = Y.y;
    i += gridSize;
  } 

}

// performs the operation z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
void caxpbypzYmbwCuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y,
                      ParitySpinor z, ParitySpinor w) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  checkSpinor(x,w);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbypzYmbwKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, 
					  (double2*)z.spinor, (double2*)w.spinor, length); 
  } else {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbypzYmbwKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, 
					  (float2*)z.spinor, (float2*)w.spinor, length); 
  }
}


// Computes c = a + b in "double single" precision.
__device__ void dsadd(QudaSumFloat &c0, QudaSumFloat &c1, const QudaSumFloat a0, 
		      const QudaSumFloat a1, const float b0, const float b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0 + b0;
  QudaSumFloat e = t1 - a0;
  QudaSumFloat t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;
  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (complex version)
__device__ void zcadd(QudaSumComplex &c0, QudaSumComplex &c1, const QudaSumComplex a0, 
		      const QudaSumComplex a1, const QudaSumComplex b0, const QudaSumComplex b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (float3 version)
__device__ void dsadd3(QudaSumFloat3 &c0, QudaSumFloat3 &c1, const QudaSumFloat3 a0, 
		       const QudaSumFloat3 a1, const QudaSumFloat3 b0, const QudaSumFloat3 b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.z + b0.z;
  e = t1 - a0.z;
  t2 = ((b0.z - e) + (a0.z - (t1 - e))) + a1.z + b1.z;
  // The result is t1 + t2, after normalization.
  c0.z = e = t1 + t2;
  c1.z = t2 - (e - t1);
}

//
// double sumCuda(float *a, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) sum##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) a[i]
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double sumQuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return sumCuda((double*)a.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return sumCuda((float*)a.spinor, a.length);
  } else {
    printf("Error, this kernel does not support QUDA_HALF_PRECISION\n");
    exit(-1);
  }
}

//
// double normCuda(float *a, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) norm##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*a[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double normQuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return normCuda((double*)a.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return normCuda((float*)a.spinor, a.length);
  } else {
    printf("Error, this kernel does not support QUDA_HALF_PRECISION\n");
    exit(-1);
  }
}

//
// double reDotProductCuda(float *a, float *b, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProduct##suffix
#define REDUCE_TYPES Float *a, Float *b
#define REDUCE_PARAMS a, b
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*b[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double reDotProductQuda(ParitySpinor a, ParitySpinor b) {
  checkSpinor(a, b);
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return reDotProductCuda((double*)a.spinor, (double*)b.spinor, a.length);
  } else {
    return reDotProductCuda((float*)a.spinor, (float*)b.spinor, a.length);
  }
}

//
// double axpyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = a*x[i] + y[i]
// Second returns the norm of y
//

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNorm##suffix
#define REDUCE_TYPES Float a, Float *x, Float *y
#define REDUCE_PARAMS a, x, y
#define REDUCE_AUXILIARY(i) y[i] = a*x[i] + y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double axpyNormQuda(double a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return axpyNormCuda(a, (double*)x.spinor, (double*)y.spinor, x.length);
  } else {
    return axpyNormCuda((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
  }
}


//
// double2 cDotProductCuda(float2 *a, float2 *b, int n) {}
//
template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProduct##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b, Float c
#define REDUCE_PARAMS a, b, c
#define REDUCE_REAL_AUXILIARY(i)
#define REDUCE_IMAG_AUXILIARY(i)
#define REDUCE_REAL_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_IMAG_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 cDotProductQuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    char c = NULL;
    return cDotProductCuda((double2*)x.spinor, (double2*)y.spinor, c, length);
  } else {
    char c = NULL;
    return cDotProductCuda((float2*)x.spinor, (float2*)y.spinor, c, length);
  }
}

//
// double2 xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int n) {}
//
// First performs the operation y = x - a*y
// Second returns complex dot product (z,y)
//

template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) xpaycDotzy##suffix
#define REDUCE_TYPES Float2 *x, Float a, Float2 *y, Float2 *z
#define REDUCE_PARAMS x, a, y, z
#define REDUCE_REAL_AUXILIARY(i) y[i].x = x[i].x + a*y[i].x
#define REDUCE_IMAG_AUXILIARY(i) y[i].y = x[i].y + a*y[i].y
#define REDUCE_REAL_OPERATION(i) (z[i].x*y[i].x + z[i].y*y[i].y)
#define REDUCE_IMAG_OPERATION(i) (z[i].x*y[i].y - z[i].y*y[i].x)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 xpayDotzyQuda(ParitySpinor x, double a, ParitySpinor y, ParitySpinor z) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return xpaycDotzyCuda((double2*)x.spinor, a, (double2*)y.spinor, (double2*)z.spinor, length);
  } else {
    return xpaycDotzyCuda((float2*)x.spinor, (float)a, (float2*)y.spinor, (float2*)z.spinor, length);
  }
}


//
// double3 cDotProductNormACuda(float2 *a, float2 *b, int n) {}
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormA##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b
#define REDUCE_PARAMS a, b
#define REDUCE_X_AUXILIARY(i)
#define REDUCE_Y_AUXILIARY(i)
#define REDUCE_Z_AUXILIARY(i)
#define REDUCE_X_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_Y_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#define REDUCE_Z_OPERATION(i) (a[i].x*a[i].x + a[i].y*a[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

double3 cDotProductNormAQuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormACuda((double2*)x.spinor, (double2*)y.spinor, length);
  } else {
    return cDotProductNormACuda((float2*)x.spinor, (float2*)y.spinor, length);
  }
}

//
// double3 cDotProductNormBCuda(float2 *a, float2 *b, int n) {}
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormB##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b
#define REDUCE_PARAMS a, b
#define REDUCE_X_AUXILIARY(i)
#define REDUCE_Y_AUXILIARY(i)
#define REDUCE_Z_AUXILIARY(i)
#define REDUCE_X_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_Y_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#define REDUCE_Z_OPERATION(i) (b[i].x*b[i].x + b[i].y*b[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

double3 cDotProductNormBQuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormBCuda((double2*)x.spinor, (double2*)y.spinor, length);
  } else {
    return cDotProductNormBCuda((float2*)x.spinor, (float2*)y.spinor, length);
  }
}


//
// double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormY##suffix
#define REDUCE_TYPES Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, Float2 *u
#define REDUCE_PARAMS a, x, b, y, z, w, u
#define REDUCE_X_AUXILIARY(i) \
  Float2 W, X, Y;	      \
  X.x = x[i].x;		      \
  X.y = x[i].y;		      \
  Y.x = y[i].x;		      \
  Y.y = y[i].y;		      \
  W.x = w[i].x;		      \
  W.y = w[i].y;
#define REDUCE_Y_AUXILIARY(i)	    \
  Float2 Z;			    \
  Z.x = a.x*X.x - a.y*X.y;	    \
  Z.y = a.y*X.x + a.x*X.y;	    \
  Z.x += b.x*Y.x - b.y*Y.y;	    \
  Z.y += b.y*Y.x + b.x*Y.y;	    \
  Y.x -= b.x*W.x - b.y*W.y;	    \
  Y.y -= b.y*W.x + b.x*W.y;	
#define REDUCE_Z_AUXILIARY(i)	      \
  z[i].x += Z.x;		      \
  z[i].y += Z.y;		      \
  y[i].x = Y.x;			      \
  y[i].y = Y.y;
#define REDUCE_X_OPERATION(i) (u[i].x*y[i].x + u[i].y*y[i].y)
#define REDUCE_Y_OPERATION(i) (u[i].x*y[i].y - u[i].y*y[i].x)
#define REDUCE_Z_OPERATION(i) (y[i].x*y[i].x + y[i].y*y[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

double3 caxpbypzYmbwcDotProductWYNormYQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y,
					   ParitySpinor z, ParitySpinor w, ParitySpinor u) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  checkSpinor(x,w);
  checkSpinor(x,u);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return caxpbypzYmbwcDotProductWYNormYCuda(a, (double2*)x.spinor, b, (double2*)y.spinor, (double2*)z.spinor, 
					      (double2*)w.spinor, (double2*)u.spinor, length);
  } else {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    return caxpbypzYmbwcDotProductWYNormYCuda(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, (float2*)z.spinor,
					      (float2*)w.spinor, (float2*)u.spinor, length);
  }
}



double cpuDouble(float *data, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++)
        sum += data[i];
    return sum;
}

void blasTest() {
    int n = 3*1<<24;
    float *h_data = (float *)malloc(n*sizeof(float));
    float *d_data;
    if (cudaMalloc((void **)&d_data,  n*sizeof(float))) {
      printf("Error allocating d_data\n");
      exit(0);
    }
    
    for (int i = 0; i < n; i++) {
        h_data[i] = rand()/(float)RAND_MAX - 0.5; // n-1.0-i;
    }
    
    cudaMemcpy(d_data, h_data, n*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaThreadSynchronize();
    stopwatchStart();
    int LOOPS = 20;
    for (int i = 0; i < LOOPS; i++) {
        sumCuda(d_data, n);
    }
    cudaThreadSynchronize();
    float secs = stopwatchReadSeconds();
    
    printf("%f GiB/s\n", 1.e-9*n*sizeof(float)*LOOPS / secs);
    printf("Device: %f MiB\n", (float)n*sizeof(float) / (1 << 20));
    printf("Shared: %f KiB\n", (float)REDUCE_THREADS*sizeof(float) / (1 << 10));

    float correctDouble = cpuDouble(h_data, n);
    printf("Total: %f\n", correctDouble);
    printf("CUDA db: %f\n", fabs(correctDouble-sumCuda(d_data, n)));
    
    cudaFree(d_data) ;
    free(h_data);
}

/*
void axpbyTest() {
    int n = 3 * 1 << 20;
    float *h_x = (float *)malloc(n*sizeof(float));
    float *h_y = (float *)malloc(n*sizeof(float));
    float *h_res = (float *)malloc(n*sizeof(float));
    
    float *d_x, *d_y;
    if (cudaMalloc((void **)&d_x,  n*sizeof(float))) {
      printf("Error allocating d_x\n");
      exit(0);
    }
    if (cudaMalloc((void **)&d_y,  n*sizeof(float))) {
      printf("Error allocating d_y\n");
      exit(0);
    }
    
    for (int i = 0; i < n; i++) {
        h_x[i] = 1;
        h_y[i] = 2;
    }
    
    cudaMemcpy(d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice);
    
    axpbyCuda(4, d_x, 3, d_y, n/2);
    
    cudaMemcpy( h_res, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float expect = (i < n/2) ? 4*h_x[i] + 3*h_y[i] : h_y[i];
        if (h_res[i] != expect)
            printf("FAILED %d : %f != %f\n", i, h_res[i], h_y[i]);
    }
    
    cudaFree(d_y);
    cudaFree(d_x);
    free(h_x);
    free(h_y);
}
*/
