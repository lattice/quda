#include <stdlib.h>
#include <stdio.h>

#include <quda_internal.h>

#define REDUCE_MAX_BLOCKS 2048

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

// These are used for reduction kernels
QudaSumFloat *d_reduceFloat=0;
QudaSumComplex *d_reduceComplex=0;
QudaSumFloat3 *d_reduceFloat3=0;

QudaSumFloat *h_reduceFloat=0;
QudaSumComplex *h_reduceComplex=0;
QudaSumFloat3 *h_reduceFloat3=0;

unsigned long long blas_quda_flops;
unsigned long long blas_quda_bytes;

// Number of threads used for each blas kernel
int blas_threads[3][22];
// Number of thread blocks for each blas kernel
int blas_blocks[3][22];

dim3 blasBlock;
dim3 blasGrid;

void initBlas() {
  
  if (!d_reduceFloat) {
    if (cudaMalloc((void**) &d_reduceFloat, REDUCE_MAX_BLOCKS*sizeof(QudaSumFloat)) == cudaErrorMemoryAllocation) {
      printf("Error allocating device reduction array\n");
      exit(0);
    }
  }

  if (!d_reduceComplex) {
    if (cudaMalloc((void**) &d_reduceComplex, REDUCE_MAX_BLOCKS*sizeof(QudaSumComplex)) == cudaErrorMemoryAllocation) {
      printf("Error allocating device reduction array\n");
      exit(0);
    }
  }
  
  if (!d_reduceFloat3) {
    if (cudaMalloc((void**) &d_reduceFloat3, REDUCE_MAX_BLOCKS*sizeof(QudaSumFloat3)) == cudaErrorMemoryAllocation) {
      printf("Error allocating device reduction array\n");
      exit(0);
    }
  }

  if (!h_reduceFloat) {
    if (cudaMallocHost((void**) &h_reduceFloat, REDUCE_MAX_BLOCKS*sizeof(QudaSumFloat)) == cudaErrorMemoryAllocation) {
      printf("Error allocating host reduction array\n");
      exit(0);
    }
  }

  if (!h_reduceComplex) {
    if (cudaMallocHost((void**) &h_reduceComplex, REDUCE_MAX_BLOCKS*sizeof(QudaSumComplex)) == cudaErrorMemoryAllocation) {
      printf("Error allocating host reduction array\n");
      exit(0);
    }
  }
  
  if (!h_reduceFloat3) {
    if (cudaMallocHost((void**) &h_reduceFloat3, REDUCE_MAX_BLOCKS*sizeof(QudaSumFloat3)) == cudaErrorMemoryAllocation) {
      printf("Error allocating host reduction array\n");
      exit(0);
    }
  }

  // Output from blas_test
#include<blas_param.h>

}

void endBlas() {
  if (d_reduceFloat) cudaFree(d_reduceFloat);
  if (d_reduceComplex) cudaFree(d_reduceComplex);
  if (d_reduceFloat3) cudaFree(d_reduceFloat3);
  if (h_reduceFloat) cudaFreeHost(h_reduceFloat);
  if (h_reduceComplex) cudaFreeHost(h_reduceComplex);
  if (h_reduceFloat3) cudaFreeHost(h_reduceFloat3);
}

void setBlock(int kernel, int length, QudaPrecision precision) {
  int prec;
  switch(precision) {
  case QUDA_HALF_PRECISION:
    prec = 0;
    break;
  case QUDA_SINGLE_PRECISION:
    prec = 1;
    break;
  case QUDA_DOUBLE_PRECISION:
    prec = 2;
    break;
  }

  int blocks = min(blas_blocks[prec][kernel], max(length/blas_threads[prec][kernel], 1));
  blasBlock.x = blas_threads[prec][kernel];
  blasBlock.y = 1;
  blasBlock.z = 1;

  blasGrid.x = blocks;
  blasGrid.y = 1;
  blasGrid.z = 1;
}

#if (__CUDA_ARCH__ == 130)
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#else
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  // do nothing
  return make_double2(0.0, 0.0);
}
#endif

#define RECONSTRUCT_HALF_SPINOR(a, texHalf, texNorm, length)		\
  float4 a##0 = tex1Dfetch(texHalf, i + 0*length);			\
  float4 a##1 = tex1Dfetch(texHalf, i + 1*length);			\
  float4 a##2 = tex1Dfetch(texHalf, i + 2*length);			\
  float4 a##3 = tex1Dfetch(texHalf, i + 3*length);			\
  float4 a##4 = tex1Dfetch(texHalf, i + 4*length);			\
  float4 a##5 = tex1Dfetch(texHalf, i + 5*length);			\
  {float b = tex1Dfetch(texNorm, i);					\
  (a##0).x *= b; (a##0).y *= b; (a##0).z *= b; (a##0).w *= b;		\
  (a##1).x *= b; (a##1).y *= b; (a##1).z *= b; (a##1).w *= b;		\
  (a##2).x *= b; (a##2).y *= b; (a##2).z *= b; (a##2).w *= b;		\
  (a##3).x *= b; (a##3).y *= b; (a##3).z *= b; (a##3).w *= b;		\
  (a##4).x *= b; (a##4).y *= b; (a##4).z *= b; (a##4).w *= b;		\
  (a##5).x *= b; (a##5).y *= b; (a##5).z *= b; (a##5).w *= b;}

#define CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, n, a, length)		\
  {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));			\
  float c1 = fmaxf(fabsf((a##0).z), fabsf((a##0).w));			\
  float c2 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));			\
  float c3 = fmaxf(fabsf((a##1).z), fabsf((a##1).w));			\
  float c4 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));			\
  float c5 = fmaxf(fabsf((a##2).z), fabsf((a##2).w));			\
  float c6 = fmaxf(fabsf((a##3).x), fabsf((a##3).y));			\
  float c7 = fmaxf(fabsf((a##3).z), fabsf((a##3).w));			\
  float c8 = fmaxf(fabsf((a##4).x), fabsf((a##4).y));			\
  float c9 = fmaxf(fabsf((a##4).z), fabsf((a##4).w));			\
  float c10 = fmaxf(fabsf((a##5).x), fabsf((a##5).y));			\
  float c11 = fmaxf(fabsf((a##5).z), fabsf((a##5).w));			\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); c2 = fmaxf(c4, c5);		\
  c3 = fmaxf(c6, c7); c4 = fmaxf(c8, c9); c5 = fmaxf(c10, c11);		\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); c2 = fmaxf(c4, c5);		\
  c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);				\
  n[i] = c0;								\
  float C = __fdividef(MAX_SHORT, c0);					\
  h[i+0*length] = make_short4((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y), \
			      (short)(C*(float)(a##0).z), (short)(C*(float)(a##0).w)); \
  h[i+1*length] = make_short4((short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y), \
			      (short)(C*(float)(a##1).z), (short)(C*(float)(a##1).w)); \
  h[i+2*length] = make_short4((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y), \
			      (short)(C*(float)(a##2).z), (short)(C*(float)(a##2).w)); \
  h[i+3*length] = make_short4((short)(C*(float)(a##3).x), (short)(C*(float)(a##3).y), \
			      (short)(C*(float)(a##3).z), (short)(C*(float)(a##3).w)); \
  h[i+4*length] = make_short4((short)(C*(float)(a##4).x), (short)(C*(float)(a##4).y), \
			      (short)(C*(float)(a##4).z), (short)(C*(float)(a##4).w)); \
  h[i+5*length] = make_short4((short)(C*(float)(a##5).x), (short)(C*(float)(a##5).y),	\
			      (short)(C*(float)(a##5).z), (short)(C*(float)(a##5).w));}

#define CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, n, a, length)		\
  {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));			\
  float c1 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));		     	\
  float c2 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));			\
  float c3 = fmaxf(fabsf((a##3).x), fabsf((a##3).y));			\
  float c4 = fmaxf(fabsf((a##4).x), fabsf((a##4).y));			\
  float c5 = fmaxf(fabsf((a##5).x), fabsf((a##5).y));			\
  float c6 = fmaxf(fabsf((a##6).x), fabsf((a##6).y));			\
  float c7 = fmaxf(fabsf((a##7).x), fabsf((a##7).y));			\
  float c8 = fmaxf(fabsf((a##8).x), fabsf((a##8).y));			\
  float c9 = fmaxf(fabsf((a##9).x), fabsf((a##9).y));			\
  float c10 = fmaxf(fabsf((a##10).x), fabsf((a##10).y));		\
  float c11 = fmaxf(fabsf((a##11).x), fabsf((a##11).y));		\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3);  c2 = fmaxf(c4, c5); c3 = fmaxf(c6, c7); \
  c4 = fmaxf(c8, c9); c5 = fmaxf(c10, c11); c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); \
  c2 = fmaxf(c4, c5); c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);		\
  n[i] = c0;								\
  float C = __fdividef(MAX_SHORT, c0);					\
  h[i+0*length] = make_short4((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y), \
			      (short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y)); \
  h[i+1*length] = make_short4((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y), \
			      (short)(C*(float)(a##3).x), (short)(C*(float)(a##3).y)); \
  h[i+2*length] = make_short4((short)(C*(float)(a##4).x), (short)(C*(float)(a##4).y), \
			      (short)(C*(float)(a##5).x), (short)(C*(float)(a##5).y)); \
  h[i+3*length] = make_short4((short)(C*(float)(a##6).x), (short)(C*(float)(a##6).y), \
			      (short)(C*(float)(a##7).x), (short)(C*(float)(a##7).y)); \
  h[i+4*length] = make_short4((short)(C*(float)(a##8).x), (short)(C*(float)(a##8).y), \
			      (short)(C*(float)(a##9).x), (short)(C*(float)(a##9).y)); \
  h[i+5*length] = make_short4((short)(C*(float)(a##10).x), (short)(C*(float)(a##10).y),	\
			      (short)(C*(float)(a##11).x), (short)(C*(float)(a##11).y));}

#define SUM_FLOAT4(sum, a)			\
  float sum = a.x + a.y + a.z + a.w;

#define REAL_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w

#define IMAG_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z

#define AX_FLOAT4(a, X)				\
  X.x *= a; X.y *= a; X.z *= a; X.w *= a;

#define XPY_FLOAT4(X, Y)		     \
  Y.x += X.x; Y.y += X.y; Y.z += X.z; Y.w += X.w;

#define XMY_FLOAT4(X, Y)		     \
  Y.x = X.x - Y.x; Y.y = X.y - X.y; Y.z = X.z - Y.z; Y.w = X.w - Y.w;

#define MXPY_FLOAT4(X, Y)		     \
  Y.x -= X.x; Y.y -= X.y; Y.z -= X.z; Y.w -= X.w;

#define AXPY_FLOAT4(a, X, Y)		     \
  Y.x += a*X.x;	Y.y += a*X.y;		     \
  Y.z += a*X.z;	Y.w += a*X.w;

#define AXPBY_FLOAT4(a, X, b, Y)		\
  Y.x = a*X.x + b*Y.x; Y.y = a*X.y + b*Y.y;	\
  Y.z = a*X.z + b*Y.z; Y.w = a*X.w + b*Y.w;

#define XPAY_FLOAT4(X, a, Y)			     \
  Y.x = X.x + a*Y.x; Y.y = X.y + a*Y.y;		     \
  Y.z = X.z + a*Y.z; Y.w = X.w + a*Y.w;

#define CAXPY_FLOAT4(a, X, Y) \
  Y.x += a.x*X.x - a.y*X.y;   \
  Y.y += a.y*X.x + a.x*X.y;   \
  Y.z += a.x*X.z - a.y*X.w;   \
  Y.w += a.y*X.z + a.x*X.w;

#define CMAXPY_FLOAT4(a, X, Y)			\
  Y.x -= (a.x*X.x - a.y*X.y);			\
  Y.y -= (a.y*X.x + a.x*X.y);			\
  Y.z -= (a.x*X.z - a.y*X.w);			\
  Y.w -= (a.y*X.z + a.x*X.w);

#define CAXPBY_FLOAT4(a, X, b, Y)		 \
  Y.x = a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Y.y = a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Y.z = a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Y.w = a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

#define CXPAYPBZ_FLOAT4(X, a, Y, b, Z)		\
  {float2 z;					       \
  z.x = X.x + a.x*Y.x - a.y*Y.y + b.x*Z.x - b.y*Z.y;   \
  z.y = X.y + a.y*Y.x + a.x*Y.y + b.y*Z.x + b.x*Z.y;   \
  Z.x = z.x; Z.y = z.y;				       \
  z.x = X.z + a.x*Y.z - a.y*Y.w + b.x*Z.z - b.y*Z.w;   \
  z.y = X.w + a.y*Y.z + a.x*Y.w + b.y*Z.z + b.x*Z.w;   \
  Z.z = z.x; Z.w = z.y;}

#define CAXPBYPZ_FLOAT4(a, X, b, Y, Z)		  \
  Z.x += a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Z.y += a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Z.z += a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Z.w += a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf1;
texture<float, 1, cudaReadModeElementType> texNorm1;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf2;
texture<float, 1, cudaReadModeElementType> texNorm2;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf3;
texture<float, 1, cudaReadModeElementType> texNorm3;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf4;
texture<float, 1, cudaReadModeElementType> texNorm4;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf5;
texture<float, 1, cudaReadModeElementType> texNorm5;

inline void checkSpinor(ParitySpinor &a, ParitySpinor &b) {
  if (a.precision != b.precision) {
    printf("checkSpinor error, precisions do not match: %d %d\n", a.precision, b.precision);
    exit(-1);
  }

  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// For kernels with precision conversion built in
inline void checkSpinorLength(ParitySpinor &a, ParitySpinor &b) {
  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// cuda's floating point format, IEEE-754, represents the floating point
// zero as 4 zero bytes
void zeroCuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(double));
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(float));
  } else {
    cudaMemset(a.spinor, 0, a.length*sizeof(short));
    cudaMemset(a.spinorNorm, 0, a.length*sizeof(float)/(a.Nc*a.Ns*2));
  }
}

__global__ void convertDSKernel(double2 *dst, float4 *src, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    for (int k=0; k<6; k++) {
      dst[2*k*length+i].x = src[k*length+i].x;
      dst[2*k*length+i].y = src[k*length+i].y;
      dst[(2*k+1)*length+i].x = src[k*length+i].z;
      dst[(2*k+1)*length+i].y = src[k*length+i].w;
    }
    i += gridSize;
  }   
}

__global__ void convertSDKernel(float4 *dst, double2 *src, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    for (int k=0; k<6; k++) {
      dst[k*length+i].x = src[2*k*length+i].x;
      dst[k*length+i].y = src[2*k*length+i].y;
      dst[k*length+i].z = src[(2*k+1)*length+i].x;
      dst[k*length+i].w = src[(2*k+1)*length+i].y;
    }
    i += gridSize;
  }   
}

__global__ void convertHSKernel(short4 *h, float *norm, int length, int real_length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < real_length) {
    float4 F0 = tex1Dfetch(spinorTexSingle, i + 0*length);
    float4 F1 = tex1Dfetch(spinorTexSingle, i + 1*length);
    float4 F2 = tex1Dfetch(spinorTexSingle, i + 2*length);
    float4 F3 = tex1Dfetch(spinorTexSingle, i + 3*length);
    float4 F4 = tex1Dfetch(spinorTexSingle, i + 4*length);
    float4 F5 = tex1Dfetch(spinorTexSingle, i + 5*length);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, norm, F, length);
    i += gridSize;
  }

}

__global__ void convertSHKernel(float4 *res, int length, int real_length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  while (i<real_length) {
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
    res[0*length+i] = I0;
    res[1*length+i] = I1;
    res[2*length+i] = I2;
    res[3*length+i] = I3;
    res[4*length+i] = I4;
    res[5*length+i] = I5;
    i += gridSize;
  }
}

__global__ void convertHDKernel(short4 *h, float *norm, int length, int real_length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < real_length) {
    double2 F0 = fetch_double2(spinorTexDouble, i+0*length);
    double2 F1 = fetch_double2(spinorTexDouble, i+1*length);
    double2 F2 = fetch_double2(spinorTexDouble, i+2*length);
    double2 F3 = fetch_double2(spinorTexDouble, i+3*length);
    double2 F4 = fetch_double2(spinorTexDouble, i+4*length);
    double2 F5 = fetch_double2(spinorTexDouble, i+5*length);
    double2 F6 = fetch_double2(spinorTexDouble, i+6*length);
    double2 F7 = fetch_double2(spinorTexDouble, i+7*length);
    double2 F8 = fetch_double2(spinorTexDouble, i+8*length);
    double2 F9 = fetch_double2(spinorTexDouble, i+9*length);
    double2 F10 = fetch_double2(spinorTexDouble, i+10*length);
    double2 F11 = fetch_double2(spinorTexDouble, i+11*length);
    CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, norm, F, length);
    i += gridSize;
  }
}

__global__ void convertDHKernel(double2 *res, int length, int real_length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < real_length) {
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
    res[0*length+i] = make_double2(I0.x, I0.y);
    res[1*length+i] = make_double2(I0.z, I0.w);
    res[2*length+i] = make_double2(I1.x, I1.y);
    res[3*length+i] = make_double2(I1.z, I1.w);
    res[4*length+i] = make_double2(I2.x, I2.y);
    res[5*length+i] = make_double2(I2.z, I2.w);
    res[6*length+i] = make_double2(I3.x, I3.y);
    res[7*length+i] = make_double2(I3.z, I3.w);
    res[8*length+i] = make_double2(I4.x, I4.y);
    res[9*length+i] = make_double2(I4.z, I4.w);
    res[10*length+i] = make_double2(I5.x, I5.y);
    res[11*length+i] = make_double2(I5.z, I5.w);
    i += gridSize;
  }

}

void copyCuda(ParitySpinor dst, ParitySpinor src) {
  checkSpinorLength(dst, src);

  setBlock(0, dst.stride, dst.precision);

  blas_quda_bytes += src.real_length*(src.precision + dst.precision);

  if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
    convertDSKernel<<<blasGrid, blasBlock>>>((double2*)dst.spinor, (float4*)src.spinor, src.stride);
  } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
    convertSDKernel<<<blasGrid, blasBlock>>>((float4*)dst.spinor, (double2*)src.spinor, src.stride);
  } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = dst.length*sizeof(short);
    cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, src.spinorNorm, spinor_bytes/12);
    convertSHKernel<<<blasGrid, blasBlock>>>((float4*)dst.spinor, src.stride, src.volume);
  } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
    int spinor_bytes = dst.length*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, src.spinor, spinor_bytes); 
    convertHSKernel<<<blasGrid, blasBlock>>>((short4*)dst.spinor, (float*)dst.spinorNorm, src.stride, src.volume);
  } else if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = dst.length*sizeof(short);
    cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, src.spinorNorm, spinor_bytes/12);
    convertDHKernel<<<blasGrid, blasBlock>>>((double2*)dst.spinor, src.stride, src.volume);
  } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
    int spinor_bytes = dst.length*sizeof(double);
    cudaBindTexture(0, spinorTexDouble, src.spinor, spinor_bytes); 
    convertHDKernel<<<blasGrid, blasBlock>>>((short4*)dst.spinor, (float*)dst.spinorNorm, src.stride, src.volume);
  } else if (dst.precision == QUDA_DOUBLE_PRECISION) {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(double), cudaMemcpyDeviceToDevice);
  } else if (dst.precision == QUDA_SINGLE_PRECISION) {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(float), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(short), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst.spinorNorm, src.spinorNorm, dst.length*sizeof(float)/(dst.Nc*dst.Ns*2), cudaMemcpyDeviceToDevice);
  }
}


template <typename Float>
__global__ void axpbyKernel(Float a, Float *x, Float b, Float *y, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    y[i] = a*x[i] + b*y[i];
    i += gridSize;
  } 
}

__global__ void axpbyHKernel(float a, float b, short4 *yH, float *yN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    AXPBY_FLOAT4(a, x0, b, y0);
    AXPBY_FLOAT4(a, x1, b, y1);
    AXPBY_FLOAT4(a, x2, b, y2);
    AXPBY_FLOAT4(a, x3, b, y3);
    AXPBY_FLOAT4(a, x4, b, y4);
    AXPBY_FLOAT4(a, x5, b, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpbyCuda(double a, ParitySpinor x, double b, ParitySpinor y) {
  setBlock(1, x.length, x.precision);
  checkSpinor(x, y);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpbyKernel<<<blasGrid, blasBlock>>>(a, (double*)x.spinor, b, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpbyKernel<<<blasGrid, blasBlock>>>((float)a, (float*)x.spinor, (float)b, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    axpbyHKernel<<<blasGrid, blasBlock>>>((float)a, (float)b, (short4*)y.spinor, 
					(float*)y.spinorNorm, y.stride, y.volume);
  }
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += 3*x.real_length;
}

template <typename Float>
__global__ void xpyKernel(Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += x[i];
    i += gridSize;
  } 
}

__global__ void xpyHKernel(short4 *yH, float *yN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    XPY_FLOAT4(x0, y0);
    XPY_FLOAT4(x1, y1);
    XPY_FLOAT4(x2, y2);
    XPY_FLOAT4(x3, y3);
    XPY_FLOAT4(x4, y4);
    XPY_FLOAT4(x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = x[i] + y[i]
void xpyCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  setBlock(2, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    xpyKernel<<<blasGrid, blasBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    xpyKernel<<<blasGrid, blasBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    xpyHKernel<<<blasGrid, blasBlock>>>((short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += x.real_length;
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

__global__ void axpyHKernel(float a, short4 *yH, float *yN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    AXPY_FLOAT4(a, x0, y0);
    AXPY_FLOAT4(a, x1, y1);
    AXPY_FLOAT4(a, x2, y2);
    AXPY_FLOAT4(a, x3, y3);
    AXPY_FLOAT4(a, x4, y4);
    AXPY_FLOAT4(a, x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = a*x[i] + y[i]
void axpyCuda(double a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  setBlock(3, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpyKernel<<<blasGrid, blasBlock>>>(a, (double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpyKernel<<<blasGrid, blasBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    axpyHKernel<<<blasGrid, blasBlock>>>((float)a, (short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += 2*x.real_length;
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

__global__ void xpayHKernel(float a, short4 *yH, float *yN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    XPAY_FLOAT4(x0, a, y0);
    XPAY_FLOAT4(x1, a, y1);
    XPAY_FLOAT4(x2, a, y2);
    XPAY_FLOAT4(x3, a, y3);
    XPAY_FLOAT4(x4, a, y4);
    XPAY_FLOAT4(x5, a, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = x[i] + a*y[i]
void xpayCuda(ParitySpinor x, double a, ParitySpinor y) {
  checkSpinor(x,y);
  setBlock(4, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    xpayKernel<<<blasGrid, blasBlock>>>((double*)x.spinor, a, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    xpayKernel<<<blasGrid, blasBlock>>>((float*)x.spinor, (float)a, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    xpayHKernel<<<blasGrid, blasBlock>>>((float)a, (short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += 2*x.real_length;
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

__global__ void mxpyHKernel(short4 *yH, float *yN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    MXPY_FLOAT4(x0, y0);
    MXPY_FLOAT4(x1, y1);
    MXPY_FLOAT4(x2, y2);
    MXPY_FLOAT4(x3, y3);
    MXPY_FLOAT4(x4, y4);
    MXPY_FLOAT4(x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}


// performs the operation y[i] -= x[i] (minus x plus y)
void mxpyCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  setBlock(5, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    mxpyKernel<<<blasGrid, blasBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    mxpyKernel<<<blasGrid, blasBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    mxpyHKernel<<<blasGrid, blasBlock>>>((short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += x.real_length;
}

float2 __device__ make_Float2(float x, float y) {
  return make_float2(x, y);
}

double2 __device__ make_Float2(double x, double y) {
  return make_double2(x, y);
}

template <typename Float, typename Float2>
__global__ void axKernel(Float a, Float2 *x, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    x[i].x *= a;
    x[i].y *= a;
    i += gridSize;
  } 
}

__global__ void axHKernel(float a, short4 *xH, float *xN, int stride, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    AX_FLOAT4(a, x0); AX_FLOAT4(a, x1); AX_FLOAT4(a, x2);
    AX_FLOAT4(a, x3); AX_FLOAT4(a, x4); AX_FLOAT4(a, x5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, stride);
    i += gridSize;
  } 
  
}

// performs the operation x[i] = a*x[i]
void axCuda(double a, ParitySpinor x) {
  setBlock(6, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axKernel<<<blasGrid, blasBlock>>>(a, (double2*)x.spinor, x.length/2);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axKernel<<<blasGrid, blasBlock>>>((float)a, (float2*)x.spinor, x.length/2);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    axHKernel<<<blasGrid, blasBlock>>>((float)a, (short4*)x.spinor, (float*)x.spinorNorm, x.stride, x.volume);
  }
  blas_quda_bytes += 2*x.real_length*x.precision;
  blas_quda_flops += x.real_length;
}

template <typename Float2>
__global__ void caxpyKernel(Float2 a, Float2 *x, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z = make_Float2(x[i].x, x[i].y);
    y[i].x += a.x*Z.x - a.y*Z.y;
    y[i].y += a.y*Z.x + a.x*Z.y;
    i += gridSize;
  } 
  
}

__global__ void caxpyHKernel(float2 a, short4 *yH, float *yN, int stride, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    CAXPY_FLOAT4(a, x0, y0);
    CAXPY_FLOAT4(a, x1, y1);
    CAXPY_FLOAT4(a, x2, y2);
    CAXPY_FLOAT4(a, x3, y3);
    CAXPY_FLOAT4(a, x4, y4);
    CAXPY_FLOAT4(a, x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  } 
  
}

// performs the operation y[i] += a*x[i]
void caxpyCuda(double2 a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  setBlock(7, length, x.precision);
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += 4*x.real_length;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpyKernel<<<blasGrid, blasBlock>>>(a, (double2*)x.spinor, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    caxpyKernel<<<blasGrid, blasBlock>>>(af2, (float2*)x.spinor, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    caxpyHKernel<<<blasGrid, blasBlock>>>(af2, (short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
}

template <typename Float2>
__global__ void caxpbyKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z1 = make_Float2(x[i].x, x[i].y);
    Float2 Z2 = make_Float2(y[i].x, y[i].y);
    y[i].x = a.x*Z1.x + b.x*Z2.x - a.y*Z1.y - b.y*Z2.y;
    y[i].y = a.y*Z1.x + b.y*Z2.x + a.x*Z1.y + b.x*Z2.y;
    i += gridSize;
  } 
}

__global__ void caxpbyHKernel(float2 a, float2 b, short4 *yH, float *yN, int stride, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    CAXPBY_FLOAT4(a, x0, b, y0);
    CAXPBY_FLOAT4(a, x1, b, y1);
    CAXPBY_FLOAT4(a, x2, b, y2);
    CAXPBY_FLOAT4(a, x3, b, y3);
    CAXPBY_FLOAT4(a, x4, b, y4);
    CAXPBY_FLOAT4(a, x5, b, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    i += gridSize;
  }   
}


// performs the operation y[i] = c*x[i] + b*y[i]
void caxpbyCuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  setBlock(8, length, x.precision);
  blas_quda_bytes += 3*x.real_length*x.precision;
  blas_quda_flops += 7*x.real_length;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbyKernel<<<blasGrid, blasBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbyKernel<<<blasGrid, blasBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbyHKernel<<<blasGrid, blasBlock>>>(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume);
  }
}

template <typename Float2>
__global__ void cxpaypbzKernel(Float2 *x, Float2 a, Float2 *y, Float2 b, Float2 *z, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 T1 = make_Float2(x[i].x, x[i].y);
    Float2 T2 = make_Float2(y[i].x, y[i].y);
    Float2 T3 = make_Float2(z[i].x, z[i].y);

    T1.x += a.x*T2.x - a.y*T2.y;
    T1.y += a.y*T2.x + a.x*T2.y;
    T1.x += b.x*T3.x - b.y*T3.y;
    T1.y += b.y*T3.x + b.x*T3.y;
    
    z[i] = make_Float2(T1.x, T1.y);
    i += gridSize;
  } 
  
}

__global__ void cxpaypbzHKernel(float2 a, float2 b, short4 *zH, float *zN, int stride, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, stride);
    CXPAYPBZ_FLOAT4(x0, a, y0, b, z0);
    CXPAYPBZ_FLOAT4(x1, a, y1, b, z1);
    CXPAYPBZ_FLOAT4(x2, a, y2, b, z2);
    CXPAYPBZ_FLOAT4(x3, a, y3, b, z3);
    CXPAYPBZ_FLOAT4(x4, a, y4, b, z4);
    CXPAYPBZ_FLOAT4(x5, a, y5, b, z5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, stride);
    i += gridSize;
  }   
}


// performs the operation z[i] = x[i] + a*y[i] + b*z[i]
void cxpaypbzCuda(ParitySpinor x, double2 a, ParitySpinor y, double2 b, ParitySpinor z) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  setBlock(9, length, x.precision);
  blas_quda_bytes += 4*x.real_length*x.precision;
  blas_quda_flops += 8*x.real_length;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    cxpaypbzKernel<<<blasGrid, blasBlock>>>((double2*)x.spinor, a, (double2*)y.spinor, b, (double2*)z.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    cxpaypbzKernel<<<blasGrid, blasBlock>>>((float2*)x.spinor, af2, (float2*)y.spinor, bf2, (float2*)z.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    cxpaypbzHKernel<<<blasGrid, blasBlock>>>(af2, bf2, (short4*)z.spinor, (float*)z.spinorNorm, z.stride, z.volume);
  }
}

template <typename Float, typename Float2>
__global__ void axpyZpbxKernel(Float a, Float2 *x, Float2 *y, Float2 *z, Float b, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 x_i = make_Float2(x[i].x, x[i].y);
    y[i].x += a*x_i.x;
    y[i].y += a*x_i.y;
    x[i].x = z[i].x + b*x_i.x;
    x[i].y = z[i].y + b*x_i.y;
    i += gridSize;
  }
}

__global__ void axpyZpbxHKernel(float a, float b, short4 *xH, float *xN, short4 *yH, float *yN, int stride, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, stride);
    AXPY_FLOAT4(a, x0, y0);
    XPAY_FLOAT4(z0, b, x0);
    AXPY_FLOAT4(a, x1, y1);
    XPAY_FLOAT4(z1, b, x1);
    AXPY_FLOAT4(a, x2, y2);
    XPAY_FLOAT4(z2, b, x2);
    AXPY_FLOAT4(a, x3, y3);
    XPAY_FLOAT4(z3, b, x3);
    AXPY_FLOAT4(a, x4, y4);
    XPAY_FLOAT4(z4, b, x4);
    AXPY_FLOAT4(a, x5, y5);
    XPAY_FLOAT4(z5, b, x5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, stride);
    i += gridSize;
  }   
}


// performs the operations: {y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]}
void axpyZpbxCuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  setBlock(10, x.length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpyZpbxKernel<<<blasGrid, blasBlock>>>(a, (double2*)x.spinor, (double2*)y.spinor, (double2*)z.spinor, b, x.length/2);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpyZpbxKernel<<<blasGrid, blasBlock>>>((float)a, (float2*)x.spinor, (float2*)y.spinor, (float2*)z.spinor, (float)b, x.length/2);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    axpyZpbxHKernel<<<blasGrid, blasBlock>>>((float)a, (float)b, (short4*)x.spinor, (float*)x.spinorNorm,
					   (short4*)y.spinor, (float*)y.spinorNorm, z.stride, z.volume);
  }
  blas_quda_bytes += 5*x.real_length*x.precision;
  blas_quda_flops += 8*x.real_length;
}

template <typename Float2>
__global__ void caxpbypzYmbwKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, int len) {

  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 X = make_Float2(x[i].x, x[i].y);
    Float2 Z = make_Float2(z[i].x, z[i].y);

    Z.x += a.x*X.x - a.y*X.y;
    Z.y += a.y*X.x + a.x*X.y;

    Float2 Y = make_Float2(y[i].x, y[i].y);
    Z.x += b.x*Y.x - b.y*Y.y;
    Z.y += b.y*Y.x + b.x*Y.y;
    z[i] = make_Float2(Z.x, Z.y);

    Float2 W = make_Float2(w[i].x, w[i].y);

    Y.x -= b.x*W.x - b.y*W.y;
    Y.y -= b.y*W.x + b.x*W.y;	
    
    y[i] = make_Float2(Y.x, Y.y);
    i += gridSize;
  } 
}

__global__ void caxpbypzYmbwHKernel(float2 a, float2 b, short4 *yH, float *yN, short4 *zH, float *zN, int stride, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, stride);
    RECONSTRUCT_HALF_SPINOR(w, texHalf4, texNorm4, stride);
    CAXPBYPZ_FLOAT4(a, x0, b, y0, z0);
    CAXPBYPZ_FLOAT4(a, x1, b, y1, z1);
    CAXPBYPZ_FLOAT4(a, x2, b, y2, z2);
    CAXPBYPZ_FLOAT4(a, x3, b, y3, z3);
    CAXPBYPZ_FLOAT4(a, x4, b, y4, z4);
    CAXPBYPZ_FLOAT4(a, x5, b, y5, z5);
    CMAXPY_FLOAT4(b, w0, y0);
    CMAXPY_FLOAT4(b, w1, y1);
    CMAXPY_FLOAT4(b, w2, y2);
    CMAXPY_FLOAT4(b, w3, y3);
    CMAXPY_FLOAT4(b, w4, y4);
    CMAXPY_FLOAT4(b, w5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, stride);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
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
  setBlock(11, length, x.precision);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbypzYmbwKernel<<<blasGrid, blasBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, 
					  (double2*)z.spinor, (double2*)w.spinor, length); 
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbypzYmbwKernel<<<blasGrid, blasBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, 
					  (float2*)z.spinor, (float2*)w.spinor, length); 
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf4, w.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm4, w.spinorNorm, spinor_bytes/12); 
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbypzYmbwHKernel<<<blasGrid, blasBlock>>>(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm,
					       (short4*)z.spinor, (float*)z.spinorNorm, z.stride, z.volume);
  }
  blas_quda_bytes += 6*x.real_length*x.precision;
  blas_quda_flops += 12*x.real_length;  
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
template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) sumF##suffix
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

template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) sumH##suffix
#define REDUCE_TYPES Float *a, int stride
#define REDUCE_PARAMS a, stride
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, stride);			\
  SUM_FLOAT4(s0, I0);							\
  SUM_FLOAT4(s1, I1);							\
  SUM_FLOAT4(s2, I2);							\
  SUM_FLOAT4(s3, I3);							\
  SUM_FLOAT4(s4, I4);							\
  SUM_FLOAT4(s5, I5);							\
  s0 += s1; s2 += s3; s4 += s5; s0 += s2; s0 += s4;
#define REDUCE_OPERATION(i) (I0.x)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double sumCuda(ParitySpinor a) {
  blas_quda_flops += a.real_length;
  blas_quda_bytes += a.real_length*a.precision;
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return sumFCuda((double*)a.spinor, a.length, 12, a.precision);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return sumFCuda((float*)a.spinor, a.length, 12, a.precision);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    return sumHCuda((char*)0, a.stride, a.volume, 12, a.precision);
  }
}

//
// double normCuda(float *a, int n) {}
//
template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) normF##suffix
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

//
// double normHCuda(char *, int n) {}
//
template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) normH##suffix
#define REDUCE_TYPES Float *a, int stride // dummy type
#define REDUCE_PARAMS a, stride
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, stride);			\
  REAL_DOT_FLOAT4(norm0, I0, I0);					\
  REAL_DOT_FLOAT4(norm1, I1, I1);					\
  REAL_DOT_FLOAT4(norm2, I2, I2);					\
  REAL_DOT_FLOAT4(norm3, I3, I3);					\
  REAL_DOT_FLOAT4(norm4, I4, I4);					\
  REAL_DOT_FLOAT4(norm5, I5, I5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double normCuda(ParitySpinor a) {
  blas_quda_flops += 2*a.real_length;
  blas_quda_bytes += a.real_length*a.precision;
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return normFCuda((double*)a.spinor, a.length, 13, a.precision);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return normFCuda((float*)a.spinor, a.length, 13, a.precision);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    return normHCuda((char*)0, a.stride, a.volume, 13, a.precision);
  }
}



//
// double reDotProductFCuda(float *a, float *b, int n) {}
//
template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductF##suffix
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

//
// double reDotProductHCuda(float *a, float *b, int n) {}
//
template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductH##suffix
#define REDUCE_TYPES Float *a, Float *b, int stride
#define REDUCE_PARAMS a, b, stride
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(aH, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(bH, texHalf2, texNorm2, stride);		\
  REAL_DOT_FLOAT4(rdot0, aH0, bH0);					\
  REAL_DOT_FLOAT4(rdot1, aH1, bH1);					\
  REAL_DOT_FLOAT4(rdot2, aH2, bH2);					\
  REAL_DOT_FLOAT4(rdot3, aH3, bH3);					\
  REAL_DOT_FLOAT4(rdot4, aH4, bH4);					\
  REAL_DOT_FLOAT4(rdot5, aH5, bH5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_OPERATION(i) (rdot0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double reDotProductCuda(ParitySpinor a, ParitySpinor b) {
  blas_quda_flops += 2*a.real_length;
  checkSpinor(a, b);
  blas_quda_bytes += 2*a.real_length*a.precision;
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return reDotProductFCuda((double*)a.spinor, (double*)b.spinor, a.length, 14, a.precision);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return reDotProductFCuda((float*)a.spinor, (float*)b.spinor, a.length, 14, a.precision);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, b.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, b.spinorNorm, spinor_bytes/12);    
    return reDotProductHCuda((char*)0, (char*)0, a.stride, a.volume, 14, a.precision);
  }
}

//
// double axpyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = a*x[i] + y[i]
// Second returns the norm of y
//

template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormF##suffix
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

template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormH##suffix
#define REDUCE_TYPES Float a, short4 *yH, float *yN, int stride
#define REDUCE_PARAMS a, yH, yN, stride
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  AXPY_FLOAT4(a, x0, y0);						\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  AXPY_FLOAT4(a, x1, y1);						\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  AXPY_FLOAT4(a, x2, y2);						\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  AXPY_FLOAT4(a, x3, y3);						\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  AXPY_FLOAT4(a, x4, y4);						\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  AXPY_FLOAT4(a, x5, y5);						\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2; norm0 += norm4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double axpyNormCuda(double a, ParitySpinor x, ParitySpinor y) {
  blas_quda_flops += 4*x.real_length;
  checkSpinor(x,y);
  blas_quda_bytes += 3*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return axpyNormFCuda(a, (double*)x.spinor, (double*)y.spinor, x.length, 15, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return axpyNormFCuda((float)a, (float*)x.spinor, (float*)y.spinor, x.length, 15, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return axpyNormHCuda((float)a, (short4*)y.spinor, (float*)y.spinorNorm, x.stride, x.volume, 15, x.precision);
  }
}


//
// double xmyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = x[i] - y[i]
// Second returns the norm of y
//

template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormF##suffix
#define REDUCE_TYPES Float *x, Float *y
#define REDUCE_PARAMS x, y
#define REDUCE_AUXILIARY(i) y[i] = x[i] - y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <int reduce_threads, typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormH##suffix
#define REDUCE_TYPES Float *d1, Float *d2, short4 *yH, float *yN, int stride
#define REDUCE_PARAMS d1, d2, yH, yN, stride
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  XMY_FLOAT4(x0, y0);							\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  XMY_FLOAT4(x1, y1);							\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  XMY_FLOAT4(x2, y2);							\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  XMY_FLOAT4(x3, y3);							\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  XMY_FLOAT4(x4, y4);							\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  XMY_FLOAT4(x5, y5);							\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2; norm0 += norm4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double xmyNormCuda(ParitySpinor x, ParitySpinor y) {
  blas_quda_flops +=3*x.real_length;
  checkSpinor(x,y);
  blas_quda_bytes += 3*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return xmyNormFCuda((double*)x.spinor, (double*)y.spinor, x.length, 16, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return xmyNormFCuda((float*)x.spinor, (float*)y.spinor, x.length, 16, x.precision);
  } else { 
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return xmyNormHCuda((char*)0, (char*)0, (short4*)y.spinor, (float*)y.spinorNorm, y.stride, y.volume, 16, x.precision);
  }
}


//
// double2 cDotProductCuda(float2 *a, float2 *b, int n) {}
//
template <int reduce_threads, typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductF##suffix
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

template <int reduce_threads, typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductH##suffix
#define REDUCE_TYPES Float2 *a, Float b, int stride
#define REDUCE_PARAMS a, b, stride
#define REDUCE_REAL_AUXILIARY(i)					\
  RECONSTRUCT_HALF_SPINOR(aH, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(bH, texHalf2, texNorm2, stride);		\
  REAL_DOT_FLOAT4(rdot0, aH0, bH0);					\
  REAL_DOT_FLOAT4(rdot1, aH1, bH1);					\
  REAL_DOT_FLOAT4(rdot2, aH2, bH2);					\
  REAL_DOT_FLOAT4(rdot3, aH3, bH3);					\
  REAL_DOT_FLOAT4(rdot4, aH4, bH4);					\
  REAL_DOT_FLOAT4(rdot5, aH5, bH5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_IMAG_AUXILIARY(i)					\
  IMAG_DOT_FLOAT4(idot0, aH0, bH0);					\
  IMAG_DOT_FLOAT4(idot1, aH1, bH1);					\
  IMAG_DOT_FLOAT4(idot2, aH2, bH2);					\
  IMAG_DOT_FLOAT4(idot3, aH3, bH3);					\
  IMAG_DOT_FLOAT4(idot4, aH4, bH4);					\
  IMAG_DOT_FLOAT4(idot5, aH5, bH5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;
#define REDUCE_REAL_OPERATION(i) (rdot0)
#define REDUCE_IMAG_OPERATION(i) (idot0)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 cDotProductCuda(ParitySpinor x, ParitySpinor y) {
  blas_quda_flops += 4*x.real_length;
  checkSpinor(x,y);
  int length = x.length/2;
  blas_quda_bytes += 2*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    char c = 0;
    return cDotProductFCuda((double2*)x.spinor, (double2*)y.spinor, c, length, 17, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    char c = 0;
    return cDotProductFCuda((float2*)x.spinor, (float2*)y.spinor, c, length, 17, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductHCuda((char*)0, (char*)0, x.stride, x.volume, 17, x.precision);
  }
}

//
// double2 xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int n) {}
//
// First performs the operation y = x + a*y
// Second returns complex dot product (z,y)
//

template <int reduce_threads, typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) xpaycDotzyF##suffix
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

template <int reduce_threads, typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) xpaycDotzyH##suffix
#define REDUCE_TYPES Float a, short4 *yH, Float2 *yN, int stride
#define REDUCE_PARAMS a, yH, yN, stride
#define REDUCE_REAL_AUXILIARY(i)					\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, stride);		\
  XPAY_FLOAT4(x0, a, y0);						\
  XPAY_FLOAT4(x1, a, y1);						\
  XPAY_FLOAT4(x2, a, y2);						\
  XPAY_FLOAT4(x3, a, y3);						\
  XPAY_FLOAT4(x4, a, y4);						\
  XPAY_FLOAT4(x5, a, y5);						\
  REAL_DOT_FLOAT4(rdot0, z0, y0);					\
  REAL_DOT_FLOAT4(rdot1, z1, y1);					\
  REAL_DOT_FLOAT4(rdot2, z2, y2);					\
  REAL_DOT_FLOAT4(rdot3, z3, y3);					\
  REAL_DOT_FLOAT4(rdot4, z4, y4);					\
  REAL_DOT_FLOAT4(rdot5, z5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_IMAG_AUXILIARY(i)					\
  IMAG_DOT_FLOAT4(idot0, z0, y0);					\
  IMAG_DOT_FLOAT4(idot1, z1, y1);					\
  IMAG_DOT_FLOAT4(idot2, z2, y2);					\
  IMAG_DOT_FLOAT4(idot3, z3, y3);					\
  IMAG_DOT_FLOAT4(idot4, z4, y4);					\
  IMAG_DOT_FLOAT4(idot5, z5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);
#define REDUCE_REAL_OPERATION(i) (rdot0)
#define REDUCE_IMAG_OPERATION(i) (idot0)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 xpaycDotzyCuda(ParitySpinor x, double a, ParitySpinor y, ParitySpinor z) {
  blas_quda_flops += 6*x.real_length;
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  blas_quda_bytes += 4*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return xpaycDotzyFCuda((double2*)x.spinor, a, (double2*)y.spinor, (double2*)z.spinor, length, 18, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return xpaycDotzyFCuda((float2*)x.spinor, (float)a, (float2*)y.spinor, (float2*)z.spinor, length, 18, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    return xpaycDotzyHCuda((float)a, (short4*)y.spinor, (float*)y.spinorNorm, x.stride, x.volume, 18, x.precision);
  }
}


//
// double3 cDotProductNormACuda(float2 *a, float2 *b, int n) {}
//
template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormAF##suffix
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

template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormAH##suffix
#define REDUCE_TYPES Float2 *a, int stride
#define REDUCE_PARAMS a, stride
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  REAL_DOT_FLOAT4(norm0, x0, x0);					\
  REAL_DOT_FLOAT4(norm1, x1, x1);					\
  REAL_DOT_FLOAT4(norm2, x2, x2);					\
  REAL_DOT_FLOAT4(norm3, x3, x3);					\
  REAL_DOT_FLOAT4(norm4, x4, x4);					\
  REAL_DOT_FLOAT4(norm5, x5, x5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_Y_AUXILIARY(i)						\
  REAL_DOT_FLOAT4(rdot0, x0, y0);					\
  REAL_DOT_FLOAT4(rdot1, x1, y1);					\
  REAL_DOT_FLOAT4(rdot2, x2, y2);					\
  REAL_DOT_FLOAT4(rdot3, x3, y3);					\
  REAL_DOT_FLOAT4(rdot4, x4, y4);					\
  REAL_DOT_FLOAT4(rdot5, x5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_Z_AUXILIARY(i)						\
  IMAG_DOT_FLOAT4(idot0, x0, y0);					\
  IMAG_DOT_FLOAT4(idot1, x1, y1);					\
  IMAG_DOT_FLOAT4(idot2, x2, y2);					\
  IMAG_DOT_FLOAT4(idot3, x3, y3);					\
  IMAG_DOT_FLOAT4(idot4, x4, y4);					\
  IMAG_DOT_FLOAT4(idot5, x5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;  
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
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

double3 cDotProductNormACuda(ParitySpinor x, ParitySpinor y) {
  blas_quda_flops += 6*x.real_length;
  checkSpinor(x,y);
  int length = x.length/2;
  blas_quda_bytes += 2*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormAFCuda((double2*)x.spinor, (double2*)y.spinor, length, 19, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return cDotProductNormAFCuda((float2*)x.spinor, (float2*)y.spinor, length, 19, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductNormAHCuda((char*)0,  x.stride, x.volume, 19, x.precision);
  }
}

//
// double3 cDotProductNormBCuda(float2 *a, float2 *b, int n) {}
//
template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormBF##suffix
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

template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormBH##suffix
#define REDUCE_TYPES Float2 *a, int stride
#define REDUCE_PARAMS a, stride
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_Y_AUXILIARY(i)						\
  REAL_DOT_FLOAT4(rdot0, x0, y0);					\
  REAL_DOT_FLOAT4(rdot1, x1, y1);					\
  REAL_DOT_FLOAT4(rdot2, x2, y2);					\
  REAL_DOT_FLOAT4(rdot3, x3, y3);					\
  REAL_DOT_FLOAT4(rdot4, x4, y4);					\
  REAL_DOT_FLOAT4(rdot5, x5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_Z_AUXILIARY(i)						\
  IMAG_DOT_FLOAT4(idot0, x0, y0);					\
  IMAG_DOT_FLOAT4(idot1, x1, y1);					\
  IMAG_DOT_FLOAT4(idot2, x2, y2);					\
  IMAG_DOT_FLOAT4(idot3, x3, y3);					\
  IMAG_DOT_FLOAT4(idot4, x4, y4);					\
  IMAG_DOT_FLOAT4(idot5, x5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;  
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
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

double3 cDotProductNormBCuda(ParitySpinor x, ParitySpinor y) {
  blas_quda_flops += 6*x.real_length;
  checkSpinor(x,y);
  int length = x.length/2;
  blas_quda_bytes += 2*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormBFCuda((double2*)x.spinor, (double2*)y.spinor, length, 20, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return cDotProductNormBFCuda((float2*)x.spinor, (float2*)y.spinor, length, 20, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductNormBHCuda((char*)0, x.stride, x.volume, 20, x.precision);
  }
}


//
// double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormYF##suffix
#define REDUCE_TYPES Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, Float2 *u
#define REDUCE_PARAMS a, x, b, y, z, w, u
#define REDUCE_X_AUXILIARY(i)				\
  Float2 X = make_Float2(x[i].x, x[i].y);		\
  Float2 Y = make_Float2(y[i].x, y[i].y);		\
  Float2 W = make_Float2(w[i].x, w[i].y);		
#define REDUCE_Y_AUXILIARY(i)			\
  Float2 Z = make_Float2(z[i].x, z[i].y);	\
  Z.x += a.x*X.x - a.y*X.y;			\
  Z.y += a.y*X.x + a.x*X.y;			\
  Z.x += b.x*Y.x - b.y*Y.y;			\
  Z.y += b.y*Y.x + b.x*Y.y;			\
  Y.x -= b.x*W.x - b.y*W.y;			\
  Y.y -= b.y*W.x + b.x*W.y;	
#define REDUCE_Z_AUXILIARY(i)	      \
  z[i] = make_Float2(Z.x, Z.y);	      \
  y[i] = make_Float2(Y.x, Y.y);	      
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

//
// double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
template <int reduce_threads, typename Float2>
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormYH##suffix
#define REDUCE_TYPES Float2 a, Float2 b, short4 *yH, float *yN, short4 *zH, float *zN, int stride
#define REDUCE_PARAMS a, b, yH, yN, zH, zN, stride
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, stride);		\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, stride);		\
  RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, stride);		\
  CAXPBYPZ_FLOAT4(a, x0, b, y0, z0);					\
  CAXPBYPZ_FLOAT4(a, x1, b, y1, z1);					\
  CAXPBYPZ_FLOAT4(a, x2, b, y2, z2);					\
  CAXPBYPZ_FLOAT4(a, x3, b, y3, z3);					\
  CAXPBYPZ_FLOAT4(a, x4, b, y4, z4);					\
  CAXPBYPZ_FLOAT4(a, x5, b, y5, z5);					\
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, stride);			\
  RECONSTRUCT_HALF_SPINOR(w, texHalf4, texNorm4, stride);		\
  CMAXPY_FLOAT4(b, w0, y0);						\
  CMAXPY_FLOAT4(b, w1, y1);						\
  CMAXPY_FLOAT4(b, w2, y2);						\
  CMAXPY_FLOAT4(b, w3, y3);						\
  CMAXPY_FLOAT4(b, w4, y4);						\
  CMAXPY_FLOAT4(b, w5, y5);						\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, stride);			
#define REDUCE_Y_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(u, texHalf5, texNorm5, stride);		\
  REAL_DOT_FLOAT4(rdot0, u0, y0);					\
  REAL_DOT_FLOAT4(rdot1, u1, y1);					\
  REAL_DOT_FLOAT4(rdot2, u2, y2);					\
  REAL_DOT_FLOAT4(rdot3, u3, y3);					\
  REAL_DOT_FLOAT4(rdot4, u4, y4);					\
  REAL_DOT_FLOAT4(rdot5, u5, y5);					\
  IMAG_DOT_FLOAT4(idot0, u0, y0);					\
  IMAG_DOT_FLOAT4(idot1, u1, y1);					\
  IMAG_DOT_FLOAT4(idot2, u2, y2);					\
  IMAG_DOT_FLOAT4(idot3, u3, y3);					\
  IMAG_DOT_FLOAT4(idot4, u4, y4);					\
  IMAG_DOT_FLOAT4(idot5, u5, y5);					
#define REDUCE_Z_AUXILIARY(i)						\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4; \
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4; \
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4; 
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
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

// This convoluted kernel does the following: z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
double3 caxpbypzYmbwcDotProductWYNormYQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y,
					   ParitySpinor z, ParitySpinor w, ParitySpinor u) {
  blas_quda_flops += 18*x.real_length;
  checkSpinor(x,y);
  checkSpinor(x,z);
  checkSpinor(x,w);
  checkSpinor(x,u);
  int length = x.length/2;
  blas_quda_bytes += 7*x.real_length*x.precision;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return caxpbypzYmbwcDotProductWYNormYFCuda(a, (double2*)x.spinor, b, (double2*)y.spinor, (double2*)z.spinor, 
					       (double2*)w.spinor, (double2*)u.spinor, length, 21, x.precision);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    return caxpbypzYmbwcDotProductWYNormYFCuda(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, (float2*)z.spinor,
					       (float2*)w.spinor, (float2*)u.spinor, length, 21, x.precision);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf4, w.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm4, w.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf5, u.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm5, u.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    return caxpbypzYmbwcDotProductWYNormYHCuda(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm, 
					       (short4*)z.spinor, (float*)z.spinorNorm, y.stride, y.volume, 21, x.precision);
  }
}



double cpuDouble(float *data, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++) sum += data[i];
  return sum;
}

